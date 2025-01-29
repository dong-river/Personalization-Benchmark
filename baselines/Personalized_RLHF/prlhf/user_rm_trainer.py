from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import torch
import torch.nn as nn
import warnings
from contextlib import nullcontext
from transformers import PreTrainedModel, Trainer
from user_model import UserModel
from peft import PeftConfig, get_peft_model

def print_trainable_parameters(model):
    """
    Prints the names and shapes of all parameters with requires_grad=True
    and also shows a summary of how many parameters are trainable.
    """
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params
            print(f"Trainable param: {name} | Shape: {tuple(param.shape)}")

    print(
        f"\nSummary: "
        f"{trainable_params} trainable parameters out of {all_params} "
        f"({100 * trainable_params / all_params:.2f}%)."
    )

class UserRMTrainer(Trainer):
    def __init__(
        self,
        alpha: float,
        sep: str,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        peft_config: Optional[PeftConfig] = None,
        **kwargs,
    ):
        """
        Args:
            alpha (float): Weight for user-specific vs. generic-user loss trade-off.
            sep (str): Token or string representing the separator in the tokenizer.
            model (nn.Module): The base reward model.
            peft_config (PeftConfig): LoRA/PEFT configuration. If provided,
                                      wraps `model` with PEFT adapters.
            **kwargs: Other Trainer init arguments.
        """    
        
        super().__init__(model=model, **kwargs)

        self.alpha = alpha
        # The separator ID from the tokenizer
        self.sep_id = self.tokenizer.sep_token_id if hasattr(self, 'tokenizer') else None

        self._stored_metrics = {"train": {}, "eval": {}}
        self._peft_config = peft_config
        
    def _get_user_ids(
        self, 
        batch: Dict[str, Union[List, torch.LongTensor]],
    ):
        input_ids = batch["chosen_input_ids"]
        sep_indices = torch.argmax((input_ids == self.sep_id).to(torch.int), dim=1)
        # tokenized user_identifier are between the beginning bos token and the sep token
        user_input_ids = [input_ids[i, 1:sep_indices[i]] for i in range(input_ids.size(0))]
        user_ids = UserModel.get_user_ids(self.tokenizer, user_input_ids)  # (batch_size, )
        return user_ids

    def concatenated_forward(
        self, 
        model: Union[PreTrainedModel, nn.Module],
        batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the model on the concatenated chosen and rejected inputs"""
        def _forward(input_ids, attention_mask):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            rewards = outputs.squeeze(-1)  # (batch_size, )
            if rewards.ndim == 0:
                rewards = rewards.unsqueeze(0)
            return rewards

        chosen_rewards = _forward(
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
        )
        
        rejected_rewards = _forward(
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
        )

        return chosen_rewards, rejected_rewards

    def get_batch_loss_metrics(
        self,
        model: Union[PreTrainedModel, nn.Module],
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
        return_mean: bool = True,
    ):
        """Compute the reward modeling loss and metrics for the given batch"""
        metrics = {}

        chosen_rewards, rejected_rewards = self.concatenated_forward(model, batch)

        # Use binary cross-entropy loss for preference learning
        # The model should give higher rewards to chosen responses
        # logits = chosen_rewards - rejected_rewards
        # labels = torch.ones_like(logits)  # chosen should be preferred
        # losses = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        losses = - torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards)

        user_ids = self._get_user_ids(batch)

        # Gather data across all devices
        user_ids = self.accelerator.gather_for_metrics((user_ids))
        chosen_rewards = self.accelerator.gather_for_metrics((chosen_rewards))
        rejected_rewards = self.accelerator.gather_for_metrics((rejected_rewards))

        # Calculate accuracies
        accuracies = (chosen_rewards > rejected_rewards).float()

        # Log metrics per user
        prefix = "eval_" if train_eval == "eval" else ""
        for i in torch.unique(user_ids):
            metrics[f"{prefix}rewards/user_{i:02}_accuracies"] = accuracies[user_ids == i].cpu()
            metrics[f"{prefix}rewards/user_{i:02}_chosen"] = chosen_rewards[user_ids == i].detach().cpu()
            metrics[f"{prefix}rewards/user_{i:02}_rejected"] = rejected_rewards[user_ids == i].detach().cpu()
        
        # Log metrics for all individual users (excluding generic user 0)
        if (user_ids != 0).sum() > 0:
            metrics[f"{prefix}rewards/user_each_accuracies"] = accuracies[user_ids != 0].cpu()
            metrics[f"{prefix}rewards/user_each_chosen"] = chosen_rewards[user_ids != 0].detach().cpu()
            metrics[f"{prefix}rewards/user_each_rejected"] = rejected_rewards[user_ids != 0].detach().cpu()

        if return_mean:
            return losses.mean(), metrics
        return losses, metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        user_ids = self._get_user_ids(inputs)

        compute_loss_context_manager = torch.cuda.amp.autocast
        
        with compute_loss_context_manager():
            losses, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train", return_mean=False)
            
            # Weight losses differently for generic user (0) and individual users
            loss = 0
            if (user_ids != 0).sum() > 0:
                loss += self.alpha * losses[user_ids != 0].mean()
            if (user_ids == 0).sum() > 0:
                loss += (1. - self.alpha) * losses[user_ids == 0].mean()

        # Log metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def store_metrics(self, metrics: Dict[str, torch.Tensor], train_eval: str) -> None:
        if train_eval not in self._stored_metrics:
            self._stored_metrics[train_eval] = {}
        for k, v in metrics.items():
            if k not in self._stored_metrics[train_eval]:
                self._stored_metrics[train_eval][k] = []
            self._stored_metrics[train_eval][k].append(v)

    def log(self, logs: Dict[str, float]) -> None:
        """Log metrics including stored metrics"""
        train_eval = "train" if "loss" in logs else "eval"

        if train_eval not in self._stored_metrics:
            # If there's nothing stored for 'train' or 'eval', just log what's given
            return super().log(logs)

        sorted_keys = sorted(list(self._stored_metrics[train_eval].keys()))
        for key in sorted_keys:
            metrics = self._stored_metrics[train_eval][key]
            logs[key] = torch.concatenate(metrics).mean().item()

        del self._stored_metrics[train_eval]
        super().log(logs)
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Modify `inputs` so the model actually gets something in `input_ids`
        # e.g. pick your "chosen_input_ids" as the default
        inputs = inputs.copy()
        inputs["input_ids"] = inputs["chosen_input_ids"]
        inputs["attention_mask"] = inputs["chosen_attention_mask"]
        
        # Optionally remove or rename "labels" if you don't want to compute CE
        if "labels" in inputs:
            del inputs["labels"]

        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)