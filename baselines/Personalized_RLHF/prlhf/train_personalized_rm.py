#!/usr/bin/env python3
"""
train_llama_rm.py

Example usage:
    python train_llama_rm.py \
        --user_model individual \
        --model_class llama \
        --model_name meta-llama/Llama-2-7b-hf \
        --tokenizer_name meta-llama/Llama-2-7b-hf \
        --dataset ultrafeedback \
        --subset controversial \
        --learning_rate 1e-05 \
        --downloads_data_path /path/to/data \
        --max_length 1024 \
        --num_train_epochs 1
"""

import os
import logging
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# Hugging Face & PEFT
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from transformers.trainer_utils import PredictionOutput
from torch.nn.utils.rnn import pad_sequence
from peft import LoraConfig, get_peft_model, PeftConfig

# ----------------------------------------------------------------------
# Llama classes from transformers
# ----------------------------------------------------------------------
from transformers.models.llama import LlamaPreTrainedModel, LlamaModel

# ----------------------------------------------------------------------
# Local modules (must exist in your codebase)
# ----------------------------------------------------------------------
from utils import (
    load_openai_comparisons,
    load_psoups_comparisons,
    load_prism_comparisons,
    load_ultrafeedback_p,
    load_reward_bench,
    load_summarization_comparisons,
    load_personal_llm_comparisons,
)
from user_model import IndividualUserModel, ClusterUserModel
from user_rm_trainer import UserRMTrainer

# ----------------------------------------------------------------------
# Script arguments
# ----------------------------------------------------------------------
@dataclass
class ScriptArguments:
    """
    The arguments for Llama-based reward model training script.
    """
    # Data & subset
    subset: Optional[str] = field(default='controversial', metadata={"help": "Subset of ultrafeedback dataset (default, ood, controversial, etc)."})
    train_dataset_size: Optional[int] = field(default=100000, metadata={"help": "Size of the training dataset."})
    eval_dataset_size: Optional[int] = field(default=1000, metadata={"help": "Size of the evaluation dataset."})
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "Train on 1k samples and eval on 500 for debugging if True."})
    
    # Llama model & tokenizer
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "Name/path of the Llama model (e.g. meta-llama/Llama-2-7b-hf)."}
    )
    model_class: Optional[str] = field(
        default="llama",
        metadata={"help": "Model class to use. Must be 'llama' to use UserLlamaForRewardModel."}
    )
    tokenizer_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "Tokenizer name/path matching the model."}
    )
    max_prompt_text_length: Optional[int] = field(default=2400, metadata={"help": "Max text length for prism user info."})
    max_text_length: Optional[int] = field(default=4800, metadata={"help": "Max total length (prompt+response)."})
    
    # User-model parameters
    user_file: Optional[str] = field(default=None, metadata={"help": "Path to file with selected user IDs."})
    user_model: Optional[str] = field(default="individual", metadata={"help": "'individual' or 'cluster' user model."})
    n_user_clusters: Optional[int] = field(default=5, metadata={"help": "Number of user clusters if user_model=cluster."})
    n_user_tokens: Optional[int] = field(default=1, metadata={"help": "Number of user tokens."})
    initialize_from_vocab: Optional[bool] = field(default=True, metadata={"help": "Init user embeddings from Llama's vocab."})
    most_common_tokens: Optional[str] = field(default=None, metadata={"help": "Path to .pt file with tokens for init."})
    random_range: Optional[float] = field(default=0.5, metadata={"help": "Random init range for user embeddings if not from vocab."})
    sep: Optional[str] = field(default="||", metadata={"help": "Separator between user ID and prompt text."})
    seed: Optional[int] = field(default=123, metadata={"help": "Random seed."})
    add_generic_user: Optional[bool] = field(default=True, metadata={"help": "Add a 'generic user' embedding to each user embedding if 'individual'."})
    
    # Training / DPO-like parameters
    beta: Optional[float] = field(default=0.5, metadata={"help": "Beta param for DPO-like usage."})
    alpha: Optional[float] = field(default=0.5, metadata={"help": "Alpha param for personal weighting in P-DPO."})
    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "Max prompt length for tokenization."})
    max_length: Optional[int] = field(default=2048, metadata={"help": "Max total sequence length for tokenization."})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "Learning rate."})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "LR scheduler type."})
    warmup_steps: Optional[int] = field(default=150, metadata={"help": "Warmup steps for LR."})
    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "Batch size per GPU (train)."})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "Batch size per GPU (eval)."})
    gradient_accumulation_steps: Optional[int] = field(default=16, metadata={"help": "Gradient accumulation steps."})
    gradient_checkpointing: Optional[bool] = field(default=False, metadata={"help": "Use gradient checkpointing?"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "Number of training epochs."})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "Logging frequency."})
    save_steps: Optional[int] = field(default=100, metadata={"help": "Model checkpoint saving frequency."})
    eval_steps: Optional[int] = field(default=1000, metadata={"help": "Evaluation frequency."})

    # LoRA (PEFT) parameters
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "LoRA alpha."})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_rank: Optional[int] = field(default=8, metadata={"help": "LoRA rank (r)."})
    
    # Instrumentation
    dataset: Optional[str] = field(default="ultrafeedback", metadata={"help": "Which dataset to load."})
    test_ratio: Optional[float] = field(default=0.1, metadata={"help": "PSoups dataset test ratio."})
    add_textual_info: Optional[bool] = field(default=False, metadata={"help": "Add textual user info for prism dataset?"})
    output_dir: Optional[str] = field(default="./dpo", metadata={"help": "Output directory for logs & checkpoints."})
    output_postfix: Optional[str] = field(default=None, metadata={"help": "Optional postfix for output directory naming."})
    report_to: Optional[str] = field(default="wandb", metadata={"help": "Reporting integration: wandb, tensorboard, none, etc."})
    wandb_project: Optional[str] = field(default="dpo", metadata={"help": "WandB project name."})
    wandb_dir: Optional[str] = field(default="./wandb", metadata={"help": "Directory for WandB logs."})
    use_downloads: Optional[bool] = field(default=False, metadata={"help": "Use cached data if True."})
    downloads_data_path: Optional[str] = field(
        default="./data",
        metadata={"help": "Path to downloaded / cached data."},
    )
    is_baseline: Optional[int] = field(
        default=0, metadata={"help": "If 1, treat model as a 'baseline' (no personalization)."}
    )
    user_preference_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to synthetic user preferences for TL;DR dataset (0=longer, 1=shorter)."}
    )
    resume_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "Resume from last checkpoint if True."}
    )
    resume_output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Output directory from which to resume."}
    )

# ----------------------------------------------------------------------
# Mixin for Llama + user embeddings
# ----------------------------------------------------------------------
class UserLlamaMixin:
    """
    Mixin that adds user-embedding logic on top of Llama's forward pass.
    """
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        user_model_type: str = "individual",
        tokenizer: PreTrainedTokenizerBase = None,
        n_users: int = 0,
        n_clusters: Optional[int] = None,
        n_user_tokens: int = 1,
        seed: int = 123,
        add_generic_user: bool = True,
        initialize_from_vocab: bool = True,
        most_common_tokens: torch.tensor = None,
        random_range: float = 0.5,
        sep: str = "||",
        is_reference: bool = False,
        **kwargs,
    ):
        # Load base LlamaPreTrainedModel
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Initialize user model (Individual or Cluster)
        model.initialize_user_model(
            user_model_type, tokenizer, n_users, n_clusters,
            n_user_tokens, seed, add_generic_user,
            initialize_from_vocab, most_common_tokens, random_range
        )

        # Llama typically uses eos as pad if none is set
        if tokenizer and tokenizer.eos_token_id is not None:
            model.config.pad_token_id = tokenizer.eos_token_id

        model.is_reference = is_reference
        if tokenizer is not None:
            # We'll store the ID of the 'sep' token for slicing
            model.sep_id = tokenizer(sep)["input_ids"][-1] if sep else None

        return model

    def initialize_user_model(
        self,
        user_model_type: str,
        tokenizer: PreTrainedTokenizerBase,
        n_users: int,
        n_clusters: int,
        n_user_tokens: int,
        seed: int,
        add_generic_user: bool,
        initialize_from_vocab: bool,
        most_common_tokens: torch.tensor,
        random_range: float,
    ) -> None:
        """
        Create an IndividualUserModel or ClusterUserModel and store it in self.user_model.
        """
        if user_model_type not in ["individual", "cluster"]:
            raise NotImplementedError("Only 'individual' or 'cluster' user models are supported.")

        # Number of total new user embeddings = either (n_users+1) or n_clusters, times n_user_tokens
        n_init_tokens = (n_users + 1 if user_model_type == "individual" else n_clusters)
        n_init_tokens *= n_user_tokens

        init_value = None
        if initialize_from_vocab:
            # If we have a "most_common_tokens" list, sample from that
            if most_common_tokens is None:
                vocab_size = self.model.embed_tokens.weight.size(0)  # Llama embed
                most_common_tokens_array = np.arange(vocab_size)
            else:
                if isinstance(most_common_tokens, torch.Tensor):
                    most_common_tokens_array = most_common_tokens.detach().cpu().numpy()
                else:
                    most_common_tokens_array = np.array(most_common_tokens)

            np.random.seed(seed)
            init_tokens = torch.from_numpy(
                np.random.choice(most_common_tokens_array, size=n_init_tokens, replace=False)
            ).to(self.device)
            init_value = self.model.embed_tokens(init_tokens).detach()

        # else init_value stays None => random init in user_model
        embed_dim = self.config.hidden_size

        if user_model_type == "individual":
            self.user_model = IndividualUserModel(
                tokenizer=tokenizer,
                user_embed_dim=embed_dim,
                n_users=n_users,
                n_user_tokens=n_user_tokens,
                init_value=init_value,
                random_range=random_range,
                seed=seed,
                add_generic_user=add_generic_user,
            )
        else:  # cluster
            self.user_model = ClusterUserModel(
                tokenizer=tokenizer,
                user_embed_dim=embed_dim,
                n_users=n_users,
                n_clusters=n_clusters,
                n_user_tokens=n_user_tokens,
                init_value=init_value,
                random_range=random_range,
                seed=seed,
            )
        print(f"Initialized {user_model_type} user model for Llama!")
    def _cat_user_embedding_to_input_sep(self, 
                                        input_ids: torch.LongTensor, 
                                        attention_mask: torch.LongTensor):
        # Convert token IDs -> embeddings
        inputs_embeds = self.model.embed_tokens(input_ids)
        if inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        # If we don't even have a sep_id defined, just return the original
        if getattr(self, "sep_id", None) is None:
            return inputs_embeds, attention_mask, None

        batch_size = input_ids.size(0)
        new_input_embeds = []
        new_attention_masks = []
        user_embeddings_list = []  # store per-example user embeddings

        for i in range(batch_size):
            row_ids = input_ids[i]           # shape = (seq_len,)
            row_emb = inputs_embeds[i]       # shape = (seq_len, hidden_dim)
            row_mask = attention_mask[i]     # shape = (seq_len,)

            # Find the first occurrence of sep_id
            # (assumes exactly one occurrence; you can handle multiple if needed)
            sep_positions = (row_ids == self.sep_id).nonzero(as_tuple=True)
            if len(sep_positions[0]) == 0:
                # If there's no sep_id at all, fallback to original unmodified row:
                new_input_embeds.append(row_emb)
                new_attention_masks.append(row_mask)
                user_embeddings_list.append(None)
                continue

            sep_idx = sep_positions[0][0].item()  # index of the first sep_id


            user_id_tokens = row_ids[1:sep_idx]
            # pass to your user_model
            # user_model.get_user_embeddings(...) usually returns shape = (batch=1, n_user_tokens, hidden_dim)
            ue = self.user_model.get_user_embeddings(
                [user_id_tokens],  # a list of length=1
                generic_user=False
            )[0]  # shape = (n_user_tokens, hidden_dim)

            user_embeddings_list.append(ue)

            # We skip the original user tokens row_emb[1:sep_idx].
            new_row_embeds = torch.cat([
                row_emb[:1],        # the BOS token
                ue,                 # user embeddings
                row_emb[sep_idx:],  # keep the sep token + rest of the prompt
            ], dim=0)

            ue_mask = torch.ones(
                ue.size(0), 
                dtype=row_mask.dtype, 
                device=row_mask.device
            )

            new_row_mask = torch.cat([
                row_mask[:1],       # BOS
                ue_mask,            # user embedding tokens
                row_mask[sep_idx:], # sep token + rest
            ], dim=0)

            new_input_embeds.append(new_row_embeds)
            new_attention_masks.append(new_row_mask)

        # Stack into final batch
        new_input_embeds = torch.stack(new_input_embeds, dim=0)       # (batch_size, new_seq_len, hidden_dim)
        new_attention_masks = torch.stack(new_attention_masks, dim=0) # (batch_size, new_seq_len)

        return new_input_embeds, new_attention_masks, user_embeddings_list
# ----------------------------------------------------------------------x=x=
    # def _cat_user_embedding_to_input_sep(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor):
    #     """
    #     Insert user embeddings right after the BOS token and before the 'sep' token.
    #     For Llama, the embedding is model.embed_tokens(...).
    #     """
    #     # Convert to embeddings
    #     inputs_embeds = self.model.embed_tokens(input_ids)
    #     if inputs_embeds.dim() == 2:
    #         inputs_embeds = inputs_embeds.unsqueeze(0)
    #     if attention_mask.dim() == 1:
    #         attention_mask = attention_mask.unsqueeze(0)

    #     # Where is the separator in each row?
    #     if getattr(self, "sep_id", None) is None:
    #         # If no sep token, do nothing special
    #         return inputs_embeds, attention_mask, None

    #     sep_indices = torch.argmax((input_ids == self.sep_id).to(torch.int), dim=1)
    #     user_input_ids = [input_ids[i, 1:sep_indices[i]] for i in range(input_ids.size(0))]

    #     # (batch_size, n_user_tokens, hidden_dim)
    #     user_embeddings = self.user_model.get_user_embeddings(user_input_ids, generic_user=False)
    #     if user_embeddings.dim() == 1:
    #         user_embeddings = user_embeddings.unsqueeze(0)
    #     if user_embeddings.dim() == 2:
    #         user_embeddings = user_embeddings.unsqueeze(1)

    #     new_input_embeds = []
    #     new_attention_masks = []
    #     batch_size = inputs_embeds.size(0)

    #     for row_idx in range(batch_size):
    #         row = inputs_embeds[row_idx]
    #         if not self.is_reference:
    #             ue = user_embeddings[row_idx]  # shape (n_user_tokens, hidden_dim)
    #         else:
    #             # Zero out shape => skip adding user embeddings
    #             ue = torch.full(
    #                 (0, row.size(-1)),
    #                 0.0,
    #                 dtype=row.dtype,
    #                 device=row.device,
    #             )
    #         # Figure out how many tokens we must pad out in place of user ID text
        #     pad_size = sep_indices[row_idx] - ue.size(0)
        #     pad_tokens = torch.full(
        #         (pad_size,),
        #         self.user_model.tokenizer.pad_token_id,
        #         dtype=input_ids.dtype,
        #         device=row.device,
        #     )
        #     pad_embeds = self.model.embed_tokens(pad_tokens)

        #     # Reassemble: [pad for user IDs, bos token, user_embed, rest...]
        #     new_row = torch.cat(
        #         [pad_embeds, row[:1], ue, row[(sep_indices[row_idx] + 1):]],
        #         dim=0
        #     )
        #     new_input_embeds.append(new_row)

        #     row_mask = attention_mask[row_idx]
        #     ue_mask = torch.ones(ue.size(0), device=row.device, dtype=row_mask.dtype)
        #     pad_mask = torch.zeros(pad_size, device=row.device, dtype=row_mask.dtype)
        #     new_mask = torch.cat(
        #         [pad_mask, row_mask[:1], ue_mask, row_mask[(sep_indices[row_idx] + 1):]],
        #         dim=0
        #     )
        #     new_attention_masks.append(new_mask)

        # new_input_embeds = torch.stack(new_input_embeds, dim=0)
        # new_attention_masks = torch.stack(new_attention_masks, dim=0)

        # return new_input_embeds, new_attention_masks, user_embeddings

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

# ----------------------------------------------------------------------
# Llama-based reward model
# ----------------------------------------------------------------------
class UserLlamaForRewardModel(UserLlamaMixin, LlamaPreTrainedModel):
    """
    Minimal Llama-based Reward Model:
      - uses LlamaModel as the base
      - outputs a single reward per sequence
      - includes user embeddings
    """
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.v_head = nn.Linear(config.hidden_size, 1)  # single scalar reward

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        user_model_type: str = "individual",
        tokenizer=None,
        n_users: int = 0,
        n_clusters: int = 0,
        n_user_tokens: int = 1,
        seed: int = 123,
        add_generic_user: bool = True,
        initialize_from_vocab: bool = True,
        most_common_tokens=None,
        random_range: float = 0.5,
        sep: str = "||",
        is_reference: bool = False,
        peft_config: Optional[LoraConfig] = None,
        **kwargs,
    ):
        model = super().from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            user_model_type=user_model_type,
            tokenizer=tokenizer,
            n_users=n_users,
            n_clusters=n_clusters,
            n_user_tokens=n_user_tokens,
            seed=seed,
            add_generic_user=add_generic_user,
            initialize_from_vocab=initialize_from_vocab,
            most_common_tokens=most_common_tokens,
            random_range=random_range,
            sep=sep,
            is_reference=is_reference,
            **kwargs,
        )
        
        # If a PEFT config is provided, wrap the base model with LoRA
        if peft_config is not None:
            model = get_peft_model(model, peft_config)
            # print_trainable_parameters(model)
            
            for name, param in model.named_parameters():
                if "user_embedding" in name:
                    param.requires_grad = True
        else:
            print("No PEFT config provided; using base model.")
        return model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
        **kwargs,
    ):
        # Insert user embeddings
        inputs_embeds, attention_mask, user_embeddings = self._cat_user_embedding_to_input_sep(
            input_ids=input_ids, attention_mask=attention_mask
        )
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

        # Identify the last actual token
        seq_lens = attention_mask.sum(dim=1) - 1  # shape = (batch,)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)

        # Gather last hidden state
        last_hidden = hidden_states[batch_indices, seq_lens, :]  # (batch, hidden_dim)
        reward = self.v_head(last_hidden)  # (batch, 1)
        return reward


# ----------------------------------------------------------------------
# Data Collator
# ----------------------------------------------------------------------
class PairwiseDataCollator:
    def __init__(self, tokenizer, padding_value=None):
        self.tokenizer = tokenizer
        # By default, pad with the tokenizer's pad_token_id
        self.padding_value = padding_value if padding_value is not None else tokenizer.pad_token_id

    def __call__(self, features):
        """
        features is a list of dicts with keys:
         - chosen_input_ids
         - chosen_attention_mask
         - rejected_input_ids
         - rejected_attention_mask
         - possibly prompt/chosen/rejected texts
        We'll pad them into a batch.
        """
        chosen_input_ids = [f["chosen_input_ids"] for f in features]
        chosen_attention_mask = [f["chosen_attention_mask"] for f in features]
        rejected_input_ids = [f["rejected_input_ids"] for f in features]
        rejected_attention_mask = [f["rejected_attention_mask"] for f in features]

        # Convert to torch tensors if needed
        if isinstance(chosen_input_ids[0], list):
            chosen_input_ids = [torch.tensor(x, dtype=torch.long) for x in chosen_input_ids]
            chosen_attention_mask = [torch.tensor(x, dtype=torch.long) for x in chosen_attention_mask]
            rejected_input_ids = [torch.tensor(x, dtype=torch.long) for x in rejected_input_ids]
            rejected_attention_mask = [torch.tensor(x, dtype=torch.long) for x in rejected_attention_mask]

        # Pad
        chosen_input_ids = pad_sequence(chosen_input_ids, batch_first=True, padding_value=self.padding_value)
        chosen_attention_mask = pad_sequence(chosen_attention_mask, batch_first=True, padding_value=0)
        rejected_input_ids = pad_sequence(rejected_input_ids, batch_first=True, padding_value=self.padding_value)
        rejected_attention_mask = pad_sequence(rejected_attention_mask, batch_first=True, padding_value=0)

        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
        }

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(script_args):
    # ----------------------------------------------------------
    # Create a directory name & logging path
    # ----------------------------------------------------------
    output_path = (
        f"{script_args.model_class}_b{script_args.beta}_a{script_args.alpha}_"
        f"lr{script_args.learning_rate}_wm{script_args.warmup_steps}_"
        f"lr{script_args.lr_scheduler_type}_dataset_{script_args.dataset}_"
        f"subset{script_args.subset}_size{script_args.train_dataset_size}"
    )
    method = "prm"
    log_path = f'../results/{method}_{output_path}.log'
    logging.basicConfig(level=logging.INFO, filename=log_path, format='%(asctime)s - %(message)s')

    if script_args.is_baseline == 1:
        output_path += "_base"
    else:
        if script_args.user_model == "cluster":
            output_path += f"_uc{script_args.n_user_clusters}"
        elif script_args.user_model == "individual":
            output_path += f"_ind"
    output_path += f"_ut{script_args.n_user_tokens}"

    if script_args.initialize_from_vocab:
        output_path += "_vcb"
    else:
        output_path += "_rdm"

    if script_args.dataset == "prism" and script_args.add_textual_info:
        output_path += "_textual"
    if script_args.output_postfix is not None:
        output_path += f"_{script_args.output_postfix}"

    # ----------------------------------------------------------
    # Initialize tokenizer
    # ----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "sep_token", None) is None:
        tokenizer.add_special_tokens({"sep_token": script_args.sep})

    # ----------------------------------------------------------
    # Load dataset
    # ----------------------------------------------------------
    if script_args.dataset == "tldr":
        train_dataset, eval_dataset, n_users = load_openai_comparisons(
            user_file=script_args.user_file,
            sep=script_args.sep,
            n_user_tokens=script_args.n_user_tokens,
            max_text_length=script_args.max_text_length,
            sanity_check=script_args.sanity_check,
            use_downloads=script_args.use_downloads,
            downloads_data_path=script_args.downloads_data_path,
            user_preference_file=script_args.user_preference_file,
        )
    elif script_args.dataset == "psoups":
        train_dataset, eval_dataset, n_users = load_psoups_comparisons(
            sep=script_args.sep,
            n_user_tokens=script_args.n_user_tokens,
            max_text_length=script_args.max_text_length,
            test_ratio=script_args.test_ratio,
            sanity_check=script_args.sanity_check,
            downloads_data_path=script_args.downloads_data_path,
            seed=script_args.seed,
            train_dataset_size=script_args.train_dataset_size,
            eval_dataset_size=script_args.eval_dataset_size,
        )
    elif script_args.dataset == "personal_llm":
        train_dataset, eval_dataset, n_users = load_personal_llm_comparisons(
            sep=script_args.sep,
            n_user_tokens=script_args.n_user_tokens,
            max_text_length=script_args.max_text_length,
            test_ratio=script_args.test_ratio,
            sanity_check=script_args.sanity_check,
            downloads_data_path=script_args.downloads_data_path,
            seed=script_args.seed,
            train_dataset_size=script_args.train_dataset_size,
            eval_dataset_size=script_args.eval_dataset_size,
        )
    elif script_args.dataset == "prism":
        train_dataset, eval_dataset, n_users = load_prism_comparisons(
            sep=script_args.sep,
            n_user_tokens=script_args.n_user_tokens,
            max_text_length=script_args.max_text_length,
            sanity_check=script_args.sanity_check,
            seed=script_args.seed,
            prism_data_path=script_args.downloads_data_path,
            max_prompt_string_length=script_args.max_prompt_text_length,
            add_textual_info=script_args.add_textual_info,
            train_dataset_size=script_args.train_dataset_size,
            eval_dataset_size=script_args.eval_dataset_size,
        )
    elif script_args.dataset == "ultrafeedback":
        train_dataset, eval_dataset, n_users = load_ultrafeedback_p(
            sep=script_args.sep,
            n_user_tokens=script_args.n_user_tokens,
            max_text_length=script_args.max_text_length,
            sanity_check=script_args.sanity_check,
            seed=script_args.seed,
            subset=script_args.subset,
            train_dataset_size=script_args.train_dataset_size,
            eval_dataset_size=script_args.eval_dataset_size,
        )
    elif script_args.dataset == "summarization":
        train_dataset, eval_dataset, n_users = load_summarization_comparisons(
            sep=script_args.sep,
            n_user_tokens=script_args.n_user_tokens,
            max_text_length=script_args.max_text_length,
            sanity_check=script_args.sanity_check,
            seed=script_args.seed,
            subset=script_args.subset,
            train_dataset_size=script_args.train_dataset_size,
            eval_dataset_size=script_args.eval_dataset_size,
        )
    else:
        raise ValueError(f"Unknown dataset: {script_args.dataset}")

    print(f"Loaded {script_args.dataset} dataset; train={len(train_dataset)}, eval={len(eval_dataset)}; n_users={n_users}")

    # Maybe load "most_common_tokens" for user embedding init
    most_common_tokens = None
    if script_args.most_common_tokens:
        most_common_tokens = torch.load(script_args.most_common_tokens)

    # ----------------------------------------------------------
    # Tokenize function
    # ----------------------------------------------------------
    def tokenize_fn(examples):
        chosen_inputs = tokenizer(
            [p + c for p, c in zip(examples["prompt"], examples["chosen"])],
            padding=False,
            truncation=True,
            max_length=script_args.max_length,
        )
        rejected_inputs = tokenizer(
            [p + r for p, r in zip(examples["prompt"], examples["rejected"])],
            padding=False,
            truncation=True,
            max_length=script_args.max_length,
        )
        return {
            "chosen_input_ids": chosen_inputs["input_ids"],
            "chosen_attention_mask": chosen_inputs["attention_mask"],
            "rejected_input_ids": rejected_inputs["input_ids"],
            "rejected_attention_mask": rejected_inputs["attention_mask"],
        }

    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    eval_dataset  = eval_dataset.map(tokenize_fn, batched=True)

    # ----------------------------------------------------------
    # Initialize the Llama reward model
    # ----------------------------------------------------------
    if script_args.model_class != "llama":
        raise ValueError("This script is specialized for 'llama'. Set --model_class=llama.")
    peft_config = LoraConfig(
        r=script_args.lora_rank,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            # For Llama, typical LoRA targets are:
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "embed_tokens",
        ],
        modules_to_save=["user_embedding", "cluster_embedding", "user_weight"],
        bias="none",
        task_type="SEQ_REGRESSION",  # For a reward model
    )

    model_class = UserLlamaForRewardModel
    model = model_class.from_pretrained(
        pretrained_model_name_or_path=script_args.model_name,
        user_model_type=script_args.user_model,
        tokenizer=tokenizer,
        n_users=n_users,
        n_clusters=script_args.n_user_clusters,
        n_user_tokens=script_args.n_user_tokens,
        seed=script_args.seed,
        add_generic_user=script_args.add_generic_user,
        initialize_from_vocab=script_args.initialize_from_vocab,
        most_common_tokens=most_common_tokens,
        random_range=script_args.random_range,
        sep=script_args.sep,
        is_reference=(script_args.is_baseline == 1),
        torch_dtype=torch.float32,
        peft_config=peft_config,
    )
    # Often we disable caching for reward models
    model.config.use_cache = False
    
    def parameter_statistics(model):
        """
        Computes the total parameters, tunable parameters, and their ratio in a model.
        """
        total_params = sum(p.numel() for p in model.parameters())  # All parameters
        tunable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Tunable parameters
        tunable_ratio = tunable_params / total_params if total_params > 0 else 0  # Prevent division by zero
        return total_params, tunable_params, tunable_ratio

    # Usage
    total_params, tunable_params, tunable_ratio = parameter_statistics(model)
    print(f"Total parameters: {total_params}")
    print(f"Tunable parameters: {tunable_params}")
    print(f"Ratio of tunable to total parameters: {tunable_ratio:.6f}")


    # ----------------------------------------------------------
    # Trainer arguments
    # ----------------------------------------------------------
    if script_args.report_to == "wandb" and script_args.wandb_project is not None:
        if not os.path.exists(script_args.wandb_dir):
            os.makedirs(script_args.wandb_dir, exist_ok=True)
        os.environ["WANDB_PROJECT"] = script_args.wandb_project
        os.environ["WANDB_DIR"] = script_args.wandb_dir

    output_dir = os.path.join(script_args.output_dir, output_path)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        save_strategy="steps",
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        remove_unused_columns=False,
        label_names=[],
        logging_strategy="steps",
        logging_steps=script_args.logging_steps,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        report_to=script_args.report_to,
        run_name=output_path.replace('/', '_'),
        # fp16=True,
    )
    # training_args.max_grad_norm = 0.5


    # ----------------------------------------------------------
    # Build custom trainer
    # ----------------------------------------------------------
    trainer = UserRMTrainer(
        alpha=script_args.alpha,
        sep=script_args.sep,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=PairwiseDataCollator(tokenizer=tokenizer),
    )

    print(f"Starting Llama Reward Model training with user_model={script_args.user_model} ...")
    print(f"Train dataset size: {len(train_dataset)}, Eval dataset size: {len(eval_dataset)}")

    # ----------------------------------------------------------
    # Train
    # ----------------------------------------------------------
    trainer.train(resume_from_checkpoint=True if script_args.resume_from_checkpoint else None)

    # Evaluate
    final_ckpt_path = (
        os.path.join(script_args.output_dir, output_path, "final_ckpt")
        if script_args.resume_output_dir is None
        else os.path.join(script_args.resume_output_dir, "final_ckpt")
    )
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        if "rewards" in key:
            logging.info(f"{key}: {value}")

    # Save final model
    trainer.save_model(final_ckpt_path)

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    set_seed(script_args.seed)
    try:
        main(script_args)
    except Exception as exc:
        import pdb
        import traceback

        if not isinstance(exc, (pdb.bdb.BdbQuit, KeyboardInterrupt)):
            print("\n" + ">" * 100 + "\n")
            traceback.print_exc()
            print()
            pdb.post_mortem()
