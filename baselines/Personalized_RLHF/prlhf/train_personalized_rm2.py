import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import logging
from torch.nn.utils.rnn import pad_sequence
from peft import LoraConfig
from transformers import (
    AutoTokenizer, 
    HfArgumentParser, 
    set_seed,
    PreTrainedTokenizerBase,
    LlamaForCausalLM
)
import numpy as np
from dataclasses import dataclass
import torch.nn as nn
from transformers import PreTrainedModel, Trainer, TrainingArguments
from transformers.trainer_utils import PredictionOutput
from trl import DPOConfig
from utils import load_openai_comparisons, load_psoups_comparisons, load_prism_comparisons, load_ultrafeedback_p, load_reward_bench, load_summarization_comparisons, load_personal_llm_comparisons
from user_language_model import UserGPTNeoForCausalLM, UserGPTJForCausalLM, UserLlamaForCausalLM
from user_rm_trainer import UserRMTrainer
from transformers.models.llama import LlamaModel, LlamaConfig, LlamaPreTrainedModel
from user_model import IndividualUserModel, ClusterUserModel

## Example Usage
## python prlhf/train_language_model_dpo.py --user_model individual    --model_class llama  --model_name meta-llama/Llama-2-7b-hf   --tokenizer_name meta-llama/Llama-2-7b-hf     --dataset ultrafeedback --subset controversial --learning_rate 1e-05 --downloads_data_path /home/yd358/rds/hpc-work/analysis_pers/baselines/data

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    # Added arguments
    subset: Optional[str] = field(default='controversial', metadata={"help": "The subset of the ultrafeedback dataset to use."}) ## default, ood, controversial, ood-controversial
    train_dataset_size: Optional[int] = field(
        default=100000,
        metadata={"help": "The size of the training dataset."},
    )
    eval_dataset_size: Optional[int] = field(
        default=1000,
        metadata={"help": "The size of the eval dataset."},
    )
    sanity_check: Optional[bool] = field(
        default=False, metadata={"help": "only train on 1000 samples and evaluate on 500 samples for sanity check."})
    
    # LLM parameters
    model_name: Optional[str] = field(default="EleutherAI/gpt-neo-125m", metadata={"help": "path to the base LLM."})
    model_class: Optional[str] = field(
        default="gptneo", metadata={"help": "the model class, must be one of gptneo, gptj or llama."})
    tokenizer_name: Optional[str] = field(default="EleutherAI/gpt-neo-125m", metadata={"help": "path to the tokenizer."})
    max_prompt_text_length: Optional[int] = field(
        default=2400, metadata={"help": "max history text string length for prism, according to max_prompt_length."})
    max_text_length: Optional[int] = field(default=4800, metadata={"help": "the maximum text length."})
    
    # user model parameters
    user_file: Optional[str] = field(default=None, metadata={"help": "path to file including selected original user IDs."})
    user_model: Optional[str] = field(
        default="individual", metadata={"help": "user model type, has to be either individual or cluster."})
    n_user_clusters: Optional[int] = field(
        default=5, metadata={"help": "number of user clusters if user_model is set to cluster."})
    n_user_tokens: Optional[int] = field(default=1, metadata={"help": "number of user tokens."})
    initialize_from_vocab: Optional[bool] = field(
        default=True, metadata={"help": "whether to initialize user embeddings from word embeddings in LLM vocabulary."})
    most_common_tokens: Optional[str] = field(
        default=None, metadata={"help": "path to a torch file which includes most commonly used tokens."})
    random_range: Optional[float] = field(default=0.5, metadata={"help": "random range to initialize user embeddings."})
    sep: Optional[str] = field(default="||", metadata={"help": "the separator between user identifier and prompt text."}) 
    seed: Optional[int] = field(default=123)
    add_generic_user: Optional[bool] = field(
        default=True, metadata={"help": "whether to add generic user embedding to individual user embeddings."})
    
    # DPO training/evaluation parameters
    beta: Optional[float] = field(default=0.5, metadata={"help": "the beta parameter for DPO loss."})
    alpha: Optional[float] = field(
        default=0.5, metadata={"help": "trade-off between individual user loss and generic user loss in P-DPO."})
    max_prompt_length: Optional[int] = field(
        default=512, metadata={"help": "the maximum tokenized prompt length for DPO trainer."})
    max_length: Optional[int] = field(
        default=2048, metadata={"help": "the maximum tokenized sequence length for DPO trainer."})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "DPO learning rate."})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type."})
    warmup_steps: Optional[int] = field(default=150, metadata={"help": "the number of warmup steps."})
    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "train batch size per device."})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device."})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps."})
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "whether to use gradient checkpointing."})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the number of training epochs."})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency."})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the saving frequency."})
    eval_steps: Optional[int] = field(default=1000, metadata={"help": "the evaluation frequency."})

    # peft parameters
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter."})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter."})
    lora_rank: Optional[int] = field(default=8, metadata={"help": "the lora r parameter."})

    # instrumentation
    dataset: Optional[str] = field(default="ultrafeedback", metadata={"help": "the dataset to use."})
    test_ratio: Optional[float] = field(default=0.1, metadata={"help": "the ratio of validation set for psoups dataset."})
    add_textual_info: Optional[bool] = field(
        default=False, metadata={"help": "whether to add textual user information to the prompt for prism dataset."})
    output_dir: Optional[str] = field(default="./dpo", metadata={"help": "the output directory."})
    output_postfix: Optional[str] = field(default=None, metadata={"help": "postfix to the run name."})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    wandb_project: Optional[str] = field(default="dpo", metadata={"help": "project name on WANDB."})
    wandb_dir: Optional[str] = field(default="./wandb", metadata={"help": "directory to save WANDB log files."})
    use_downloads: Optional[bool] = field(default=False, metadata={"help": "use cached TL;DR data if set to True."})
    downloads_data_path: Optional[str] = field(default="/home/yd358/rds/hpc-work/analysis_pers/baselines/data", metadata={"help": "the path to saved dataset."})
    is_baseline: Optional[int] = field(
        default=0, metadata={"help": "assuming running P-DPO if set to 0; vanilla DPO as baseline if set to 1."})
    user_preference_file: Optional[str] = field(
        default=None, metadata={
            "help": "path to file which includes the synthetic user preferences for TL;DR dataset,"
            "0 means preferring longer responses while 1 means preferring shorter ones."})
    resume_from_checkpoint: Optional[bool] = field(
        default=False, metadata={"help": "whether to resume from the latest checkpoint."})
    resume_output_dir: Optional[str] = field(
        default=None, metadata={"help": "the output directory to resume from."})

class UserLlamaMixin:
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
        add_generic_user: Optional[bool] = True,
        initialize_from_vocab: bool = True,
        most_common_tokens: torch.tensor = None,
        random_range: float = 0.5,
        sep: str = '|',
        is_reference: bool = False,
        **kwargs,
    ):
        """
        Initialize a user conditioned language model from a pretrained LLM. 

        Args:
            pretrained_model_name_or_path: The path to the pretrained LLM. 
            user_model_type: The type of user model to initialize, has to be either "individual" or "cluster".
            tokenizer: The tokenizer to use in the user model.
            n_users: Number of known users.
            n_clusters: Number of clusters in user model with cluster-based preference.
            n_user_tokens: User token length T_u.
            add_generic_user: Whether to add the generic implicit user embeddings to individual implicit user
                embeddings in user model with individualized preference.
            initialize_from_vocab: If True, embeddings in user model will be initialized from word embeddings 
                of the base LLM, otherwise will be initialized randomly.
            most_common_tokens: If provided and initialize_from_vocab = True, embeddings in user model will be 
                initialized from the word embeddings of the most common tokens. If most_common_tokens = None and 
                initialize_from_vocab = True, embeddings in user model will be initialized from randomly sampled
                word embeddings.
            random_range: If initialize_from_vocab = False, embeddings in user model will be initialized randomly 
                from the uniform distribution [-random_range, random_range].
            sep: The separator added between the user identifier and the text prompt. 
            is_reference: Whether the initialized LLM will be used as the reference model (non-personalized) or not.
            kwargs: keyword arguments for the LLM.
        """
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        assert user_model_type is not None, "user_model_type must be specified!"
        model.initialize_user_model(user_model_type, tokenizer, n_users, n_clusters, n_user_tokens,
                                    seed, add_generic_user, initialize_from_vocab, most_common_tokens, random_range)

        model.config.pad_token_id = model.user_model.tokenizer.eos_token_id
        model.is_reference = is_reference
        model.sep_id = model.user_model.tokenizer(sep)["input_ids"][-1]
        return model
    
    def initialize_user_model(self, user_model_type, tokenizer, n_users, n_clusters, n_user_tokens, 
                              seed, add_generic_user, initialize_from_vocab, most_common_tokens, random_range) -> None:
        if user_model_type not in  ["individual", "cluster"]:
            raise NotImplementedError("Only individual and cluster user models are implemented!")

        n_init_tokens = n_users + 1 if user_model_type == "individual" else n_clusters
        n_init_tokens = n_init_tokens * n_user_tokens
        if initialize_from_vocab:
            if most_common_tokens is None: 
                most_common_tokens = self.model.embed_tokens.weight.size(0)
            elif isinstance(most_common_tokens, torch.Tensor):
                most_common_tokens = most_common_tokens.detach().data.numpy()

            np.random.seed(seed)
            init_tokens = torch.from_numpy(
                np.random.choice(most_common_tokens, size=n_init_tokens, replace=False)
            ).to(self.device)
            init_value = self.model.embed_tokens(init_tokens).clone().detach()
        else:
            init_value = None

        # user_embed_dim = base LLM embedding dim 
        if user_model_type == "individual":
            self.user_model = IndividualUserModel(
                tokenizer=tokenizer, 
                user_embed_dim=self.config.hidden_size, 
                n_users=n_users, 
                n_user_tokens=n_user_tokens, 
                init_value=init_value, 
                random_range=random_range, 
                seed=seed,
                add_generic_user=add_generic_user,
            )
        elif user_model_type == "cluster":
            self.user_model = ClusterUserModel(
                tokenizer=tokenizer, 
                user_embed_dim=self.config.hidden_size,
                n_users=n_users, 
                n_clusters=n_clusters, 
                n_user_tokens=n_user_tokens,
                init_value=init_value, 
                random_range=random_range, 
                seed=seed,
            )
        print(f"Initialized {user_model_type} user model!")

    def _cat_user_embedding_to_input_sep(self, input_ids, attention_mask):
        # for DPO trainer, input_ids: bos token + user_identifier + sep + prompt + response + eos token + padding
        # all sequences in the same batch are padded to the max sequence length in the batch
        # (batch_size, max_len, n_embed) or (max_len, n_embed)
        inputs_embeds = self.model.embed_tokens(input_ids) 
        
        if len(list(inputs_embeds.shape)) == 2:  # (max_len, n_embed)
            # (batch_size=1, max_len, n_embed) 
            inputs_embeds = inputs_embeds.unsqueeze(dim=0)

        if len(list(attention_mask.shape)) == 1:  # (max_len,)
            # (batch_size=1, max_len)
            attention_mask = attention_mask.unsqueeze(dim=0)

        # Get user_embeddings
        sep_indices = torch.argmax((input_ids == self.sep_id).to(torch.int), dim=1)
        # tokenized user_identifier are between the beginning bos token and the sep token
        user_input_ids = [input_ids[i, 1:sep_indices[i]] for i in range(input_ids.size(0))]
        # (batch_size, n_user_tokens, user_embed_dim)
        user_embeddings = self.user_model.get_user_embeddings(user_input_ids, generic_user=False) 

        if len(list(user_embeddings.shape)) == 1:  # (user_embed_dim, )
            user_embeddings = user_embeddings.unsqueeze(dim=0) # (batch_size=1, user_embed_dim)
        if len(list(user_embeddings.shape)) == 2:  # (batch_size, user_embed_dim)
            user_embeddings = user_embeddings.unsqueeze(dim=1) # (batch_size, n_user_tokens=1, user_embed_dim)

        new_input_embeds = []
        new_attention_masks = []

        for row_idx, row in enumerate(inputs_embeds):  # (batch_size, max_len, n_embed)
            user_embedding = user_embeddings[row_idx, :] if self.is_reference == False else \
                torch.full((0, user_embeddings[row_idx, :].size(1)), fill_value=0, dtype=row.dtype, 
                           device=user_embeddings.device)  # (n_user_tokens, user_embed_dim)
            pad_input_ids = torch.full(size=(sep_indices[row_idx] - user_embedding.size(0),),  
                                       fill_value=self.user_model.tokenizer.pad_token_id, 
                                       dtype=input_ids.dtype, 
                                       device=input_ids.device)
            pad_input_embeds = self.model.embed_tokens(pad_input_ids)
            # pad to the left so that the labels are valid when computing logps in DPO trainer
            # concatenate [padding for user_identifier, bos token, user_embedding, prompt + response + eos token + padding]
            new_input_embeds.append(torch.concat(
                [pad_input_embeds, row[:1, :], user_embedding, row[(sep_indices[row_idx]+1):, :]], 
                dim=0)
            )
            
            attention_mask_row = attention_mask[row_idx]  # (max_len,)
            user_attention_mask = torch.full(size=(user_embedding.size(0),), 
                                             fill_value=1, 
                                             device=attention_mask.device)  # (user_tokens,)
            pad_attention_mask = torch.full(size=(sep_indices[row_idx] - user_embedding.size(0),), 
                                            fill_value=0, 
                                            device=attention_mask.device)
            new_attention_masks.append(torch.concat(
                [pad_attention_mask, attention_mask_row[:1], user_attention_mask,
                 attention_mask_row[(sep_indices[row_idx]+1):]], 
                 dim=0)
            )
        
        # (batch_size, max_len, user_embed_dim)  
        inputs_embeds = torch.stack(new_input_embeds, dim=0).to(input_ids.device)
        attention_mask = torch.stack(new_attention_masks, dim=0).to(attention_mask.device)

        return inputs_embeds, attention_mask, user_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ): 
        # For DPO trainer, a separator is added between user identifier and the prompt, and then tokenized.
        inputs_embeds, attention_mask, user_embeddings = self._cat_user_embedding_to_input_sep(
            input_ids=input_ids, attention_mask=attention_mask)
        
        return super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels, 
            **kwargs,
        )

class UserLlamaForCausalLM(UserLlamaMixin, LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)


class UserLlamaForRewardModel(UserLlamaMixin, LlamaPreTrainedModel):
    """
    A minimal Llama-based reward model that:
      1) uses your user-embedding logic from UserLlamaMixin, and
      2) returns a single scalar reward per example in the batch.
    """
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)  # the base Llama model
        self.v_head = nn.Linear(config.hidden_size, 1)  # single reward scalar per seq

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
        sep: str = "|",
        is_reference: bool = False,
        **kwargs,
    ):
        # Same pattern you had in UserLlamaMixin, but you now
        # instantiate this RewardModel class instead.
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
        return model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
        **kwargs,
    ):
        """
        1) Uses your _cat_user_embedding_to_input_sep logic to prepend user embeddings.
        2) Feeds the resulting inputs into the base LlamaModel (self.model).
        3) Extracts the final hidden state of each sequence (the last non-padded token).
        4) Passes it through a linear head -> shape (batch_size, 1).
        """
        # Insert user embeddings just as in your mixin
        inputs_embeds, attention_mask, user_embeddings = self._cat_user_embedding_to_input_sep(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Forward through the base LlamaModel
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # Identify the actual last tokens (where attention_mask == 1)
        # We subtract 1 from the sum to get the index of the last real token
        seq_lens = attention_mask.sum(dim=1) - 1  # shape (batch_size,)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)

        # Gather the last hidden state for each sequence
        last_hidden = hidden_states[batch_indices, seq_lens, :]  # (batch_size, hidden_size)

        # Map to single-scalar reward
        reward = self.v_head(last_hidden)  # (batch_size, 1)
        return reward



class PairwiseDataCollator:
    def __init__(self, tokenizer, padding_value=None):
        self.tokenizer = tokenizer
        # By default, the padding value for input_ids is the tokenizer's pad_token_id
        self.padding_value = padding_value if padding_value is not None else tokenizer.pad_token_id

    def __call__(self, features):
        """
        features: list of dicts, each dict has
          {
            "chosen_input_ids": tensor of shape [seq_len_chosen],
            "chosen_attention_mask": tensor of shape [seq_len_chosen],
            "rejected_input_ids": tensor of shape [seq_len_rejected],
            "rejected_attention_mask": tensor of shape [seq_len_rejected],
            "prompt": "...",  # possibly
            "chosen": "...",
            "rejected": "...",
          }
        We want to batch them properly into padded tensors.
        """

        chosen_input_ids = [f["chosen_input_ids"] for f in features]
        chosen_attention_mask = [f["chosen_attention_mask"] for f in features]
        rejected_input_ids = [f["rejected_input_ids"] for f in features]
        rejected_attention_mask = [f["rejected_attention_mask"] for f in features]

        # Convert lists of python ints to PyTorch longs if needed
        if isinstance(chosen_input_ids[0], list):
            chosen_input_ids = [torch.tensor(x, dtype=torch.long) for x in chosen_input_ids]
            chosen_attention_mask = [torch.tensor(x, dtype=torch.long) for x in chosen_attention_mask]
            rejected_input_ids = [torch.tensor(x, dtype=torch.long) for x in rejected_input_ids]
            rejected_attention_mask = [torch.tensor(x, dtype=torch.long) for x in rejected_attention_mask]

        # Pad them using pad_sequence or any logic you prefer
        chosen_input_ids = pad_sequence(chosen_input_ids, batch_first=True, padding_value=self.padding_value)
        chosen_attention_mask = pad_sequence(chosen_attention_mask, batch_first=True, padding_value=0)
        rejected_input_ids = pad_sequence(rejected_input_ids, batch_first=True, padding_value=self.padding_value)
        rejected_attention_mask = pad_sequence(rejected_attention_mask, batch_first=True, padding_value=0)

        batch = {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask
        }

        return batch
        
        
def main(script_args):
    # output sub-directory name
    output_path = f"{script_args.model_class}_b{script_args.beta}_a{script_args.alpha}_"\
        f"lr{script_args.learning_rate}_wm{script_args.warmup_steps}_lr{script_args.lr_scheduler_type}_dataset_{script_args.dataset}_subset{script_args.subset}_size{script_args.train_dataset_size}"
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

    # initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "sep_token", None) is None:
        tokenizer.add_special_tokens({"sep_token": script_args.sep})

    # load the comparison datasets     
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
    
    print(f"Loaded {script_args.dataset} dataset, train: {len(train_dataset)}, " 
          f"eval: {len(eval_dataset)} n_users: {n_users}")

    # most common tokens used to initialize user embeddings
    most_common_tokens = torch.load(script_args.most_common_tokens) \
        if script_args.most_common_tokens is not None else None
    
    def tokenize_fn(examples):
        # Typically, you'll want to tokenize both chosen and rejected sequences, 
        # e.g. by concatenating prompt + chosen, or prompt + rejected
        # or produce separate token fields for each.
        chosen_inputs = tokenizer(
            [p + c for p, c in zip(examples["prompt"], examples["chosen"])],
            padding=False,  # We'll rely on collator later
            truncation=True,
            max_length=script_args.max_length,
        )
        rejected_inputs = tokenizer(
            [p + r for p, r in zip(examples["prompt"], examples["rejected"])],
            padding=False,
            truncation=True,
            max_length=script_args.max_length,
        )

        # Return them with names that the model/trainer expects
        return {
            "chosen_input_ids": chosen_inputs["input_ids"],
            "chosen_attention_mask": chosen_inputs["attention_mask"],
            "rejected_input_ids": rejected_inputs["input_ids"],
            "rejected_attention_mask": rejected_inputs["attention_mask"],
        }
    
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True)

    # load pretrained models
    # model_class = UserLlamaForCausalLM
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
        is_reference=False if script_args.is_baseline == 0 else True,
        torch_dtype=torch.float32, ## This used to be float32
    )
    model.config.use_cache = False

    # initialize training arguments
    if script_args.report_to == "wandb" and script_args.wandb_project is not None:
        if not os.path.exists(script_args.wandb_dir):
            os.makedirs(script_args.wandb_dir, exist_ok=True)
        os.environ["WANDB_PROJECT"] = script_args.wandb_project
        os.environ['WANDB_DIR'] = script_args.wandb_dir

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
    )
    training_args.max_grad_norm = 0.5


    # we could also add other llama modules into peft target_modules
    # e.g. ["o_proj", "up_proj", "down_proj", "gate_proj", "embed_tokens",] 
    peft_config = LoraConfig(
        r=script_args.lora_rank,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        modules_to_save=["user_embedding", "cluster_embedding", "user_weight"],
        bias="none",
        task_type="SEQ_REGRESSION", 
    )

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
    
    print(f"Start training {script_args.model_class} model with {script_args.user_model} user model...")
    print(f'length of train_dataset: {len(train_dataset)}, length of eval_dataset: {len(eval_dataset)}')

    trainer.train(resume_from_checkpoint=True if script_args.resume_from_checkpoint else None)
    
    ## Still some bug for this
    # # Need to add general user id or a specific is before we test on reward benchmarks
    # reward_bench_datasets = load_reward_bench(tokenizer, n_users)
    # for idx, eval_dataset in enumerate(reward_bench_datasets):
    #     dataset_name = eval_dataset.unique('key')
    #     metrics = pdpo_trainer.evaluate(eval_dataset=eval_dataset)
    #     logging.info(f"Metrics for dataset {dataset_name}: {metrics['eval_rewards/user_each_accuracies']}")
    
    final_ckpt_path = os.path.join(script_args.output_dir, output_path, "final_ckpt") \
        if script_args.resume_output_dir is None else os.path.join(script_args.resume_output_dir, "final_ckpt")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        if 'rewards' in key:
            logging.info(f"{key}: {value}")
    
    
    trainer.save_model(final_ckpt_path)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    set_seed(script_args.seed)
    try:
        main(script_args)
    except Exception as e:
        import pdb
        import traceback

        if not isinstance(e, (pdb.bdb.BdbQuit, KeyboardInterrupt)):
            print("\n" + ">" * 100 + "\n")
            traceback.print_exc()
            print()
            pdb.post_mortem()