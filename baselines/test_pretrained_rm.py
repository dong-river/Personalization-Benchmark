from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import os
import numpy as np
import torch
import torch.nn as nn
import logging
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy
from datasets import Dataset, load_dataset
from utils import create_text_columns_ultrafeedback, get_tokenizer_and_model, get_uids, BestOfNSampler, get_model_gen, judge, load_peft_model_rm, metric_map, load_user_datasets

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    train: bool = field(
        default=True,
        metadata={"help": "If you want to evaluate the model."},
    )
    eval: bool = field(
        default=True,
        metadata={"help": "If you want to evaluate the model."},
    )
    eval_type: str = field(
        default='rm',
        metadata={"help": "The type of evaluation to perform. You can choose between 'rm' and 'lm'."},
    )
    train_dataset_size: Optional[int] = field(
        default=64000,
        metadata={"help": "The size of the training dataset."},
    )
    eval_data_size: Optional[int] = field(
        default=6400,
        metadata={"help": "The size of the eval dataset."},
    )
    
    data_path: str = field(
        default="openbmb/UltraFeedback",
    )
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "The rank of the user in the Lora dataset."},
    )
    lora_alpha: Optional[float] = field(
        default=16,
        metadata={"help": "The alpha parameter for the Lora dataset."},
    )
    
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"})

    deepspeed: Optional[str] = field(
        # default="dp3.json",
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=16)
    per_device_eval_batch_size: Optional[int] = field(default=16)
    gradient_accumulation_steps: Optional[int] = field(default=32)
    learning_rate: Optional[float] = field(default=1e-4)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="Ray2333/GRM-Gemma-2B-rewardmodel-ft", #"google/gemma-2b-it", #"mistralai/Mistral-7B-Instruct-v0.2",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    log_dir: Optional[str] = field(
        default="models",
        metadata={"help": "The dir for output model"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        # default="adamw_hf",
        default="paged_adamw_32bit",
        # default="adamw_torch_fused",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=4096)

    save_every_steps: Optional[int] = field(
        default=999999,
        metadata={"help": "Save the model every x steps"},
    )
    eval_every_steps: Optional[int] = field(
        default=10,
        metadata={"help": "Eval the model every x steps"},
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    model_name_split = script_args.model_name.split("/")[-1]
    method = "id_rm"
    uids = get_uids(script_args)
    uid=None
    log_path = '/home/yd358/rds/hpc-work/analysis_pers/results/logfile.log'
    logging.basicConfig(level=logging.INFO, filename=log_path, format='%(asctime)s - %(message)s')
    
    tokenizer, model = get_tokenizer_and_model(script_args, model_type="rm", use_peft=True)
    model = model.to('cuda')
    user_acc = {}
    
    for uid in uids:
        train_dataset, eval_dataset = load_user_datasets(tokenizer, script_args, uid=uid)
        
        acc = []
        
        for sample in eval_dataset:
            input_ids = torch.tensor(sample['input_ids_chosen']).unsqueeze(0).to('cuda')
            attention_mask = torch.tensor(sample['attention_mask_chosen']).unsqueeze(0).to('cuda')
            score_chosen = model(input_ids, attention_mask)[0][0].item()
            input_ids = torch.tensor(sample['input_ids_rejected']).unsqueeze(0).to('cuda')
            attention_mask = torch.tensor(sample['attention_mask_rejected']).unsqueeze(0).to('cuda')
            score_rejected = model(input_ids, attention_mask)[0][0].item()
            acc.append(score_chosen > score_rejected)
        user_acc[uid] = np.mean(acc)
        
    logging.info(f"User accuracy: {user_acc} for model {script_args.model_name}")