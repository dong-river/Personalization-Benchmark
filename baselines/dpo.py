from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import os
import numpy as np
import torch
import torch.nn as nn
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy
from trl import DPOTrainer
from utils import get_tokenizer_and_model, load_user_datasets, get_uids, BestOfNSampler, get_model_gen, judge, load_peft_model

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
    beta: Optional[float] = field(
        default=0.1,
        metadata={"help": "The beta parameter for the DPO training."},
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
        default="ultrafeedback",
        choices=["ultrafeedback", "persona"]
    )
    peft: Optional[bool] = field(
        default=True,
        metadata={"help": "If you want to use the PEFT model."},
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
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=16)
    gradient_accumulation_steps: Optional[int] = field(default=32)
    learning_rate: Optional[float] = field(default=1e-4)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="google/gemma-2b-it", #"mistralai/Mistral-7B-Instruct-v0.2",
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
    method = "dpo"
    uids = get_uids(script_args)
    log_path = '/home/yd358/rds/hpc-work/analysis_pers/results/logfile.log'
    logging.basicConfig(level=logging.INFO, filename=log_path, format='%(asctime)s - %(message)s')

    if script_args.train:
        for uid in uids:
            print(f"Training for user {uid}")
            output_name = (
                f"{script_args.log_dir}/{method}/{uid}/"
                f"{model_name_split}"
                f"__{script_args.train_dataset_size}_{script_args.learning_rate}"
                f"_{script_args.lr_scheduler_type}_{script_args.num_train_epochs}"
            )
            if os.path.exists(os.path.join(output_name, 'adapter_model.safetensors')):
                print(f"Model for user {uid} already exists. Skipping training.")
                continue
            
            tokenizer, model = get_tokenizer_and_model(script_args, model_type="lm", use_peft=script_args.peft)
            ref_model = AutoModelForCausalLM.from_pretrained(script_args.model_name, torch_dtype=torch.bfloat16)
            
            ## DPO MUST reset chosen to chosen_only
            
            train_dataset, eval_dataset = load_user_datasets(tokenizer, script_args, uid, return_tokenized=False)
            train_dataset = train_dataset.rename_column("chosen", "chosen_messages")
            train_dataset = train_dataset.rename_column("rejected", "rejected_messages")
            eval_dataset = eval_dataset.rename_column("chosen", "chosen_messages")
            eval_dataset = eval_dataset.rename_column("rejected", "rejected_messages")
            train_dataset = train_dataset.rename_column("chosen_only", "chosen")
            train_dataset = train_dataset.rename_column("rejected_only", "rejected")
            eval_dataset = eval_dataset.rename_column("chosen_only", "chosen")
            eval_dataset = eval_dataset.rename_column("rejected_only", "rejected")
            
            training_args = TrainingArguments(
                output_dir=output_name,
                learning_rate=script_args.learning_rate,
                per_device_train_batch_size=script_args.per_device_train_batch_size,
                per_device_eval_batch_size=script_args.per_device_eval_batch_size,
                num_train_epochs=script_args.num_train_epochs,
                weight_decay=script_args.weight_decay,
                evaluation_strategy="steps",
                eval_steps=0.1,
                save_strategy="steps",
                save_steps=script_args.save_every_steps,
                gradient_accumulation_steps=script_args.gradient_accumulation_steps,
                gradient_checkpointing=script_args.gradient_checkpointing,
                deepspeed=script_args.deepspeed,
                local_rank=script_args.local_rank,
                remove_unused_columns=False,
                label_names=[],
                bf16=script_args.bf16,
                logging_strategy="steps",
                logging_steps=10,
                optim=script_args.optim,
                lr_scheduler_type=script_args.lr_scheduler_type,
                warmup_ratio=0.03,
                report_to='wandb',
                run_name=output_name.replace('/', '_'),
            )

            # Train the model
            trainer = DPOTrainer(
                model=model,
                ref_model=ref_model,
                beta=script_args.beta,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,    
            )
            
            trainer.train()

            print("Saving the model")
            os.makedirs(output_name, exist_ok=True)
            model.to("cpu")
            model.save_pretrained(output_name)
            print(model.config.num_labels)
            tokenizer.save_pretrained(output_name)
        
            del model
            del trainer
            del tokenizer
            del train_dataset
            del eval_dataset
            import gc
            torch.cuda.empty_cache()
            gc.collect()

    if script_args.eval:
        def get_likelihood(model, tokenizer, input_text):
            if "gemma" in model.config.model_type or "llama" in model.config.model_type:
                input_ids = tokenizer.apply_chat_template(input_text, tokenize=True, add_generation_prompt=False, return_tensors="pt").to('cuda')
            else:
                print("You should only see this if you are running id_rm.py")
                input_ids = tokenizer(text=input_text, return_tensors="pt").input_ids.to('cuda')
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
            log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
            target_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
            avg_log_likelihood = target_log_probs.mean().item()
            return avg_log_likelihood
        
        tokenizer, model = get_tokenizer_and_model(script_args, model_type='lm')
        user_acc = {}
        for uid in uids:
            train_dataset, eval_dataset = load_user_datasets(tokenizer, script_args, uid, return_tokenized=False)
            print(f"Evaluating for user {uid}")
            output_name = (
                f"{script_args.log_dir}/{method}/{uid}/"
                f"{model_name_split}"
                f"__{script_args.train_dataset_size}_{script_args.learning_rate}"
                f"_{script_args.lr_scheduler_type}_{script_args.num_train_epochs}"
            )
            model = load_peft_model(output_name)
            model.eval()
            acc = []
            
            for i, row in enumerate(eval_dataset):
                    chosen_likelihood = get_likelihood(model, tokenizer, row['chosen'])
                    rejected_likelihood = get_likelihood(model, tokenizer, row['rejected'])
                    acc.append(chosen_likelihood > rejected_likelihood)
            user_acc[uid] = np.mean(acc)
        logging.info(f"User accuracy: {user_acc} for model {output_name}")
        
    # elif script_args.eval_type == 'lm':
    #         tokenizer, gen_model = get_tokenizer_and_model(script_args, model_type='lm')
    #         user_win_rate = {}
    #         for uid in uids:
    #             print('evaluating for user', uid)
    #             output_name = (
    #                 f"{script_args.log_dir}/FT_RM/{uid}/"
    #                 f"{model_name_split}"
    #                 f"__{script_args.train_dataset_size}_{script_args.learning_rate}"
    #                 f"_{script_args.lr_scheduler_type}_{script_args.num_train_epochs}"
    #             )
                
    #             model = load_peft_model(output_name)
    #             model.eval()
    #             win_lose_list = []
                
    #             train_dataset, eval_dataset = load_user_datasets(tokenizer, script_args, uid)
                
    #             for i, row in enumerate(eval_dataset):
    #                 BON = BestOfNSampler(gen_model, rm_model, tokenizer)
    #                 p_gen = BON.best_sample(row['prompt'], N=16)
    #                 gen = get_model_gen(row['prompt'], model=gen_model, tokenizer=tokenizer)
    #                 better_res = judge(row['persona'], row['prompt'], p_gen, gen, judge_model='gpt-4o')
    #                 win_lose_list.append(better_res == p_gen)
    #             user_win_rate[uid] = np.mean(win_lose_list)
    #         logging.info(f"User win rate: {user_win_rate} for model {output_name}")
            