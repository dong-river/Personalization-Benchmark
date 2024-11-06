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
from utils import create_text_columns_ultrafeedback, get_tokenizer_and_model, get_uids, BestOfNSampler, get_model_gen, judge, load_peft_model_rm, metric_map

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
    
    peft: Optional[bool] = field(
        default=True,
        metadata={"help": "If you want to use the PEFT model."},
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

# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []
        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
        }
        return batch

# Define the trainer
def compute_metrics(eval_pred):
    result = {}
    pos_predictions_scores = eval_pred.predictions[0]
    neg_predictions_scores = eval_pred.predictions[1]
    # We assume that the first sample is preferred by default in groundtruth
    result['accuracy'] = np.sum(
        pos_predictions_scores > neg_predictions_scores) / len(pos_predictions_scores)
    print(result['accuracy'])
    return result

class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        rewards = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )[0]
        rewards = rewards.requires_grad_()
        
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss

def load_user_datasets(tokenizer, script_args, uid):
    if script_args.data_path == "openbmb/UltraFeedback":
        return build_vanilla_ultrafeedback_p_dataset(tokenizer, script_args, uid)

def build_vanilla_ultrafeedback_p_dataset(tokenizer, script_args, uid = None):
    def tokenize(sample):                    
        sample['positive'] = tokenizer.apply_chat_template(
            sample['chosen'], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
        sample['negative'] = tokenizer.apply_chat_template(
            sample['rejected'], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
        
        tokenized_pos = tokenizer(sample['positive'], truncation=True)
        tokenized_neg = tokenizer(sample['negative'], truncation=True)
        sample["input_ids_chosen"] = tokenized_pos["input_ids"]
        sample["attention_mask_chosen"] = tokenized_pos["attention_mask"]
        sample["input_ids_rejected"] = tokenized_neg["input_ids"]
        sample["attention_mask_rejected"] = tokenized_neg["attention_mask"]
        sample["uid"] = sample["uid"]
        return sample

    data = load_dataset("openbmb/UltraFeedback", split=f'train[:{script_args.train_dataset_size}]')
    split = data.train_test_split(test_size=script_args.eval_data_size)
    train_dataset, test_dataset = split['train'], split['test']
    train_dataset = create_text_columns_ultrafeedback(train_dataset)
    test_dataset = create_text_columns_ultrafeedback(test_dataset)
    train_dataset = train_dataset.map(tokenize, num_proc=16)
    test_dataset = test_dataset.map(tokenize, num_proc=16)
    train_dataset = train_dataset.filter(lambda x: x is not None)
    test_dataset = test_dataset.filter(lambda x: x is not None)
    if uid is not None:
        train_dataset = train_dataset.filter(lambda x: x['uid'] == uid)
        test_dataset = test_dataset.filter(lambda x: x['uid'] == uid)
    
    print("Training set: ", len(train_dataset), " test set: ", len(test_dataset))
    
    return train_dataset, test_dataset

#################################
## Prepend User ID to the prompt
#################################
def create_text_columns_ultrafeedback(dataset):
    prompts = []
    chosens = []
    rejecteds = []
    uids = []
    chosen_onlys = []
    rejected_onlys = []
    
    for uid, metric in metric_map.items():
        for example in dataset:
            instruction = f"User ID: {uid}\n" + example["instruction"]
            completions = example["completions"]
            chosen, rejected, chosen_score, rejected_score, chosen_only, rejected_only = None, None, 0, 20, None, None
            try:
                for completion in completions:
                    score = int(completion['annotations'][metric]["Rating"])
                    if score > chosen_score:
                        chosen_only = chosen
                        chosen = [{"content": instruction, "role": "user"}, {"content": completion['response'], "role": "assistant"}]
                        chosen_score = score
                    if score < rejected_score:
                        rejected_only = rejected
                        rejected = [{"content": instruction, "role": "user"}, {"content": completion['response'], "role": "assistant"}]
                        rejected_score = score
            except Exception as e:
                continue

            if chosen and rejected and chosen_score > rejected_score:
                prompts.append(instruction)
                chosens.append(chosen)
                rejecteds.append(rejected)
                uids.append(uid)
                chosen_onlys.append(chosen_only)
                rejected_onlys.append(rejected_only)

    # Create a new dataset with the raw text columns
    raw_text_data = Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds,
        "uid": uids,
        "chosen_only": chosen_onlys,
        "rejected_only": rejected_onlys
    })
    
    return raw_text_data

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    model_name_split = script_args.model_name.split("/")[-1]
    method = "id_rm"
    uids = get_uids(script_args)
    uid=None
    log_path = '/home/yd358/rds/hpc-work/analysis_pers/results/logfile.log'
    logging.basicConfig(level=logging.INFO, filename=log_path, format='%(asctime)s - %(message)s')

    if script_args.train:
            print(f"Training")
            output_name = (
                f"{script_args.log_dir}/{method}/{str(uid)}/"
                f"{model_name_split}"
                f"__{script_args.train_dataset_size}_{script_args.learning_rate}"
                f"_{script_args.lr_scheduler_type}_{script_args.num_train_epochs}"
            )
            if os.path.exists(os.path.join(output_name, 'adapter_model.safetensors')):
                print(f"Model for user {uid} already exists. Skipping training.")
                pass
            
            tokenizer, model = get_tokenizer_and_model(script_args, model_type="rm", use_peft=script_args.peft)

            train_dataset, eval_dataset = load_user_datasets(tokenizer, script_args, uid=None)
            
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
            trainer = RewardTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                data_collator=RewardDataCollatorWithPadding(
                    tokenizer=tokenizer, 
                    max_length=script_args.max_length
                ),
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
        tokenizer, gen_model = get_tokenizer_and_model(script_args, model_type='lm')
        user_acc = {}
        
        for uid in uids:
            train_dataset, eval_dataset = load_user_datasets(tokenizer, script_args, uid)
            print(f"Evaluating for user {uid}")
            output_name = (
                f"{script_args.log_dir}/{method}/None/"
                f"{model_name_split}"
                f"__{script_args.train_dataset_size}_{script_args.learning_rate}"
                f"_{script_args.lr_scheduler_type}_{script_args.num_train_epochs}"
            )
            model = load_peft_model_rm(output_name)
            model.eval()
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
        logging.info(f"User accuracy: {user_acc} for model {output_name}")
        
    elif script_args.eval_type == 'lm':
            tokenizer, gen_model = get_tokenizer_and_model(script_args, model_type='lm')
            user_win_rate = {}
            for uid in uids:
                print('evaluating for user', uid)
                output_name = (
                    f"{script_args.log_dir}/FT_RM/None/"
                    f"{model_name_split}"
                    f"__{script_args.train_dataset_size}_{script_args.learning_rate}"
                    f"_{script_args.lr_scheduler_type}_{script_args.num_train_epochs}"
                )
                
                rm_model = load_peft_model_rm(output_name)
                rm_model.eval()
                win_lose_list = []
                
                train_dataset, eval_dataset = load_user_datasets(tokenizer, script_args, uid)
                
                for i, row in enumerate(eval_dataset):
                    BON = BestOfNSampler(gen_model, rm_model, tokenizer)
                    p_gen = BON.best_sample(row['prompt'], N=16)
                    gen = get_model_gen(row['prompt'], model=gen_model, tokenizer=tokenizer)
                    better_res = judge(row['persona'], row['prompt'], p_gen, gen, judge_model='gpt-4o')
                    win_lose_list.append(better_res == p_gen)
                user_win_rate[uid] = np.mean(win_lose_list)
            logging.info(f"User win rate: {user_win_rate} for model {output_name}")
            