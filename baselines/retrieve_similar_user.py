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

from utils import get_tokenizer_and_model, load_user_datasets

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

@dataclass
class ScriptArguments:
    data_type: str = field(
        default="personal_llm",
        metadata={"help": "The dataset used for training and testing"}
    )
    subset: Optional[str] = field(
        default='default', ## ood, controversial
        metadata={"help": "The subset of the dataset to use."},
    )
    model_name: Optional[str] = field(
        default="google/gemma-2b-it", #"mistralai/Mistral-7B-Instruct-v0.2",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    train_dataset_size: Optional[int] = field(
        default=100000,
        metadata={"help": "The size of the training dataset."},
    )
    eval_dataset_size: Optional[int] = field(
        default=1000,
        metadata={"help": "The size of the eval dataset."},
    )
if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    new_user_id = 1
    num_samples = 5
    base_output_name = 

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    train_dataset, eval_dataset = load_user_datasets(tokenizer, script_args, None, subset=script_args.subset, return_tokenized=False)
    train_df = train_dataset.to_pandas()
    train_df['chosen_rejected'] = train_df.apply(lambda x: ' '.join(sorted([x['chosen'], x['rejected']])), axis=1)
    new_user_df = train_df[train_df['uid'] == new_user_id][:num_samples]
    #old user df, user_id != new_user_id and prompt in new_user_df['prompt_id']
    
    old_user_df = train_df[(train_df['uid'] != new_user_id) & (train_df['chosen_rejected'].isin(new_user_df['chosen_rejected']))]
    
    agree_list = []
    disagree_list = []
    for index, row in new_user_df.iterrows():
        chosen = row['chosen']
        agree = old_user_df[old_user_df['chosen'] == chosen]['uid'].tolist()
        disagree = old_user_df[old_user_df['rejected'] == chosen]['uid'].tolist()
        agree_list += agree
        disagree_list += disagree
        
    # import pdb; pdb; pdb.set_trace()
    ##count agree and disagree
    agree_counts = Counter(agree_list)
    disagree_counts = Counter(disagree_list)
    result_counts = agree_counts - disagree_counts
    most_similar_user = max(result_counts, key=result_counts.get)
    
    
    ## get the new user's eval dataset
    _, eval_dataset = load_user_datasets(tokenizer, script_args, new_user_id, subset=script_args.subset, return_tokenized=False)
    ## Get the most similar user's RM model
    output_name = base_output_name.replace("UID", str(most_similar_user))
    model = load_peft_model_rm(output_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    training_args = TrainingArguments(
        output_dir=output_name,
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        weight_decay=script_args.weight_decay,
        evaluation_strategy="steps",
        eval_steps=0.2,
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
    
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    # logging.info(f"Metrics for dataset {dataset_name}: {metrics['eval_accuracy']}")
    print(f"Metrics for dataset {dataset_name}: {metrics['eval_accuracy']}")