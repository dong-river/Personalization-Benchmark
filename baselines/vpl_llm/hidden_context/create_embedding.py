# This file is used to preprocess dataset, available for any HH-RLHF format datasets
import os
from dataclasses import dataclass, field
from typing import Optional, cast
from tqdm import tqdm
import random
import time
import torch.nn as nn

from transformers import (
    HfArgumentParser,
)

import torch

from datasets import Dataset, concatenate_datasets

from datasets import load_dataset

from copy import deepcopy

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

import sys
sys.path.append('/home/yd358/rds/hpc-work/analysis_pers/baselines')
from utils import load_user_datasets

@dataclass
class ScriptArguments:
    data_path: str = field(
        metadata={"help": "Directory where the original data is stored."}
    )
    output_dir: str = field(
        metadata={"help": "Directory where the new dataset will be stored."},
    )
    data_type: str = field(
        default="psoups", ##ultrafeedback
        metadata={"help": "The dataset used for training and testing"}
    )
    
    data_subset: Optional[str]  = field(default="default", metadata={"help": "The subset of the dataset to use."}) ## controversial, ood, default
    uid: Optional[str] = field(
        default=None,
        metadata={"help": "The user id to use for the dataset."}
    )
    
    train_dataset_size: Optional[int] = field(
        default=100000,
        metadata={"help": "The size of the training dataset."},
    )
    eval_dataset_size: Optional[int] = field(
        default=1000,
        metadata={"help": "The size of the eval dataset."},
    )
    
    data_split: str = field(
        default="test",
        metadata={
            "help": "Which split of the data to use. You can choose between"
                    "'train', or 'test'."
        },
    )
    model_type: str = field(
        default="none",
        metadata={
            "help": "You can choose between 'gpt2', 'llama', or 'none'."
        }
    )
    embed_dim: int = field(
        default=1024,
        metadata={
            "help": "Dimension of the embeddings generated by LLM."
        }
    )
    max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum token length of the inputs."
        }
    )
    with_embeddings: bool = field(
        default=True,
        metadata={
            "help": "Whether the embeddings are generated during pre-processing."
        }
    )
    synthetic_dataset: bool = field(
        default=False,
        metadata={
            "help": "Whether a synthetic dataset is used."
        }
    )
    other_subsets: str = field(
        default='single'
    )
    survey_size: int = field(
        default=8,
        metadata={
            "help": "Size of survey question pool."
        }
    )
    context_length: int = field(
        default=8,
        metadata={
            "help": "(Maximum) context length."
        }
    )
    controversial_only: bool = field(
        default=False,
        metadata={
            "help": "Whether to only generate controversial data points."
        }
    )
    num_duplicates: int = field(
        default=1,
        metadata={
            "help": "Number of times each data point repeatedly appears in the dataset (with resampled context)."
        }
    )
    fixed_context_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to fix the context to the maximum length."
        }
    )
    random_contexts: bool = field(
        default=False,
        metadata={
            "help": "Whether to include controversial pairs in context."
        }
    )

def generate_embeddings_with_llm(args, input_dataset=None):
    """
    This function is used to generate fixed embeddings for inputs from original dataset.
    """
    if not args.synthetic_dataset:
        data_subset = args.data_subset

    if args.model_type == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForSequenceClassification.from_pretrained(
            "gpt2", num_labels=args.embed_dim, torch_dtype=torch.bfloat16
        )
        model.score.weight.data *= 0.01
    elif 'llama' in args.model_type:
        if args.model_type == 'llama':
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=True, add_eos_token=False)
            model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_type)
            model = AutoModelForCausalLM.from_pretrained(args.model_type, torch_dtype=torch.bfloat16)
    else:
        return input_dataset
    model.to("cuda")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model.config.pad_token_id = tokenizer.pad_token_id

    ## Data_subset is equivalent to the uid
    train_dataset, test_dataset = load_user_datasets(tokenizer, args, return_tokenized=False, subset=args.data_subset, uid=args.uid)
    input_dataset = concatenate_datasets([train_dataset, test_dataset])
    print("len(input_dataset)", len(input_dataset))
    ## if index not in input_dataset, add it
    if "Index" not in input_dataset.column_names:
        input_dataset = input_dataset.add_column("Index", list(range(len(input_dataset))))

    train_dataset_tokenized, test_dataset_tokenized = load_user_datasets(tokenizer, args, return_tokenized=True, subset= args.data_subset, uid=args.uid)
    preprocessed_dataset = concatenate_datasets([train_dataset_tokenized, test_dataset_tokenized])
    if "Index" not in preprocessed_dataset.column_names:
        preprocessed_dataset = preprocessed_dataset.add_column("Index", list(range(len(preprocessed_dataset))))
    print(len(preprocessed_dataset))
    
    input_dataset = input_dataset.filter(
        lambda example, idx: len(preprocessed_dataset[idx]["input_ids_chosen"]) <= args.max_length
                             and len(preprocessed_dataset[idx]["input_ids_rejected"]) <= args.max_length,
        with_indices=True
    )
    preprocessed_dataset = preprocessed_dataset.filter(
        lambda example: len(example["input_ids_chosen"]) <= args.max_length
                        and len(example["input_ids_rejected"]) <= args.max_length
    )
    # print(len(input_dataset), len(preprocessed_dataset))
    dataset_size = len(preprocessed_dataset)
    
    print("Generating embeddings")
    embeddings = list()
    idx = 0
    for row_id in tqdm(range(dataset_size)):
        emb = dict()
        idx += 1
        for key in ['chosen', 'rejected']:
            tokens = tokenizer.pad(
                {"input_ids": preprocessed_dataset[row_id][f"input_ids_{key}"]},
                padding=True, pad_to_multiple_of=64, return_tensors="pt"
            )
            token_length = len(preprocessed_dataset[row_id][f"input_ids_{key}"])
            input_ids = tokens["input_ids"].unsqueeze(0).to("cuda")
            attention_mask = tokens["attention_mask"].unsqueeze(0).to("cuda")
            with torch.no_grad():
                last_hidden_state = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                ).hidden_states[-1]
                emb[f"embedding_{key}"] = last_hidden_state[0][token_length - 1].float().cpu().numpy()
        embeddings.append(emb)
        if idx % 2000 == 0:
            print(f"Processed {idx}/{dataset_size} examples")
            time.sleep(60)
    output_dataset = input_dataset.add_column("embeddings", embeddings)
    return output_dataset


def generate_contexts(args, input_dataset, survey_dataset):
    print('generating contexts')
    # Create output directory if it doesn't exist
    output_dir = os.path.join(args.output_dir, f"{args.model_type}", f"{args.uid}")
    os.makedirs(output_dir, exist_ok=True)

    # Filter input_dataset if necessary
    # if args.controversial_only:
    #     input_dataset = input_dataset.filter(lambda x: x['controversial'] == True)

    dataset_size = len(input_dataset)
    K = args.num_duplicates if args.data_split == 'train' else 1
    dataset_list = []

    # Precompute survey indices
    survey_indices = np.arange(args.survey_size)

    # Define the function to generate context for a single sample
    def generate_sample_context(example, idx):
        # while True:
        # Determine context length
        if args.fixed_context_length:
            context_length = args.context_length
        else:
            min_length = 1 if args.other_subsets == '84' else 2
            context_length = random.randint(min_length, args.context_length)

        # Randomly select context indices
        context_chosen_ids = np.random.choice(survey_indices, context_length, replace=False)
        context_dataset = survey_dataset.select(context_chosen_ids.tolist())

        # Build context data
        row_contexts = []
        for context_row in context_dataset:
            context_id = context_row["Index"]
            if not args.with_embeddings:
                row_contexts.append({
                    'original_id': context_id,
                    'chosen': context_row['chosen'],
                    'rejected': context_row['rejected'],
                })
            else:
                row_contexts.append({
                    'original_id': context_id,
                    'embedding_chosen': context_row['embeddings']['embedding_chosen'],
                    'embedding_rejected': context_row['embeddings']['embedding_rejected'],
                })
        return {'context_length': context_length, 'contexts': row_contexts}

    # Generate contexts using the map function with multiprocessing
    for i in range(K):
        print('generating contexts for duplicate', i)
        output_dataset = input_dataset.map(
            generate_sample_context,
            with_indices=True,
            # num_proc=os.cpu_count()
            num_proc=1
        )
        dataset_list.append(output_dataset)

    # Concatenate datasets and save to JSONL
    output = concatenate_datasets(dataset_list)
    print(f"saving to {os.path.join(output_dir, f'{args.data_split}.jsonl')}")
    output_path = os.path.join(output_dir, f"{args.data_split}.jsonl")
    if not os.path.exists(output_path):
        output.to_json(os.path.join(output_dir, f"{args.data_split}.jsonl"))
    else:
        print("File already exists")
    return output


if __name__ == "__main__":
    # default setting on HH-RLHF dataset, please iterate over data subsets and data splits
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)
    
    print('generating embeddings')
    dataset = generate_embeddings_with_llm(script_args)
    print("generated embeddings")
    dataset = dataset.map(lambda x: {"survey_options": True}, num_proc=1)
    if not script_args.random_contexts:
        survey_options = dataset.filter(lambda x: x['survey_options'] == True)
    else:
        survey_options = dataset.filter(lambda x: x['survey_options'] == True or x['survey_options'] == False)
    survey_ids = np.random.choice(range(len(survey_options)), script_args.survey_size, replace=False)
    print(survey_ids)
    if script_args.data_split == "train":
        survey_data = survey_options.filter(lambda example, idx: idx in survey_ids, with_indices=True)
        survey_data.to_json(os.path.join(script_args.data_path, script_args.data_subset, "survey_{}.jsonl".format(script_args.survey_size)))
    else:
        survey_data = load_dataset('json', data_files=os.path.join(script_args.data_path, script_args.data_subset, "survey_{}.jsonl".format(script_args.survey_size)))
        survey_data = survey_data['train']
    print('generating context and saving')
    generate_contexts(script_args, dataset, survey_data)
    print('done')