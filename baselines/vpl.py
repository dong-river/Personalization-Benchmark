import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    PreTrainedTokenizerBase,
    TrainingArguments,
    AutoModelForCausalLM,
)
from transformers.utils import PaddingStrategy
from utils import create_text_columns_ultrafeedback, get_tokenizer_and_model, load_user_datasets, get_uids, BestOfNSampler, get_model_gen, judge, load_peft_model_rm
from datasets import concatenate_datasets
import random
from datasets import load_dataset
from copy import deepcopy
from vae_utils import VAEModel, VAETrainer
from peft import LoraConfig, TaskType, get_peft_model


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    
    survey_size: Optional[int] = field(
        default=100,
        metadata={"help": "The size of the survey for VPL Embeddings."},
    )
    output_dir: Optional[str] = field(
        default='./cache_data',
        metadata={"help": "The output directory for the embedding data."},
    )
    context_length: int = field(
        default=8,
        metadata={
            "help": "(Maximum) context length."
        }
    )
    reward_model_type: str = field(
        default="base",
        metadata={
            "help": "The type of reward model to use. You can choose between "
                    "'base', 'mean_and_variance', or 'categorical'."
        },
    )
    fixed_contexts: bool = field(
        default=True,
        metadata={"help": "whether to use pre-calculated embeddings for contexts (encoder inputs)"}
    )
    fixed_llm_embeddings: bool = field(
        default=False,
        metadata={"help": "whether to use pre-calculated embeddings for decoder inputs"}
    )
    with_embeddings: bool = field(
        default=True,
        metadata={"help": "whether to use pre-calculated embeddings for decoder inputs"}
    )
    latent_dim: int = field(default=512, metadata={"help": "dimension of latent user vector"})    # todo: 64
    hidden_dim: int = field(default=512, metadata={"help": "dimension of hidden layer in vae"})    # todo: 256
    encoder_embed_dim: int = field(default=1024, metadata={"help": "dimension of LLM embeddings for encoder"})
    decoder_embed_dim: int = field(default=1024, metadata={"help": "dimension of LLM embeddings for decoder"})
    kl_loss_weight: float = field(default=0.01, metadata={"help": "weight for KLD loss"})
    use_annealing: bool = field(default=True, metadata={"help": "Whether to use annealing for learning rate"})
    resume_from_checkpoint: bool = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    
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
    peft: bool = field(
        default=True,
        metadata={"help": "Whether to use PEFT or not."},
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
    per_device_train_batch_size: Optional[int] = field(default=2)
    per_device_eval_batch_size: Optional[int] = field(default=16)
    gradient_accumulation_steps: Optional[int] = field(default=32)
    learning_rate: Optional[float] = field(default=1e-4)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="gpt2", #"mistralai/Mistral-7B-Instruct-v0.2",
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
    fp16: bool = field(
        default=False,
        metadata={
            "help": "This essentially cuts the training time in half if you want to "
                    "sacrifice a little precision and have a supported GPU."
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
        default=False,
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

@dataclass
class RewardDataCollatorWithPadding:
    args: ScriptArguments
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    uids: List[Union[str, int]] = field(default=None)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # if self.args.other_subsets is None:
        #     user_mapping = {
        #         "helpful": 0,
        #         "harmless": 1,
        #     }
        # else:   # TODO: set subsets here
        #     if self.args.other_subsets == 'ultra_feedback':
        #         subsets = ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']
        #     elif self.args.other_subsets == 'single' or self.args.other_subsets == '84':
        #         subsets = ['8', '4', '2', '1']
        #     elif self.args.other_subsets == 'persona':
        #         subsets = ['3', '2', '1', '0']
        #     else:
        #         subsets = []
        # if uids is list or int ty
        
        user_mapping = {subset: int(idx) for idx, subset in enumerate(uids)}
        if self.args.fixed_llm_embeddings:
            batch_size = len(features)
            embeddings_chosen = []
            embeddings_rejected = []
            contexts_embeddings_chosen = []
            contexts_embeddings_rejected = []
            contexts_lengths = [0]
            for feature in features:
                embeddings_chosen.append(
                    feature["embedding_chosen"]
                )
                embeddings_rejected.append(
                    feature["embedding_rejected"]
                )
                contexts_embeddings_chosen.extend(
                    [
                        context["embedding_chosen"] for context in feature["contexts_embeddings"]
                    ]
                )
                contexts_embeddings_rejected.extend(
                    [
                        context["embedding_rejected"] for context in feature["contexts_embeddings"]
                    ]
                )
                contexts_lengths.append(len(feature["contexts_embeddings"]))
            contexts_lengths = torch.cumsum(torch.tensor(contexts_lengths), dim=0)
            seq_start_end = torch.stack(
                [contexts_lengths[:-1], contexts_lengths[1:]], dim=1
            )
            user_type = [int(user_mapping[feature["user_type"]]) for feature in features]
            assert len(seq_start_end) == batch_size
            return {
                "embeddings_chosen": embeddings_chosen,
                "embeddings_rejected": embeddings_rejected,
                "contexts_embeddings_chosen": contexts_embeddings_chosen,
                "contexts_embeddings_rejected": contexts_embeddings_rejected,
                "seq_start_end": seq_start_end,
                "return_loss": True,
                "user_type": user_type,
            }
        if self.args.fixed_contexts:
            batch_size = len(features)
            features_chosen = []
            features_rejected = []
            contexts_embeddings_chosen = []
            contexts_embeddings_rejected = []
            contexts_lengths = [0]
            for feature in features:
                features_chosen.append(
                    {
                        "input_ids": feature["input_ids_chosen"],
                        "attention_mask": feature["attention_mask_chosen"],
                    }
                )
                features_rejected.append(
                    {
                        "input_ids": feature["input_ids_rejected"],
                        "attention_mask": feature["attention_mask_rejected"],
                    }
                )
                # Creating a flattened list of contexts.
                contexts_embeddings_chosen.extend(
                    [
                        context["embedding_chosen"] for context in feature["contexts_embeddings"]
                    ]
                )
                contexts_embeddings_rejected.extend(
                    [
                        context["embedding_rejected"] for context in feature["contexts_embeddings"]
                    ]
                )
                # Keep track of the start and end of each sequence.
                contexts_lengths.append(len(feature["contexts_embeddings"]))

            batch = self.tokenizer.pad(
                features_chosen + features_rejected,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )

            input_ids = batch["input_ids"].view(
                2, batch_size, batch["input_ids"].shape[-1]
            )
            attention_mask = batch["attention_mask"].view(
                2, batch_size, batch["attention_mask"].shape[-1]
            )

            context_lengths = torch.cumsum(torch.tensor(contexts_lengths), dim=0)
            seq_start_end = torch.stack(
                [context_lengths[:-1], context_lengths[1:]], dim=1
            )
            user_type = [int(user_mapping[feature["user_type"]]) for feature in features]
            assert len(seq_start_end) == batch_size

            return {
                "input_ids_chosen": input_ids[0],
                "attention_mask_chosen": attention_mask[0],
                "input_ids_rejected": input_ids[1],
                "attention_mask_rejected": attention_mask[1],
                "contexts_embeddings_chosen": contexts_embeddings_chosen,
                "contexts_embeddings_rejected": contexts_embeddings_rejected,
                "seq_start_end": seq_start_end,
                "return_loss": True,
                "user_type": user_type,
            }

        batch_size = len(features)
        features_chosen = []
        features_rejected = []
        contexts_features_chosen = []
        contexts_features_rejected = []
        contexts_lengths = [0]
        for feature in features:
            features_chosen.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            features_rejected.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )

            # Creating a flattened list of contexts.
            contexts_features_chosen.extend(
                [
                    {
                        "input_ids": context["input_ids_chosen"],
                        "attention_mask": context["attention_mask_chosen"],
                    }
                    for context in feature["contexts_tokens"]
                ]
            )
            contexts_features_rejected.extend(
                [
                    {
                        "input_ids": context["input_ids_rejected"],
                        "attention_mask": context["attention_mask_rejected"],
                    }
                    for context in feature["contexts_tokens"]
                ]
            )
            # Keep track of the start and end of each sequence.
            contexts_lengths.append(len(feature["contexts_tokens"]))

        batch = self.tokenizer.pad(
            features_chosen + features_rejected + contexts_features_chosen + contexts_features_rejected,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        input_ids = batch["input_ids"][:2 * batch_size].view(
            2, batch_size, batch["input_ids"].shape[-1]
        )
        attention_mask = batch["attention_mask"][:2 * batch_size].view(
            2, batch_size, batch["attention_mask"].shape[-1]
        )

        contexts_lengths = torch.cumsum(torch.tensor(contexts_lengths), dim=0)
        seq_start_end = torch.stack(
            [contexts_lengths[:-1], contexts_lengths[1:]], dim=1
        )
        user_type = [int(user_mapping[feature["user_type"]]) for feature in features]
        assert len(seq_start_end) == batch_size
        context_ids = batch["input_ids"][2 * batch_size:].view(
            2, contexts_lengths[-1], batch["input_ids"].shape[-1]
        )
        context_attention_mask = batch["attention_mask"][2 * batch_size:].view(
            2, contexts_lengths[-1], batch["attention_mask"].shape[-1]
        )

        return {
            "input_ids_chosen": input_ids[0],
            "attention_mask_chosen": attention_mask[0],
            "input_ids_rejected": input_ids[1],
            "attention_mask_rejected": attention_mask[1],
            "contexts_input_ids_chosen": context_ids[0],
            "contexts_attention_mask_chosen": context_attention_mask[0],
            "contexts_input_ids_rejected": context_ids[1],
            "contexts_attention_mask_rejected": context_attention_mask[1],
            "seq_start_end": seq_start_end,
            "return_loss": True,
            "user_type": user_type,
        }

class GenDataPreprocessor(object):
    def __init__(self, tokenizer, **tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs

    def __call__(self, examples):
        new_examples: dict = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = self.tokenizer(chosen, **self.tokenizer_kwargs)
            tokenized_rejected = self.tokenizer(rejected, **self.tokenizer_kwargs)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(
                tokenized_chosen["attention_mask"]
            )
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(
                tokenized_rejected["attention_mask"]
            )

        return new_examples

class HHRLHFPreprocessor(object):
    def __init__(self, args, tokenizer, **tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.args = args
        self.tokenizer_kwargs = tokenizer_kwargs

    def __call__(self, examples):
        if self.args.fixed_llm_embeddings:
            new_examples: dict = {
                "embedding_chosen": [],
                "embedding_rejected": [],
                "contexts_embeddings": [],
                "max_lengths": []
            }
            for embeddings, contexts in zip(
                    examples["embeddings"], examples["contexts"]
            ):
                new_examples["embedding_chosen"].append(embeddings["embedding_chosen"])
                new_examples["embedding_rejected"].append(embeddings["embedding_rejected"])
                contexts_embeddings = [{"embedding_chosen": context["embedding_chosen"],
                                        "embedding_rejected": context["embedding_rejected"]}
                                       for context in contexts]
                new_examples["contexts_embeddings"].append(contexts_embeddings)
                new_examples["max_lengths"].append(0)
            new_examples["user_type"] = examples["data_subset"]
            return new_examples

        new_examples: dict = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
            "max_lengths": []
        }
        if self.args.fixed_contexts:
            new_examples["contexts_embeddings"] = []
        else:
            new_examples["contexts_tokens"] = []
        for chosen, rejected, contexts, user_type in zip(
                examples["chosen"], examples["rejected"], examples["contexts"], examples["data_subset"]
        ):
            max_length = 0
            tokenized_chosen = self.tokenizer(chosen, **self.tokenizer_kwargs)
            tokenized_rejected = self.tokenizer(rejected, **self.tokenizer_kwargs)
            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(
                tokenized_chosen["attention_mask"]
            )
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(
                tokenized_rejected["attention_mask"]
            )
            max_length = max(max_length, len(tokenized_chosen["input_ids"]))
            max_length = max(max_length, len(tokenized_rejected["input_ids"]))

            if self.args.fixed_contexts:
                contexts_embeddings = [{"embedding_chosen": context["embedding_chosen"],
                                        "embedding_rejected": context["embedding_rejected"]}
                                       for context in contexts]
                new_examples["contexts_embeddings"].append(contexts_embeddings)
            else:
                tokenized_context = []
                # Tokenize the contexts.
                for context in contexts:
                    chosen, rejected = context["chosen"], context["rejected"]
                    tokenized_chosen = self.tokenizer(chosen, **self.tokenizer_kwargs)
                    tokenized_rejected = self.tokenizer(rejected, **self.tokenizer_kwargs)
                    tokenized_context.append(
                        {
                            "input_ids_chosen": tokenized_chosen["input_ids"],
                            "attention_mask_chosen": tokenized_chosen["attention_mask"],
                            "input_ids_rejected": tokenized_rejected["input_ids"],
                            "attention_mask_rejected": tokenized_rejected["attention_mask"],
                        }
                    )
                    max_length = max(max_length, len(tokenized_chosen["input_ids"]))
                    max_length = max(max_length, len(tokenized_rejected["input_ids"]))
                new_examples["contexts_tokens"].append(tokenized_context)
            new_examples["max_lengths"].append(max_length)
        new_examples["user_type"] = examples["data_subset"]
        return new_examples

def generate_contexts(args, input_dataset, survey_dataset, uid, data_split, output_path, num_duplicates=2):
    # Generate context with survey question pool
    output_dir = os.path.join(args.output_dir, f"{args.model_name}", str(uid))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_size = len(input_dataset)
    if data_split == 'train':
        K = num_duplicates  # repeat samples for K times
    else:
        K = 1
    dataset_list = list()

    def random_choice(max_context_length, survey_size):
        if max_context_length <= survey_size:
            from functools import reduce
            while True:
                context_length = max_context_length 
                context_chosen_ids = np.random.choice(survey_size, context_length, replace=False)
                chosen_dataset = [d for idx, d in enumerate(survey_dataset) if idx in context_chosen_ids]
                # if args.other_subsets != 'single':
                return chosen_dataset, context_length
                # satisfied_sets = list()
                # for row in chosen_dataset:
                #     satisfied_sets.append(set(row["satisfied_subset"]))
                # if len(reduce(lambda x, y: x.intersection(y), satisfied_sets)) == 1:
                #     return chosen_dataset, context_length
                # elif context_length == survey_size:
                #     raise ValueError("Please choose another random seed!")
        else:
            raise ValueError("Context length is larger than survey size!")

    for idx in range(K):
        output_dataset = deepcopy(input_dataset)
        context_lengths = list()
        contexts = list()
        for _ in tqdm(range(dataset_size)):  # iterate over all samples in original dataset
            row_contexts = list()

            context_dataset, context_length = random_choice(args.context_length, args.survey_size)
            context_lengths.append(context_length)
            for context_row in context_dataset:
                context_id = context_row["Index"]
                context_data = context_row
                if not args.with_embeddings:
                    row_contexts.append({
                        'original_id': context_id,
                        'chosen': context_data['chosen'],
                        'rejected': context_data['rejected'],
                    })
                else:
                    row_contexts.append({
                        'original_id': context_id,
                        'embedding_chosen': context_data['embeddings']['embedding_chosen'],
                        'embedding_rejected': context_data['embeddings']['embedding_rejected'],
                    })
            contexts.append(row_contexts)
        output_dataset = output_dataset.add_column("context_length", context_lengths)
        output_dataset = output_dataset.add_column("contexts", contexts)
        output_dataset.map()
        dataset_list.append(output_dataset)
    output = concatenate_datasets(dataset_list)
    output.to_json(output_path)
    return output

def generate_embeddings_with_llm(args, input_dataset=None):
    """
    This function is used to generate fixed embeddings for inputs from original dataset.
    """
    # if not args.synthetic_dataset:
    #     data_subset = cast(DataSubset, args.data_subset)
    #     input_dataset = get_hh_rlhf_dataset(
    #         data_subset,
    #         args.data_split,
    #         args.dataset_size,
    #         data_path=args.data_path,
    #         use_subset_as_dir=True,
    #         other_subsets=args.other_subsets,
    #     )

    if args.model_name == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained("gpt2", use_auth_token=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            "gpt2", num_labels=args.embed_dim, torch_dtype=torch.bfloat16
        )
        model.score.weight.data *= 0.01
    elif args.model_name == "llama" or args.model_name == "meta-llama/Llama-2-7b-hf" or args.model_name == "google/gemma-2b-it":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=True, add_eos_token=False)
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16
        )
    else:
        return input_dataset
    model.to("cuda")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_length

    model.config.pad_token_id = tokenizer.pad_token_id
    dataset_size = len(input_dataset)
    print(dataset_size)

    preprocessed_dataset = input_dataset.map(
        GenDataPreprocessor(tokenizer),
        batched=True,
        num_proc=24,
        remove_columns=input_dataset.column_names,
        load_from_cache_file=False,
    )

    input_dataset = input_dataset.filter(
        lambda example, idx: len(preprocessed_dataset[idx]["input_ids_chosen"]) <= args.max_length
                             and len(preprocessed_dataset[idx]["input_ids_rejected"]) <= args.max_length,
        with_indices=True
    )
    preprocessed_dataset = preprocessed_dataset.filter(
        lambda example: len(example["input_ids_chosen"]) <= args.max_length
                        and len(example["input_ids_rejected"]) <= args.max_length
    )
    print(len(input_dataset), len(preprocessed_dataset))
    dataset_size = len(preprocessed_dataset)

    embeddings = list()
    for row_id in tqdm(range(dataset_size)):
        emb = dict()
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
    output_dataset = input_dataset.add_column("embeddings", embeddings)
    return output_dataset

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    torch.set_default_dtype(torch.bfloat16 if script_args.bf16 else torch.float32)
    model_name_split = script_args.model_name.split("/")[-1]
    method = "vpl"
    uids = get_uids(script_args)
    log_path = '/home/yd358/rds/hpc-work/analysis_pers/results/logfile.log'
    logging.basicConfig(level=logging.INFO, filename=log_path, format='%(asctime)s - %(message)s')
    # tokenizer, _ = get_tokenizer_and_model(script_args, model_type="rm", use_peft=False)
    tokenizer_name = script_args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True, add_eos_token=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = script_args.max_length
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
        
    if script_args.train:
        for uid in uids:
            output_survey_path = os.path.join(script_args.output_dir, script_args.data_path.split("/")[-1], str(uid), model_name_split, "survey_{}.jsonl".format(script_args.survey_size))
            train_data_path = os.path.join(script_args.output_dir, script_args.data_path.split("/")[-1], str(uid), model_name_split,  "train.jsonl")
            test_data_path = os.path.join(script_args.output_dir, script_args.data_path.split("/")[-1], str(uid), model_name_split, "test.jsonl")
            if os.path.exists(output_survey_path) and os.path.exists(train_data_path) and os.path.exists(test_data_path):
                continue
            
            train_dataset, eval_dataset = load_user_datasets(tokenizer, script_args, uid)
            train_dataset = train_dataset.rename_column("uid", "data_subset")
            eval_dataset = eval_dataset.rename_column("uid", "data_subset")
            train_dataset = train_dataset.rename_column("chosen", "chosen_messages")
            train_dataset = train_dataset.rename_column("rejected", "rejected_messages")
            eval_dataset = eval_dataset.rename_column("chosen", "chosen_messages")
            eval_dataset = eval_dataset.rename_column("rejected", "rejected_messages")
            train_dataset = train_dataset.rename_column("chosen_only", "chosen")
            train_dataset = train_dataset.rename_column("rejected_only", "rejected")
            eval_dataset = eval_dataset.rename_column("chosen_only", "chosen")
            eval_dataset = eval_dataset.rename_column("rejected_only", "rejected")
            ## add Index
            train_dataset = train_dataset.add_column("Index", list(range(len(train_dataset))))
            eval_dataset = eval_dataset.add_column("Index", list(range(len(eval_dataset))))
            
            train_dataset = generate_embeddings_with_llm(script_args, train_dataset)
            eval_dataset = generate_embeddings_with_llm(script_args, eval_dataset)
            survey_ids = np.random.choice(range(len(train_dataset)), script_args.survey_size, replace=False)
            survey_data = train_dataset.filter(lambda example, idx: idx in survey_ids, with_indices=True)
            survey_data.to_json(output_survey_path)
            
            generate_contexts(script_args, train_dataset, survey_data, uid, 'train', train_data_path)
            generate_contexts(script_args, eval_dataset, survey_data, uid, 'test', test_data_path)
        
        ## Load data
        train_datasets, eval_datasets = [], []
        for uid in uids:
            train_data_path = os.path.join(script_args.output_dir, script_args.data_path.split("/")[-1], str(uid), model_name_split, "train.jsonl")
            test_data_path = os.path.join(script_args.output_dir, script_args.data_path.split("/")[-1], str(uid), model_name_split, "test.jsonl")
            train_datasets.append(load_dataset('json', data_files=train_data_path)['train'])
            eval_datasets.append(load_dataset('json', data_files=test_data_path)['train'])

        train_dataset = concatenate_datasets(train_datasets)
        eval_dataset = concatenate_datasets(eval_datasets)
                    
        if script_args.model_name == 'gpt2':
            script_args.decoder_embed_dim = 768
            script_args.encoder_embed_dim = 768
        if script_args.model_name == 'meta-llama/Llama-2-7b-hf':
            script_args.decoder_embed_dim = 4096
            script_args.encoder_embed_dim = 4096
        
        output_name = (
            f"{script_args.log_dir}/VAE/{uid}/"
            f"{model_name_split}_{script_args.train_dataset_size}_{script_args.learning_rate}"
            f"_{script_args.lr_scheduler_type}_{script_args.num_train_epochs}"
        )
        

        training_args = TrainingArguments(
            output_dir=output_name,
            learning_rate=script_args.learning_rate,
            per_device_train_batch_size=script_args.per_device_train_batch_size,
            per_device_eval_batch_size=script_args.per_device_eval_batch_size,
            num_train_epochs=script_args.num_train_epochs,
            weight_decay=script_args.weight_decay,
            evaluation_strategy="steps",
            eval_steps=0.05,
            save_strategy="steps",
            save_steps=10000,
            gradient_accumulation_steps=script_args.gradient_accumulation_steps,
            gradient_checkpointing=script_args.gradient_checkpointing,
            deepspeed=script_args.deepspeed,
            local_rank=script_args.local_rank,
            remove_unused_columns=False,
            label_names=[],
            bf16=script_args.bf16,
            fp16=script_args.fp16,
            logging_strategy="steps",
            logging_steps=100,
            optim=script_args.optim,
            lr_scheduler_type="cosine",
            report_to="wandb",
            run_name=output_name.replace('/', '_'),
        )
        
        decoder_embed_dim = script_args.decoder_embed_dim
        encoder_embed_dim = script_args.encoder_embed_dim
        model = AutoModelForSequenceClassification.from_pretrained(
            script_args.model_name, num_labels=decoder_embed_dim, torch_dtype=torch.bfloat16
        )
        model.score.weight.data *= 0.01
        model.config.pad_token_id = tokenizer.pad_token_id
        
        if script_args.peft:
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=128,
                lora_alpha=256,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        
        model.config.use_cache = not script_args.gradient_checkpointing
        num_proc = 24  # Can adjust to be higher if you have more processors.
        original_columns = train_dataset.column_names
        
        train_dataset = train_dataset.map(
            HHRLHFPreprocessor(script_args, tokenizer),
            batched=True,
            num_proc=num_proc,
            remove_columns=original_columns,
        )
        
        train_dataset = train_dataset.filter(
            lambda x: x["max_lengths"] <= script_args.max_length
        )
        
        eval_dataset = eval_dataset.map(
            HHRLHFPreprocessor(script_args, tokenizer),
            batched=True,
            num_proc=num_proc,
            remove_columns=original_columns,
        )
        eval_dataset = eval_dataset.filter(
            lambda x: x["max_lengths"] <= script_args.max_length
        )
        
        latent_dim = script_args.latent_dim
        hidden_dim = script_args.hidden_dim
        contexts_model = None
                
        vae_model = VAEModel(encoder_embed_dim, decoder_embed_dim, hidden_dim, latent_dim, model, contexts_model,
                            fixed_contexts=script_args.fixed_contexts,
                            fixed_llm_embeddings=script_args.fixed_llm_embeddings,)
        
        trainer = VAETrainer(
            model=vae_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=VAETrainer.compute_metrics,
            data_collator=RewardDataCollatorWithPadding(
                args=script_args,
                tokenizer=tokenizer,
                max_length=script_args.max_length,
                pad_to_multiple_of=64,
                uids=uids,
            ),
            kl_loss_weight=script_args.kl_loss_weight,
            use_annealing=script_args.use_annealing,
            # **trainer_kwargs,
        )
        trainer.train(script_args.resume_from_checkpoint)
        print("Saving last checkpoint of the model")

        os.makedirs(output_name, exist_ok=True)
        model.save_pretrained(output_name)
        tokenizer.save_pretrained(output_name)
        
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
    # except Exception as e:
    #     import pdb
    #     import traceback

    #     if not isinstance(e, (pdb.bdb.BdbQuit, KeyboardInterrupt)):
    #         print("\n" + ">" * 100 + "\n")
    #         traceback.print_exc()
    #         print()
    #         pdb.post_mortem()