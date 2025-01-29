import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union, cast

import numpy as np
import torch
import random
import logging
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    TrainingArguments,
    TrainerCallback,
)
from transformers.utils import PaddingStrategy
from .vae_utils import VAETrainer, VAEModel

from .train_llm_preference_model import (
    get_step_decay_lr_lambda,
    get_cosine_decay_lr_lambda,
    RewardModelType,
    DataSubset,
    get_hh_rlhf_dataset,
    concatenate_datasets
)

import sys
sys.path.append('/home/yd358/rds/hpc-work/analysis_pers/baselines')
from utils import load_user_datasets, load_reward_bench

@dataclass
class ScriptArguments:
    local_rank: int = field(default=-1, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: bool = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this "
                    "if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=3e-6)
    weight_decay: float = field(default=0.001)
    lora_rank: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.1)
    model_name: str = field(
        default="gpt2",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. "
                    "E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    data_path: str = field(
        default="Anthropic/hh-rlhf",
    )
    ## Newly added
    data_type: str = field(default='summarization') #psoups, ultrafeedback-p
    
    data_subset: str = field(
        default="both",
        metadata={
            "help": "Which subset of the data to use. You can choose between 'both', "
                    "'helpful', or 'harmless'."
        },
    )
    reward_model_type: str = field(
        default="base",
        metadata={
            "help": "The type of reward model to use. You can choose between "
                    "'base', 'mean_and_variance', or 'categorical'."
        },
    )
    num_atoms: int = field(
        default=10,
        metadata={
            "help": "The number of atoms to use for the categorical reward model."
        },
    )
    entropy_coeff: float = field(
        default=0.1,
        metadata={"help": "The entropy coefficient for the categorical reward model."},
    )
    variance_penalty: float = field(
        default=0.0,
        metadata={
            "help": "The variance penalty for the mean and variance reward model."
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer for your model, if left empty will use the default "
                    "for your model",
        },
    )
    bf16: bool = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to "
                    "sacrifice a little precision and have a supported GPU."
        },
    )
    fp16: bool = field(
        default=False,
        metadata={
            "help": "This essentially cuts the training time in half if you want to "
                    "sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_dataset_size: int = field(
        default=0,
        metadata={"help": "The size of the subset of the training data to use"},
    )
    eval_dataset_size: int = field(
        default=0,
        metadata={"help": "The size of the subset of the eval data to use"},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: str = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )
    max_length: int = field(default=1024)
    eval_first_step: bool = field(
        default=True,
        metadata={"help": "Whether to run eval after the first step"},
    )
    log_dir: str = field(default="data/reward_models/hh_rlhf")
    kl_loss_weight: float = field(default=0.01, metadata={"help": "weight for KLD loss"})
    latent_dim: int = field(default=512, metadata={"help": "dimension of latent user vector"})    # todo: 64
    hidden_dim: int = field(default=512, metadata={"help": "dimension of hidden layer in vae"})    # todo: 256
    encoder_embed_dim: int = field(default=1024, metadata={"help": "dimension of LLM embeddings for encoder"})
    decoder_embed_dim: int = field(default=1024, metadata={"help": "dimension of LLM embeddings for decoder"})
    use_annealing: bool = field(default=True, metadata={"help": "Whether to use annealing for learning rate"})
    fixed_contexts: bool = field(
        default=False,
        metadata={"help": "whether to use pre-calculated embeddings for contexts (encoder inputs)"}
    )
    fixed_llm_embeddings: bool = field(
        default=False,
        metadata={"help": "whether to use pre-calculated embeddings for decoder inputs"}
    )
    seed: int = field(default=0)
    controversial_only: bool = field(
        default=False,
        metadata={"help": "whether to only include controversial data"}
    )
    up_sampling: bool = field(
        default=False,
        metadata={"help": "whether to upsample controversial data during training phase"}
    )
    one_user: str = field(
        default=None,
        metadata={"help": "whether to only train and evaluate on one single user"}
    )
    other_subsets: str = field(
        default=None,
        metadata={"help": "specify the group of subsets if not using helpful/harmless. You can choose between"
                          "ultra_feedback, pos_neg, set, single."},
    )
    use_last_token_embedding: bool = field(
        default=False,
        metadata={"help": "whether to use the last token embedding of last layer as LLM embeddings"}
    )

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


trainer_classes: Dict[RewardModelType, Type[VAETrainer]] = {
    "vae": VAETrainer,
}


# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    args: ScriptArguments
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.args.data_type is None:
            user_mapping = {
                "helpful": 0,
                "harmless": 1,
            }
        else:   # TODO: set subsets here
            if self.args.data_type == 'ultra_feedback':
                subsets = ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']
            elif self.args.data_type == 'single' or self.args.data_type == '84':
                subsets = ['8', '4', '2', '1']
            else:
                subsets = []
            user_mapping = {subset: idx for idx, subset in enumerate(subsets)}
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

            user_type = [user_mapping[feature["user_type"]] for feature in features]
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
            # import pdb; pdb.set_trace()
            # user_type = [user_mapping[feature["user_type"]] for feature in features]
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
                "user_type": feature["user_type"], ##user_type
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
        user_type = [user_mapping[feature["user_type"]] for feature in features]
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


def up_sample_controversial(dataset, seed):
    cont = dataset.filter(lambda example: example['controversial'] == True)
    up_sampled_dataset = concatenate_datasets([cont] * 4 + [dataset])
    up_sampled_dataset = up_sampled_dataset.shuffle(seed=seed)
    return up_sampled_dataset


def customized_optimizer(model, lr):
    encoder_params = [p for p in model.parameters() if p not in model.decoder.parameters()]
    decoder_params = [p for p in model.parameters() if p in model.decoder.parameters()]
    grouped_parameters = [
        {'params': encoder_params, 'lr': lr},
        {'params': decoder_params, 'lr': lr / 10},
    ]
    return


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

    seed = script_args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.set_default_dtype(torch.bfloat16 if script_args.bf16 else torch.float32)

    if script_args.use_last_token_embedding:
        if script_args.model_name == 'gpt2':
            script_args.decoder_embed_dim = 768
            script_args.encoder_embed_dim = 768
        if script_args.model_name == 'meta-llama/Llama-2-7b-hf':
            script_args.decoder_embed_dim = 4096
            script_args.encoder_embed_dim = 4096

    data_subset = cast(DataSubset, script_args.data_subset)
    train_dataset = get_hh_rlhf_dataset(
        data_subset,
        "train",
        script_args.train_dataset_size,
        data_path=script_args.data_path,
        other_subsets=script_args.data_type
    )
    eval_dataset = get_hh_rlhf_dataset(
        data_subset,
        "test",
        script_args.eval_dataset_size,
        data_path=script_args.data_path,
        other_subsets=script_args.data_type
    )
    print(len(train_dataset), len(eval_dataset))
    # if script_args.controversial_only:
    #     train_dataset = train_dataset.filter(lambda example: example['controversial'] == True)
    #     eval_dataset = eval_dataset.filter(lambda example: example['controversial'] == True)
    # elif script_args.up_sampling:
    #     train_dataset = up_sample_controversial(train_dataset, seed)

    # if script_args.one_user:
    #     train_dataset = train_dataset.filter(lambda example: example['data_subset'] == script_args.one_user)
    #     eval_dataset = eval_dataset.filter(lambda example: example['data_subset'] == script_args.one_user)
    reward_model_type = cast(RewardModelType, script_args.reward_model_type)

    # Define the training args. Needs to be done before the model is loaded if you
    # are using deepspeed.
    model_name_split = script_args.model_name.split("/")[-1]
    output_name = (
        f"{script_args.log_dir}/VPL_{data_subset}/"
        f"{reward_model_type}_{model_name_split}_{script_args.data_type}"
        f"_{script_args.train_dataset_size}_{script_args.learning_rate}_psoups_default"
        f"_{script_args.lr_scheduler_type}_{script_args.num_train_epochs}"
    )
    output_name += f"_{script_args.kl_loss_weight}_{script_args.latent_dim}_{script_args.decoder_embed_dim}_seed{script_args.seed}"
    log_path = f'../results/{output_name.replace("/", "_")}.log'
    logging.basicConfig(level=logging.INFO, filename=log_path, format='%(asctime)s - %(message)s')
    
    trainer_kwargs: Dict[str, Any] = {}
    if script_args.lr_scheduler_type == "step":
        lr_scheduler_type = "constant"
        trainer_kwargs["lr_lambda"] = get_step_decay_lr_lambda
    elif script_args.lr_scheduler_type == "cosine":
        lr_scheduler_type = "constant"
        trainer_kwargs["lr_lambda"] = get_cosine_decay_lr_lambda
    else:
        lr_scheduler_type = script_args.lr_scheduler_type

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
        lr_scheduler_type=lr_scheduler_type,
        report_to="wandb",
        run_name=output_name.split("/")[-1],
    )
    # Load the value-head model and tokenizer.
    tokenizer_name = (
        script_args.tokenizer_name
        if script_args.tokenizer_name is not None
        else script_args.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True, add_eos_token=False)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=script_args.lora_rank,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
    )

    torch.set_anomaly_enabled(True)

    trainer_class = trainer_classes[reward_model_type]
    decoder_embed_dim = script_args.decoder_embed_dim
    encoder_embed_dim = script_args.encoder_embed_dim

    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name, num_labels=decoder_embed_dim, torch_dtype=torch.bfloat16
    )
    # We multiply the final linear layer's weights by 0.01 because this seems to
    # significantly stabilize training and lead to better optimization of the loss.
    model.score.weight.data *= 0.01
    if not script_args.fixed_contexts:
        contexts_model = AutoModelForSequenceClassification.from_pretrained(
            script_args.model_name, num_labels=encoder_embed_dim, torch_dtype=torch.bfloat16
        )
        contexts_model.score.weight.data *= 0.01
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print('rank', script_args.lora_rank, 'alpha', script_args.lora_alpha, 'dropout', script_args.lora_dropout)

    if not script_args.fixed_contexts:
        contexts_model = get_peft_model(contexts_model, peft_config)
        contexts_model.print_trainable_parameters()
        contexts_model.config.pad_token_id = tokenizer.pad_token_id
        contexts_model.config.use_cache = not script_args.gradient_checkpointing
    else:
        contexts_model = None

    # Need to do this for GPT2 and Llama because they don't have official pad tokens.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "right"

    model.config.use_cache = not script_args.gradient_checkpointing
    num_proc = 24  # Can adjust to be higher if you have more processors.
    
    train_dataset = train_dataset.remove_columns(["chosen", 'rejected'])
    eval_dataset = eval_dataset.remove_columns(["chosen", 'rejected'])
    train_dataset = train_dataset.rename_column("chosen_only", "chosen")
    train_dataset = train_dataset.rename_column("rejected_only", "rejected")
    eval_dataset = eval_dataset.rename_column("chosen_only", "chosen")
    eval_dataset = eval_dataset.rename_column("rejected_only", "rejected")
    train_dataset = train_dataset.rename_column("uid", "data_subset")
    eval_dataset = eval_dataset.rename_column("uid", "data_subset")
    original_columns = train_dataset.column_names
    
    train_dataset = train_dataset.map(
        HHRLHFPreprocessor(script_args, tokenizer),
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    train_dataset = train_dataset.filter(
        lambda x: x["max_lengths"] <= script_args.max_length,
        num_proc=num_proc,
    )

    eval_dataset = eval_dataset.map(
        HHRLHFPreprocessor(script_args, tokenizer),
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    eval_dataset = eval_dataset.filter(
        lambda x: x["max_lengths"] <= script_args.max_length,
        num_proc=num_proc,
    )

    # Train the model.
    latent_dim = script_args.latent_dim
    hidden_dim = script_args.hidden_dim
    vae_model = VAEModel(encoder_embed_dim, decoder_embed_dim, hidden_dim, latent_dim, model, contexts_model,
                         fixed_contexts=script_args.fixed_contexts,
                         fixed_llm_embeddings=script_args.fixed_llm_embeddings,)

    trainer = trainer_class(
        model=vae_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=trainer_class.compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            args=script_args,
            tokenizer=tokenizer,
            max_length=script_args.max_length,
            pad_to_multiple_of=64,
        ),
        kl_loss_weight=script_args.kl_loss_weight,
        use_annealing=script_args.use_annealing,
        **trainer_kwargs,
    )

    class EvaluateFirstStepCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            if state.global_step == 0:
                control.should_evaluate = True


    trainer.add_callback(EvaluateFirstStepCallback())

    trainer.train(script_args.resume_from_checkpoint)
    
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    logging.info(f"Metrics for {metrics}")
    reward_bench_datasets = load_reward_bench(tokenizer)
    for idx, eval_dataset in enumerate(reward_bench_datasets):
        dataset_name = eval_dataset.unique('key')
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        logging.info(f"Metrics for dataset {dataset_name}: {metrics['eval_accuracy']}")

    print("Saving last checkpoint of the model")

    model.save_pretrained(output_name + "_peft_last_checkpoint")
    output_name += "_peft_last_checkpoint"
    os.makedirs(output_name, exist_ok=True)

    output_name = os.path.join(output_name, "model.pt")
    vae_model.save_model(output_name)
