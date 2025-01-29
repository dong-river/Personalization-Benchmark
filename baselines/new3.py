from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import os
import torch
import torch.nn as nn
import logging
from pathlib import Path
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    PretrainedConfig
)
from sentence_transformers import SentenceTransformer
from utils import load_user_datasets, get_uids

log_path = '/home/yd358/rds/hpc-work/analysis_pers/tmp.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_path}")

@dataclass
class ScriptArguments:
    """Arguments for the training script"""
    train: bool = field(
        default=True,
        metadata={"help": "Whether to run training."},
    )
    eval: bool = field(
        default=True,
        metadata={"help": "Whether to run evaluation."},
    )
    train_dataset_size: Optional[int] = field(
        default=10000,
        metadata={"help": "Size of the training dataset."},
    )
    eval_dataset_size: Optional[int] = field(
        default=1000,
        metadata={"help": "Size of the evaluation dataset."},
    )
    data_type: str = field(
        default="psoups",
        metadata={"help": "Type of dataset to use."},
    )
    subset: str = field(
        default="default",
        metadata={"help": "Subset of the dataset to use."},
    )
    per_device_train_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size per device during training."},
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per device during evaluation."},
    )
    learning_rate: float = field(
        default=3e-4,
        metadata={"help": "Learning rate."},
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "Number of training epochs."},
    )
    weight_decay: float = field(
        default=0.001,
        metadata={"help": "Weight decay."},
    )
    model_name: str = field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        metadata={"help": "Model name or path."},
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Whether to use bf16 precision."},
    )
    log_dir: str = field(
        default="models",
        metadata={"help": "Directory for logs and outputs."},
    )
    eval_every_steps: int = field(
        default=10,
        metadata={"help": "Number of steps between evaluations."},
    )
    save_every_steps: int = field(
        default=100,
        metadata={"help": "Number of steps between model saves."},
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length."},
    )

class RewardModelConfig(PretrainedConfig):
    """Configuration class for UserRewardModel"""
    model_type = "user_reward_model"
    
    def __init__(
        self,
        st_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_users: int = 1000,
        user_embed_dim: int = 64,
        mlp_hidden_dim: int = 128,
        num_labels: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.st_model_name = st_model_name
        self.num_users = num_users
        self.user_embed_dim = user_embed_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.num_labels = num_labels

class UserRewardModel(nn.Module):
    """A reward model that combines user embeddings with text embeddings."""
    
    def __init__(self, config: RewardModelConfig):
        super().__init__()
        self.config = config
        self.device_name = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize components
        self._init_sentence_transformer()
        self._init_user_embedding()
        self._init_mlp()
        
        self.to(self.device_name)
        
    def _init_sentence_transformer(self):
        try:
            self.sentence_transformer = SentenceTransformer(
                self.config.st_model_name, 
                device=self.device_name
            )
            logger.info(f"Initialized sentence transformer: {self.config.st_model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize sentence transformer: {str(e)}")
            raise
            
    def _init_user_embedding(self):
        self.user_embedding = nn.Embedding(
            self.config.num_users, 
            self.config.user_embed_dim
        )
        logger.info(f"Initialized user embeddings: {self.config.num_users} users")
        
    def _init_mlp(self):
        st_emb_dim = self.sentence_transformer.get_sentence_embedding_dimension()
        combined_dim = self.config.user_embed_dim + st_emb_dim
        
        self.mlp = nn.Sequential(
            # First layer: Combined input to first hidden layer
            nn.Linear(combined_dim, self.config.mlp_hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.config.mlp_hidden_dim * 2),
            nn.Dropout(0.2),
            
            # Second layer
            nn.Linear(self.config.mlp_hidden_dim * 2, self.config.mlp_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.config.mlp_hidden_dim),
            nn.Dropout(0.2),
            
            # Third layer
            nn.Linear(self.config.mlp_hidden_dim, self.config.mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.config.mlp_hidden_dim // 2),
            nn.Dropout(0.1),
            
            # Fourth layer
            nn.Linear(self.config.mlp_hidden_dim // 2, self.config.mlp_hidden_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(self.config.mlp_hidden_dim // 4),
            nn.Dropout(0.1),
            
            # Final output layer
            nn.Linear(self.config.mlp_hidden_dim // 4, 1)
        )
        logger.info(f"Initialized enhanced MLP with input dim: {combined_dim}")
        
    # def _init_mlp(self):
    #     st_emb_dim = self.sentence_transformer.get_sentence_embedding_dimension()
    #     combined_dim = self.config.user_embed_dim + st_emb_dim
        
    #     self.mlp = nn.Sequential(
    #         nn.Linear(combined_dim, self.config.mlp_hidden_dim),
    #         nn.ReLU(),
    #         nn.Dropout(0.1),
    #         nn.Linear(self.config.mlp_hidden_dim, self.config.mlp_hidden_dim // 2),
    #         nn.ReLU(),
    #         nn.Dropout(0.1),
    #         nn.Linear(self.config.mlp_hidden_dim // 2, 1)
    #     )
    #     logger.info(f"Initialized MLP with input dim: {combined_dim}")

    def forward(self, texts: List[str], user_ids: torch.Tensor) -> Tuple[torch.Tensor]:
        user_ids = user_ids.to(self.device_name)
        
        # Get user embeddings
        user_emb = self.user_embedding(user_ids)
        
        # Get text embeddings
        try:
            text_emb = self.sentence_transformer.encode(
                texts, 
                convert_to_tensor=True,
                show_progress_bar=False
            )
            text_emb = text_emb.to(self.device_name)
        except Exception as e:
            logger.error(f"Failed to encode texts: {str(e)}")
            raise

        if len(text_emb.shape) == 1:
            text_emb = text_emb.unsqueeze(0)
            
        if user_emb.shape[0] != text_emb.shape[0]:
            raise ValueError(
                f"Batch size mismatch: user_emb: {user_emb.shape}, text_emb: {text_emb.shape}"
            )
            
        combined = torch.cat([user_emb, text_emb], dim=-1)
        rewards = self.mlp(combined).squeeze(-1)
        
        return (rewards,)

    def save_pretrained(self, save_directory: str):
        """Save the model configuration and state dict."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the config
        self.config.save_pretrained(save_directory)
        
        # Save the model state
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        logger.info(f"Model saved to {save_directory}")

@dataclass
class RewardDataCollator:
    """Collates data for the reward model."""
    
    tokenizer: Any
    max_length: Optional[int] = None
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {
            "texts": [],
            "user_ids": []
        }
        
        for feature in features:
            # Get texts
            chosen = self.tokenizer.decode(
                feature["input_ids_chosen"], 
                skip_special_tokens=True
            )
            rejected = self.tokenizer.decode(
                feature["input_ids_rejected"], 
                skip_special_tokens=True
            )
            
            # Add both versions
            batch["texts"].extend([chosen, rejected])
            
            # Add user IDs (converting from 1-indexed to 0-indexed)
            user_id = feature["uid"] - 1
            batch["user_ids"].extend([user_id, user_id])
            
        # Convert user IDs to tensor
        batch["user_ids"] = torch.tensor(
            batch["user_ids"],
            dtype=torch.long
        )
        
        return batch

class RewardTrainer(Trainer):
    """Custom trainer for the reward model."""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards = model(
            texts=inputs["texts"],
            user_ids=inputs["user_ids"]
        )[0]
        
        # Get chosen/rejected pairs
        batch_size = len(rewards)
        rewards_chosen = rewards[0:batch_size:2]
        rewards_rejected = rewards[1:batch_size:2]
        
        # Compute pairwise ranking loss
        loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        
        if return_outputs:
            outputs = {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected
            }
            return loss, outputs
            
        return loss
        
    def compute_metrics(self, eval_preds):
        rewards = eval_preds.predictions
        if isinstance(rewards, tuple):
            rewards = rewards[0]
            
        batch_size = len(rewards)
        rewards_chosen = rewards[0:batch_size:2]
        rewards_rejected = rewards[1:batch_size:2]
        
        # Compute accuracy and average reward difference
        accuracy = (rewards_chosen > rewards_rejected).mean()
        avg_reward_diff = (rewards_chosen - rewards_rejected).mean()
        
        metrics = {
            "accuracy": float(accuracy),
            "avg_reward_diff": float(avg_reward_diff)
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        # Run the default evaluation loop
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )
        
        # Add metrics to output if they're not already there
        if hasattr(output, 'metrics') and output.metrics is not None:
            if 'accuracy' not in output.metrics:
                metrics = self.compute_metrics(output)
                output.metrics.update(metrics)
        
        return output
    
def print_model_info(model):
    """Print detailed information about model components and parameters."""
    print("\n" + "="*50)
    print("Model Component Analysis")
    print("="*50)
    
    # 1. User Embedding Layer
    emb_params = sum(p.numel() for p in model.user_embedding.parameters())
    print(f"\nUser Embedding Layer:")
    print(f"Structure: {model.user_embedding}")
    print(f"Number of parameters: {emb_params:,}")
    
    # 2. Sentence Transformer
    st_params = sum(p.numel() for p in model.sentence_transformer.parameters())
    print(f"\nSentence Transformer:")
    print(f"Model: {model.sentence_transformer}")
    print(f"Embedding dimension: {model.sentence_transformer.get_sentence_embedding_dimension()}")
    print(f"Number of parameters: {st_params:,}")
    
    # 3. MLP
    print(f"\nMLP Structure:")
    total_mlp_params = 0
    for idx, layer in enumerate(model.mlp):
        layer_params = sum(p.numel() for p in layer.parameters())
        total_mlp_params += layer_params
        print(f"Layer {idx}: {layer}")
        print(f"Parameters: {layer_params:,}")
    print(f"Total MLP parameters: {total_mlp_params:,}")
    
    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Model Parameters: {total_params:,}")
    
    # Parameter distribution
    print("\nParameter Distribution:")
    print(f"User Embedding: {emb_params/total_params*100:.2f}%")
    print(f"Sentence Transformer: {st_params/total_params*100:.2f}%")
    print(f"MLP: {total_mlp_params/total_params*100:.2f}%")


def train_reward_model(
    config: RewardModelConfig,
    train_dataset,
    eval_dataset,
    training_args: TrainingArguments,
    tokenizer,
    output_dir: str = "reward_model",
    max_length: int = 512,
):
    """Main training function."""
    # Initialize model
    model = UserRewardModel(config)
    print_model_info(model)
    
    # Initialize trainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=RewardDataCollator(
            tokenizer=tokenizer,
            max_length=max_length
        ),
        compute_metrics=lambda x: RewardTrainer.compute_metrics(None, x)
    )
    
    try:
        if training_args.do_train:
            trainer.train()
        
        if training_args.do_eval:
            metrics = trainer.evaluate()
            logger.info("Final evaluation metrics:")
            for key, value in metrics.items():
                logger.info(f"{key}: {value}")
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_path)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
        
    return model, metrics

def main():
    # Parse arguments
    parser = HfArgumentParser((ScriptArguments))
    script_args = parser.parse_args_into_dataclasses()[0]
    
    # Setup output directory
    model_name_split = script_args.model_name.split("/")[-1]
    output_name = os.path.join(
        script_args.log_dir,
        f"reward_model_{model_name_split}_{script_args.train_dataset_size}"
    )
    
    # Get user IDs
    uids = get_uids(script_args)
    num_users = len(uids)
    
    # Create proper config
    config = RewardModelConfig(
        st_model_name=script_args.model_name,
        num_users=num_users,
        user_embed_dim=384,
        mlp_hidden_dim=128,
        num_labels=1
    )
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    train_dataset, eval_dataset = load_user_datasets(
        tokenizer, 
        script_args, 
        uid=None,
        subset=script_args.subset
    )
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=output_name,
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        weight_decay=script_args.weight_decay,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_every_steps,
        save_strategy="steps",
        save_steps=script_args.save_every_steps,
        logging_steps=10,
        bf16=script_args.bf16,
        do_train=script_args.train,
        do_eval=script_args.eval,
        remove_unused_columns=False,
        report_to="none"
    )
    
    # Train model
    model, metrics = train_reward_model(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=training_args,
        tokenizer=tokenizer,
        output_dir=output_name,
        max_length=script_args.max_length
    )
    
    return model, metrics

if __name__ == "__main__":
    main()