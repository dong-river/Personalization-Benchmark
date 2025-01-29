import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaConfig
)
import pandas as pd
from datasets import load_dataset
from typing import Dict, List, Tuple
from tqdm import tqdm

# PEFT imports
from peft import LoraConfig, TaskType, get_peft_model

class PreferenceDataset(Dataset):
    def __init__(self, user_ids: List[int], prompts: List[str], 
                 chosen: List[str], rejected: List[str], tokenizer):
        self.user_ids = user_ids
        self.prompts = prompts
        self.chosen = chosen
        self.rejected = rejected
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.user_ids)
        
    def __getitem__(self, idx):
        chosen_input = self.tokenizer(
            self.prompts[idx],
            self.chosen[idx],
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
        
        rejected_input = self.tokenizer(
            self.prompts[idx],
            self.rejected[idx],
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'user_id': torch.tensor(self.user_ids[idx]),
            'chosen_input_ids': chosen_input['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_input['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected_input['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_input['attention_mask'].squeeze(0),
        }

class TextEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        if model_name == 'meta-llama/Llama-3.2-1B':
            config = LlamaConfig.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            config.rope_scaling = {
                "type": "linear",  # or "dynamic"
                "factor": 4        # or some other integer/float
            }

            self.model = AutoModelForCausalLM.from_config(config)
            
        else:
            base_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

            base_model.config.output_hidden_states = True  
            base_model.config.use_cache = False
            # self.model = base_model

            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )

            # 4) Wrap base model with PEFT
            self.model = get_peft_model(base_model, lora_config)

    def forward(self, input_ids, attention_mask):
        # 5) Forward pass 
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False
        )
        
        # "outputs" is a CausalLMOutputWithPast. We want the final hidden state from hidden_states[-1].
        hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]

        # If we want a single embedding per sequence, let's take the last token 
        # (commonly done in LLaMA if the last token is most representative).
        # Alternatively, you can average-pool or take the first token.
        # E.g. last token embedding:
        embedding_2d = hidden_state[:, -1, :]  # [batch_size, hidden_dim]

        # Add tiny noise to break ties
        embedding_2d = embedding_2d + torch.randn_like(embedding_2d) * 1e-6
        
        return embedding_2d

class RewardModel(nn.Module):
    def __init__(self, model_name: str, num_users: int, embedding_dim: int):
        super().__init__()
        self.text_encoder = TextEncoder(model_name)
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)

        # For a LLaMA-based model, hidden_dim might be in self.text_encoder.model.config.hidden_size or .n_embd
        # Check which attribute the config uses:
        hidden_dim = getattr(self.text_encoder.model.config, "hidden_size", None)
        if hidden_dim is None:
            hidden_dim = getattr(self.text_encoder.model.config, "n_embd", None)
            if hidden_dim is None:
                raise ValueError("Could not find 'hidden_size' or 'n_embd' in config.")

        # A simple reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # optional bounding
        )
        
    def get_reward(self, user_id, input_ids, attention_mask):
        text_embedding = self.text_encoder(input_ids, attention_mask)  # [batch_size, hidden_dim]
        user_embedding = self.user_embeddings(user_id)                 # [batch_size, embedding_dim]

        # Combine
        combined = torch.cat([text_embedding, user_embedding], dim=1)  # [batch_size, hidden_dim + embedding_dim]
        return self.reward_head(combined).squeeze(-1)
    
    def forward(self, batch):
        chosen_reward = self.get_reward(
            batch["user_id"],
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"]
        )
        rejected_reward = self.get_reward(
            batch["user_id"],
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"]
        )
        return chosen_reward, rejected_reward

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        chosen_reward, rejected_reward = model(batch)
        
        # Bradley-Terry style preference loss
        eps = 1e-6
        loss = -torch.log(torch.sigmoid(chosen_reward - rejected_reward + eps)).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            chosen_reward, rejected_reward = model(batch)
            # chosen should be > rejected
            correct += (chosen_reward > rejected_reward).sum().item()
            total += len(chosen_reward)
    return correct / total

def main():
    learning_rate = 1e-4
    batch_size = 8
    train_size = 1000
    eval_size = 50
    num_epochs = 3

    # e.g. "meta-llama/Llama-2-7b-hf" or "google/gemma-2b" "meta-llama/Llama-3.2-1B"
    model_name = "google/gemma-2b"  #"google/gemma-2b"  #"meta-llama/Llama-3.2-1B" #"meta-llama/Llama-2-7b-hf"

    # Load some example data
    train_dataset = load_dataset("RiverDong/psoups", split='train').select(range(train_size))
    val_dataset = load_dataset("RiverDong/psoups", split='test').select(range(eval_size))

    train_df = pd.DataFrame(train_dataset)
    train_df.rename(columns={'uid': 'user_id'}, inplace=True)
    train_df['user_id'] = train_df['user_id'] - 1
    ##sort by prompt
    train_df = train_df.sort_values(by=['prompt'])

    val_df = pd.DataFrame(val_dataset)
    val_df.rename(columns={'uid': 'user_id'}, inplace=True)
    val_df['user_id'] = val_df['user_id'] - 1

    # Tokenizer
    # For LLaMA 2, you might want LlamaTokenizer if it exists on HF. 
    # Often, AutoTokenizer is enough if the config is recognized:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = 2048
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    
    # Datasets
    train_dataset = PreferenceDataset(
        train_df['user_id'].tolist(),
        train_df['prompt'].tolist(),
        train_df['chosen'].tolist(),
        train_df['rejected'].tolist(),
        tokenizer
    )
    val_dataset = PreferenceDataset(
        val_df['user_id'].tolist(),
        val_df['prompt'].tolist(),
        val_df['chosen'].tolist(),
        val_df['rejected'].tolist(),
        tokenizer
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RewardModel(
        model_name=model_name,
        num_users=train_df['user_id'].nunique(),
        embedding_dim=128
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print("Start training...")
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_accuracy = evaluate(model, train_loader, device)
        val_accuracy = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Train Accuracy: {train_accuracy:.4f}")

if __name__ == "__main__":
    main()
