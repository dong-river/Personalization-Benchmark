import os
import torch
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from attrdict import AttrDict
from torch.nn.functional import softmax
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding, HfArgumentParser
from utils import load_user_datasets, get_tokenizer_and_model

@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    
    max_ctx_num_qs: Optional[int] = field(
        default=100,
        metadata={"help": "The maximum number of context questions."},
    )
    min_ctx_num_qs: Optional[int] = field(
        default=10,
        metadata={"help": "The minimum number of context questions."},
    )
    max_tar_num_qs: Optional[int] = field(
        default=100,
        metadata={"help": "The maximum number of target questions."},
    )
    min_tar_num_qs: Optional[int] = field(
        default=10,
        metadata={"help": "The minimum number of target questions."},
    )
    group_split_ratio: Optional[float] = field(
        default=0.8,
        metadata={"help": "The ratio of users to use for training."},
    )
    num_steps: Optional[int] = field(
        default=100,
        metadata={"help": "The number of training steps."},
    )
    
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
        default=640,
        metadata={"help": "The size of the training dataset."},
    )
    eval_data_size: Optional[int] = field(
        default=100,
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
    ),
    per_device_train_batch_size: Optional[int] = field(default=2)
    per_device_eval_batch_size: Optional[int] = field(default=16)
    gradient_accumulation_steps: Optional[int] = field(default=32)
    learning_rate: Optional[float] = field(default=1e-4)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
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

class llmodel:
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer 
        self.model = model
        self.model.eval()        

    def get_avg_sentence_embeddings(self, sentences):
        # Tokenize a batch of sentences and feed them to the model.
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=False, max_length=2048)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_hidden_states = outputs.hidden_states[-1]
            mask = inputs['attention_mask'].to(self.device)
            mask_expanded = mask.unsqueeze(-1).expand(all_hidden_states.size())
            sum_hidden_states = torch.sum(all_hidden_states * mask_expanded, 1)
            sentence_embeddings = sum_hidden_states / mask_expanded.sum(1)
        return sentence_embeddings


class GPODataset(Dataset):
    def __init__(self, df, config=None, device='cuda', seed=41):
        self.device = device
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        # Group by 'uid' and then by 'qkey'
        grouped_by_group = df.groupby('uid')
        
        q_key_dict = {}
        q_key_count = 0
        
        self.data = []
        for group_name, group_data in grouped_by_group:
            total_num_options = 0
            qkey_list = []
            for query, sub_group in group_data.groupby('prompt'): 
                ## I think we only have one prompt for each user so the prob_y is also either 0 or 1
                embedding = torch.tensor(sub_group['embedding'].tolist())
                prob_y = torch.tensor([1, 0], dtype=torch.float).unsqueeze(0)
                total_num_options += len(embedding)
                if query not in q_key_dict:
                    q_key_dict[query] = q_key_count
                    q_key_count += 1
                    qkey = q_key_dict[query] 
                else:
                    qkey = q_key_dict[query]    
                qkey_list.append({
                    'q_emb': embedding,
                    'prob_ys': prob_y,
                    'qkey': qkey
                })
            self.data.append({
                'groups': group_name,
                'qkeys': qkey_list,
                'total_nqs': total_num_options,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        group_data = self.data[idx]

        return {
            'questions': group_data['qkeys'],
            'groups': group_data['groups'],
            'total_nqs': group_data['total_nqs'],
        }

def load_GPO_datasets(embed_save_path):
    df = pd.read_pickle(embed_save_path)
    uids = df['uid'].unique()
    train_users = np.random.choice(uids, int(len(uids) * script_args.group_split_ratio), replace=False)
    eval_users = [uid for uid in uids if uid not in train_users]
    train_df = df[df['uid'].isin(train_users)]
    eval_df = df[df['uid'].isin(eval_users)]
    train_dataset = GPODataset(train_df)
    eval_dataset = GPODataset(eval_df)
    return train_df, eval_df, train_dataset, eval_dataset

# class CollateFunction:
#     def __init__(self, max_ctx_num_points, min_ctx_num_points, max_tar_num_points, min_tar_num_points, dataset='oqa'):
#         self.max_ctx_num_points = max_ctx_num_points
#         self.min_ctx_num_points = min_ctx_num_points
#         self.max_tar_num_points = max_tar_num_points
#         self.min_tar_num_points = min_tar_num_points
#         self.dataset = dataset
        
#     def __call__(self, batch):
#         # Ensure min_ctx_num_points <= max_ctx_num_points and min_tar_num_points <= max_tar_num_points
#         assert self.min_ctx_num_points <= self.max_ctx_num_points and self.min_tar_num_points <= self.max_tar_num_points, "The range values are not properly defined."
#         # assert self.max_ctx_num_points < total_num_qs
#         # Randomly select the number of context points from range [min_ctx_num_points, max_ctx_num_points]
#         num_ctx = torch.randint(low=self.min_ctx_num_points, high=self.max_ctx_num_points + 1, size=(1,)).item()

#         min_tar = self.min_tar_num_points
#         max_tar = self.max_tar_num_points
#         # max_tar = min(self.max_tar_num_points, total_num_qs - num_ctx)
#         num_tar = torch.randint(low=min_tar, high=max_tar + 1, size=(1,)).item()
#         assert num_ctx + num_tar <= self.max_ctx_num_points + self.max_tar_num_points, "The total number of points exceeded the maximum limit."
#         # Data holders
#         collated_batch = { 
#             'x': [],
#             'xc': [],
#             'xt': [],
#             'y': [],
#             'yc': [],
#             'yt': [],
#             'tarqlen':[],
#         }
#         # perm_indices = torch.randperm(total_num_qs)
#         # ctx_indices = perm_indices[:num_ctx]
#         # tar_indices = perm_indices[num_ctx:num_ctx+num_tar]
#         # import pdb; pdb.set_trace()
#         for b in batch:
#             num_questions = len(b['questions'])
#             perm_indices = torch.randperm(num_questions)
#             ctx_indices = perm_indices[:num_ctx]
#             tar_indices = perm_indices[num_ctx:num_ctx+num_tar]
            
#             ctx_qs = [b['questions'][i] for i in ctx_indices]
#             tar_qs = [b['questions'][i] for i in tar_indices]
#             ctx_pa = torch.cat([q['q_emb'] for q in ctx_qs], dim=0)
#             ctx_prob_ys = torch.cat([q['prob_ys'] for q in ctx_qs], dim=0)
#             tar_pa = torch.cat([q['q_emb'] for q in tar_qs], dim=0)
#             tar_prob_ys = torch.cat([q['prob_ys'] for q in tar_qs], dim=0)
#             tar_q_len = torch.tensor([len(q['prob_ys']) for q in tar_qs])

#             collated_batch['x'].append(torch.cat([ctx_pa, tar_pa], dim=0))
#             collated_batch['xc'].append(ctx_pa)
#             collated_batch['xt'].append(tar_pa)
#             collated_batch['y'].append(torch.cat([ctx_prob_ys, tar_prob_ys], dim=0))
#             collated_batch['yc'].append(ctx_prob_ys)
#             collated_batch['yt'].append(tar_prob_ys)
#             collated_batch['tarqlen'].append(tar_q_len)

#         for key in collated_batch:
#             collated_batch[key] = torch.stack(collated_batch[key]).to('cuda')
        
#         return collated_batch

class CollateFunction:
    def __init__(self, max_ctx_num_points, min_ctx_num_points, max_tar_num_points, min_tar_num_points, dataset='oqa'):
        self.max_ctx_num_points = max_ctx_num_points
        self.min_ctx_num_points = min_ctx_num_points
        self.max_tar_num_points = max_tar_num_points
        self.min_tar_num_points = min_tar_num_points
        self.dataset = dataset
        
    def __call__(self, batch):
        # Ensure min_ctx_num_points <= max_ctx_num_points and min_tar_num_points <= max_tar_num_points
        assert self.min_ctx_num_points <= self.max_ctx_num_points and self.min_tar_num_points <= self.max_tar_num_points, "The range values are not properly defined."
        num_ctx = torch.randint(low=self.min_ctx_num_points, high=self.max_ctx_num_points + 1, size=(1,)).item()

        min_tar = self.min_tar_num_points
        max_tar = self.max_tar_num_points
        num_tar = torch.randint(low=min_tar, high=max_tar + 1, size=(1,)).item()
        assert num_ctx + num_tar <= self.max_ctx_num_points + self.max_tar_num_points, "The total number of points exceeded the maximum limit."
        # Data holders
        collated_batch = { 
            'x': [],
            'xc': [],
            'xt': [],
            'y': [],
            'yc': [],
            'yt': [],
            'tarqlen':[],
        }
        
        temp_ctx_pa = []
        temp_tar_pa = []
        temp_ctx_prob_ys = []
        temp_tar_prob_ys = []
        temp_tarqlen = []
        temp_ctxqlen = []

        for b in batch:
            num_questions = len(b['questions'])
            perm_indices = torch.randperm(num_questions)
            ctx_indices = perm_indices[:num_ctx]
            tar_indices = perm_indices[num_ctx:num_ctx+num_tar]
            
            ctx_qs = [b['questions'][i] for i in ctx_indices]
            tar_qs = [b['questions'][i] for i in tar_indices]
            ctx_pa = torch.cat([q['q_emb'] for q in ctx_qs], dim=0)
            ctx_prob_ys = torch.cat([q['prob_ys'] for q in ctx_qs], dim=0)
            tar_pa = torch.cat([q['q_emb'] for q in tar_qs], dim=0)
            tar_prob_ys = torch.cat([q['prob_ys'] for q in tar_qs], dim=0)
            tar_q_len = torch.tensor([len(q['prob_ys']) for q in tar_qs])

            temp_ctx_pa.append(ctx_pa)
            temp_tar_pa.append(tar_pa)
            temp_ctx_prob_ys.append(ctx_prob_ys)
            temp_tar_prob_ys.append(tar_prob_ys)
            temp_tarqlen.append(torch.tensor([len(q['prob_ys'][0]) for q in tar_qs]))
            temp_ctxqlen.append(torch.tensor([len(q['prob_ys'][0]) for q in ctx_qs]))
        
        pad_ctx_pa = pad_sequence(temp_ctx_pa, batch_first=True, padding_value=0)
        pad_tar_pa = pad_sequence(temp_tar_pa, batch_first=True, padding_value=0)
        pad_ctx_prob_ys = pad_sequence(temp_ctx_prob_ys, batch_first=True, padding_value=0)
        pad_tar_prob_ys = pad_sequence(temp_tar_prob_ys, batch_first=True, padding_value=0)

        # for key in collated_batch:
        #     collated_batch[key] = torch.stack(collated_batch[key]).to('cuda')
        for ctx_pa, tar_pa, ctx_prob_ys, tar_prob_ys in zip(pad_ctx_pa, pad_tar_pa, pad_ctx_prob_ys, pad_tar_prob_ys):
            collated_batch['x'].append(torch.cat([ctx_pa, tar_pa], dim=0))
            collated_batch['xc'].append(ctx_pa)
            collated_batch['xt'].append(tar_pa)
            collated_batch['y'].append(torch.cat([ctx_prob_ys, tar_prob_ys], dim=0))
            collated_batch['yc'].append(ctx_prob_ys)
            collated_batch['yt'].append(tar_prob_ys)
        collated_batch['tarqlen'] = temp_tarqlen

        for key in collated_batch:
            collated_batch[key] = torch.stack(collated_batch[key]).to('cuda')
        import pdb; pdb.set_trace()
        collated_batch['y'] = collated_batch['y'].reshape(len(batch), -1, 1)
        collated_batch['yc'] = collated_batch['yc'].reshape(len(batch), -1, 1)
        collated_batch['yt'] = collated_batch['yt'].reshape(len(batch), -1, 1)
        
        return collated_batch


def build_mlp(dim_in, dim_hid, dim_out, depth):
    modules = [nn.Linear(dim_in, dim_hid), nn.ReLU(True)]
    for _ in range(depth-2):
        modules.append(nn.Linear(dim_hid, dim_hid))
        modules.append(nn.ReLU(True))
    modules.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*modules)

class TNP(nn.Module):
    def __init__(
        self,
        dim_x,
        dim_y,
        d_model,
        emb_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers,
        bound_std
    ):
        super(TNP, self).__init__()

        self.embedder = build_mlp(dim_x + dim_y, d_model, d_model, emb_depth)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.bound_std = bound_std



    def construct_input(self, batch, autoreg=False):
        
        x_y_ctx = torch.cat((batch['xc'], batch['yc']), dim=-1)
        x_0_tar = torch.cat((batch['xt'], torch.zeros_like(batch['yt'])), dim=-1)
        if not autoreg:
            inp = torch.cat((x_y_ctx, x_0_tar), dim=1)
        else:
            if self.training and self.bound_std:
                yt_noise = batch['yt'] + 0.05 * torch.randn_like(batch['yt']) # add noise to the past to smooth the model
                x_y_tar = torch.cat((batch['xt'], yt_noise), dim=-1)
            else:
                x_y_tar = torch.cat((batch['xt'], batch['yt']), dim=-1)
            inp = torch.cat((x_y_ctx, x_y_tar, x_0_tar), dim=1)
        return inp

    def create_mask(self, batch, autoreg=False):
        num_ctx = batch['xc'].shape[1]
        num_tar = batch['xt'].shape[1]
        num_all = num_ctx + num_tar

       # Create source key padding mask [batch_size, sequence_length]
        padding_mask_ctx = (torch.sum(batch['xc'], dim=-1) == 0)
        padding_mask_tar = (torch.sum(batch['xt'], dim=-1) == 0)

        src_key_padding_mask = torch.cat([padding_mask_ctx, padding_mask_tar], dim=1)
        if not autoreg:
            mask = torch.zeros(num_all, num_all, device='cuda').fill_(float('-inf'))
            mask[:, :num_ctx] = 0.0
        else:
            mask = torch.zeros((num_all+num_tar, num_all+num_tar), device='cuda').fill_(float('-inf'))
            mask[:, :num_ctx] = 0.0 # all points attend to context points
            mask[num_ctx:num_all, num_ctx:num_all].triu_(diagonal=1) # each real target point attends to itself and precedding real target points
            mask[num_all:, num_ctx:num_all].triu_(diagonal=0) # each fake target point attends to preceeding real target points
            src_key_padding_mask = torch.cat([padding_mask_ctx, padding_mask_tar, padding_mask_tar], dim=1)
        return mask, src_key_padding_mask, num_tar

    def encode(self, batch, autoreg=False):
        inp = self.construct_input(batch, autoreg)
        mask, src_key_padding_mask, num_tar = self.create_mask(batch, autoreg)
        embeddings = self.embedder(inp)
        out = self.encoder(embeddings, mask=mask, src_key_padding_mask=src_key_padding_mask)
        return out[:, -num_tar:]
    
class GPO(TNP):
    def __init__(
        self,
        dim_x,
        dim_y,
        d_model,
        emb_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers,
        bound_std=False
    ):
        super(GPO, self).__init__(
            dim_x,
            dim_y,
            d_model,
            emb_depth,
            dim_feedforward,
            nhead,
            dropout,
            num_layers,
            bound_std
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_y*2)
        )

    def forward(self, batch, reduce_ll=True):
        batch_size = batch['xc'].shape[0]
        target_real_lens = (torch.sum(batch['xt'], dim=-1) != 0).sum(1)
        assert torch.max(target_real_lens) == batch['yt'].shape[1], "Max target real lens is not equal to the number of target points"
        z_target = self.encode(batch, autoreg=False)
        out = self.predictor(z_target)
        mean, std = torch.chunk(out, 2, dim=-1)
        tar_q_len = batch['tarqlen']
        start_idx = 0
        softmax_mean = torch.zeros_like(mean)
        for bidx in range(batch_size):
            start_idx = 0
            for num_options in tar_q_len[bidx]: 
                    segment = mean[bidx, start_idx:start_idx + num_options]
                    softmax_segment = softmax(segment, dim=0)
                    softmax_mean[bidx, start_idx:start_idx + num_options] = softmax_segment  # Update the corresponding segment
                    start_idx += num_options
        mean = softmax_mean
        if self.bound_std:
            std = 0.05 + 0.95 * F.softplus(std)
        else:
            std = torch.exp(std)

        pred_tar = Normal(mean, std)
        log_probs = pred_tar.log_prob(batch['yt'])
        masked_log_probs = torch.zeros_like(log_probs)
        # Mask the log probabilities
        for i, length in enumerate(target_real_lens):
            masked_log_probs[i, :length] = log_probs[i, :length]

        outs = AttrDict()
        if reduce_ll:
            outs.tar_ll = masked_log_probs.sum(-1).mean()
        else:
            outs.tar_ll = masked_log_probs.sum(-1)
 
        outs.loss = - (outs.tar_ll)

        return outs

    def predict(self, xc, yc, xt):
        batch = AttrDict()
        batch.xc = xc
        batch.yc = yc
        batch.xt = xt
        batch.yt = torch.zeros((xt.shape[0], xt.shape[1], yc.shape[2]), device='cuda')

        z_target = self.encode(batch, autoreg=False)
        out = self.predictor(z_target)
        mean, std = torch.chunk(out, 2, dim=-1)
        if self.bound_std:
            std = 0.05 + 0.95 * F.softplus(std)
        else:
            std = torch.exp(std)

        return Normal(mean, std)
    
if __name__ == "__main__":
    try:
        parser = HfArgumentParser(ScriptArguments)
        script_args = parser.parse_args_into_dataclasses()[0]
        model_name_split = script_args.model_name.replace("/", "_")
        
        tokenizer, embed_model = get_tokenizer_and_model(script_args, model_type="lm", use_peft=True)
        train_dataset, eval_dataset = load_user_datasets(tokenizer, script_args, uid=None, return_tokenized=False)
        
        embed_save_path = f'./cache_data/{model_name_split}_{script_args.train_dataset_size}_embeddings.pkl'
        
        if not os.path.exists(embed_save_path):
            def make_joint_text(sample):
                if random.random() > 0.5:
                    res_A = sample['chosen']
                    res_B = sample['rejected']
                else:
                    res_A = sample['rejected']
                    res_B = sample['chosen']
                joint_text = f"Prompt: {sample['prompt']}\n\nOption A: {res_A}\n\nOption B: {res_B}"
                joint_text = [{"content": joint_text, "role": "user"}]
                sample['joint_text'] = tokenizer.apply_chat_template(
                    joint_text, tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
                return sample
            train_dataset = train_dataset.map(make_joint_text, num_proc=16)
            df = train_dataset.data.to_pandas()
            print("Training model...")
            embeddings = []
            embed_model = llmodel(script_args.model_name)
            with torch.no_grad():
                for i in tqdm(range(0, len(df), script_args.per_device_train_batch_size)):
                    batch_sentences = df['joint_text'].iloc[i:i+script_args.per_device_train_batch_size].tolist()
                    batch_embeddings = embed_model.get_avg_sentence_embeddings(batch_sentences)
                    embeddings.extend(batch_embeddings.cpu().numpy().tolist())
            df['embedding'] = embeddings    
            df.to_pickle(embed_save_path)
        del embed_model
        
        
        if script_args.model_name == "meta-llama/Llama-2-7b-hf":
            dim_x = 5120
        else:
            raise ValueError("Enter input dimension for the model.")
        model = GPO(dim_x=dim_x, dim_y=1, d_model=128, emb_depth=4, dim_feedforward=128, nhead=4, dropout=0.0, num_layers=6)
        model.cuda()
        
        train_df, eval_df, train_dataset, eval_dataset = load_GPO_datasets(embed_save_path)
        collate_function = CollateFunction(script_args.max_ctx_num_qs, script_args.min_ctx_num_qs, script_args.max_tar_num_qs, script_args.min_tar_num_qs)
        
        train_dataloader = DataLoader(train_dataset, batch_size=script_args.per_device_train_batch_size, collate_fn=collate_function, num_workers=0)
        eval_dataloader = DataLoader(eval_dataset, batch_size=script_args.per_device_eval_batch_size, collate_fn=collate_function, num_workers=0)

        optimizer = torch.optim.Adam(model.parameters(), lr=script_args.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=script_args.num_steps)

        start_step = 1
        best_alignscore = 0
        assert next(model.parameters()).is_cuda
        
        for step in tqdm(range(start_step, script_args.num_steps+1)):
            model.train()
            optimizer.zero_grad()
            
            for batch in train_dataloader:
                batch = {k: v.to('cuda') for k, v in batch.items()}
                # import pdb; pdb.set_trace()
                outs = model(batch)
                outs.loss.backward()
                optimizer.step()
                scheduler.step()

    except Exception as e:
        import pdb
        import traceback

        if not isinstance(e, (pdb.bdb.BdbQuit, KeyboardInterrupt)):
            print("\n" + ">" * 100 + "\n")
            traceback.print_exc()
            print()
            pdb.post_mortem()
        