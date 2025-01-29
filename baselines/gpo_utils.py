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
from torch.nn import DataParallel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding, HfArgumentParser


class llmodel:
    def __init__(self, model_name, quantize=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if quantize:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        else:
            # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
            model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(self.device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for inference.")
            model = DataParallel(model, device_ids=[0,1,2,3])
            
        tokenizer.pad_token = tokenizer.eos_token
        # model.config.pad_token_id = tokenizer.eos_token_id
        if model.module if hasattr(model, 'module') else model is not None:
            (model.module if hasattr(model, 'module') else model).config.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer 
        self.model = model
        self.model.eval()        

    def get_avg_sentence_embeddings(self, sentences):
        # Tokenize a batch of sentences and feed them to the model.
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_hidden_states = outputs.hidden_states[-1]
            mask = inputs['attention_mask'].to(self.device)
            mask_expanded = mask.unsqueeze(-1).expand(all_hidden_states.size())
            sum_hidden_states = torch.sum(all_hidden_states * mask_expanded, 1)
            sentence_embeddings = sum_hidden_states / mask_expanded.sum(1)
        return sentence_embeddings


def build_mlp(dim_in, dim_hid, dim_out, depth):
    modules = [nn.Linear(dim_in, dim_hid), nn.ReLU(True)]
    for _ in range(depth-2):
        modules.append(nn.Linear(dim_hid, dim_hid))
        modules.append(nn.ReLU(True))
    modules.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*modules)


# def build_mlp(dim_in, dim_hid, dim_out, depth):
#     modules = []
#     modules.append(nn.Linear(dim_in, dim_hid))
#     modules.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))  # Use LeakyReLU

#     for _ in range(depth - 2):
#         modules.append(nn.Linear(dim_hid, dim_hid))
#         modules.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))  # Use LeakyReLU

#     modules.append(nn.Linear(dim_hid, dim_out))
    
#     mlp = nn.Sequential(*modules)
    
#     for m in mlp:
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
                
#     return mlp

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
        if torch.isnan(inp).any() or torch.isnan(src_key_padding_mask).any():
            import pdb; pdb.set_trace()
        embeddings = self.embedder(inp)
        if torch.isnan(embeddings).any():
            import pdb; pdb.set_trace()
        out = self.encoder(embeddings, mask=mask, src_key_padding_mask=src_key_padding_mask)
        if torch.isnan(out).any():
            import pdb; pdb.set_trace()
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

        mean = torch.nan_to_num(mean, nan=0.0, posinf=1e6, neginf=-1e6)
        std = torch.nan_to_num(std, nan=1.0, posinf=1e6, neginf=1e-6)
        std = torch.clamp(std, min=1e-6)

        pred_tar = Normal(mean, std)
        log_probs = pred_tar.log_prob(batch['yt'])
        # print(log_probs)
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

class GPODataset(Dataset):
    def __init__(self, emb_df):
        grouped_by_group = emb_df.groupby('uid')
        self.data = []
        for group_name, group_data in grouped_by_group:
            qkey_list = []
            for idx, row in group_data.iterrows():
                qkey = row['qkey']
                embedding = torch.tensor(row['embedding'])
                prob_y = torch.tensor(row['dist'], dtype=torch.float).unsqueeze(0)
                qkey_list.append({
                    'q_emb': embedding,
                    'prob_ys': prob_y,
                    'qkey': qkey,
                    'is_train': row['is_train'],
                })
            self.data.append({
                'groups': group_name,
                'qkeys': qkey_list,
            })
                        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        group_data = self.data[idx]

        return {
            'questions': group_data['qkeys'],
            'groups': group_data['groups'],
        }

def calculate_acc(args, model, dataset, mode='eval', logging=True):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    acc_list = []
    for i, batch in enumerate(dataloader):
        acc_group = []
        this_group = batch['groups']
        print(this_group)
        group_questions = batch['questions']
        num_questions = len(group_questions)
        context_questions = np.random.choice(np.arange(num_questions), size=args.eval_num_qs, replace=False)
        target_questions = np.setdiff1d(np.arange(num_questions), context_questions)
        # Now, let's collect the context embeddings and probabilities.
        ctx_embeddings = []
        ctx_prob_ys = []
        tar_embeddings = []
        tar_prob_ys = []
        for context_q_idx in context_questions:
            ctx_embeddings.append(group_questions[context_q_idx]['q_emb'])
            ctx_prob_ys.append(group_questions[context_q_idx]['prob_ys'][0])
        ctx_embeddings = torch.cat(ctx_embeddings, dim=1).to('cuda')
        ctx_prob_ys = torch.cat(ctx_prob_ys, dim=1).unsqueeze(-1).to('cuda', dtype=torch.float)
        for target_q_idx in target_questions:
            tar_embeddings = group_questions[target_q_idx]['q_emb'].to('cuda')
            tar_prob_ys = group_questions[target_q_idx]['prob_ys'].to('cuda')
            with torch.no_grad():
                predicted_distribution = model.predict(ctx_embeddings, ctx_prob_ys, tar_embeddings).loc
                predicted_distribution = predicted_distribution.cpu().detach().numpy().squeeze()
                target_distribution = tar_prob_ys.cpu().detach().numpy().squeeze()
                pred = predicted_distribution.argmax()
                label = target_distribution.argmax()
                acc_list.append(pred == label)
                acc_group.append(pred == label)
        acc_group_mean = np.mean(acc_group)
        print(f"{mode.capitalize()}_alignment_score_{this_group}: {acc_group_mean}")
    acc_overall = np.mean(acc_list)
    print(f"{mode.capitalize()} Mean acc: {acc_overall}")
    return acc_overall

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
            assert num_ctx + num_tar <= len(b['questions']), "Not enough questions in this data item to form a batch."
            num_questions = len(b['questions'])
            perm_indices = torch.randperm(num_questions)
            ctx_indices = perm_indices[:num_ctx]
            tar_indices = perm_indices[num_ctx:num_ctx+num_tar]
            ctx_qs = [b['questions'][i] for i in ctx_indices]
            tar_qs = [b['questions'][i] for i in tar_indices]
            
            ctx_pa = torch.cat([q['q_emb'] for q in ctx_qs], dim=0)
            ctx_prob_ys = torch.cat([q['prob_ys'][0] for q in ctx_qs], dim=0)
            tar_pa = torch.cat([q['q_emb'] for q in tar_qs], dim=0)
            tar_prob_ys = torch.cat([q['prob_ys'][0] for q in tar_qs], dim=0)

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

        collated_batch['y'] = collated_batch['y'].reshape(len(batch), -1, 1)
        collated_batch['yc'] = collated_batch['yc'].reshape(len(batch), -1, 1)
        collated_batch['yt'] = collated_batch['yt'].reshape(len(batch), -1, 1)
        
        return collated_batch
