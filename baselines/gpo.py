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
from utils import load_user_datasets, get_tokenizer_and_model, metric_map, alphabet, reverse_metric_map

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
    controversial: Optional[bool] = field(
        default=True,
        metadata={"help": "If you want to include controversial questions."},
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
    def __init__(self, model_name, quantize=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if quantize:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=True,  # Enable 4-bit quantization
                device_map="auto",  # Automatically places model on GPU if available
                torch_dtype=torch.float16,  # Use float16 for further memory optimization
            )
        else:
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

        collated_batch['y'] = collated_batch['y'].reshape(len(batch), -1, 1)
        collated_batch['yc'] = collated_batch['yc'].reshape(len(batch), -1, 1)
        collated_batch['yt'] = collated_batch['yt'].reshape(len(batch), -1, 1)
        
        import pdb; pdb.set_trace()
        
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


def create_text_columns_ultrafeedback_gpo(script_args, embed_save_path):
    prompts = []
    options = []
    dists = []
    uids = []
    qkeys = []
    prompt_answers = []
    
    qkey = 0
    dataset = load_dataset("RiverDong/ultrafeedback-p", split=f'train')
    if script_args.controversial:
        dataset = dataset.filter(lambda x: x['controversial'] == True)
    dataset = dataset.select(range(min(script_args.train_dataset_size, len(dataset))))
    
    for example in dataset:
        qkey += 1
        uid = reverse_metric_map[example['attributes']]
        if random.random() > 0.5:
            res_A, res_B, dist = example['chosen'], example['rejected'], [1, 0]
        else:
            res_A, res_B, dist = example['rejected'], example['chosen'], [0, 1]
        mc_prompt = f"Which of the following responses would you prefer?\nPrompt: {example['prompt']}\nResponse A: {res_A}\nResponse B: {res_B}" + "\nPreferred Response: "
        prompt_answer = [mc_prompt + alphabet[i] for i in range(2)]
        prompts.append(example['prompt'])
        options.append([res_A, res_B])
        dists.append(dist)
        uids.append(uid)
        qkeys.append(qkey)
        prompt_answers.append(prompt_answer)
        
    ## make a dataframe
    df = pd.DataFrame({'uid': uids, 'prompt': prompts, 'options': options, 'dist': dists, 'qkey': qkeys, 'prompt_answer': prompt_answers})
    embeddings = []
    embed_model = llmodel(script_args.model_name, quantize=False) ## change to false in the last run
    with torch.no_grad():
        # Process the DataFrame in batches
        for start_idx in tqdm(range(0, len(df), script_args.per_device_train_batch_size)):
            end_idx = start_idx + script_args.per_device_train_batch_size
            batch = df.iloc[start_idx:end_idx]
            batch_embeddings = []

            # Gather all prompt_answer lists in the batch
            all_prompt_answers = [prompt for row in batch['prompt_answer'] for prompt in row]
            
            # Compute embeddings in a single forward pass
            if all_prompt_answers:
                all_embeddings = embed_model.get_avg_sentence_embeddings(all_prompt_answers)

                # Split embeddings back into rows
                index = 0
                for row in batch['prompt_answer']:
                    num_prompts = len(row)
                    row_embeddings = all_embeddings[index:index+num_prompts]
                    batch_embeddings.append([emb.cpu().numpy().tolist() for emb in row_embeddings])
                    index += num_prompts

            embeddings.extend(batch_embeddings)
    ## do embedding one by one
    # with torch.no_grad():
    #     for i, row in tqdm(df.iterrows()):
    #         print(i)
    #         row_embeddings = []
    #         for prompt_answer in row['prompt_answer']:
    #             emb = embed_model.get_avg_sentence_embeddings(prompt_answer)[0]
    #             row_embeddings.append(emb.cpu().numpy().tolist())
    #             if torch.isnan(emb[0]).any():
    #                 import pdb; pdb.set_trace()
    #         embeddings.append(row_embeddings)
    df['embedding'] = embeddings
    df.to_pickle(embed_save_path)
    return df

class GPODataset(Dataset):
    def __init__(self, emb_df):
        grouped_by_group = emb_df.groupby('uid')
        self.data = []
        for group_name, group_data in grouped_by_group:
            qkey_list = []
            for idx, row in group_data.iterrows():
                # import pdb; pdb.set_trace()
                qkey = row['qkey']
                embedding = torch.tensor(row['embedding'])
                prob_y = torch.tensor(row['dist'], dtype=torch.float).unsqueeze(0)
                qkey_list.append({
                    'q_emb': embedding,
                    'prob_ys': prob_y,
                    'qkey': qkey
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

if __name__ == "__main__":
    try:
        parser = HfArgumentParser(ScriptArguments)
        script_args = parser.parse_args_into_dataclasses()[0]
        model_name_split = script_args.model_name.replace("/", "_")
        
        # tokenizer, embed_model = get_tokenizer_and_model(script_args, model_type="lm", use_peft=True)
        
        embed_save_path = f'./cache_data/{model_name_split}_{script_args.train_dataset_size}_embeddings.pkl'
        if script_args.model_name == "meta-llama/Llama-2-7b-hf":
            dim_x = 5120
        else:
            raise ValueError("Enter input dimension for the model.")
        
        # emb_dataset = create_text_columns_ultrafeedback_gpo(data, script_args, embed_save_path)
        if os.path.exists(embed_save_path):
            emb_dataset = pd.read_pickle(embed_save_path)
        else:
            emb_dataset = create_text_columns_ultrafeedback_gpo(script_args, embed_save_path)
        print(emb_dataset)
        
        def filter_prompts_for_all_uids(df, uid_column='uid', prompt_column='prompt'):
            # Count unique prompts per uid
            uid_prompt_counts = df.groupby([uid_column, prompt_column]).size().unstack(fill_value=0)
            
            # Identify prompts that appear for every uid
            prompts_for_all_uids = uid_prompt_counts.columns[(uid_prompt_counts > 0).all()]
            
            # Filter dataframe to keep only prompts that appear for every uid
            filtered_df = df[df[prompt_column].isin(prompts_for_all_uids)]
            
            return filtered_df
        
        
        emb_dataset = filter_prompts_for_all_uids(emb_dataset)
        
        uids = emb_dataset['uid'].unique()
        train_users = np.random.choice(uids, int(len(uids) * script_args.group_split_ratio), replace=False)
        eval_users = [uid for uid in uids if uid not in train_users]
        train_df = emb_dataset[emb_dataset['uid'].isin(train_users)]
        eval_df = emb_dataset[emb_dataset['uid'].isin(eval_users)]

        train_dataset = GPODataset(train_df)
        test_dataset = GPODataset(eval_df)
        # import pdb; pdb.set_trace()
        
        model = GPO(dim_x=dim_x, dim_y=1, d_model=128, emb_depth=4, dim_feedforward=128, nhead=4, dropout=0.0, num_layers=6)
        model.cuda()
        
        collate_function = CollateFunction(script_args.max_ctx_num_qs, script_args.min_ctx_num_qs, script_args.max_tar_num_qs, script_args.min_tar_num_qs)
        
        train_dataloader = DataLoader(train_dataset, batch_size=script_args.per_device_train_batch_size, collate_fn=collate_function, num_workers=0)
        eval_dataloader = DataLoader(test_dataset, batch_size=script_args.per_device_eval_batch_size, collate_fn=collate_function, num_workers=0)

        optimizer = torch.optim.Adam(model.parameters(), lr=script_args.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=script_args.num_steps)

        start_step = 1
        best_alignscore = 0
        assert next(model.parameters()).is_cuda
        
        for step in tqdm(range(start_step, script_args.num_steps+1)):
            print(step)
            model.train()
            optimizer.zero_grad()
            
            for batch in train_dataloader:
                batch = {k: v.to('cuda') for k, v in batch.items()}
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
        