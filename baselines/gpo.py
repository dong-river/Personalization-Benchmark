import os
import torch
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import logging
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, HfArgumentParser
from torch.utils.data import DataLoader
from attrdict import AttrDict
from gpo_utils import GPODataset, CollateFunction, calculate_acc, llmodel, GPO
from utils import load_user_datasets
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    dropout: Optional[float] = field(
        default=0,
        metadata={"help": "The dropout rate."},
    )
    subset: Optional[str] = field(
        default='controversial', ## controversial, default
        metadata={"help": "The subset of the dataset to use."},
    )
    data_type: str = field(
        default="ultrafeedback", # ultrafeedback or psoups
    )
    embed_batch_size: Optional[int] = field(
        default=1,
    )
    eval_freq: Optional[int] = field(
        default=10,
        metadata={"help": "The frequency of evaluation."},
    )
    eval_seed: Optional[int] = field(
        default=0,
        metadata={"help": "The seed for evaluation."},
    )
    eval_num_qs: Optional[int] = field(
        default=20,
        metadata={"help": "The number of questions to evaluate."},
    )
    eval_num_steps: Optional[int] = field(
        default=10,
        metadata={"help": "The number of steps to evaluate."},
    )
    eval_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "directly load a pre-trained model to evaluate."},
    )
    
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
        default=500000,
        metadata={"help": "The number of training steps."},
    )
    train_dataset_size: Optional[int] = field(
        default=10000,
        metadata={"help": "The size of the training dataset."},
    )
    eval_dataset_size: Optional[int] = field(
        default=100,
        metadata={"help": "The size of the eval dataset."},
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
    per_device_train_batch_size: Optional[int] = field(default=32)
    per_device_eval_batch_size: Optional[int] = field(default=16)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=1e-4)
    weight_decay: Optional[float] = field(default=0.0)
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
        default="/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/models",
        metadata={"help": "The dir for output model"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=4096)

def _create_prompt_df(script_args, tokenizer):
    qkey = 0
    train_dataset, test_dataset = load_user_datasets(tokenizer, script_args, return_tokenized=False, subset=script_args.subset)
    
    def create_dataframe(dataset):
        nonlocal qkey  # Allows updating qkey across datasets
        prompts = []
        options = []
        dists = []
        uids = []
        qkeys = []
        prompt_answers = []
        for example in dataset:
            qkey += 1
            uid = example['uid']
            if random.random() > 0.5:
                res_A, res_B, dist = example['chosen_only'], example['rejected_only'], [1, 0]
            else:
                res_A, res_B, dist = example['rejected_only'], example['chosen_only'], [0, 1]
                
            prompt_answer = []
            for res in [res_A, res_B]:
                mc_prompt = [
                    # {'content': f"Prompt: {example['prompt']}\n", 'role': 'user'},
                    # {'content': f"Response: {res}\n", 'role': 'assistant'},
                    {'content': f"{example['prompt']}\n", 'role': 'user'},
                    {'content': f"{res}\n", 'role': 'assistant'},
                ]
                mc_prompt = tokenizer.apply_chat_template(mc_prompt, tokenize=False)
                prompt_answer.append(mc_prompt)
            prompts.append(example['prompt'])
            options.append([res_A, res_B])
            dists.append(dist)
            uids.append(uid)
            qkeys.append(qkey)
            prompt_answers.append(prompt_answer)
        
        # Create a DataFrame for the given dataset
        return pd.DataFrame({
            'uid': uids,
            'prompt': prompts,
            'options': options,
            'dist': dists,
            'qkey': qkeys,
            'prompt_answer': prompt_answers
        })
    
    # Create DataFrames for train and test datasets
    train_df = create_dataframe(train_dataset)
    test_df = create_dataframe(test_dataset)
    
    return train_df, test_df

def create_embeddings(script_args, train_save_path, test_save_path):
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    train_df, test_df = _create_prompt_df(script_args, tokenizer)
    embed_model = llmodel(script_args.model_name, quantize=False)  # Change to False in the final run
    
    # Wrap in DataParallel if you have multiple GPUs
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs for embedding creation.")
    #     embed_model = nn.DataParallel(embed_model)

    # Create a mapping of DataFrame to its respective save path
    df_save_mapping = {
        "train": (train_df, train_save_path),
        "test": (test_df, test_save_path),
    }

    # Process and save both DataFrames in a loop
    for split, (df, save_path) in df_save_mapping.items():
        print(f"Processing {split}_df and saving to {save_path}")
        embeddings = []
        with torch.no_grad():
            for start_idx in tqdm(range(0, len(df), script_args.embed_batch_size)):
                end_idx = start_idx + script_args.embed_batch_size
                batch = df.iloc[start_idx:end_idx]
                batch_embeddings = []
                all_prompt_answers = [prompt for row in batch['prompt_answer'] for prompt in row]
                if all_prompt_answers:
                    all_embeddings = embed_model.get_avg_sentence_embeddings(all_prompt_answers)
                    index = 0
                    for row in batch['prompt_answer']:
                        num_prompts = len(row)
                        row_embeddings = all_embeddings[index:index + num_prompts]
                        batch_embeddings.append([emb.cpu().numpy().tolist() for emb in row_embeddings])
                        index += num_prompts

                embeddings.extend(batch_embeddings)
        df['embedding'] = embeddings
        df.to_pickle(save_path)
    return train_df, test_df

    ## generating embedding one by one
    # with torch.no_grad():
    #     for i, row in tqdm(df.iterrows()):
    #         print(i)
    #         row_embeddings = []
    #         for prompt_answer in row['prompt_answer']:
    #             emb = embed_model.get_avg_sentence_embeddings(prompt_answer)[0]
    #             row_embeddings.append(emb.cpu().numpy().tolist())
    #             if torch.isnan(emb[0]).any():
    #         embeddings.append(row_embeddings)

def calculate_acc_test(args, model, dataset, mode='eval', logging=True):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    acc_list = {}
    print("Evaluation with train and test seperate")
    for i, batch in enumerate(dataloader):
        acc_group = []
        this_group = batch['groups']
        print(this_group)
        group_questions = batch['questions']
        train_questions = [q for q in group_questions if q['is_train']]
        test_questions = [q for q in group_questions if not q['is_train']]
        num_train_questions = len(train_questions)
        num_test_questions = len(test_questions)
        context_questions = np.random.choice(np.arange(num_train_questions), size=args.eval_num_qs, replace=False)
        target_questions = np.arange(num_test_questions)
        # Now, let's collect the context embeddings and probabilities.
        ctx_embeddings = []
        ctx_prob_ys = []
        tar_embeddings = []
        tar_prob_ys = []
        for context_q_idx in context_questions:
            ctx_embeddings.append(train_questions[context_q_idx]['q_emb'])
            ctx_prob_ys.append(train_questions[context_q_idx]['prob_ys'][0])
        ctx_embeddings = torch.cat(ctx_embeddings, dim=1).to('cuda')
        ctx_prob_ys = torch.cat(ctx_prob_ys, dim=1).unsqueeze(-1).to('cuda', dtype=torch.float)
        for target_q_idx in target_questions:
            tar_embeddings = test_questions[target_q_idx]['q_emb'].to('cuda')
            tar_prob_ys = test_questions[target_q_idx]['prob_ys'].to('cuda')
            with torch.no_grad():
                predicted_distribution = model.predict(ctx_embeddings, ctx_prob_ys, tar_embeddings).loc
                predicted_distribution = predicted_distribution.cpu().detach().numpy().squeeze()
                target_distribution = tar_prob_ys.cpu().detach().numpy().squeeze()
                pred = predicted_distribution.argmax()
                label = target_distribution.argmax()
                acc_group.append(pred == label)
        acc_group_mean = np.mean(acc_group)
        acc_list[this_group] = acc_group_mean
        # print(f"{mode.capitalize()}_test acc_{this_group}: {acc_group_mean}")
    acc_overall = np.mean(list(acc_list.values()))
    print(f"Test mean acc: {acc_overall}")
    return acc_overall, acc_list

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    model_name_split = script_args.model_name.replace("/", "_")
    
    ## Set paths
    base_output_name = f'GPO_{model_name_split}_{script_args.train_dataset_size}_{script_args.data_type}_{script_args.subset}_{script_args.lora_rank}_{script_args.num_steps}_{script_args.learning_rate}_{script_args.dropout}_{script_args.weight_decay}'
    base_output_name = script_args.eval_model_path.replace(".", "_") if script_args.eval_model_path else base_output_name
    embed_base_name = f'GPO_embedding_{model_name_split}_{script_args.train_dataset_size}_{script_args.data_type}_{script_args.subset}'
    embed_save_path_train = os.path.join(script_args.log_dir, embed_base_name + '_train.pkl')
    embed_save_path_test = os.path.join(script_args.log_dir, embed_base_name + '_test.pkl')
    log_path = f'results/{base_output_name.replace("/", "_")}.log'
    logging.basicConfig(level=logging.INFO, filename=log_path, format='%(asctime)s - %(message)s')

    if script_args.model_name == "meta-llama/Llama-2-7b-hf":
        dim_x = 4096
    elif script_args.model_name == "google/gemma-2b-it":
        dim_x = 2048
    else:
        raise ValueError("Enter input dimension for the model.")
    
    ## Create embeddings
    if os.path.exists(embed_save_path_train) and os.path.exists(embed_save_path_test):
        emb_train_df = pd.read_pickle(embed_save_path_train)
        emb_test_df = pd.read_pickle(embed_save_path_test)
    else:
        emb_train_df, emb_test_df = create_embeddings(script_args, embed_save_path_train, embed_save_path_test)
    print(emb_train_df)
    
    emb_train_df['is_train'] = True
    emb_test_df['is_train'] = False
    emb_test_df = pd.concat([emb_train_df, emb_test_df])
    emb_test_df = emb_test_df.sample(frac=1).reset_index(drop=True)

    ## Load data
    train_dataset = GPODataset(emb_train_df)
    test_dataset = GPODataset(emb_test_df)
    collate_function = CollateFunction(script_args.max_ctx_num_qs, script_args.min_ctx_num_qs, script_args.max_tar_num_qs, script_args.min_tar_num_qs)
    train_dataloader = DataLoader(train_dataset, batch_size=script_args.per_device_train_batch_size, collate_fn=collate_function, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=script_args.per_device_eval_batch_size, collate_fn=collate_function, num_workers=0)
    
    ## Load model
    model = GPO(dim_x=dim_x, dim_y=1, d_model=128, emb_depth=4, dim_feedforward=128, nhead=4, dropout=script_args.dropout, num_layers=6)
    model.cuda()
    
    if script_args.eval_model_path == None and not os.path.exists(os.path.join(script_args.log_dir, base_output_name)):
        optimizer = torch.optim.Adam(model.parameters(), lr=script_args.learning_rate, weight_decay=script_args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=script_args.num_steps)
        
        ## Start training
        start_step = 1
        best_acc = 0
        for step in range(start_step, script_args.num_steps+1):
            model.train()
            optimizer.zero_grad()
            
            for batch in train_dataloader:
                batch = {k: v.to('cuda') for k, v in batch.items()}
                outs = model(batch)
                outs.loss.backward()
                optimizer.step()
                scheduler.step()
            
            ## Evaluating model
            if step % script_args.eval_freq == 0:
                train_acc = calculate_acc(script_args, model, train_dataset, mode='eval')
                test_acc, test_acc_list = calculate_acc_test(script_args, model, test_dataset, mode='eval')
                logging.info(f"Step: {step}, train set accuracy: {train_acc}, test set accuracy: {test_acc}")
                logging.info(f"Test set accuracy for each user: {test_acc_list}")
                
                ## Saving checkpoints
                if train_acc > best_acc:
                    best_acc = train_acc
                    ckpt = AttrDict()
                    ckpt.model = model.state_dict()
                    ckpt.optimizer = optimizer.state_dict()
                    ckpt.scheduler = scheduler.state_dict()
                    ckpt.step = step + 1
                    torch.save(ckpt, os.path.join(script_args.log_dir, base_output_name + '.tar'))
    
    else:
        checkpoint_path = script_args.eval_model_path if script_args.eval_model_path else os.path.join(script_args.log_dir, base_output_name + '.tar')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint.model)
        
        train_acc = calculate_acc(script_args, model, train_dataset, mode='eval')
        test_acc, test_acc_list = calculate_acc_test(script_args, model, test_dataset, mode='eval')
        logging.info(f"train set accuracy: {train_acc}, test set accuracy: {test_acc}")
        for uid, user_acc in test_acc_list:
            logging.info(f"Metric for user {uid+1}: {user_acc}")
        print(f"train set accuracy: {train_acc}, test set accuracy: {test_acc}")
        print(f"Test set accuracy for each user: {test_acc_list}")