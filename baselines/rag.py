import torch
import numpy as np
import random
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModelForCausalLM, HfArgumentParser
from utils import load_user_datasets, get_uids, get_model_gen, judge, get_tokenizer_and_model, get_cohere_gen
from sentence_transformers import SentenceTransformer

@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    embed_target: Optional[str] = field(
        default='prompt', #"paired_pref"
        metadata={"help": "The txet we embed for calculating embedding"},
    )
    k: Optional[int] = field(
        default=3, 
        metadata={"help": "The number of samples for retrieval"},
    )
        
    train_dataset_size: Optional[int] = field(
        default=100000,
        metadata={"help": "The size of the training dataset."},
    )
    eval_dataset_size: Optional[int] = field(
        default=1000,
        metadata={"help": "The size of the eval dataset."},
    )
    model_name: Optional[str] = field(
        default="google/gemma-2b-it", #"mistralai/Mistral-7B-Instruct-v0.2", can also use cohere/openai
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    data_type: str = field(
        default="psoups",
        metadata={"help": "The dataset used for training and testing"}
    )
    subset: Optional[str] = field(
        default='default', ## ood, controversial
        metadata={"help": "The subset of the dataset to use."},
    )
    max_length: Optional[int] = field(default=4096)
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "The rank of the user in the Lora dataset."},
    )
    lora_alpha: Optional[float] = field(
        default=16,
        metadata={"help": "The alpha parameter for the Lora dataset."},
    )
    max_length: Optional[int] = field(default=4096)
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    

def get_embeddings(texts):
    s_bert_model = SentenceTransformer('all-MiniLM-L6-v2').to('cuda')
    embeddings = s_bert_model.encode(texts, convert_to_tensor=True).to('cuda')
    return embeddings

def retrieve_top_k(query_embed, doc_embed, k=3):
    # cos_sim = cosine_similarity(query_embed, doc_embed)
    cos_sim = doc_embed @ prompt_embed
    top_k_indices = np.argsort(cos_sim.cpu().numpy())[-k:]
    return top_k_indices

def create_rag_prompt(user_data):
    # prompt = []
    prompt = "The examples "
    for i, row in enumerate(user_data):
        chosen = row['chosen'].split("\nAssistant: ")[-1].replace('"', "").strip()
        rejected = row['rejected'].split("\nAssistant: ")[-1].replace('"', "").strip()
        if random.random() < 0.5:
            ans = "A"
            res_A, res_B = chosen, rejected
        else:
            ans = "B"
            res_A, res_B = rejected, chosen
        # prompt += [{'content': row['prompt'], "role": "user"}, 
        #            {'content': f"Response A: {res_A}\nResponse B: {res_B}\nWhich response do you prefer?", "role": "assistant"},
        #            {'content': f"Response {ans}\n===\n", "role": "user"}]
        prompt += f"{row['prompt']}\n\nResponse A: {res_A}\nResponse B: {res_B}\n\nWhich response do you prefer?\n\nResponse {ans}\n===\n"
    return prompt

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    model_name_split = script_args.model_name.split("/")[-1]
    method = "rag"
    uids = get_uids(script_args)
    log_path = f'/home/yd358/rds/hpc-work/analysis_pers/baselines/results/{method}_{script_args.embed_target}_{script_args.data_type}_{script_args.subset}_{model_name_split}.log'
    logging.basicConfig(level=logging.INFO, filename=log_path, format='%(asctime)s - %(message)s')

    user_acc = {}
    for uid in uids:
        tokenizer, model = get_tokenizer_and_model(script_args, model_type="lm", use_peft=True)
        model.eval()
        train_dataset, eval_dataset = load_user_datasets(tokenizer, script_args, uid, return_tokenized=False)
        
        train_dataset = train_dataset.map(lambda x: {"paired_pref": f"Prompt: {x['prompt']}\nChosen response: {x['chosen_only']}\nRejected response: {x['rejected_only']}"})
        eval_dataset = eval_dataset.map(lambda x: {"paired_pref": f"Prompt: {x['prompt']}\nChosen response: {x['chosen_only']}\nRejected response: {x['rejected_only']}"})
        train_dataset = train_dataset.map(lambda x: {"chosen_text_version": f"Prompt: {x['prompt']}\n{x['chosen_only']}"})
        eval_dataset = eval_dataset.map(lambda x: {"chosen_text_version": f"Prompt: {x['prompt']}\n{x['chosen_only']}"})
        train_dataset = train_dataset.map(lambda x: {"rejected_text_version": f"Prompt: {x['prompt']}\n{x['rejected_only']}"})
        eval_dataset = eval_dataset.map(lambda x: {"rejected_text_version": f"Prompt: {x['prompt']}\n{x['rejected_only']}"})
        
        if script_args.embed_target == 'prompt':
            doc_embed = get_embeddings(train_dataset['prompt'])
        elif script_args.embed_target == 'paired_pref':
            doc_embed = get_embeddings(train_dataset['paired_pref'])
        elif script_args.embed_target == 'pair_diff':
            doc_embed = get_embeddings(train_dataset['chosen']) - get_embeddings(train_dataset['rejected'])
        elif script_args.embed_target == 'random':
            doc_embed = torch.randn(len(train_dataset), 384).to('cuda')
        acc = []
        
        for i, row in enumerate(eval_dataset):
            if script_args.embed_target == 'prompt':
                prompt_embed = get_embeddings(row['prompt'])
            elif script_args.embed_target == 'paired_pref':
                prompt_embed = get_embeddings(row['paired_pref'])
            elif script_args.embed_target == 'pair_diff':
                prompt_embed = get_embeddings(row['chosen_text_version']) - get_embeddings(row['rejected_text_version'])
            elif script_args.embed_target == 'random':
                prompt_embed = torch.randn(384, 1).to('cuda')
            
            top_k_indices = retrieve_top_k(prompt_embed, doc_embed, k = script_args.k)
            selected_docs = train_dataset.select(top_k_indices)
            prompt = create_rag_prompt(selected_docs)
            
            chosen = row['chosen'].split("\nAssistant: ")[-1].replace('"', "").strip()
            rejected = row['rejected'].split("\nAssistant: ")[-1].replace('"', "").strip()
            if random.random() < 0.5:
                res_A, res_B, ans = chosen, rejected, "A"
            else:
                res_A, res_B, ans = rejected, chosen, "B"
            prompt += f"Prompt: {row['prompt']}\n\nResponse A: {res_A}\n\nResponse B: {res_B}\n\nWhich response do you prefer?\nResponse "
            print(prompt)
            # messages = [{'content': prompt, "role": "user"}]
            # res = get_model_gen(messages, model, tokenizer, max_new_tokens=10)
            if script_args.model_name == 'cohere':
                res = get_cohere_gen(prompt)
            else:
                # messages = [{'content': prompt, "role": "user"}]
                messages = prompt
                res = get_model_gen(messages, model, tokenizer, max_new_tokens=10)
            res = res.split("===")[0].strip()
            import pdb; pdb.set_trace()
            acc.append(ans in res)
            print(f"ans: {ans}, res: {res}, T/F: {ans in res}")
            if i % 100 == 0:
                print(acc)
        user_acc[uid] = np.mean(acc)
    logging.info(f"User accuracy: {user_acc} for model RAG")