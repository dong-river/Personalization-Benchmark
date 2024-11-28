import time
import torch
import random
from datasets import Dataset, load_dataset
from peft import PeftConfig, PeftModel
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModel

from openai import OpenAI
openai_api_key = "sk-proj-ljlZP2CJvxi1QwzTODFnT3BlbkFJHXacxo7EXktI8bmXs7eZ"
client = OpenAI(api_key=openai_api_key)


metric_map = {
    0: "instruction_following",
    1: "honesty",
    2: "truthfulness",
    3: "helpfulness", 
}

reverse_metric_map = {
    "instruction_following": 0,
    "honesty": 1,
    "truthfulness": 2,
    "helpfulness": 3,
}

alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

def get_uids(script_args):
    if script_args.data_path == "openbmb/UltraFeedback":
        num_user = 4
        return list(range(num_user))
    else:
        raise ValueError(f"Invalid data path: {script_args.data_path}")
    
def load_user_datasets(tokenizer, script_args, uid=None, return_tokenized=True, prepend_idx=False, controversial=True):
    if script_args.data_path == "openbmb/UltraFeedback":
        return build_vanilla_ultrafeedback_p_dataset(tokenizer, script_args, uid, return_tokenized, prepend_idx)

def build_vanilla_ultrafeedback_p_dataset(tokenizer, script_args, uid = None, return_tokenized=True, prepend_idx=False, controversial=True):
    def tokenize(sample):                    
        sample['positive'] = tokenizer.apply_chat_template(
            sample['chosen'], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
        sample['negative'] = tokenizer.apply_chat_template(
            sample['rejected'], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
        
        tokenized_pos = tokenizer(sample['positive'], truncation=True)
        tokenized_neg = tokenizer(sample['negative'], truncation=True)
        sample["input_ids_chosen"] = tokenized_pos["input_ids"]
        sample["attention_mask_chosen"] = tokenized_pos["attention_mask"]
        sample["input_ids_rejected"] = tokenized_neg["input_ids"]
        sample["attention_mask_rejected"] = tokenized_neg["attention_mask"]
        sample["uid"] = sample["uid"]
        return sample

    print("Loading UltraFeedback dataset")
    train_dataset = load_dataset("RiverDong/ultrafeedback-p", split=f'train')
    test_dataset = load_dataset("RiverDong/ultrafeedback-p", split=f'test')
    
    if controversial:
        train_dataset = train_dataset.filter(lambda x: x['controversial'] == True)
        test_dataset = test_dataset.filter(lambda x: x['controversial'] == True)
    
    train_dataset = train_dataset.select(range(min(script_args.train_dataset_size, len(train_dataset))))
    test_dataset = test_dataset.select(range(min(script_args.eval_data_size, len(test_dataset))))
    
    ## map the attributes to the uid
    train_dataset = train_dataset.map(lambda x: {"uid": reverse_metric_map[x["attributes"]]}, remove_columns=["attributes"])
    test_dataset = test_dataset.map(lambda x: {"uid": reverse_metric_map[x["attributes"]]}, remove_columns=["attributes"])
    ## set chosen and rejected to the correct format
    train_dataset = train_dataset.rename_column("chosen", "chosen_only")
    train_dataset = train_dataset.rename_column("rejected", "rejected_only")
    test_dataset = test_dataset.rename_column("chosen", "chosen_only")
    test_dataset = test_dataset.rename_column("rejected", "rejected_only")
    
    if prepend_idx: ## Only for id-conditioned Fine-tuning, add "User ID: {uid}" to the prompt
        # train_dataset = train_dataset.map(lambda x: {"prompt": f"User ID: {x['uid']}\n{x['prompt']}"})
        # test_dataset = test_dataset.map(lambda x: {"prompt": f"User ID: {x['uid']}\n{x['prompt']}"})
        train_dataset = train_dataset.map(lambda x: {"prompt": f"User ID: {x['uid']}\n{x['prompt']}"})
        test_dataset = test_dataset.map(lambda x: {"prompt": f"User ID: {x['uid']}\n{x['prompt']}"})
        
    train_dataset = train_dataset.map(lambda x: {"chosen": [{"content": x["prompt"], "role": "user"}, {"content": x["chosen_only"], "role": "assistant"}]})
    train_dataset = train_dataset.map(lambda x: {"rejected": [{"content": x["prompt"], "role": "user"}, {"content": x["rejected_only"], "role": "assistant"}]})
    test_dataset = test_dataset.map(lambda x: {"chosen": [{"content": x["prompt"], "role": "user"}, {"content": x["chosen_only"], "role": "assistant"}]})
    test_dataset = test_dataset.map(lambda x: {"rejected": [{"content": x["prompt"], "role": "user"}, {"content": x["rejected_only"], "role": "assistant"}]})
    

    train_dataset = train_dataset.filter(lambda x: x['prompt'] is not None and x['chosen'] is not None and x['rejected'] is not None)
    test_dataset = test_dataset.filter(lambda x: x['prompt'] is not None and x['chosen'] is not None and x['rejected'] is not None)
        
    if return_tokenized:
        train_dataset = train_dataset.map(tokenize, num_proc=16)
        test_dataset = test_dataset.map(tokenize, num_proc=16)

    if uid is not None:
        train_dataset = train_dataset.filter(lambda x: x['uid'] == uid)
        test_dataset = test_dataset.filter(lambda x: x['uid'] == uid)
    
    print("Training set: ", len(train_dataset), " test set: ", len(test_dataset))
    
    return train_dataset, test_dataset

# def build_vanilla_ultrafeedback_p_dataset(tokenizer, script_args, uid = None, return_tokenized=True):
#     def tokenize(sample):                    
#         sample['positive'] = tokenizer.apply_chat_template(
#             sample['chosen'], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
#         sample['negative'] = tokenizer.apply_chat_template(
#             sample['rejected'], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
        
#         tokenized_pos = tokenizer(sample['positive'], truncation=True)
#         tokenized_neg = tokenizer(sample['negative'], truncation=True)
#         sample["input_ids_chosen"] = tokenized_pos["input_ids"]
#         sample["attention_mask_chosen"] = tokenized_pos["attention_mask"]
#         sample["input_ids_rejected"] = tokenized_neg["input_ids"]
#         sample["attention_mask_rejected"] = tokenized_neg["attention_mask"]
#         sample["uid"] = sample["uid"]
#         return sample

#     print("Loading UltraFeedback dataset")
#     data = load_dataset("openbmb/UltraFeedback", split=f'train[:{script_args.train_dataset_size}]')
#     split = data.train_test_split(test_size=script_args.eval_data_size)
#     train_dataset, test_dataset = split['train'], split['test']
#     train_dataset = create_text_columns_ultrafeedback(train_dataset)
#     test_dataset = create_text_columns_ultrafeedback(test_dataset)
#     train_dataset = train_dataset.filter(lambda x: x['prompt'] is not None and x['chosen'] is not None and x['rejected'] is not None)
#     test_dataset = test_dataset.filter(lambda x: x['prompt'] is not None and x['chosen'] is not None and x['rejected'] is not None)
#     if return_tokenized:
#         train_dataset = train_dataset.map(tokenize, num_proc=16)
#         test_dataset = test_dataset.map(tokenize, num_proc=16)

#     if uid is not None:
#         train_dataset = train_dataset.filter(lambda x: x['uid'] == uid)
#         test_dataset = test_dataset.filter(lambda x: x['uid'] == uid)
    
#     print("Training set: ", len(train_dataset), " test set: ", len(test_dataset))
    
#     return train_dataset, test_dataset


# def create_text_columns_ultrafeedback(dataset, controversial=False):
#     prompts = []
#     chosens = []
#     rejecteds = []
#     uids = []
#     chosen_onlys = []
#     rejected_onlys = []
    
#     for example in dataset:
#         for uid, metric in metric_map.items():
#             instruction = example["instruction"]
#             completions = example["completions"]
#             chosen, rejected, chosen_score, rejected_score, chosen_only, rejected_only = None, None, 0, 20, None, None
#             if controversial and find_disagreement(completions):
#                 continue
#             try:
#                 for completion in completions:
#                     score = int(completion['annotations'][metric]["Rating"])
#                     if score > chosen_score:
#                         chosen_only = completion['response']
#                         chosen = [{"content": instruction, "role": "user"}, {"content": completion['response'], "role": "assistant"}]
#                         chosen_score = score
#                     if score < rejected_score:
#                         rejected_only = completion['response']
#                         rejected = [{"content": instruction, "role": "user"}, {"content": completion['response'], "role": "assistant"}]
#                         rejected_score = score
#             except Exception as e:
#                 continue

#             if chosen and rejected and chosen_score > rejected_score:
#                 prompts.append(instruction)
#                 chosens.append(chosen)
#                 rejecteds.append(rejected)
#                 uids.append(uid)
#                 chosen_onlys.append(chosen_only)
#                 rejected_onlys.append(rejected_only)

#     # Create a new dataset with the raw text columns
#     raw_text_data = Dataset.from_dict({
#         "prompt": prompts,
#         "chosen": chosens,
#         "rejected": rejecteds,
#         "uid": uids,
#         "chosen_only": chosen_onlys,
#         "rejected_only": rejected_onlys
#     })
    
#     return raw_text_data

def load_peft_model_lm(output_path):
    config = PeftConfig.from_pretrained(output_path)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, num_labels=1)
    model = PeftModel.from_pretrained(model, output_path)
    model = model.merge_and_unload()
    model = model.to('cuda') 
    model.eval()
    return model

def get_tokenizer_and_model(script_args, model_type, use_peft=True):
    tokenizer_name = script_args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True, add_eos_token=False)

    # Need to do this for GPT2 and Llama because they don't have official pad tokens.
    if 'gpt2' in script_args.model_name or 'llama' in script_args.model_name:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.model_max_length = script_args.max_length
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "left"
    elif "gemma" in script_args.model_name.lower():
        tokenizer.truncation_side = "left"
        tokenizer.model_max_length = script_args.max_length
    # elif "Ray" in script_args.model_name:
    else:
        raise ValueError(f"please config the tokenizer: {script_args.model_name}")
        

    if model_type == "seq_cls" or model_type == "rm":
        model = AutoModelForSequenceClassification.from_pretrained(
            script_args.model_name, torch_dtype=torch.bfloat16, num_labels=1
        )
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=script_args.lora_rank,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model.base_model.config.num_labels = 1
        model.config.num_labels = 1
    elif model_type == "causal_lm" or model_type == "lm":
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name, torch_dtype=torch.bfloat16, output_hidden_states=True
        ).to('cuda')
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=script_args.lora_rank,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    if use_peft:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    model.config.use_cache = not script_args.gradient_checkpointing
    model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model

def get_model_gen(prompt, model, tokenizer, max_new_tokens=1000, remove_prompt=True):
    if "gemma" in model.config.model_type or "llama" in model.config.model_type:
        input_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=False, return_tensors="pt").to('cuda')
    else:
        print("You should only see this if you are running id_rm.py")
        input_ids = tokenizer(text=prompt, return_tensors="pt").input_ids.to('cuda')
    output_ids = model.generate(input_ids, num_return_sequences=1, temperature=0.7, top_p=0.95, do_sample=True, max_new_tokens=max_new_tokens)
    ## remove the prompt from the output
    if remove_prompt:
        res = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
    else:
        res = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # res = res.replace(prompt, "").strip()
    return res

def judge(persona, query, res_A, res_B, judge_model='gpt-4o'):
    random_number = random.random()
    if random_number < 0.5: ## In that case return True if B is better
        tmp = res_A
        res_A = res_B
        res_B = tmp
        
    prompt = f"""Given the user profile provided below, select the response from AI assistant A or B that the user would most likely prefer. 

[User Profile]
{persona}

[User Question]
{query}

[The Start of Assistant A's Answer]
{res_A}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{res_B}
[The End of Assistant B's Answer]

[Answer]
[[""".format(persona=persona, query=query, res_A=res_A, res_B=res_B)
    response = get_openai_gen(prompt, model=judge_model).replace("[[","")[0]
    if response == "A":
        return res_A
    else:
        return res_B

def get_openai_gen(prompt, system_prompt = "", model = 'gpt-3.5-turbo', max_tokens = 2048, temperature = 0.7, stop_strs = None, max_depth = 3, cur_depth = 0):
    try:
        if cur_depth >= max_depth:
            return "Sorry, I am not able to answer that question."
        if type(prompt) == list:
            ## In this case, make sure the prompt list is in the correct order: user, assistant, user, assistant, ...
            messages = [{"role": "user", "content": p} if idx % 2 == 0 else {"role": "assistant", "content": p} for idx, p in enumerate(prompt)]
        elif type(prompt) == str:
            messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stop=stop_strs,
            temperature=temperature,
            logprobs=True,
            top_logprobs=5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(e)
        time.sleep(30)
        return get_openai_gen(prompt, cur_depth=cur_depth + 1)
    
def load_peft_model_rm(output_path):
    try:
        config = PeftConfig.from_pretrained(output_path)
        model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=1)
        model = PeftModel.from_pretrained(model, output_path)
        model = model.merge_and_unload()
        model = model.to('cuda') 
    except:
        model = AutoModelForSequenceClassification.from_pretrained(output_path, num_labels=1)
        model = model.to('cuda')
    return model

class BestOfNSampler:
    def __init__(self, gen_model, reward_model, tokenizer, device='cuda'):
        """
        Initialize the BestOfNSampler class with a causal language model and reward model.
        
        Args:
            gen_model_name (str): Pretrained model name or path for the causal language model.
            reward_model_name (str): Pretrained model name or path for the reward model.
            tokenizer_name (str, optional): Pretrained tokenizer name or path. If None, uses gen_model_name.
            device (str): Device to use ('cpu' or 'cuda').
        """
        self.gen_model = gen_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device = device
    
    def generate_samples(self, input_text, N=5, max_length=100, **gen_kwargs):
        """
        Generate N different samples from the causal language model.
        
        Args:
            input_text (str): The prompt or input text to generate from.
            N (int): Number of samples to generate.
            max_length (int): Maximum length of generated sequences.
            **gen_kwargs: Additional arguments for the generate() function.
        
        Returns:
            list: List of generated text samples.
        """
        input_ids = self.tokenizer.encode("Human: " + input_text, return_tensors="pt").to(self.device)
        if "User ID" in input_text:
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device) ## We already added User ID and "Human: " in the prompt for id_rm.py
            print("You should only see this if you are running id_rm.py") 
        samples = []
        for _ in range(N):
            output = self.gen_model.generate(
                input_ids, 
                max_length=max_length, 
                do_sample=True, 
                **gen_kwargs
            )
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            samples.append(generated_text)
        return samples
    
    def score_samples(self, samples, user_type=0):
        """
        Score each generated sample using the reward model.

        Args:
            samples (list): List of generated text samples.
            user_type (int): Identifier for the user type.

        Returns:
            list: List of scores corresponding to each sample.
        """
        scores = []
        for sample in samples:
            inputs = self.tokenizer(sample, return_tensors="pt", truncation=True).to(self.device)
            with torch.no_grad():
                reward = self.reward_model.compute_reward(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    user_type=user_type
                )
            score = reward.item()
            scores.append(score)
        return scores

    
    def best_sample(self, input_text, N=5, max_length=512, **gen_kwargs):
        """
        Generate N samples, score them, and return the best sample based on reward model score.
        
        Args:
            input_text (str): The prompt or input text to generate from.
            N (int): Number of samples to generate.
            max_length (int): Maximum length of generated sequences.
            **gen_kwargs: Additional arguments for the generate() function.
        
        Returns:
            dict: Dictionary containing the best sample and its score.
        """
        # Step 1: Generate N samples
        samples = self.generate_samples(input_text, N=N, max_length=max_length, **gen_kwargs)
        
        # Step 2: Score the samples
        scores = self.score_samples(samples)
        
        # Step 3: Select the best sample
        best_idx = torch.argmax(torch.tensor(scores))
        best_sample = samples[best_idx]
        best_score = scores[best_idx]
        
        return {
            'best_sample': best_sample,
            'best_score': best_score,
            'all_samples': samples,
            'all_scores': scores
        }
        
        
# def find_disagreement(responses):
#     n = len(responses)
#     annotations = metric_map.values()
#     found = False

#     # Compare each pair of responses
#     for i in range(n):
#         for j in range(n):
#             if i == j:
#                 continue
#             response_A = responses[i]
#             response_B = responses[j]

#             # Compare ratings across different annotations
#             for ann1 in annotations:
#                 for ann2 in annotations:
#                     if ann1 == ann2:
#                         continue
#                     rating_A_ann1 = int(response_A['annotations'][ann1]['Rating'])
#                     rating_B_ann1 = int(response_B['annotations'][ann1]['Rating'])
#                     rating_A_ann2 = int(response_A['annotations'][ann2]['Rating'])
#                     rating_B_ann2 = int(response_B['annotations'][ann2]['Rating'])

#                     # Check if A is preferred over B in ann1, but B is preferred over A in ann2
#                     if rating_A_ann1 > rating_B_ann1 and rating_A_ann2 < rating_B_ann2:
#                         print(f"Response {i+1} is preferred over Response {j+1} in '{ann1}', "
#                               f"but Response {j+1} is preferred over Response {i+1} in '{ann2}'.")
#                         print(f"Response {i+1} '{ann1}' Rating: {rating_A_ann1}")
#                         print(f"Response {j+1} '{ann1}' Rating: {rating_B_ann1}")
#                         print(f"Response {i+1} '{ann2}' Rating: {rating_A_ann2}")
#                         print(f"Response {j+1} '{ann2}' Rating: {rating_B_ann2}")
#                         print("-" * 50)
#                         found = True
#     return found