import os
import numpy as np
from datasets import load_dataset, load_from_disk, Dataset
import json
import random
from collections import Counter

def encode_selected_users(file_name="./selected_users.txt"):
    user_list = np.genfromtxt(file_name, dtype="str")
    selected_users = {}
    for i, user in enumerate(user_list):
        # seen workers are numbered as 1, 2, ...
        # worker 0 is reserved for the generic worker
        selected_users[user] = i+1  
    return selected_users 

from datasets import load_dataset
from collections import defaultdict

##############################################################################
# 1) A helper function, mirroring build_ultrafeedback_p_dataset_dpo
##############################################################################
def build_reward_bench_dataset_dpo(
    dataset,
    tokenizer,
    n_users,
    sep="||", 
    n_user_tokens=1,
    is_train=True,
    original_columns=None,
    num_proc=24,
):
    """
    A helper function that processes each example in reward-bench similarly to
    build_ultrafeedback_p_dataset_dpo. It constructs new 'prompt', 'chosen',
    and 'rejected' fields, optionally duplicating the example if is_train.
    """
    new_examples = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "chosen_input_ids": [],
        "chosen_attention_mask": [],
        "chosen_labels": [],
        "rejected_input_ids": [],
        "rejected_attention_mask": [],
        "rejected_labels": [],
    }
    
    # Example fields in reward-bench: "prompt", "chosen", "rejected"
    # We'll just treat all user_ids as 1 for demonstration (similar structure).
    for prompt, chosen, rejected in zip(
        dataset["prompt"],
        dataset["chosen"],
        dataset["rejected"]
    ):
        # Hardcode user_id = 1 (just to replicate the user-based duplication logic)
        user_id = str(random.randint(1, n_users))
        
        # Construct the conversation text
        question = f"<|user|>\n{prompt}\n<|assistant|>\n"
        
        if (is_train and (user_id != 0)) or (not is_train):
            user_identifier = f"USER: {user_id} " + ("</s> " * n_user_tokens)
            new_prompt = user_identifier + sep + " " + question
            new_examples["prompt"].append(new_prompt)
            new_examples["chosen"].append(chosen)
            new_examples["rejected"].append(rejected)
        
        if (is_train and (user_id != 0)):
            user_identifier = f"USER: 0 " + ("</s> " * n_user_tokens)
            new_prompt = user_identifier + sep + " " + question
            new_examples["prompt"].append(new_prompt)
            new_examples["chosen"].append(chosen)
            new_examples["rejected"].append(rejected)
        
        question_chosen = f"<|user|>\n{new_prompt}\n<|assistant|>\n{chosen}"
        question_rejected = f"<|user|>\n{new_prompt}\n<|assistant|>\n{rejected}"
        tokenized_chosen = tokenizer(question_chosen, truncation=True, padding="max_length", max_length=2048)
        tokenized_rejected = tokenizer(question_rejected, truncation=True, padding="max_length", max_length=2048)
        new_examples["chosen_input_ids"].append(tokenized_chosen["input_ids"])
        new_examples["chosen_attention_mask"].append(tokenized_chosen["attention_mask"])
        new_examples["chosen_labels"].append(tokenized_chosen["input_ids"])
        new_examples["rejected_input_ids"].append(tokenized_rejected["input_ids"])
        new_examples["rejected_attention_mask"].append(tokenized_rejected["attention_mask"])
        new_examples["rejected_labels"].append(tokenized_rejected["input_ids"])

    ds = Dataset.from_dict(new_examples)
    print("Finished building reward-bench dataset!")
    return ds



def load_reward_bench(
    tokenizer,
    n_users,
    sep="||",
    n_user_tokens=1,
    max_text_length=4800,
    test_ratio=0.1,
    sanity_check=False,
    downloads_data_path=None,  # For structural similarity; not used here
    num_proc=24,
    seed=123,
    subset="raw",
    train_dataset_size=1000000,
    eval_data_size=1000000,
):
    eval_datasets = []
    dataset = load_dataset("allenai/reward-bench", split="raw")
    original_columns = dataset.column_names
    
    for key, value in reward_bench_map.items():
        sub_dataset = dataset.filter(lambda x: x['subset'] in value)
        sub_dataset = build_reward_bench_dataset_dpo(
            dataset=sub_dataset,
            tokenizer=tokenizer,
            n_users=n_users,
            sep=sep,
            n_user_tokens=n_user_tokens,
            is_train=False,
            original_columns=original_columns,
            num_proc=num_proc,
        )
        sub_dataset = sub_dataset.map(lambda x: {"key": key})
        eval_datasets.append(sub_dataset)

    return eval_datasets


def build_tldr_dataset_dpo_synthetic(
    dataset,
    selected_users,
    sep="||", 
    n_user_tokens=1,
    is_train=True,
    user_preference_file="./users_preference.txt",
    original_columns=None,
    num_proc=24,
):
    '''Build synthetic dataset for DPO from `openai/summarize_from_feedback`.
       The preferences of users depend on the length of the responses.
    '''
    
    user_list = np.genfromtxt(user_preference_file, dtype="str")
    user_preference = {}
    for i in range(user_list.shape[0]):
        user_preference[user_list[i, 0]] = int(user_list[i, 1])

    def preprocess_function(dataset):
        new_examples = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
        }
        for info, summaries, choice, worker in \
            zip(dataset["info"], dataset["summaries"], dataset["choice"], dataset["worker"]):
            original_text_field = "post" if info["post"] is not None else "article"
            query = f"SUBREDDIT: r/{info['subreddit']}\nTITLE: {info['title']}\nPOST: {info[original_text_field]}\nTL;DR: "
            if summaries[0]["text"] == summaries[1]["text"]:
                continue

            len_0, len_1 = len(summaries[0]["text"]), len(summaries[1]["text"])
            # if preference = 0, user prefers longer summary
            # if preference = 1, user prefers shorter summary
            if ((len_0 >= len_1) and (user_preference[worker] == 0)) or \
                ((len_0 <= len_1) and (user_preference[worker] == 1)):
                summary_j = summaries[0]["text"]
                summary_k = summaries[1]["text"]
            
            if ((len_0 < len_1) and (user_preference[worker] == 0)) or \
                ((len_0 > len_1) and (user_preference[worker] == 1)):
                summary_j = summaries[1]["text"]
                summary_k = summaries[0]["text"]

            if selected_users is not None:
                if worker in selected_users.keys():
                    user_id = selected_users[worker]
                else:
                    user_id = 0
            else:
                user_id = 0

            # for personalized dpo, prompt = user identifier|query
            user_identifier = f"USER: {user_id}" + ("<|endoftext|>"*n_user_tokens)
            prompt = user_identifier + sep + query

            # is_train = True: use selected users in train split for training
            # is_train = False: use all users in validation split for evaluation
            if (is_train and (user_id != 0)) or (not is_train):
                new_examples["prompt"].append(prompt)
                new_examples["chosen"].append(summary_j)
                new_examples["rejected"].append(summary_k)

            if (is_train and (user_id != 0)):
                # for each example used for training (user_id != 0),
                # create a duplicate example with user_id = 0 to simplify the loss computation
                user_identifier = f"USER: 0" + ("<|endoftext|>"*n_user_tokens)
                prompt = user_identifier + sep + query
                new_examples["prompt"].append(prompt)
                new_examples["chosen"].append(summary_j)
                new_examples["rejected"].append(summary_k)
            
        return new_examples 

    ds = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    print("Finished loading synthetic TL;DR dataset!")
    return ds


def load_openai_comparisons(
    user_file="./selected_users.txt",
    sep="||",
    n_user_tokens=1,
    max_text_length=4800,
    sanity_check=False,
    use_downloads=False,
    downloads_data_path=None,
    user_preference_file="./users_preference.txt",
    num_proc=24,
):
    # the selected users
    selected_users = encode_selected_users(file_name=user_file) if user_file is not None else None
    n_users = len(selected_users)

    if not use_downloads:
        train_dataset = load_dataset("openai/summarize_from_feedback", 'comparisons', split='train')
        eval_dataset  = load_dataset("openai/summarize_from_feedback", 'comparisons', split='validation')
    else:
        # load data from the path which contains downloaded data
        assert downloads_data_path is not None, "downloads_data_path cannot be None if use_downloads is True!"
        train_dataset = load_from_disk(os.path.join(downloads_data_path, "train"))
        eval_dataset  = load_from_disk(os.path.join(downloads_data_path, "validation"))

    # only use comparisons sampled from SFTs (but not ppo policies)
    filter_func = lambda x: all(y["policy"].find("sup") >= 0 and y["policy"].find("ppo") == -1 and \
                                y["policy"].find("cnn") == -1 for y in x["summaries"])
    train_dataset = train_dataset.filter(filter_func)
    eval_dataset  = eval_dataset.filter(filter_func)

    if sanity_check: # use a small subset for sanity check
        train_dataset = train_dataset.select(range(100))
        eval_dataset  = eval_dataset.select(range(50))
    original_columns = train_dataset.column_names

    datasets = {"train": train_dataset, "eval": eval_dataset}
    for key in datasets.keys():
        assert user_preference_file is not None, \
            "user_preference_file cannot be None if is_synthetic is True!"
        datasets[key] = build_tldr_dataset_dpo_synthetic(
            dataset=datasets[key],
            selected_users=selected_users,
            sep=sep, 
            n_user_tokens=n_user_tokens, 
            is_train=True if key=="train" else False,
            user_preference_file=user_preference_file,
            original_columns=original_columns,
            num_proc=num_proc,
        )
        datasets[key] = datasets[key].filter(
            lambda x: len(x["prompt"]) + len(x["chosen"]) <= max_text_length \
                and len(x["prompt"]) + len(x["rejected"]) <= max_text_length
        )

    return datasets["train"], datasets["eval"], n_users


def build_psoups_dataset_dpo(
    dataset,
    sep="||", 
    n_user_tokens=1,
    is_train=True,
    original_columns=None,
    num_proc=24,
):
    '''Build dataset for DPO from personalized soups allcombo_8_cleaned.json generated by running /data/psoups_dataset.ipynb.'''
    
    def preprocess_function(dataset):
        new_examples = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
        }
        for user_id, question, response_j, response_k in \
            zip(dataset['uid'], dataset["prompt"], dataset["chosen"], dataset["rejected"]):
                # zip(dataset['user_id'], dataset["user_input"], dataset["completion_a"], dataset["completion_b"]):
            question = f"<|user|>\n{question} \n<|assistant|>\n"
            # for personalized dpo, prompt = user identifier|query
            user_identifier = f"USER: {user_id} " + ("</s> "*n_user_tokens)
            prompt = user_identifier + sep + ' ' + question

            if (is_train and (user_id != 0)) or (not is_train):
                new_examples["prompt"].append(prompt)
                new_examples["chosen"].append(response_j)
                new_examples["rejected"].append(response_k)
                    
            if (is_train and (user_id != 0)):
                # for each example used for training (user_id != 0),
                # create a duplicate example with user_id = 0 to simply the loss computation
                user_identifier = f"USER: 0 " + ("</s> "*n_user_tokens)
                prompt = user_identifier + sep + ' ' + question
                new_examples["prompt"].append(prompt)
                new_examples["chosen"].append(response_j)
                new_examples["rejected"].append(response_k)
        
        return new_examples    

    ds = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    print("Finished loading psoups dataset!")
    return ds   

def build_summarization_comparison_dataset_dpo(
    dataset,
    id_map,
    sep="||", 
    n_user_tokens=1,
    is_train=True,
    original_columns=None,
    num_proc=24,
):
    """
    A helper function that processes each example in the Summarize-from-Feedback 
    dataset (similar in spirit to build_ultrafeedback_p_dataset_dpo).
    
    - We map each row to a new prompt format that includes a user ID, if desired.
    - We create 'prompt', 'chosen', and 'rejected' fields that can then be used 
      for DPO or RLHF training.
    
    Returns: A `datasets.Dataset` object with columns: [prompt, chosen, rejected].
    """

    new_examples = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }

    # Each row has: 
    #   'summaries' = [ { 'text': summary_0 }, { 'text': summary_1 } ]
    #   'choice'    = index of the chosen summary (0 or 1)
    #   'info'      = dict with 'post' containing the text to summarize
    #   'worker'    = worker ID, which we map to a small integer via id_map
    for summaries, choice, info, worker in zip(
        dataset["summaries"], 
        dataset["choice"],
        dataset["info"],
        dataset["worker"]
    ):
        # Map worker => user_id (1..5 if you filtered top 5 workers)
        user_id = id_map.get(worker, 0)  # fallback to 0 if not in id_map
        post_text = info["post"]
        
        # Construct the question (or user content) in a style similar to <|user|> ... <|assistant|>
        question = f"<|user|>\nSummarize the following text:\n{post_text}\n<|assistant|>\n"

        # Retrieve chosen/rejected texts
        chosen_text   = summaries[choice]["text"]
        rejected_text = summaries[1 - choice]["text"]

        # Following the same duplication logic from build_ultrafeedback_p_dataset_dpo:
        #   if (is_train and user_id != 0) or (not is_train): we build an example with the real user_id
        #   if (is_train and user_id != 0): we also create a duplicate with user_id=0
        def add_example(uid):
            # Format the user identifier
            user_identifier = f"USER: {uid} " + ("</s> " * n_user_tokens)
            prompt = user_identifier + sep + ' ' + question
            new_examples["prompt"].append(prompt)
            new_examples["chosen"].append(chosen_text)
            new_examples["rejected"].append(rejected_text)

        # 1) Add the example with user_id (unless user_id == 0 and is_train? 
        #    We replicate the same condition used in build_ultrafeedback_p.)
        if (is_train and user_id != 0) or (not is_train):
            add_example(user_id)

        # 2) If training and user_id != 0, add a duplicate with user_id=0
        if (is_train and user_id != 0):
            add_example(0)

    processed_dataset = Dataset.from_dict(new_examples)
    return processed_dataset

def load_summarization_comparisons(
    sep="||",
    n_user_tokens=1,
    max_text_length=4800,
    test_ratio=0.1,        # not used directly but kept for structural similarity
    sanity_check=False,
    downloads_data_path=None,  # not used here but kept for similarity
    num_proc=24,
    seed=123,
    subset="comparisons",
    train_dataset_size=1000000,
    eval_dataset_size=1000000
):
    """
    Loads and processes the 'openai/summarize_from_feedback' dataset (subset='comparisons')
    in a style similar to load_ultrafeedback_p.

    1) Loads the train/validation splits.
    2) Filters to the top 5 workers, discards rows with `info['post'] = None`.
    3) Subsets to train_dataset_size/eval_dataset_size, optionally does a small 
       sanity_check sample.
    4) Calls a helper function to convert each row into a {prompt, chosen, rejected} format 
       with user IDs in the prompt, duplicating examples as needed for training.
    5) Filters out examples exceeding max_text_length.
    6) Returns (train_dataset, val_dataset, n_users).
    """

    # ----------------------------
    # 1) Load Summarize-from-Feedback (comparisons)
    # ----------------------------
    train_dataset = load_dataset("openai/summarize_from_feedback", 'comparisons', split='train')
    val_dataset   = load_dataset("openai/summarize_from_feedback", 'comparisons', split='validation')

    # ----------------------------
    # 2) Filter to top 5 worker IDs & remove rows with empty 'post'
    # ----------------------------
    # Count top 5 frequent worker IDs in the training set
    selected_worker_ids = Counter(train_dataset["worker"]).most_common(5)
    selected_worker_ids = [wid for wid, _ in selected_worker_ids]

    # Map each worker ID -> integer (1..5)
    id_map = {worker_id: i+1 for i, worker_id in enumerate(selected_worker_ids)}

    # Filter train/val to only keep selected workers
    train_dataset = train_dataset.filter(lambda x: x["worker"] in selected_worker_ids)
    val_dataset   = val_dataset.filter(lambda x: x["worker"] in selected_worker_ids)

    # Drop rows where info['post'] is None
    train_dataset = train_dataset.filter(lambda x: x["info"]["post"] is not None)
    val_dataset   = val_dataset.filter(lambda x: x["info"]["post"] is not None)

    # ----------------------------
    # 3) Subset the dataset sizes and optional sanity check
    # ----------------------------
    train_dataset = train_dataset.select(range(min(train_dataset_size, len(train_dataset))))
    val_dataset   = val_dataset.select(range(min(eval_dataset_size, len(val_dataset))))

    if sanity_check:
        train_dataset = train_dataset.select(range(min(1000, len(train_dataset))))
        val_dataset   = val_dataset.select(range(min(500,  len(val_dataset))))

    original_columns = train_dataset.column_names

    # ----------------------------
    # 4) Process each split via build_summarization_comparison_dataset_dpo
    # ----------------------------
    datasets = {"train": train_dataset, "eval": val_dataset}
    for key in datasets.keys():
        # is_train = True if "train" else False
        is_train = (key == "train")

        # Process to create prompt, chosen, rejected
        processed_ds = build_summarization_comparison_dataset_dpo(
            dataset         = datasets[key],
            id_map          = id_map,
            sep             = sep, 
            n_user_tokens   = n_user_tokens,
            is_train        = is_train,
            original_columns= original_columns,
            num_proc        = num_proc,
        )
        print(f"Number of examples in {key} dataset before filtering by max_length: {len(processed_ds)}")

        # ----------------------------
        # 5) Filter out examples that exceed max_text_length
        # ----------------------------
        processed_ds = processed_ds.filter(
            lambda x: (
                len(x["prompt"]) + len(x["chosen"]) <= max_text_length
                and len(x["prompt"]) + len(x["rejected"]) <= max_text_length
            ),
            num_proc=num_proc
        )

        print(f"Number of examples in {key} dataset after filtering: {len(processed_ds)}")

        datasets[key] = processed_ds

    # We used the top 5 workers
    n_users = 5

    # Return train/eval and the number of users
    return datasets["train"], datasets["eval"], n_users

reverse_metric_map = {
    "instruction_following": 3,
    "honesty": 1,
    "truthfulness": 4,
    "helpfulness": 2,
}

def build_ultrafeedback_p_dataset_dpo(
    dataset,
    sep="||", 
    n_user_tokens=1,
    is_train=True,
    original_columns=None,
    num_proc=24,
):
    new_examples = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }
    
    for attributes, prompt, chosen, rejected in zip(dataset["attributes"], dataset["prompt"], dataset["chosen"], dataset["rejected"]):
        user_id = reverse_metric_map[attributes]
        question = f"<|user|>\n{prompt} \n<|assistant|>\n"

        if (is_train and (user_id != 0)) or (not is_train):
            
            user_identifier = f"USER: {user_id} " + ("</s> "*n_user_tokens)
            prompt = user_identifier + sep + ' ' + question
            new_examples["prompt"].append(prompt)
            new_examples["chosen"].append(chosen)
            new_examples["rejected"].append(rejected)
                
        if (is_train and (user_id != 0)):
            # for each example used for training (user_id != 0),
            # create a duplicate example with user_id = 0 to simply the loss computation
            user_identifier = f"USER: 0 " + ("</s> "*n_user_tokens)
            prompt = user_identifier + sep + ' ' + question
            new_examples["prompt"].append(prompt)
            new_examples["chosen"].append(chosen)
            new_examples["rejected"].append(rejected)
        

    ds = Dataset.from_dict(new_examples)
    print("Finished loading psoups dataset!")
    return ds   

def load_ultrafeedback_p(
    sep="||",
    n_user_tokens=1,
    max_text_length=4800,
    test_ratio=0.1,
    sanity_check=False,
    downloads_data_path=None,
    num_proc=24,
    seed=123,
    subset="conversational",
    train_dataset_size=1000000,
    eval_dataset_size=1000000
):
    train_dataset = load_dataset("RiverDong/ultrafeedback-personalized", subset, split="train")
    test_dataset  = load_dataset("RiverDong/ultrafeedback-personalized", subset, split="test")
    
    train_dataset = train_dataset.filter(lambda x: x['attributes'] in ['helpfulness', 'honesty'])
    test_dataset = test_dataset.filter(lambda x: x['attributes'] in ['helpfulness', 'honesty'])
    
    train_dataset = train_dataset.select(range(min(train_dataset_size, len(train_dataset))))
    test_dataset = test_dataset.select(range(min(eval_dataset_size, len(test_dataset))))
    
    if sanity_check: # use a small subset for sanity check
        train_dataset = train_dataset.select(range(1000))
        test_dataset  = test_dataset.select(range(500))
    original_columns = train_dataset.column_names
    
    datasets = {"train": train_dataset, "eval": test_dataset}
    for key in datasets.keys():
        datasets[key] = build_ultrafeedback_p_dataset_dpo(
            dataset=datasets[key],
            sep=sep, 
            n_user_tokens=n_user_tokens,
            is_train=True if key=="train" else False,
            original_columns=original_columns,
            num_proc=num_proc,
        )
        
        print(f"Number of examples in {key} dataset before filtering by max_length: {len(datasets[key])}")
    
        datasets[key] = datasets[key].filter(
            lambda x: len(x["prompt"]) + len(x["chosen"]) <= max_text_length \
                and len(x["prompt"]) + len(x["rejected"]) <= max_text_length
        )
        print(f"Number of examples in {key} dataset after filtering: {len(datasets[key])}")

    # return datasets["train"], datasets["eval"], 4  # n_users = 6 for psoups
    return datasets["train"], datasets["eval"], 2

reward_bench_map = {
    "chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-medium"
    ],
    "chat hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual"
    ],
    "safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "do not answer"
    ],
    "reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust"
    ]
}


def load_psoups_comparisons(
    sep="||",
    n_user_tokens=1,
    max_text_length=4800,
    test_ratio=0.1,
    sanity_check=False,
    downloads_data_path=None,
    num_proc=24,
    seed=123,
    train_dataset_size=1000000,
    eval_dataset_size=1000000
):
    train_dataset = load_dataset("RiverDong/psoups", split="train")
    eval_dataset  = load_dataset("RiverDong/psoups", split="test")
    train_dataset = train_dataset.select(range(min(train_dataset_size, len(train_dataset))))
    eval_dataset = eval_dataset.select(range(min(eval_dataset_size, len(eval_dataset))))

    if sanity_check: # use a small subset for sanity check
        train_dataset = train_dataset.select(range(100))
        eval_dataset  = eval_dataset.select(range(50))
    original_columns = train_dataset.column_names

    datasets = {"train": train_dataset, "eval": eval_dataset}
    for key in datasets.keys():
        datasets[key] = build_psoups_dataset_dpo(
            dataset=datasets[key],
            sep=sep, 
            n_user_tokens=n_user_tokens,
            is_train=True if key=="train" else False,
            original_columns=original_columns,
            num_proc=num_proc,
        )
    
        datasets[key] = datasets[key].filter(
            lambda x: len(x["prompt"]) + len(x["chosen"]) <= max_text_length \
                and len(x["prompt"]) + len(x["rejected"]) <= max_text_length
        )

    return datasets["train"], datasets["eval"], 6  # n_users = 6 for psoups


def load_personal_llm_comparisons(
    sep="||",
    n_user_tokens=1,
    max_text_length=4800,
    test_ratio=0.1,
    sanity_check=False,
    downloads_data_path=None,
    num_proc=24,
    seed=123,
    train_dataset_size=1000000,
    eval_dataset_size=1000000
):
    train_dataset = load_dataset("RiverDong/Personal-LLM-DPO", split="train")
    eval_dataset  = load_dataset("RiverDong/Personal-LLM-DPO", split="test")
    
    train_dataset = train_dataset.select(range(min(train_dataset_size, len(train_dataset))))
    eval_dataset = eval_dataset.select(range(min(eval_dataset_size, len(eval_dataset))))

    if sanity_check: # use a small subset for sanity check
        train_dataset = train_dataset.select(range(100))
        eval_dataset  = eval_dataset.select(range(50))
    original_columns = train_dataset.column_names

    datasets = {"train": train_dataset, "eval": eval_dataset}
    for key in datasets.keys():
        datasets[key] = build_psoups_dataset_dpo( ## Psoups and Personal-LLM-DPO have EXACTLY the same structure so we can use the same function
            dataset=datasets[key],
            sep=sep, 
            n_user_tokens=n_user_tokens,
            is_train=True if key=="train" else False,
            original_columns=original_columns,
            num_proc=num_proc,
        )
    
        datasets[key] = datasets[key].filter(
            lambda x: len(x["prompt"]) + len(x["chosen"]) <= max_text_length \
                and len(x["prompt"]) + len(x["rejected"]) <= max_text_length
        )

    return datasets["train"], datasets["eval"], 8  # n_users = 6 for psoups


def load_prism_comparisons(
    sep="||",
    n_user_tokens=10,
    max_text_length=2400,
    max_prompt_string_length = 1500,  # about 500 tokens, corresponds to max_prompt_length 550
    sanity_check=False,
    prism_data_path=None,
    seed=123,
    add_textual_info=False,
):
    with open(os.path.join(prism_data_path, "prism_data_dialog.json"), 'r') as f:
        data_dialog = json.load(f)
    with open(os.path.join(prism_data_path, "prism_data_user.json"), 'r') as f:
        data_user = json.load(f)
    with open(os.path.join(prism_data_path, "prism_split_ids.json"), 'r') as f:
        split_ids = json.load(f)

    n_users = len(data_user)

    def preprocess_function(is_train, add_textual_info=False):
        new_examples = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
        }

        if is_train:
            dialog_ids = split_ids["train_dialog_ids"]
            user_ids = split_ids["seen_user_ids"]
        else:
            dialog_ids = split_ids["test_dialog_ids"]
            user_ids = split_ids["seen_user_ids"]
            user_ids.update(split_ids["unseen_user_ids"])

        for idx, dialog_id in enumerate(dialog_ids):
            # if idx > 100:
            #     break
            history = ""
            # textual info of the user
            if add_textual_info:
                preference = ", ".join(data_user[data_dialog[dialog_id]["user_id"]]["demographics"]["preference"])
                textual_info = f"preference: {preference}; "
                for k in data_user[data_dialog[dialog_id]["user_id"]]["demographics"]:
                    if k != "preference":
                        textual_info += f"{k}: {data_user[data_dialog[dialog_id]['user_id']]['demographics'][k]}; "
            for turn in data_dialog[dialog_id]["turns"]:
                # add user utterance to history
                history += f"<|start_header_id|>user<|end_header_id|>\n\n{turn['user_utterance'][0]}<|eot_id|>\n"

                # prepare examples, skip empty or too long examples
                if (    turn['user_utterance'] != [] 
                    and turn['chosen_utterance'] != [] 
                    and turn['rejected_utterance'] != [] 
                    and len(turn["user_utterance"][0]) + len(turn["chosen_utterance"][0]) < max_text_length
                ):
                    # build user identifier
                    user_identifier = f"USER: {user_ids[data_dialog[dialog_id]['user_id']]} " + ("<|end_of_text|>"*n_user_tokens)
                        
                    # build prompt
                    prompt = user_identifier + sep
                    if add_textual_info:
                        prompt += 'User textual information: ' + textual_info 
                    # truncate history
                    max_history_string_length = max_prompt_string_length - len(prompt)
                    if len(history) > max_history_string_length:
                        history = history[-max_history_string_length:]
                    prompt += '\n' + history + "<|start_header_id|>assistant<|end_header_id|>\n\n"

                    # append to examples
                    for rejected in turn["rejected_utterance"]:
                        # skip too long examples
                        if len(turn["user_utterance"][0]) + len(rejected) > max_text_length:
                            continue
                        new_examples["prompt"].append(prompt)
                        new_examples["chosen"].append(turn['chosen_utterance'][0])
                        new_examples["rejected"].append(rejected)
                        # if train, duplicate 0
                        if is_train:
                            user_identifier_0 = f"USER: {0} " + ("<|end_of_text|>"*n_user_tokens)
                            prompt_0 = user_identifier_0 + sep
                            if add_textual_info:
                                user_identifier_0 += 'User textual information: ' + textual_info 
                            prompt_0 += '\n' + history + "<|start_header_id|>assistant<|end_header_id|>\n\n"
                            new_examples["prompt"].append(prompt_0)
                            new_examples["chosen"].append(turn['chosen_utterance'][0]) 
                            new_examples["rejected"].append(rejected)
                    
                # add the first chosen utterance to history for next turn
                if turn['chosen_utterance'] != []:
                    history += f"<|start_header_id|>assistant<|end_header_id|>\n\n{turn['chosen_utterance'][0]}<|eot_id|>\n"

        return new_examples

    train_examples = preprocess_function(is_train=True, add_textual_info=add_textual_info)
    test_examples = preprocess_function(is_train=False, add_textual_info=add_textual_info)
    
    train_dataset = Dataset.from_dict(train_examples)
    test_dataset = Dataset.from_dict(test_examples)

    # use a small subset for sanity check
    if sanity_check: 
        train_dataset = train_dataset.select(range(100))
        test_dataset  = test_dataset.select(range(50))
    
    return train_dataset, test_dataset, n_users


def build_tldr_prompts(script_args):
    tldr_selected_prompts = np.load(script_args.tldr_selected_prompts_path)

    if script_args.is_baseline == 1:
        # for vanilla DPO baseline, only need to generate once 
        user_ids = [0]
    else:
        # for known users seen during training, user_id = 1, ..., n_users
        # for unknown users, user_id = 0
        user_ids = range(0, script_args.n_users + 1) 

    tldr_prompts = {} 
    for prompt_id, prompt in enumerate(tldr_selected_prompts):
        for user_id in user_ids:
            # DPO trainer adds a bos token to the beginning of the input text
            # since GPT-J tokenizer does not add bos token automatically, we add it manually to the beginning
            user_identifier = "<|endoftext|>" + f"USER: {user_id}" + ("<|endoftext|>"*script_args.n_user_tokens)
            input_text = user_identifier + script_args.sep + prompt
            tldr_prompt = {
                "prompt_id": f"{user_id}_{prompt_id}",
                "our_user_id": user_id,
                "prompt": prompt,
                "input_text": input_text,
                "output_text": "",
                "output_length": 0,
            }
            tldr_prompts[tldr_prompt["prompt_id"]] = tldr_prompt

    return tldr_prompts


def build_psoups_prompts(script_args):
    with open(script_args.koala_prompts_path, "r") as f:
        koala_prompts = json.load(f)
    
    if script_args.is_baseline == 1:
        # for vanilla DPO baseline, only need to generate once 
        user_ids = [0]
    else:
        # for known users seen during training, user_id = 1, ..., n_users
        # for unknown users, user_id = 0
        user_ids = range(0, script_args.n_users + 1) 

    psoups_prompts = {} 
    for prompt_id, prompt in enumerate(koala_prompts):
        prompt = prompt["prompt"]
        prompt = f"<|user|>\n{prompt} \n<|assistant|>\n"

        for user_id in user_ids:
            user_identifier = f"USER: {user_id} " + ("</s> "*script_args.n_user_tokens)
            input_text = user_identifier + script_args.sep + ' ' + prompt
            psoups_prompt = {
                "prompt_id": f"{user_id}_{prompt_id}",
                "our_user_id": user_id,
                "prompt": prompt,
                "input_text": input_text,
                "output_text": "",
                "output_length": 0,
            }
            psoups_prompts[psoups_prompt["prompt_id"]] = psoups_prompt
    
    return psoups_prompts


def build_prism_prompts(script_args):
    with open(script_args.selected_examples_path, 'r') as f:
        selected_examples = json.load(f)

    prism_prompts = {}

    for dialog_id, dialog in selected_examples.items():
        # build user identifier
        user_identifier = f"USER: {dialog['our_id']} " + ("<|end_of_text|>"*script_args.n_user_tokens)
        for turn_nb, turn in dialog["turns"].items():
            # add textual user info
            if script_args.add_textual_info:
                preference = ", ".join(dialog["demographics"]["preference"])
                textual_info = f"preference: {preference}; "
                for k in dialog["demographics"]:
                    if k != "preference":
                        textual_info += f"{k}: {dialog['demographics'][k]}; "
            # build prompt
            prompt = user_identifier + script_args.sep
            if script_args.add_textual_info:
                prompt += 'User textual information: ' + textual_info 
            # truncate history
            max_history_string_length = script_args.max_prompt_text_length - len(prompt)
            if len(turn["history"]) > max_history_string_length:
                history = turn["history"][-max_history_string_length:]
            else:
                history = turn["history"]
            # append history to input
            input_text = prompt + '\n' + history + "<|start_header_id|>assistant<|end_header_id|>\n\n"
            
            prism_prompt = {
                "prompt_id": f"{dialog['user_id']}_{dialog_id}_{turn['turn_nb']}",
                "our_user_id": dialog['our_id'],
                "prompt": prompt,
                "truncated_history": history,
                "input_text": input_text,
                "output_text": "",
                "output_length": 0,
            }
            prism_prompts[prism_prompt["prompt_id"]] = prism_prompt

    return prism_prompts