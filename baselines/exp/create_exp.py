# method_list = ['ft_rm', 'id_rm', 'rag', 'dpo', 'vpl', 'gpo', 'prm', 'vpl_embed]
method_list = ['ft_rm']

train_dataset_size_list = [300, 1000, 3000, 10000, 100000]
# train_dataset_size_list = [10000]
eval_dataset_size = 1000

data_type = "personal_llm"  #ultrafeedback, psoups, summarization, personal_llm
# subset = ['default', 'controversial', 'ood', 'ood-controversial']
subset_list = ['default']

if data_type == 'ultrafeedback':
    assert 'controversial' in subset_list or 'ood-controversial' in subset_list
elif data_type == 'psoups':
    assert 'controversial' not in subset_list and 'ood-controversial' not in subset_list
elif data_type == 'summarization':
    print ("Summarization dataset")
elif data_type == 'personal_llm':
    print ("Personal LLM dataset")
else:
    raise ValueError("data_type must be ultrafeedback or psoups")

learning_rate_list = [3e-4]
# learning_rate_list = [1e-5, 3e-5]

## maybe should be the chat version meta-llama/Llama-2-7b-chat-hf
# weqweasdas/RM-Mistral-7B "Ray2333/GRM-Gemma-2B-rewardmodel-ft", meta-llama/Llama-2-7b-hf, meta-llama/Llama-3.1-8B, Ray2333/GRM-Llama3-8B-rewardmodel-ft
model_name_list = ["meta-llama/Llama-2-7b-hf"]
num_train_epochs_list = [1]
per_device_train_batch_size = 2  ##32 for gemma (trains about 4 hour for 100k 1 epoch), 16 or 8 for llama 8B?
gradient_accumulation_steps = 16

log_dir = '/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/models'

header = """#!/bin/bash

source /home/yd358/rds/hpc-work/miniconda3/etc/profile.d/conda.sh
conda activate personalization
cd /home/yd358/rds/hpc-work/analysis_pers/baselines

"""

count = 0
commands = []
if 'rag' in method_list:
    ## Must check this model_name before running
    model_name = 'cohere'
    method_list.remove('rag')
    with open(f'/home/yd358/rds/hpc-work/analysis_pers/baselines/exp/rag_{count}.sh', 'w') as f:
        f.write(header)
    for subset in subset_list:
        for train_dataset_size in train_dataset_size_list:
            for embed_target in ['prompt', 'paired_pref', 'pair_diff', 'random']:
                
                model_name = model_name_list[0]
                assert 'rewardmodel' not in model_name ## Must be a LLM to do RAG, not a RM
                
                command = f"python rag.py  --train_dataset_size {train_dataset_size} --eval_dataset_size {eval_dataset_size} --data_type {data_type} --subset {subset} --model_name {model_name} --embed_target {embed_target} --model_name {model_name} --log_dir {log_dir}\n"
                with open(f'/home/yd358/rds/hpc-work/analysis_pers/baselines/exp/rag_{count}.sh', 'a') as f:
                    
                    f.write(command)

output_dir = '/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/models'

# import pdb; pdb.set_trace()
if 'vpl_embed' in method_list:
    method_list.remove('vpl_embed')
    header = """#!/bin/bash

source /home/yd358/rds/hpc-work/miniconda3/etc/profile.d/conda.sh
conda activate personalization
cd /home/yd358/rds/hpc-work/analysis_pers/baselines/vpl_llm

"""

    if data_type == 'psoups':
        uids = ['1', '2', '3', '4', '5', '6']
    elif data_type == 'ultrafeedback':
        uids = ['0', '1', '2', '3']
    elif data_type == 'summarization':
        uids = ['1', '2', '3', '4', '5']
    elif data_type == 'personal_llm':
        uids = ['1', '2', '3', '4', '5', '6', '7', '8']
    
    ## python -m hidden_context.create_embedding --output_dir "/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/data/P_4_survey_${survey_size}/" \ --data_path "/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/data/UltraFeedback_single_P_4" --data_subset default --data_split train --model_type llama  --with_embeddings True --survey_size 100 --num_duplicates 4 --data_type summarization --uid 1
    
    for uid in uids:
        with open(f'/home/yd358/rds/hpc-work/analysis_pers/baselines/exp/vpl_embed_{count}.sh', 'w') as f:
            f.write(header)
            f.write(f"""python -m hidden_context.create_embedding  \\
--output_dir "/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/data/{data_type}_4_survey_100/" \\
--data_path "/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/data/{data_type}_single_P_4" \\
--data_subset {subset_list[0]} --data_split train --model_type llama  --with_embeddings True \\
--survey_size 100 --num_duplicates 4 --data_type {data_type} --uid {uid}\n\n""")
            f.write(f"""python -m hidden_context.create_embedding  \\
--output_dir "/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/data/{data_type}_4_survey_100/"\\
--data_path "/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/data/{data_type}_single_P_4" \\
--data_subset {subset_list[0]} --data_split test --model_type llama  --with_embeddings True  \\
--survey_size 100 --num_duplicates 4 --data_type {data_type} --uid {uid}\n\n""")
            count += 1

    # for uid in uids:
    #     with open(f'/home/yd358/rds/hpc-work/analysis_pers/baselines/exp/vpl_embed_{count}.sh', 'w') as f:
    #         # f.write(header + f'\nbash /home/yd358/rds/hpc-work/analysis_pers/baselines/vpl_llm/generate_llm_embeddings_psoups.sh llama {data_type} {uid}')
    #         f.write(header + f'\nbash /home/yd358/rds/hpc-work/analysis_pers/baselines/vpl_llm/generate_llm_embeddings.sh llama single {uid} {data_type}')
    #         count += 1

# python -m hidden_context.train_llm_vae_preference_model \
#         --model_name=${model_name} \
#         --data_path="/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/data/psoups_4_survey_100/llama" \
#         --num_train_epochs=2 \
#         --reward_model_type=vae \
#         --data_subset=all \
#         --log_dir="/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/models" \
#         --bf16 True \
#         --fp16 False \
#         --per_device_train_batch_size 4 \
#         --gradient_accumulation_steps 8 \
#         --latent_dim 512 \
#         --hidden_dim 512 \
#         --learning_rate 1e-4 \
#         --use_annealing True \
#         --kl_loss_weight 3e-6 \
#         --controversial_only True \
#         --fixed_contexts True \
#         --fixed_llm_embeddings False \
#         --up_sampling False \
#         --other_subsets single \
#         --use_last_token_embedding True \
#         --seed 0

if 'vpl' in method_list:
    method_list.remove('vpl')
    if data_type == 'psoups':
        other_subset = 'psoups'
    elif data_type == 'ultrafeedback':
        other_subset = 'single'
    assert per_device_train_batch_size < 8
    header = """#!/bin/bash

source /home/yd358/rds/hpc-work/miniconda3/etc/profile.d/conda.sh
conda activate personalization
cd /home/yd358/rds/hpc-work/analysis_pers/baselines/vpl_llm

"""
    # assert data_type == 'psoups' ## otherwise need to change train_llm_vae_preference_model_psoups
    for lr in learning_rate_list:
        for train_dataset_size in train_dataset_size_list:
            with open(f'/home/yd358/rds/hpc-work/analysis_pers/baselines/exp/vpl_train_{count}.sh', 'w') as f:
                f.write(header + f"""torchrun --nproc_per_node=4 --nnodes=1 --master_port=29522 -m hidden_context.train_llm_vae_preference_model \\
--model_name meta-llama/Llama-2-7b-hf --data_path /home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/data/{data_type}_4_survey_100/llama  \\
--num_train_epochs 1 --reward_model_type vae --data_subset all --log_dir /home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/models  \\
--bf16 True --fp16 False --per_device_train_batch_size {per_device_train_batch_size} --gradient_accumulation_steps {gradient_accumulation_steps}  \\
--latent_dim 1024 --hidden_dim 1024 --learning_rate {lr} --use_annealing True --kl_loss_weight 3e-6 --controversial_only True --fixed_contexts True \\
--fixed_llm_embeddings False --up_sampling False --use_last_token_embedding True --seed 0 \\
--train_dataset_size {train_dataset_size} --eval_dataset_size {eval_dataset_size}""")
            count += 1
            
if 'prm' in method_list:
    method_list.remove('prm')
    prm_learning_rate_list = [1e-5, 5e-5]
    prm_header = """#!/bin/bash

source /home/yd358/rds/hpc-work/miniconda3/etc/profile.d/conda.sh
conda activate prm
cd /home/yd358/rds/hpc-work/analysis_pers/baselines/Personalized_RLHF

"""
    for lr in prm_learning_rate_list:
        for subset in subset_list:
            count += 1
            command = f"torchrun --nproc_per_node=4 --nnodes=1 --master_port=29522  prlhf/train_language_model_dpo.py --user_model individual     --model_class llama  --model_name meta-llama/Llama-2-7b-hf     --tokenizer_name meta-llama/Llama-2-7b-hf     --dataset ultrafeedback --subset {subset} --learning_rate {lr} --output_dir {output_dir} --train_dataset_size {train_dataset_size} --eval_dataset_size {eval_dataset_size}"
            with open(f'/home/yd358/rds/hpc-work/analysis_pers/baselines/exp/prm_{count}.sh', 'w') as f:
                f.write(prm_header + command)

for method in method_list:
    for subset in subset_list:
        for lr in learning_rate_list:
            for epoch in num_train_epochs_list:
                for model_name in model_name_list:
                    if 'ood' not in subset:
                        for train_dataset_size in train_dataset_size_list:
                            count += 1
                            command = f"python {method}.py --train --train_dataset_size {train_dataset_size} --eval_dataset_size {eval_dataset_size} --data_type {data_type} --subset {subset} --per_device_train_batch_size {per_device_train_batch_size} --gradient_accumulation_steps {gradient_accumulation_steps} --learning_rate {lr} --num_train_epochs {epoch} --model_name {model_name} --log_dir {log_dir}"
                            with open(f'/home/yd358/rds/hpc-work/analysis_pers/baselines/exp/{method}_{count}.sh', 'w') as f:
                                f.write(header + command)
                    else:
                        train_dataset_size = train_dataset_size_list[-1]
                        count += 1
                        command = f"python {method}.py --train --train_dataset_size {train_dataset_size} --eval_dataset_size {eval_dataset_size} --data_type {data_type} --subset {subset} --per_device_train_batch_size {per_device_train_batch_size} --gradient_accumulation_steps {gradient_accumulation_steps} --learning_rate {lr} --num_train_epochs {epoch} --model_name {model_name} --log_dir {log_dir}"
                        with open(f'/home/yd358/rds/hpc-work/analysis_pers/baselines/exp/{method}_{count}.sh', 'w') as f:
                            f.write(header + command)


