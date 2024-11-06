path = "/home/yd358/rds/hpc-work/analysis_pers/exp/oct27_id_rm.sh"
max_command_per_file = 5

with open(path, "w") as f:
    f.write("#!/bin/bash\n")
    f.write("source /home/yd358/rds/hpc-work/miniconda3/etc/profile.d/conda.sh\n")
    f.write("conda activate personalization\n")
    f.write("cd /home/yd358/rds/hpc-work/analysis_pers/baselines\n")

count = 0
command_list = []

for method in ['id_rm']: # ['ft_rm', 'ft', 'dpo']:
    for train_dataset_size in [10000, 0]: # [10**3, 10**4, 10**5, 10**6]
        for lr in [1e-4, 3e-4]:
            for num_train_epochs in [1, 2]:
                command = f"""python -m {method} \
--model_name='meta-llama/Llama-2-7b-hf' \
--tokenizer_name 'meta-llama/Llama-2-7b-hf' \
--data_path="/home/yd358/rds/hpc-work/analysis_pers/baselines/vpl_llm/data/P_4_survey_100/gpt2"  \
--num_train_epochs={num_train_epochs} \
--data_subset=all \
--log_dir="logs/gpt2_P_4_survey_100" \
--bf16 True \
--fp16 False \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--learning_rate {lr} \
--other_subsets single \
--seed 0 \
--train_dataset_size {train_dataset_size} \
--eval_dataset_size 500 \
--lr_scheduler_type linear \n\n
"""
                command_list.append(command)
                with open(path, "a") as f:
                    f.write(command)    

# for 