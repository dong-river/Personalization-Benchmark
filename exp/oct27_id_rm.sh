#!/bin/bash
source /home/yd358/rds/hpc-work/miniconda3/etc/profile.d/conda.sh
conda activate personalization
cd /home/yd358/rds/hpc-work/analysis_pers/baselines
python -m id_rm --model_name='meta-llama/Llama-2-7b-hf' --tokenizer_name 'meta-llama/Llama-2-7b-hf' --data_path="/home/yd358/rds/hpc-work/analysis_pers/baselines/vpl_llm/data/P_4_survey_100/gpt2"  --num_train_epochs=1 --data_subset=all --log_dir="logs/gpt2_P_4_survey_100" --bf16 True --fp16 False --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --learning_rate 0.0001 --other_subsets single --seed 0 --train_dataset_size 10000 --eval_dataset_size 500 --lr_scheduler_type linear 


python -m id_rm --model_name='meta-llama/Llama-2-7b-hf' --tokenizer_name 'meta-llama/Llama-2-7b-hf' --data_path="/home/yd358/rds/hpc-work/analysis_pers/baselines/vpl_llm/data/P_4_survey_100/gpt2"  --num_train_epochs=2 --data_subset=all --log_dir="logs/gpt2_P_4_survey_100" --bf16 True --fp16 False --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --learning_rate 0.0001 --other_subsets single --seed 0 --train_dataset_size 10000 --eval_dataset_size 500 --lr_scheduler_type linear 


python -m id_rm --model_name='meta-llama/Llama-2-7b-hf' --tokenizer_name 'meta-llama/Llama-2-7b-hf' --data_path="/home/yd358/rds/hpc-work/analysis_pers/baselines/vpl_llm/data/P_4_survey_100/gpt2"  --num_train_epochs=1 --data_subset=all --log_dir="logs/gpt2_P_4_survey_100" --bf16 True --fp16 False --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --learning_rate 0.0003 --other_subsets single --seed 0 --train_dataset_size 10000 --eval_dataset_size 500 --lr_scheduler_type linear 


python -m id_rm --model_name='meta-llama/Llama-2-7b-hf' --tokenizer_name 'meta-llama/Llama-2-7b-hf' --data_path="/home/yd358/rds/hpc-work/analysis_pers/baselines/vpl_llm/data/P_4_survey_100/gpt2"  --num_train_epochs=2 --data_subset=all --log_dir="logs/gpt2_P_4_survey_100" --bf16 True --fp16 False --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --learning_rate 0.0003 --other_subsets single --seed 0 --train_dataset_size 10000 --eval_dataset_size 500 --lr_scheduler_type linear 


python -m id_rm --model_name='meta-llama/Llama-2-7b-hf' --tokenizer_name 'meta-llama/Llama-2-7b-hf' --data_path="/home/yd358/rds/hpc-work/analysis_pers/baselines/vpl_llm/data/P_4_survey_100/gpt2"  --num_train_epochs=1 --data_subset=all --log_dir="logs/gpt2_P_4_survey_100" --bf16 True --fp16 False --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --learning_rate 0.0001 --other_subsets single --seed 0 --train_dataset_size 0 --eval_dataset_size 500 --lr_scheduler_type linear 


python -m id_rm --model_name='meta-llama/Llama-2-7b-hf' --tokenizer_name 'meta-llama/Llama-2-7b-hf' --data_path="/home/yd358/rds/hpc-work/analysis_pers/baselines/vpl_llm/data/P_4_survey_100/gpt2"  --num_train_epochs=2 --data_subset=all --log_dir="logs/gpt2_P_4_survey_100" --bf16 True --fp16 False --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --learning_rate 0.0001 --other_subsets single --seed 0 --train_dataset_size 0 --eval_dataset_size 500 --lr_scheduler_type linear 


python -m id_rm --model_name='meta-llama/Llama-2-7b-hf' --tokenizer_name 'meta-llama/Llama-2-7b-hf' --data_path="/home/yd358/rds/hpc-work/analysis_pers/baselines/vpl_llm/data/P_4_survey_100/gpt2"  --num_train_epochs=1 --data_subset=all --log_dir="logs/gpt2_P_4_survey_100" --bf16 True --fp16 False --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --learning_rate 0.0003 --other_subsets single --seed 0 --train_dataset_size 0 --eval_dataset_size 500 --lr_scheduler_type linear 


python -m id_rm --model_name='meta-llama/Llama-2-7b-hf' --tokenizer_name 'meta-llama/Llama-2-7b-hf' --data_path="/home/yd358/rds/hpc-work/analysis_pers/baselines/vpl_llm/data/P_4_survey_100/gpt2"  --num_train_epochs=2 --data_subset=all --log_dir="logs/gpt2_P_4_survey_100" --bf16 True --fp16 False --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --learning_rate 0.0003 --other_subsets single --seed 0 --train_dataset_size 0 --eval_dataset_size 500 --lr_scheduler_type linear 


