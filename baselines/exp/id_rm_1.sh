#!/bin/bash

source /home/yd358/rds/hpc-work/miniconda3/etc/profile.d/conda.sh
conda activate personalization
cd /home/yd358/rds/hpc-work/analysis_pers/baselines


python id_rm.py --train --eval --train_dataset_size 1000 --eval_dataset_size 1000 --data_type personal_llm --subset default  --per_device_train_batch_size 8 --gradient_accumulation_steps 8 --lora_rank 8 --lora_alpha 16 --learning_rate 3e-4 --num_train_epochs 1 --model_name meta-llama/Llama-2-7b-hf --log_dir /home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/models

python id_rm.py --train --eval --train_dataset_size 1000 --eval_dataset_size 1000 --data_type personal_llm --subset default  --per_device_train_batch_size 8 --gradient_accumulation_steps 8 --lora_rank 8 --lora_alpha 16 --learning_rate 3e-4 --num_train_epochs 10 --model_name meta-llama/Llama-2-7b-hf --log_dir /home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/models

python id_rm.py --train --eval --train_dataset_size 1000 --eval_dataset_size 1000 --data_type personal_llm --subset default  --per_device_train_batch_size 8 --gradient_accumulation_steps 8 --lora_rank 8 --lora_alpha 16 --learning_rate 3e-4 --num_train_epochs 30 --model_namemeta-llama/Llama-2-7b-hf --log_dir /home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/models

python id_rm.py --train --eval --train_dataset_size 10000 --eval_dataset_size 1000 --data_type personal_llm --subset default  --per_device_train_batch_size 8 --gradient_accumulation_steps 8 --lora_rank 8 --lora_alpha 16 --learning_rate 3e-4 --num_train_epochs 1 --model_name meta-llama/Llama-2-7b-hf --log_dir /home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/models
