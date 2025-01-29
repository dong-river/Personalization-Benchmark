#!/bin/bash

source /home/yd358/rds/hpc-work/miniconda3/etc/profile.d/conda.sh
conda activate personalization
cd /home/yd358/rds/hpc-work/analysis_pers/baselines

python ft_rm_general.py --train --eval --train_dataset_size 200000 --eval_data_size 1000 --data_type ultrafeedback --subset controversial  --per_device_train_batch_size 8 --gradient_accumulation_steps 8 --lora_rank 8 --lora_alpha 16 --learning_rate 5e-5 --num_train_epochs 1 --model_name google/gemma-2b-it --log_dir /home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/models

python ft_rm_general.py --train --eval --train_dataset_size 200000 --eval_data_size 1000 --data_type ultrafeedback --subset controversial  --per_device_train_batch_size 8 --gradient_accumulation_steps 8 --lora_rank 8 --lora_alpha 16 --learning_rate 5e-5 --num_train_epochs 2 --model_name google/gemma-2b-it --log_dir /home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/models
