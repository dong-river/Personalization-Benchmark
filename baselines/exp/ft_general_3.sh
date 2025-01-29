#!/bin/bash

source /home/yd358/rds/hpc-work/miniconda3/etc/profile.d/conda.sh
conda activate personalization
cd /home/yd358/rds/hpc-work/analysis_pers/baselines


python ft_rm_general.py --train --eval --train_dataset_size 100000 --eval_dataset_size 1000 --data_type summarization --subset default  --per_device_train_batch_size 8 --gradient_accumulation_steps 8 --lora_rank 8 --lora_alpha 16 --learning_rate 3e-4 --num_train_epochs 1 --model_name weqweasdas/RM-Mistral-7B --log_dir /home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/models
