#!/bin/bash

source /home/yd358/rds/hpc-work/miniconda3/etc/profile.d/conda.sh
conda activate personalization
cd /home/yd358/rds/hpc-work/analysis_pers/baselines

python ft_rm.py --train --eval --train_dataset_size 100000 --eval_dataset_size 1000 --data_type summarization --subset default --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 0.0003 --num_train_epochs 1 --model_name meta-llama/Llama-2-7b-hf --log_dir /home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/models
