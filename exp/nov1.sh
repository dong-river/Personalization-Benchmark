#!/bin/bash
source /home/yd358/rds/hpc-work/miniconda3/etc/profile.d/conda.sh
conda activate personalization
cd /home/yd358/rds/hpc-work/analysis_pers/baselines


python id_rm.py --train_dataset_size 60000 --eval_data_size 1000  --data_path "openbmb/UltraFeedback"  --per_device_train_batch_size 16 --gradient_accumulation_steps 64 --learning_rate 3e-4 --num_train_epochs 1 --model_name "google/gemma-2b-it" --max_length 4096

python id_rm.py --train_dataset_size 60000 --eval_data_size 1000  --data_path "openbmb/UltraFeedback"  --per_device_train_batch_size 16 --gradient_accumulation_steps 64 --learning_rate 1e-4 --num_train_epochs 2 --model_name "google/gemma-2b-it" --max_length 4096
