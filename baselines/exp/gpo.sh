#!/bin/bash

source /home/yd358/rds/hpc-work/miniconda3/etc/profile.d/conda.sh
conda activate personalization
cd /home/yd358/rds/hpc-work/analysis_pers/baselines

python gpo.py --train_dataset_size 1000 --eval_dataset_size 1000 --learning_rate 1e-5 --data_type psoups --num_steps 20000 --subset default --eval_freq 10 --max_ctx_num_qs 70 --min_ctx_num_qs 10 --max_tar_num_qs 70 --min_tar_num_qs 10 --embed_batch_size 2 --weight_decay 0.01

python gpo.py --train_dataset_size 10000 --eval_dataset_size 1000 --learning_rate 1e-5 --data_type psoups --num_steps 20000 --subset default --eval_freq 100 --max_ctx_num_qs 100 --min_ctx_num_qs 10 --max_tar_num_qs 100 --min_tar_num_qs 10 --embed_batch_size 2 --weight_decay 0.01
