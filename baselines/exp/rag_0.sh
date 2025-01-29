#!/bin/bash

source /home/yd358/rds/hpc-work/miniconda3/etc/profile.d/conda.sh
conda activate personalization
cd /home/yd358/rds/hpc-work/analysis_pers/baselines

python rag.py  --train_dataset_size 50000 --eval_dataset_size 1000 --data_type psoups --subset default --model_name meta-llama/Llama-2-7b-hf --embed_target paired_pref
