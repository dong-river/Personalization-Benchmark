#!/bin/bash

source /home/yd358/rds/hpc-work/miniconda3/etc/profile.d/conda.sh
conda activate prm
cd /home/yd358/rds/hpc-work/analysis_pers/baselines/Personalized_RLHF

python prlhf/train_personalized_rm.py --user_model individual   --model_class llama  --model_name meta-llama/Llama-2-7b-hf    --tokenizer_name meta-llama/Llama-2-7b-hf   --dataset psoups  --learning_rate 3e-05 --output_dir /home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/models --num_train_epochs 1 --train_dataset_size 100000 --lora_rank 32 --lora_alpha 64