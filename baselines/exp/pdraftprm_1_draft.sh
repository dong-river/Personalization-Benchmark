#!/bin/bash

source /home/yd358/rds/hpc-work/miniconda3/etc/profile.d/conda.sh
conda activate prm
cd /home/yd358/rds/hpc-work/analysis_pers/baselines/Personalized_RLHF

python prlhf/train_language_model_dpo.py --user_model individual   --model_class llama  --model_name meta-llama/Llama-2-7b-hf     --tokenizer_name meta-llama/Llama-2-7b-hf   --dataset psoups  --learning_rate 5e-06 --output_dir /home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/models --num_train_epochs 1 --train_dataset_size 1000

python prlhf/train_personalized_rm.py --user_model individual   --model_class llama  --model_name meta-llama/Llama-2-7b-hf     --tokenizer_name meta-llama/Llama-2-7b-hf   --dataset psoups  --learning_rate 5e-06 --output_dir /home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/models --num_train_epochs 1 --train_dataset_size 1000

python prlhf/train_personalized_rm2.py --user_model individual   --model_class llama  --model_name meta-llama/Llama-2-7b-hf     --tokenizer_name meta-llama/Llama-2-7b-hf   --dataset psoups  --learning_rate 5e-06 --output_dir /home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/models --num_train_epochs 1 --train_dataset_size 1000

python prlhf/train_gptneo_rm.py --user_model individual   --model_class gptneo  --model_name EleutherAI/gpt-neo-1.3B     --tokenizer_name EleutherAI/gpt-neo-1.3B   --dataset psoups  --learning_rate 5e-05 --output_dir /home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/models --num_train_epochs 1 --train_dataset_size 1000


python prlhf/train_personalized_rm.py --user_model individual   --model_class llama  --model_name meta-llama/Llama-2-7b-hf    --tokenizer_name meta-llama/Llama-2-7b-hf   --dataset psoups  --learning_rate 1e-05 --output_dir /home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/models --num_train_epochs 2 --train_dataset_size 1000 --lora_rank 16 --lora_alpha 32


python prlhf/train_personalized_rm.py --user_model individual   --model_class llama  --model_name meta-llama/Llama-2-7b-hf    --tokenizer_name meta-llama/Llama-2-7b-hf   --dataset psoups  --learning_rate 1e-05 --output_dir /home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/models --num_train_epochs 3 --train_dataset_size 1000 --lora_rank 32 --lora_alpha 64



python prlhf/train_gptneo_rm.py --user_model individual   --model_class gptneo  --model_name EleutherAI/gpt-neo-1.3B     --tokenizer_name EleutherAI/gpt-neo-1.3B   --dataset personal_llm  --learning_rate 5e-05 --output_dir /home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/models --num_train_epochs 1 --train_dataset_size 1000
