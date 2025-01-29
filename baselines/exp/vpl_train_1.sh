#!/bin/bash

source /home/yd358/rds/hpc-work/miniconda3/etc/profile.d/conda.sh
conda activate personalization
cd /home/yd358/rds/hpc-work/analysis_pers/baselines/vpl_llm

python -m hidden_context.train_llm_vae_preference_model \
--model_name meta-llama/Llama-2-7b-hf --data_path /home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/data/summarization_4_survey_100/llama  \
--num_train_epochs 1 --reward_model_type vae --data_subset all --log_dir /home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/models  \
--bf16 True --fp16 False --per_device_train_batch_size 2 --gradient_accumulation_steps 16  \
--latent_dim 1024 --hidden_dim 1024 --learning_rate 3e-05 --use_annealing True --kl_loss_weight 3e-6 --controversial_only True --fixed_contexts True \
--fixed_llm_embeddings False --up_sampling False --use_last_token_embedding True --seed 0 \
--train_dataset_size 10000 --eval_dataset_size 1000