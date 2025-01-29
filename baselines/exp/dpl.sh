#!/bin/bash

source /home/yd358/rds/hpc-work/miniconda3/etc/profile.d/conda.sh
conda activate personalization
cd /home/yd358/rds/hpc-work/analysis_pers/baselines/dpl_llm

python -m hidden_context.train_llm_preference_model --model_name=meta-llama/Llama-2-7b-hf --num_train_epochs=1 --reward_model_type=categorical --data_subset=both --data_path RiverDong/psoups --learning_rate 1e-5

python -m hidden_context.train_llm_preference_model --model_name=meta-llama/Llama-2-7b-hf --num_train_epochs=1 --reward_model_type=categorical --data_subset=both --data_path RiverDong/psoups --learning_rate 5e-5