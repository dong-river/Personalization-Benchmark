#!/bin/bash

source /home/yd358/rds/hpc-work/miniconda3/etc/profile.d/conda.sh
conda activate personalization
cd /home/yd358/rds/hpc-work/analysis_pers/baselines/vpl_llm

python -m hidden_context.create_embedding  \
--output_dir "/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/data/personal_llm_4_survey_100/" \
--data_path "/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/data/personal_llm_single_P_4" \
--data_subset default --data_split train --model_type llama  --with_embeddings True \
--survey_size 100 --num_duplicates 4 --data_type personal_llm --uid 2

python -m hidden_context.create_embedding  \
--output_dir "/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/data/personal_llm_4_survey_100/"\
--data_path "/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/data/personal_llm_single_P_4" \
--data_subset default --data_split test --model_type llama  --with_embeddings True  \
--survey_size 100 --num_duplicates 4 --data_type personal_llm --uid 2

