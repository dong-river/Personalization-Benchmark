#!/bin/bash

source /home/yd358/rds/hpc-work/miniconda3/etc/profile.d/conda.sh
conda activate personalization
cd /home/yd358/rds/hpc-work/analysis_pers/baselines/vpl_llm

bash /home/yd358/rds/hpc-work/analysis_pers/baselines/vpl_llm/generate_llm_embeddings_UF_P_4_Dec1.sh  llama single 8