
# Set model_type to be 'gpt2' or 'llama' for model_type
# Set other_subsets to be 'ultra_feedback', 'single', or '84'

# bash generate_llm_embeddings_UF_P_4.sh llama single 2
# bash generate_llm_embeddings_UF_P_4.sh llama single 4
# bash generate_llm_embeddings_UF_P_4.sh llama single 8


model_type=$1
other_subsets=$2
subset=$3
data_type=$4 ## 'ultrafeedback' or 'psoups'

# Generate LLM embeddings for UltraFeedback dataset
if [ "${other_subsets}" = "ultra_feedback" ]; then
    subsets="helpfulness honesty instruction_following truthfulness"
elif [ "${other_subsets}" = "single" ]; then
    subsets="8 4 2 1"
elif [ "${other_subsets}" = "84" ]; then
    subsets="8 4"
else
    echo "Invalid!"
fi

echo "${subsets}"


# Final version for four users
survey_size=100

python -m hidden_context.data_utils.add_survey_contexts --output_dir "/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/data/P_4_survey_${survey_size}/" \
--data_path "/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/data/UltraFeedback_${other_subsets}_P_4" --data_subset ${subset} --data_split train --model_type ${model_type} \
--other_subsets ${other_subsets} --with_embeddings True --survey_size $survey_size --num_duplicates 4 --data_type ${data_type}

python -m hidden_context.data_utils.add_survey_contexts --output_dir "/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/data/P_4_survey_${survey_size}/" \
--data_path "/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/data/UltraFeedback_${other_subsets}_P_4" --data_subset ${subset} --data_split test --model_type ${model_type} \
--other_subsets ${other_subsets} --with_embeddings True --survey_size $survey_size --num_duplicates 4 --data_type ${data_type}


python -m hidden_context.create_embedding --output_dir "/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/data/P_4_survey_${survey_size}/" \
--data_path "/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/data/UltraFeedback_single_P_4" --data_subset default --data_split train --model_type llama  --with_embeddings True --survey_size 100 --num_duplicates 4 --data_type summarization --uid 1
