
# Set model_type to be 'gpt2' or 'llama' for model_type
# Set other_subsets to be 'ultra_feedback', 'single', or '84', 'psoups'

# bash generate_llm_embeddings_UF_P_4.sh llama single

model_type=$1
data_type=$2 ##psoups or ultrafeedback
subset=$3 

# Final version for four users
survey_size=100
other_subsets='psoups' ## Must not set to "single" if running for poups

python -m hidden_context.data_utils.add_survey_contexts --output_dir "/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/data/${data_type}_4_survey_${survey_size}/" \
--data_path "/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/data/${data_type}_${other_subsets}_P_4" --data_subset ${subset} --data_split train --model_type ${model_type} \
--other_subsets ${other_subsets} --data_type ${data_type} --with_embeddings True --survey_size $survey_size --num_duplicates 4 --train_dataset_size 200000 --eval_data_size 100000

python -m hidden_context.data_utils.add_survey_contexts --output_dir "/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/data/${data_type}_4_survey_${survey_size}/" \
--data_path "/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/data/${data_type}_${other_subsets}_P_4" --data_subset ${subset} --data_split test --model_type ${model_type} \
--other_subsets ${other_subsets} --data_type ${data_type} --with_embeddings True --survey_size $survey_size --num_duplicates 4 --train_dataset_size 200000 --eval_data_size 100000