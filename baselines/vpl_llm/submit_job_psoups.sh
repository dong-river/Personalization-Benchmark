export WANDB_MODE=online
export WANDB_PROJECT=vpl
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# Set model_name to be 'gpt2' or 'meta-llama/Llama-2-7b-hf' here
model_name='llama'

model_type=$1

if [ ${model_type} == "vae" ];
then
# Train VPL on UltraFeedback four-user dataset
python -m hidden_context.train_llm_vae_preference_model \
        --model_name=${model_name} \
        --data_path="/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/data/psoups_4_survey_100/llama" \
        --num_train_epochs=2 \
        --reward_model_type=vae \
        --data_subset=all \
        --log_dir="/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/models" \
        --bf16 True \
        --fp16 False \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --latent_dim 512 \
        --hidden_dim 512 \
        --learning_rate 1e-4 \
        --use_annealing True \
        --kl_loss_weight 3e-6 \
        --controversial_only True \
        --fixed_contexts True \
        --fixed_llm_embeddings False \
        --up_sampling False \
        --other_subsets psoups \
        --use_last_token_embedding True \
        --seed 0
else
# Train baseline models on UltraFeedback four-user dataset
python -m hidden_context.train_llm_preference_model \
        --model_name=${model_name} \
        --data_path="/home/yd358/rds/rds-semioticsteam-gLFG2ebmCDI/river_cache/data/psoups_4_survey_100/llama" \
        --num_train_epochs=2 \
        --reward_model_type=${model_type} \
        --data_subset=all \
        --log_dir="logs/gpt2_P_4_survey_100" \
        --bf16 True \
        --fp16 False \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1e-4 \
        --controversial_only True \
        --up_sampling False \
        --other_subsets single \
        --seed 0
fi
