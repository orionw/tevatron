#!/bin/bash
encoded_save_path=$1
model=$2
dataset_name=$3
gpu_num=$4
prompt=$5

if [ -z "$prompt" ]; then
  prompt_flag=()
  final_output_path="$encoded_save_path/${dataset_name}_queries_emb.pkl"
else
  prompt_flag=(--prompt "$prompt")
  prompt_hash=$(echo -n "$prompt" | md5sum | awk '{print $1}')
  echo "Prompt hash: $prompt_hash for prompt $prompt"
  echo "Prompt flag: ${prompt_flag[*]}"
  final_output_path="$encoded_save_path/${dataset_name}_queries_emb_${prompt_hash}.pkl"
fi


# if final_output_path exists, skip
if [ -f "$final_output_path" ]; then
  echo "Skipping $dataset_name because of existing file $final_output_path"
  exit 0
fi

echo "Saving to $final_output_path"


CUDA_VISIBLE_DEVICES=$gpu_num python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --lora_name_or_path "$model" \
  --lora \
  --query_prefix "query: " \
  --passage_prefix "passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --encode_is_query \
  --per_device_eval_batch_size 16 \
  --query_max_len 512 \
  --passage_max_len 512 \
  --dataset_name Tevatron/beir \
  --dataset_config "$dataset_name" \
  --dataset_split test \
  --encode_output_path "$final_output_path" "${prompt_flag[@]}"
  
