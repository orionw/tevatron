#!/bin/bash
encoded_save_path=$1
model=$2
for dataset in dl19 dl20 dev; do
echo $dataset
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --lora_name_or_path $model \
  --lora \
  --query_prefix "query: " \
  --passage_prefix "passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --encode_is_query \
  --per_device_eval_batch_size 128 \
  --query_max_len 32 \
  --passage_max_len 156 \
  --dataset_name Tevatron/msmarco-passage \
  --dataset_split $dataset \
  --encode_output_path $encoded_save_path/${dataset}_queries_emb.pkl 
done
  
