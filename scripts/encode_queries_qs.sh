#!/bin/bash
encoded_save_path=$1
model=$2
mkdir -p $encoded_save_path
for dataset in dl19 dl20 msmarcodev; do
echo $dataset
CUDA_VISIBLE_DEVICES=6 python -m tevatron.retriever.driver.encode \
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
  --query_max_len 304 \
  --passage_max_len 156 \
  --dataset_name orionweller/msmarco-aug-question-mark \
  --dataset_config $dataset \
  --dataset_split train \
  --encode_output_path $encoded_save_path/${dataset}_queries_q_emb.pkl 
done
  
