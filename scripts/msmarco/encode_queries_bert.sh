#!/bin/bash
encoded_save_path=$1
model=$2

echo "Path to save: $encoded_save_path"
echo "Model and Model Type: $model $model_type"

for dataset in dl19 dl20 dev; do
echo $dataset
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path $model \
  --tokenizer_name ModernBERT/bert24-base-v2-2ep-decay_100B-0.08-lr \
  --bf16 \
  --pooling cls \
  --normalize \
  --encode_is_query \
  --per_device_eval_batch_size 512 \
  --query_max_len 32 \
  --passage_max_len 156 \
  --dataset_name Tevatron/msmarco-passage \
  --dataset_split $dataset \
  --encode_output_path $encoded_save_path/${dataset}_queries_emb.pkl 
done
  
#   --append_eos_token \
