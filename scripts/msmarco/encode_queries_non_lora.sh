#!/bin/bash
encoded_save_path=$1
model=$2
pooling_type=$3

# if model type is empty then use llama2
if [ -z "$model_type" ]
then
  model_type="meta-llama/Llama-2-7b-hf"
fi

echo "Path to save: $encoded_save_path"
echo "Model and Model Type: $model $model_type"
echo "Pooling type: $pooling_type"

for dataset in dl19 dl20 dev; do
echo $dataset
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path $model \
  --query_prefix "query: " \
  --passage_prefix "passage: " \
  --bf16 \
  --pooling $pooling_type \
  --normalize \
  --encode_is_query \
  --per_device_eval_batch_size 128 \
  --query_max_len 32 \
  --passage_max_len 156 \
  --dataset_name Tevatron/msmarco-passage \
  --dataset_split $dataset \
  --encode_output_path $encoded_save_path/${dataset}_queries_emb.pkl 
done
  
