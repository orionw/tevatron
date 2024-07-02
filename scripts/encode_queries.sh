#!/bin/bash
model_path=$1
pretty_name=$(basename $model_path)
echo "Encoding queries using $pretty_name"
for dataset in dl19 dl20 dev; do
    CUDA_VISIBLE_DEVICES=0 python encode.py \
    --output_dir=temp \
    --model_name_or_path $model_path \
    --tokenizer_name meta-llama/Llama-2-7b-hf \
    --fp16 \
    --per_device_eval_batch_size 16 \
    --q_max_len 512 \
    --dataset_name Tevatron/msmarco-passage/$dataset \
    --encoded_save_path ${pretty_name}_embeddings/${dataset}_queries_emb.pkl \
    --encode_is_qry

done
  
