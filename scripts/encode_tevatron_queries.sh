#!/bin/bash
model_path=$1
pretty_name=$(basename $model_path)
echo "Encoding queries using $pretty_name"
# --model_name_or_path $model_path \

for dataset in 1 2 3 4 5 6 7 8; do
    # cuda device is dataset - 1
    echo "Encoding dataset $dataset"
    device=$((dataset-1))
    CUDA_VISIBLE_DEVICES=$device python encode.py \
    --output_dir=temp \
    --model_name_or_path castorini/repllama-v1-7b-lora-passage \
    --tokenizer_name meta-llama/Llama-2-7b-hf \
    --fp16 \
    --per_device_eval_batch_size 32 \
    --q_max_len 512 \
    --dataset_name /home/ubuntu/eval/eval/tevatron_queries_0$dataset.tsv \
    --encoded_save_path ${pretty_name}_embeddings/tevatron_${dataset}_queries_emb.pkl \
    --encode_is_qry &
done
  
