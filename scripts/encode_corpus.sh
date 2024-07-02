#!/bin/bash

model_path=$1
pretty_name=$(basename $model_path)
echo "Encoding corpus using $pretty_name"
for s in $(seq -f "%01g" 0 8)
do
echo $s
CUDA_VISIBLE_DEVICES=$s python encode.py \
  --output_dir=temp \
  --model_name_or_path $model_path \
  --fp16 \
  --per_device_eval_batch_size 16 \
  --p_max_len 512 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --encoded_save_path ${pretty_name}_embeddings/corpus_emb.${s}.pkl \
  --encode_num_shard 8 \
  --encode_shard_index ${s} > index${s}.log 2>&1 & 
done
