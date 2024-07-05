#!/bin/bash

model_path=$1
pretty_name=$(basename $model_path)
echo "Encoding corpus using $pretty_name"
echo "Model path is $model_path"
for s in $(seq -f "%01g" 0 7)
do
echo $s
# add 4 to s if we want the latter gpus
# cuda_s=$((s+4))
cuda_s=$s
echo $cuda_s
CUDA_VISIBLE_DEVICES=$cuda_s python encode.py \
  --output_dir=temp \
  --model_name_or_path castorini/repllama-v1-7b-lora-passage \
  --tokenizer_name meta-llama/Llama-2-7b-hf \
  --fp16 \
  --per_device_eval_batch_size 64 \
  --p_max_len 512 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --encoded_save_path ${pretty_name}_embeddings/corpus_emb.${s}.pkl \
  --encode_num_shard 8 \
  --encode_shard_index ${s} > index_og_${s}.log 2>&1 & 
done
