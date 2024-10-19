#!/bin/bash
path_to_save=$1
model=$2

echo "Path to save: $path_to_save"
echo "Model and Model Type: $model"

mkdir -p $path_to_save
for s in $(seq -f "%01g" 0 7)
do
# add 4 to the gpu_id
# gpuid=$((s+4))
gpuid=$s
# if it's the last gpu, run without nohup
if [ $s -eq 7 ]
then
  echo $gpuid
  CUDA_VISIBLE_DEVICES=$gpuid python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --model_name_or_path $model \
      --tokenizer_name ModernBERT/bert24-base-v2-2ep-decay_100B-0.08-lr \
    --bf16 \
    --query_prefix "query: " \
    --passage_prefix "passage: " \
    --append_eos_token \
    --pooling eos \
    --normalize \
    --per_device_eval_batch_size 512 \
    --query_max_len 32 \
    --passage_max_len 156 \
    --dataset_name Tevatron/msmarco-passage-corpus \
    --dataset_number_of_shards 8 \
    --dataset_shard_index ${s} \
    --encode_output_path $path_to_save/corpus_emb.${s}.pkl 
else
  echo $gpuid
  CUDA_VISIBLE_DEVICES=$gpuid python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --model_name_or_path $model \
    --tokenizer_name ModernBERT/bert24-base-v2-2ep-decay_100B-0.08-lr \
    --bf16 \
    --query_prefix "query: " \
    --passage_prefix "passage: " \
    --append_eos_token \
    --pooling eos \
    --normalize \
    --per_device_eval_batch_size 512 \
    --query_max_len 32 \
    --passage_max_len 156 \
    --dataset_name Tevatron/msmarco-passage-corpus \
    --dataset_number_of_shards 8 \
    --dataset_shard_index ${s} \
    --encode_output_path $path_to_save/corpus_emb.${s}.pkl > logs/msmarco_encode_corpus_$s.log 2>&1 &
fi
done