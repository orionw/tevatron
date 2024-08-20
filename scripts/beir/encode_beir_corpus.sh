#!/bin/bash
path_to_save=$1
model=$2
dataset_name=$3
mkdir -p $path_to_save
# ps aux | grep "[p]ython -m tevatron.retriever.driver.encode" | awk '{print $2}' | xargs kill
echo "Encoding BEIR corpus... $dataset_name"
# reverse these 
# reverse it
for s in $(seq 3 -1 0);
do
  gpuid=$s
  echo $gpuid
  # if it's the last one (aka zero), don't run in background
  if [ "$s" == "0" ]; then
    # give it some time so that it's the last to run
    sleep 60
    CUDA_VISIBLE_DEVICES=$gpuid python -m tevatron.retriever.driver.encode \
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
      --per_device_eval_batch_size 32 \
      --query_max_len 512 \
      --passage_max_len 512 \
      --dataset_name "Tevatron/beir-corpus" \
      --dataset_config "$dataset_name" \
      --dataset_split "train" \
      --dataset_number_of_shards 4 \
      --dataset_shard_index ${s} \
      --encode_output_path $path_to_save/corpus_emb.${s}.pkl > encode_corpus_${dataset_name}_${s}.log 2>&1
  else
  CUDA_VISIBLE_DEVICES=$gpuid python -m tevatron.retriever.driver.encode \
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
    --per_device_eval_batch_size 32 \
    --query_max_len 304 \
    --passage_max_len 512 \
    --dataset_name "Tevatron/beir-corpus" \
    --dataset_config "$dataset_name" \
    --dataset_split "train" \
    --dataset_number_of_shards 4 \
    --dataset_shard_index ${s} \
    --encode_output_path $path_to_save/corpus_emb.${s}.pkl > encode_corpus_${dataset_name}_${s}.log 2>&1 &
  fi
done
  
# bash scripts/beir/encode_beir_corpus.sh reproduced-v2/scifact orionweller/repllama-reproduced-v2 scifact