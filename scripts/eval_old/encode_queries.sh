#!/bin/bash
for dataset in dl19 dl20 dev; do
    CUDA_VISIBLE_DEVICES=0 python encode.py \
    --output_dir=temp \
    --model_name_or_path retriever-llama2 \
    --tokenizer_name meta-llama/Llama-2-7b-hf \
    --fp16 \
    --per_device_eval_batch_size 16 \
    --q_max_len 512 \
    --dataset_name Tevatron/msmarco-passage/$dataset \
    --encoded_save_path msmarco_embeddings/${dataset}_queries_emb.pkl \
    --encode_is_qry

done
  
