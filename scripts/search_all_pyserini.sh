#!/bin/bash

for i in 1 2 3 4 5 6 7 8; do
    echo "Launching search for dataset $i"
    # device is i - 1
    device=$((i-1))
    bash scripts/search_pyserini.sh repllama-v1-7b-lora-passage_embeddings/faiss/full tevatron_queries_0${i}.tsv repllama-v1-7b-lora-passage_embeddings/tevatron_${i}_queries_emb repllama-v1-7b-lora-passage_embeddings/output_0${i}.txt $device > logs_search_0${i}.log 2>&1 &
done
  
