#!/bin/bash

# Example
#   bash scripts/search_pyserini.sh repllama-v1-7b-lora-passage_embeddings/faiss/full tevatron_queries_01.tsv repllama-v1-7b-lora-passage_embeddings/tevatron_1_queries_emb repllama-v1-7b-lora-passage_embeddings/output_01.txt 0


index_folder=$1
query_file=$2
query_emb_file=$3
output_file=$4
gpu=$5

echo "Index folder: $index_folder"
echo "Query file: $query_file"
echo "Query embeddings file: $query_emb_file"
echo "Output file: $output_file"
echo "GPU: $gpu"

cmd="""python -m pyserini.search.faiss \
  --threads 16 \
  --batch-size 64 \
  --index $index_folder \
  --topics $query_file \
  --encoded-queries $query_emb_file \
  --output $output_file \
  --device cuda:$gpu \
  --hits 1000"""

echo $cmd
eval $cmd