#!/bin/bash

model_path=$1
pretty_name=$(basename $model_path)
echo "Running evaluation using $pretty_name"

if [ ! -d "${pretty_name}_embeddings" ]; then
    mkdir ${pretty_name}_embeddings
fi

# if ${pretty_name}_embeddings/corpus_emb.${s}.pkl exists, skip
if [ ! -f "${pretty_name}_embeddings/corpus_emb.7.pkl" ]; then
    bash scripts/encode_corpus.sh $model_path 
fi

# if ${pretty_name}_embeddings/dl19_queries_emb.pkl exists, skip
if [ ! -f "${pretty_name}_embeddings/dl19_queries_emb.pkl" ]; then
    bash scripts/encode_queries.sh $model_path
fi

# if the eval file doesn't exist, run the search
if [ ! -f "${pretty_name}_embeddings/eval.${dataset}.txt" ]; then
    bash scripts/search.sh $model_path
fi