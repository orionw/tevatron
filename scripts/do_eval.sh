#!/bin/bash

# example:
#   bash scripts/do_eval.sh /home/ubuntu/tevatron/retriever-llama2-instruct-cl-slow

model_path=$1
pretty_name=$(basename $model_path)
echo "Running evaluation using $pretty_name"

if [ ! -d "${pretty_name}_embeddings" ]; then
    mkdir ${pretty_name}_embeddings
fi

if [ ! -f "${pretty_name}_embeddings/corpus_emb.3.pkl" ]; then
    # echo "Corpus embeddings not found. Please run encode_corpus.sh first."
    bash scripts/encode_corpus.sh $model_path 
    exit 0
fi

# if ${pretty_name}_embeddings/dl19_queries_emb.pkl exists, skip
if [ ! -f "${pretty_name}_embeddings/dl19_queries_emb.pkl" ]; then
    bash scripts/encode_queries.sh $model_path
fi


# if the eval file doesn't exist, run the search
if [ ! -f "${pretty_name}_embeddings/eval.${dataset}.txt" ]; then
    datasets=(dl19 dl20) 
    for i in "${!datasets[@]}"; do 
        cmd="CUDA_VISIBLE_DEVICES=$i bash scripts/search.sh $model_path "${datasets[$i]}" $i &"
        echo $cmd
        eval $cmd
    done
fi