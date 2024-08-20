#!/bin/bash

# example usage: bash scripts/beir/run_all.sh orionweller/repllama-reproduced-v2 reproduced-v2

export CUDA_VISIBLE_DEVICES=0,1,2,3

retriever_name=$1
nickname=$2

mkdir -p $nickname

datasets=(
    'arguana'
    # 'climate-fever'
    # 'dbpedia-entity'
    # 'fever'
    'fiqa'
    # 'hotpotqa'
    'nfcorpus'
    'quora'
    'scidocs'
    'scifact'
    'trec-covid'
    'webis-touche2020'
    'nq'
)

for dataset in "${datasets[@]}"; do
    # if the dataset already exists (corpus_emb.0.pkl exists), skip it
    if [ -d "$nickname/$dataset" ] && [ -f "$nickname/$dataset/corpus_emb.0.pkl" ]; then
        echo "Skipping $dataset"
        continue
    fi
    echo "Encoding corpus: $dataset"
    bash scripts/beir/encode_beir_corpus.sh $nickname/$dataset $retriever_name $dataset
done

# for dataset in "${datasets[@]}"; do
#     if [ -d "$nickname/$dataset" ] && [ -f "$nickname/$dataset/${dataset}_queries_emb.pkl" ]; then
#         echo "Skipping $dataset"
#         continue
#     fi
#     echo "Encoding queries: $dataset"
#     bash scripts/beir/encode_beir_queries.sh $nickname/$dataset $retriever_name $dataset
# done

# TODO: run all prompts through this in queries and search over them

# for dataset in "${datasets[@]}"; do
#     if [ -d "$nickname/$dataset" ] && [ -f "$nickname/$dataset/rank.${dataset}.eval" ]; then
#         echo "Skipping $dataset"
#         continue
#     fi
#     echo "Searching dataset: $dataset"
#     bash scripts/beir/search_beir.sh $nickname/$dataset $dataset
# done