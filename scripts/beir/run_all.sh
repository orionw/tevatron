#!/bin/bash

# example usage: bash scripts/beir/run_all.sh orionweller/repllama-reproduced-v2 reproduced-v2
# example usage: bash scripts/beir/run_all.sh orionweller/repllama-instruct-hard-positives-v2-joint joint-full

# export CUDA_VISIBLE_DEVICES="4,5,6,7"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

retriever_name=$1
nickname=$2

mkdir -p $nickname

datasets=(
    'fiqa'
    'nfcorpus'
    'scidocs'
    'scifact'
    'trec-covid'
    'webis-touche2020'
    'quora'
    'nq'
    'arguana'
    'hotpotqa'
    'fever'
    'climate-fever'
    'dbpedia-entity'
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
