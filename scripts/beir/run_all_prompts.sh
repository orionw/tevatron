#!/bin/bash

# example usage:
#   bash scripts/beir/run_all_prompts.sh orionweller/repllama-reproduced-v2 reproduced-v2
#   bash scripts/beir/run_all_prompts.sh orionweller/repllama-instruct-hard-positives-v2-joint joint-full

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
    'msmarco-dl19'
    'msmarco-dl20'
    'msmarco-dev'
)


# Read in each line of the generic_prompts.csv file where each line is a prompt
# Run it on each dataset, hashing the prompt and passing that as the fourth argument
gpu_num=0
gpu_max=7
while IFS= read -r prompt
do
    for dataset in "${datasets[@]}"; do
        echo "Running prompt on dataset: $dataset"
        echo "Prompt: '$prompt'"
        # if the gpu_num is the max, don't run it in the background, otherwise run in the background
        if [ $gpu_num -eq $gpu_max ]; then
            bash scripts/beir/encode_beir_queries.sh "$nickname/$dataset" "$retriever_name" "$dataset" "$gpu_num" "$prompt"
            # echo "Sleeping for 120 seconds..."
            # sleep 10
            # echo "Done sleeping."
        else
            bash scripts/beir/encode_beir_queries.sh "$nickname/$dataset" "$retriever_name" "$dataset" "$gpu_num" "$prompt" &
        fi
        # update the GPU num looping if it hits the max
        gpu_num=$((gpu_num+1))
        if [ $gpu_num -gt $gpu_max ]; then
            gpu_num=0
        fi
    done
done < generic_prompts.csv


# also run one without a prompt for each dataset
for dataset in "${datasets[@]}"; do
    echo "Running without prompt on dataset: $dataset"
    bash scripts/beir/encode_beir_queries.sh "$nickname/$dataset" "$retriever_name" "$dataset" "$gpu_num"
    # update the GPU num looping if it hits the max
    gpu_num=$((gpu_num+1))
    if [ $gpu_num -gt $gpu_max ]; then
        gpu_num=0
    fi
done


remove_quotes() {
    echo "$1" | sed -e 's/^"//' -e 's/"$//'
}

# Read the CSV file, handling quoted fields
while IFS=',' read -r dataset quoted_prompt
do
    # Remove surrounding quotes from the prompt
    prompt=$(remove_quotes "$quoted_prompt")
    echo "Running domain prompt on dataset: $dataset"
    echo "Prompt: '$prompt'"
    # if the gpu_num is the max, don't run it in the background, otherwise run in the background
    if [ $gpu_num -eq $gpu_max ]; then
        bash scripts/beir/encode_beir_queries.sh "$nickname/$dataset" "$retriever_name" "$dataset" "$gpu_num" "$prompt"
        echo "Sleeping for 120 seconds..."
        # sleep 120
        echo "Done sleeping."
    else
        bash scripts/beir/encode_beir_queries.sh "$nickname/$dataset" "$retriever_name" "$dataset" "$gpu_num" "$prompt" &
    fi
    # update the GPU num looping if it hits the max
    gpu_num=$((gpu_num+1))
    if [ $gpu_num -gt $gpu_max ]; then
        gpu_num=0
    fi
done < domain_prompts.csv
