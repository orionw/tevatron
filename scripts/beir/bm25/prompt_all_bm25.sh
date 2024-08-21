#!/bin/bash

# example usage:
#   bash scripts/beir/bm25/prompt_all_bm25.sh 

nickname=bm25

mkdir -p $nickname

datasets=(
    'arguana'
    'fiqa'
    'nfcorpus'
    'scidocs'
    'scifact'
    'trec-covid'
    'webis-touche2020'
    'quora'
    'nq'
    'hotpotqa'
    'climate-fever'
    'dbpedia-entity'
    'fever'
    # 'msmarco'
)


# Read in each line of the generic_prompts.csv file where each line is a prompt
# Run it on each dataset, hashing the prompt and passing that as the fourth argument
while IFS= read -r prompt
do
    prompt_hash=$(echo -n "$prompt" | md5sum | awk '{print $1}')
    for dataset in "${datasets[@]}"; do
        mkdir -p "$nickname/$dataset"
        if [ -f "$nickname/$dataset/${dataset}_${prompt_hash}.trec" ]; then
            echo "Skipping $dataset because of existing file $nickname/$dataset/$dataaset_$prompt_hash.trec"
            continue
        fi
        echo "Running prompt on dataset: $dataset"
        echo "Prompt: '$prompt'"
        python scripts/run_bm25s.py --dataset_name "$dataset" --prompt "$prompt" --top_k 1000 --output_dir "$nickname/$dataset" --prompt_hash "$prompt_hash"
    done
done < generic_prompts.csv


# also run one without a prompt for each dataset
for dataset in "${datasets[@]}"; do
    if [ -f "$nickname/$dataset/$dataset.trec" ]; then
        echo "Skipping $dataset because of existing file $nickname/$dataset/$dataset.trec"
        continue
    fi
    echo "Running without prompt on dataset: $dataset"
    python scripts/run_bm25s.py --dataset_name $dataset --top_k 1000 --output_dir "$nickname/$dataset"
done



# # now load domain_prompt.csv (dataset, prompt) and run it on each dataset
# while IFS=, read -r dataset prompt
# do
#     echo "Running prompt on dataset: $dataset"
#     echo "Prompt: '$prompt'"
#     # if the gpu_num is the max, don't run it in the background, otherwise run in the background
#     if [ $gpu_num -eq $gpu_max ]; then
#         bash scripts/beir/encode_beir_queries.sh "$nickname/$dataset" "$retriever_name" "$dataset" "$gpu_num" "$prompt"
#         echo "Sleeping for 120 seconds..."
#         sleep 120
#         echo "Done sleeping."
#     else
#         bash scripts/beir/encode_beir_queries.sh "$nickname/$dataset" "$retriever_name" "$dataset" "$gpu_num" "$prompt" &
#     fi
#     # update the GPU num looping if it hits the max
#     gpu_num=$((gpu_num+1))
#     if [ $gpu_num -gt $gpu_max ]; then
#         gpu_num=0
#     fi
# done < domain_prompt.csv
