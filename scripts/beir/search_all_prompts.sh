#!/bin/bash

# example: bash scripts/beir/search_all_prompts.sh reproduced-v2

save_path=$1

datasets=(
    'arguana'
    'fiqa'
    'nfcorpus'
    'scidocs'
    'scifact'
    'webis-touche2020'
    'trec-covid'
    'quora'
    'nq'
)


search_and_evaluate() {
    local dataset_name=$1
    local query_emb_file=$2
    local output_suffix=$3

    # if the final eval file exists and has a score, skip
    if [[ -f "${save_path}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval" ]]; then
        if [[ $(awk '/ndcg_cut_10 / {print $3}' "${save_path}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval") != "0.0000" ]]; then
            echo "Skipping ${dataset_name}${output_suffix} because of existing file ${save_path}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval"
            return
        fi
    fi

    echo "Searching and evaluating ${dataset_name} with ${query_emb_file}..."

    python -m tevatron.retriever.driver.search \
    --query_reps "${query_emb_file}" \
    --passage_reps "${save_path}/${dataset_name}/corpus_emb.*.pkl" \
    --batch_size 64 \
    --depth 1000 \
    --save_text \
    --save_ranking_to "${save_path}/${dataset_name}/rank.${dataset_name}${output_suffix}.txt"

    echo "Ranking is saved at ${save_path}/${dataset_name}/rank.${dataset_name}${output_suffix}.txt"

    python -m tevatron.utils.format.convert_result_to_trec \
    --input "${save_path}/${dataset_name}/rank.${dataset_name}${output_suffix}.txt" \
    --output "${save_path}/${dataset_name}/rank.${dataset_name}${output_suffix}.trec" \
    --remove_query

    echo "Evaluating ${dataset_name}${output_suffix}..."
    python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 \
    "beir-v1.0.0-${dataset_name}-test" \
    "${save_path}/${dataset_name}/rank.${dataset_name}${output_suffix}.trec" \
    > "${save_path}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval"

    echo "Score is saved at ${save_path}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval"
    cat "${save_path}/${dataset_name}/rank.${dataset_name}${output_suffix}.eval"
}

# Process all datasets
for dataset in "${datasets[@]}"; do
    dataset_path="${save_path}/${dataset}"
    
    # Search without prompt
    search_and_evaluate "$dataset" "${dataset_path}/${dataset}_queries_emb.pkl" ""

    # Search with generic prompts
    for query_file in "${dataset_path}/${dataset}_queries_emb_"*.pkl; do
        if [[ -f "$query_file" ]]; then
            prompt_hash=$(basename "$query_file" | sed -n 's/.*_emb_\(.*\)\.pkl/\1/p')
            search_and_evaluate "$dataset" "$query_file" "_${prompt_hash}"
        fi
    done
done

# Aggregate results
echo "Aggregating results..."
output_file="${save_path}/aggregate_results.csv"
echo "Dataset,Prompt,NDCG@10,Recall@100" > "$output_file"

for dataset in "${datasets[@]}"; do
    dataset_path="${save_path}/${dataset}"
    
    # Process results without prompt
    eval_file="${dataset_path}/rank.${dataset}.eval"
    if [[ -f "$eval_file" ]]; then
        ndcg=$(awk '/ndcg_cut_10 / {print $3}' "$eval_file")
        recall=$(awk '/recall_100 / {print $3}' "$eval_file")
        echo "${dataset},no_prompt,${ndcg},${recall}" >> "$output_file"
    fi
    
    # Process results with prompts
    for eval_file in "${dataset_path}/rank.${dataset}_"*.eval; do
        if [[ -f "$eval_file" ]]; then
            prompt_hash=$(basename "$eval_file" | sed -n 's/.*_\(.*\)\.eval/\1/p')
            ndcg=$(awk '/ndcg_cut_10 / {print $3}' "$eval_file")
            recall=$(awk '/recall_100 / {print $3}' "$eval_file")
            echo "${dataset},${prompt_hash},${ndcg},${recall}" >> "$output_file"
        fi
    done
done

echo "Aggregate results saved to ${output_file}"