#!/bin/bash
model_path=$1
pretty_name=$(basename $model_path)
echo "Retrieving using $pretty_name"
for dataset in dl19 dl20 dev; do 
    python -m tevatron.faiss_retriever \
    --query_reps "${pretty_name}_embeddings/${dataset}_queries_emb.pkl" \
    --passage_reps '${pretty_name}_embeddings/corpus_emb.*.pkl' \
    --depth 100 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to ${pretty_name}_embeddings/rank.${dataset}.txt

    python -m tevatron.utils.format.convert_result_to_trec --input ${pretty_name}_embeddings/rank.${dataset}.txt \
                                                        --output ${pretty_name}_embeddings/rank.${dataset}.trec \
                                                        --remove_query

    # if dataset is dev use msmarco-passage-dev-subset
    if [ $dataset == "dev" ]; then
        pyserini_dataset="msmarco-passage-dev-subset"
    else
        pyserini_dataset="${dataset}-passage"
    fi

    echo "Evaluating ${dataset}..."
    echo "python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 $pyserini_dataset ${pretty_name}_embeddings/rank.${dataset}.trec > ${pretty_name}_embeddings/eval.${dataset}.txt"

    python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 $pyserini_dataset ${pretty_name}_embeddings/rank.${dataset}.trec > ${pretty_name}_embeddings/eval.${dataset}.txt
done
