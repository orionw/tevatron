#!/bin/bash
model_path=$1
dataset=$2
device=$3
pretty_name=$(basename $model_path)
echo "Retrieving using $pretty_name"
echo "Dataset is $dataset"

python -m tevatron.faiss_retriever \
--query_reps ${pretty_name}_embeddings/${dataset}_queries_emb.pkl \
--passage_reps "${pretty_name}_"'embeddings/corpus_emb.*.pkl' \
--depth 100 \
--batch_size 64 \
--save_text \
--save_ranking_to ${pretty_name}_embeddings/rank.${dataset}.txt \
# --device "cuda:$device"

python -m tevatron.utils.format.convert_result_to_trec --input ${pretty_name}_embeddings/rank.${dataset}.txt \
                                                    --output ${pretty_name}_embeddings/rank.${dataset}.trec \
                                                    --remove_query

# dev has a different name in pyserini
if [ $dataset == "dev" ]; then
    pyserini_dataset="msmarco-passage-dev-subset"
    echo "Evaluating dev..."
    echo "python -m pyserini.eval.trec_eval -c -M 10 -m recip_rank $pyserini_dataset ${pretty_name}_embeddings/rank.${dataset}.trec > ${pretty_name}_embeddings/eval.${dataset}.txt"
    python -m pyserini.eval.trec_eval -c -M 10 -m recip_rank $pyserini_dataset ${pretty_name}_embeddings/rank.${dataset}.trec > ${pretty_name}_embeddings/eval.${dataset}.txt
    # now also do recall
    echo "python -m pyserini.eval.trec_eval -c -m recall.1000 $pyserini_dataset ${pretty_name}_embeddings/rank.${dataset}.trec >> ${pretty_name}_embeddings/eval.${dataset}.txt"
    python -m pyserini.eval.trec_eval -c -m recall.1000 $pyserini_dataset ${pretty_name}_embeddings/rank.${dataset}.trec >> ${pretty_name}_embeddings/eval.${dataset}.txt
else
    pyserini_dataset="${dataset}-passage"
    echo "Evaluating ${dataset}..."
    echo "python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 $pyserini_dataset ${pretty_name}_embeddings/rank.${dataset}.trec > ${pretty_name}_embeddings/eval.${dataset}.txt"
    python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 -mrecip_rank.10 $pyserini_dataset ${pretty_name}_embeddings/rank.${dataset}.trec > ${pretty_name}_embeddings/eval.${dataset}.txt
fi


