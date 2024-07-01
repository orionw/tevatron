#!/bin/bash

for dataset in dev; do # dl19 dl20 
    python -m tevatron.faiss_retriever \
    --query_reps "msmarco_embeddings/${dataset}_queries_emb.pkl" \
    --passage_reps 'msmarco_embeddings/corpus_emb.*.pkl' \
    --depth 100 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to msmarco_embeddings/rank.${dataset}.txt

    python -m tevatron.utils.format.convert_result_to_trec --input msmarco_embeddings/rank.${dataset}.txt \
                                                        --output msmarco_embeddings/rank.${dataset}.trec \
                                                        --remove_query

    # if dataset is dev use msmarco-passage-dev-subset
    if [ $dataset == "dev" ]; then
        pyserini_dataset="msmarco-passage-dev-subset"
    else
        pyserini_dataset="${dataset}-passage"
    fi

    echo "Evaluating ${dataset}..."
    echo "python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 $pyserini_dataset msmarco_embeddings/rank.${dataset}.trec"

    python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 $pyserini_dataset msmarco_embeddings/rank.${dataset}.trec
done
