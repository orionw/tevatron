#!/bin/bash
mkdir -p batch_output
num_array=(50000 100000 150000 200000)
for s in $(seq -f "%01g" 0 3)
do
# gpu = gpu + 4
gpuid=$((s+4))
num=${num_array[$s]}
echo $gpuid
CUDA_VISIBLE_DEVICES=$gpuid python scripts/filter_query_doc_pairs_from_batch_gpt.py -i batch_instances_$num.jsonl -o followir_batch_scores_$num.tsv > filter_$s.log 2>&1 &
done
  
