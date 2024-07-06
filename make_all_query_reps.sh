#!/bin/bash

for i in 1 2 3 4 5 6 7 8; do
    python scripts/tevatron_to_encoded_queries.py -q repllama-v1-7b-lora-passage_embeddings/tevatron_${i}_queries_emb.pkl -t tevatron_queries.json
done
