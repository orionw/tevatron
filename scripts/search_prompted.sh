#!/bin/bash

# Base directory where the new directories are located
BASE_DIR=~/tevatron/msmarco-passage-aug

# Array of suffixes for the new directories
SUFFIXES=("long_train" "short_train" "short_new" "long_new" "prompted_new")

# Loop through each suffix
for suffix in "${SUFFIXES[@]}"; do
    echo "bash scripts/search_prompted_ind.sh $suffix"
    bash scripts/search_prompted_ind.sh $suffix
done

echo "Processing complete."