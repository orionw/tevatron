#!/bin/bash

# Base directory where the new directories are located
BASE_DIR=~/tevatron/msmarco-passage-aug

# Array of suffixes for the new directories
SUFFIXES=("long_train" "short_train" "short_new" "long_new" "prompted_new")

# Loop through each suffix
for suffix in "${SUFFIXES[@]}"; do
    # Construct the full path to the new directory
    NEW_DIR="${BASE_DIR}-${suffix}"
    
    # Check if the directory exists
    if [ -d "$NEW_DIR" ]; then
        # echo "Processing directory: $NEW_DIR"
        
        # Call the encoder_queries.sh script with the new directory
        echo "bash scripts/encode_queries_prompts_ind.sh $suffix"
        bash scripts/encode_queries_prompts_ind.sh $suffix
    else
        echo "Directory not found: $NEW_DIR"
    fi
done

echo "Processing complete."