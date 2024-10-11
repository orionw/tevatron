#!/bin/bash

# infinite loop
while true; do
    # for each folder path in the list, upload to huggingface
    # joint-full llama3.1 llama3.1-instruct mistral-v0.1 mistral-v0.3 reproduced-v2
    for folder_path in bert-v1 bert-standard mean_pool with_tokens;
    do 
        python scripts/utils/upload_to_hf_folder.py -f $folder_path -r orionweller/promptriever-$folder_path
        # if there was an error sleep for an hour
        if [ $? -ne 0 ]; then
            sleep 500
        fi
    done
done