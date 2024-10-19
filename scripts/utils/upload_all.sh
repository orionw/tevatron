#!/bin/bash

# infinite loop
while true; do
    # for each folder path in the list, upload to huggingface
    # 
    #     for folder_path in  ;

    for folder_path in bert-v1 bert-standard mean_pool with_tokens swap-v2 standard-long-v1 standard-long-v2 joint-full reproduced-v2;
    do 
        python scripts/utils/upload_to_hf_folder.py -f $folder_path -r orionweller/promptriever-$folder_path
        # if there was an error sleep for an hour
        if [ $? -ne 0 ]; then
            sleep 500
        fi
    done
done