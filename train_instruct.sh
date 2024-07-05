#!/bin/bash
# args are (1) name of run (2) dataset, and (3) nodes e.g. "0,1,2,3" (4) port num either 1 or 2 or something
# bash train_instruct.sh 25 orionweller/instruction-msmarco-passage-aug-25-percent "0,1,2,3" 0 > 25-percent-4gpu.log 2>&1
# bash train_instruct.sh 75 orionweller/instruction-msmarco-passage-aug-75-percent "4,5,6,7" 1 > 75-percent-4gpu.log 2>&1
# bash train_instruct.sh 50 orionweller/instruction-msmarco-passage-aug-50-percent "4,5,6,7" 2 > 50-percent-4gpu.log 2>&1
# bash train_instruct.sh cl-slow orionweller/instruction-msmarco-passage-aug-cl-slow "0,1,2,3" 0 > cl-slow-percent-4gpu.log 2>&1
# bash train_instruct.sh cl-reverse orionweller/instruction-msmarco-passage-aug-cl-reverse "4,5,6,7" 1 > cl-reverse-percent-4gpu.log 2>&1
# bash train_instruct.sh cl-fast orionweller/instruction-msmarco-passage-aug-cl-fast "4,5,6,7" 2 > cl-fast-percent-4gpu.log 2>&1

echo "Args are $1 $2 $3 $4"
deepspeed --include localhost:$3 --master_port "6000$4" --module tevatron.retriever.driver.train \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir retriever-llama2-instruct-$1 \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --lora \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 200 \
  --dataset_name $2 \
  --query_prefix "Query: " \
  --passage_prefix "Passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 8 \
  --gradient_checkpointing \
  --train_group_size 16 \
  --learning_rate 1e-4 \
  --query_max_len 32 \
  --passage_max_len 196 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --warmup_steps 100 \
  --gradient_accumulation_steps 4 \
  --dont_shuffle
  