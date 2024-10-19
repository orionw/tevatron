#!/bin/bash
# args are (1) name of run (2) dataset, and (3) nodes e.g. "0,1,2,3" (4) port num either 1 or 2 or something
# bash scripts/training/train_modernbert_cls.sh cls orionweller/joint-msmarco-passage-aug  "0,1,2,3,4,5,6,7" 0 > modernbert_cls.log 2>&1
echo "Args are $1 $2 $3 $4"
deepspeed --include localhost:$3 --master_port "6000$4" --module tevatron.retriever.driver.train \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir retriever-modernbert-cls_v2 \
  --model_name_or_path ModernBERT/bert24-base-v2-2ep-decay_100B-0.08-lr \
  --save_steps 1000 \
  --dataset_name $2 \
  --bf16 \
  --pooling cls \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 24 \
  --train_group_size 24 \
  --learning_rate 8e-5 \
  --query_max_len 304 \
  --passage_max_len 196 \
  --num_train_epochs 1 \
  --logging_steps 5 \
  --overwrite_output_dir \
  --warmup_steps 200 \
  --negatives_first_n 3 \
  --gradient_accumulation_steps 10


deepspeed --include localhost:$3 --master_port "6000$4" --module tevatron.retriever.driver.train \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir retriever-modernbert-eos \
  --model_name_or_path ModernBERT/bert24-base-v2-2ep-decay_100B-0.08-lr \
  --save_steps 1000 \
  --dataset_name $2 \
  --bf16 \
  --query_prefix "query: " \
  --passage_prefix "passage: " \
  --append_eos_token \
  --pooling eos \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 24 \
  --train_group_size 24 \
  --learning_rate 8e-5 \
  --query_max_len 304 \
  --passage_max_len 196 \
  --num_train_epochs 1 \
  --logging_steps 5 \
  --overwrite_output_dir \
  --warmup_steps 200 \
  --negatives_first_n 3 \
  --gradient_accumulation_steps 1


  