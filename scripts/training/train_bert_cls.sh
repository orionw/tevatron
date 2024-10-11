#!/bin/bash
# args are (1) name of run (2) dataset, and (3) nodes e.g. "0,1,2,3" (4) port num either 1 or 2 or something
# bash scripts/training/train_bert_cls.sh standard Tevatron/msmarco-passage-aug "0,1,2,3,4,5,6,7" 0 > bert_standard.log 2>&1
echo "Args are $1 $2 $3 $4"
deepspeed --include localhost:$3 --master_port "6000$4" --module tevatron.retriever.driver.train \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir retriever-bert-$1 \
  --model_name_or_path bert-base-uncased \
  --save_steps 500 \
  --dataset_name $2 \
  --query_prefix "query: " \
  --passage_prefix "passage: " \
  --bf16 \
  --pooling cls \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 32 \
  --train_group_size 16 \
  --learning_rate 1e-5 \
  --query_max_len 32 \
  --passage_max_len 196 \
  --num_train_epochs 5 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --warmup_steps 100 \
  --gradient_accumulation_steps 1
  # --dont_shuffle
  