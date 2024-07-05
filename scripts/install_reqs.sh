#!/bin/bash
# conda create -n tevatron-eval python=3.10 -y
# conda activate tevatron-eval

pip install deepspeed accelerate
pip install transformers datasets peft
pip install faiss-cpu
pip install xformers
pip install pyserini==0.25.0 
pip install -r requirements.txt
pip install -e .
# huggingface-cli login --token $TOKEN --add-to-git-credential
# conda install openjdk=11 -y