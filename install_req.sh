#!/bin/bash
# conda create -n tevatron python=3.10 -y
# conda activate tevatron

pip install deepspeed accelerate
pip install transformers datasets peft
pip install faiss-cpu
pip install -r requirements.txt
pip install -e .
# huggingface-cli login --token $TOKEN --add-to-git-credential