#!/bin/bash

name=$1
# if it's empty, quit
if [ -z "$name" ]
then
  echo "Please provide a name"
  exit 1
fi
bash ./scripts/msmarco/encode_corpus_bert_eos.sh modernbert-$name retriever-modernbert-$name
bash ./scripts/msmarco/encode_queries_bert_eos.sh modernbert-$name retriever-modernbert-$name
bash ./scripts/msmarco/search.sh modernbert-$name