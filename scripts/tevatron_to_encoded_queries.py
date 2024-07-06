import os
import glob
import pickle
import json
import subprocess
import argparse
import pandas as pd
import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np

QUERY_MAP = {}

def process_chunk(args):
    global QUERY_MAP
    chunk, lookup_chunk = args
    return [{"id": l, "embedding": r.tolist(), "text": QUERY_MAP[str(l)]} for r, l in zip(chunk, lookup_chunk)]


def convert(args):
    global QUERY_MAP
    with open(args.query_reps, 'rb') as f:
        print(f"Loading...")
        reps, lookup = pickle.load(f)
        assert len(reps) == len(lookup)

    # load the text file
    with open(args.query_text, 'r') as f:
        QUERY_MAP = json.load(f)

    print(f"Converting and writing...")
    output_path = args.query_reps.replace(".pkl", ".jsonl")
    print(f"Saving to {output_path}")
    chunk_size = 1000
    all_chunks = []
    with Pool(processes=10) as pool:
        for chunk in tqdm.tqdm(pool.imap(process_chunk, 
                                            ((reps[i:i+chunk_size], lookup[i:i+chunk_size]) 
                                            for i in range(0, len(reps), chunk_size)),
                                            chunksize=1),
                                total=(len(reps)-1)//chunk_size + 1):
            all_chunks.extend(chunk)

    # save it out as as pickle file with columns ['id', 'embedding', 'text'] in a new directory with embeddings.pkl
    os.makedirs(args.query_reps.replace(".pkl", ""), exist_ok=True)
    df = pd.DataFrame(all_chunks)
    df.to_pickle(args.query_reps.replace(".pkl", "/embedding.pkl"))


 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Tevatron index to Pyserini encoded query file")
    parser.add_argument("-q", "--query_reps", type=str, help="The query reps to convert", required=True)
    parser.add_argument("-t", "--query_text", type=str, help="The query text to use", required=True)
    args = parser.parse_args()
    convert(args)

    # example usage
    #   python scripts/tevatron_to_encoded_queries.py -q repllama-v1-7b-lora-passage_embeddings/tevatron_1_queries_emb.pkl -t tevatron_queries.json

