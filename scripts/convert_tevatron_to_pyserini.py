import os
import glob
import pickle
import json
import subprocess
import argparse
import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np

def process_chunk(args):
    chunk, lookup_chunk = args
    return [json.dumps({"id": l, "vector": r.tolist()}) for r, l in zip(chunk, lookup_chunk)]

def process_file(args):
    pkl_file, folder, chunk_size = args
    print(f"Converting {pkl_file}")
    number = pkl_file.split(".")[-2]
    output_path = os.path.join(folder, "faiss", number, os.path.basename(pkl_file)).replace(".pkl", ".jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving to {output_path}")

    with open(pkl_file, 'rb') as f:
        print(f"Loading...")
        reps, lookup = pickle.load(f)
        assert len(reps) == len(lookup)

    print(f"Converting and writing...")
    with Pool(processes=10) as pool:
        with open(output_path, 'w') as out_f:
            for chunk in tqdm.tqdm(pool.imap(process_chunk, 
                                             ((reps[i:i+chunk_size], lookup[i:i+chunk_size]) 
                                              for i in range(0, len(reps), chunk_size)),
                                             chunksize=1),
                                   total=(len(reps)-1)//chunk_size + 1):
                out_f.write('\n'.join(chunk))
                out_f.write('\n')

    return output_path

def convert(args):
    pkl_files = glob.glob(os.path.join(args.folder, f"corpus_emb.{args.index}.pkl"))
    chunk_size = 1000  # Adjust this value based on your system's memory and number of cores

    # Process files sequentially, but parallelize within each file
    output_paths = []
    for pkl_file in tqdm.tqdm(pkl_files):
        # output_paths.append(process_file((pkl_file, args.folder, chunk_size)))
        # path_to_faiss = os.path.dirname(output_paths[-1])
        path_to_faiss = os.path.join(args.folder, "faiss", args.index)
        print(f"Converting {path_to_faiss} to FAISS")
        # run it in the background
        print(f"python -m pyserini.index.faiss --dim 4096 --input {path_to_faiss} --output {path_to_faiss}_faiss")
        subprocess.Popen(["python", "-m", "pyserini.index.faiss", "--dim", "4096", "--input", path_to_faiss, "--output", f"{path_to_faiss}_faiss"])


    # when done run this in the faiss folder after moving all non-faiss folders
    #  python -m pyserini.index.merge_faiss_indexes --prefix ./ --shard-num 8 --dim 4096

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Tevatron index to Pyserini index")
    parser.add_argument("-f", "--folder", type=str, help="The folder to convert", required=True)
    parser.add_argument("-i", "--index", type=str, help="The index to convert", required=True)
    args = parser.parse_args()
    convert(args)

    # python scripts/convert_tevatron_to_pyserini.py -f repllama-v1-7b-lora-passage_embeddings