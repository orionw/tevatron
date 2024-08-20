import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import tqdm

def main():
    parser = argparse.ArgumentParser(description="Count tokens in the 'query' field of a dataset")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf", help="Model name for tokenizer")
    parser.add_argument("--dataset", default="orionweller/instruction-msmarco-passage-aug-50-fixed-standard", help="Dataset name")
    parser.add_argument("--split", default="train", help="Dataset split to use")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load dataset
    dataset = load_dataset(args.dataset, split=args.split)

    # Count tokens
    total_tokens = 0
    token_list = []
    pos_token_list = []
    for example in tqdm.tqdm(dataset):
        # query = example.get("query", "")
        # tokens = tokenizer.encode(query)
        # total_tokens += len(tokens)
        # token_list.append(len(tokens))
        pos = example.get("positive_passages", "")[0]["title"] + " " + example.get("positive_passages", "")[0]["text"]
        pos_tokens = tokenizer.encode(pos)
        pos_token_list.append(len(pos_tokens))


    # print(f"Total number of tokens in the 'query' field: {total_tokens}")
    # print(f"Average number of tokens per query: {total_tokens / len(dataset):.2f}")

    # # print the percentiles
    # percentiles = np.percentile(token_list, [25, 50, 75, 90, 95, 99])
    # print("Percentiles:")
    # print(percentiles)
    """
    Total number of tokens in the 'query' field: 37049308
    Average number of tokens per query: 75.46
    Percentiles:
    [ 10.  35. 128. 201. 242. 316.]
    {
        "25": 10.0,
        "50": 35.0,
        "75": 128.0,
        "90": 201.0,
        "95": 242.0,
        "99": 316.0
    }
    """

    """
    for docs:
        [ 74.  92. 123. 158. 178. 222.]
    
    {
        "25": 74.0,
        "50": 92.0,
        "75": 123.0,
        "90": 158.0,
        "95": 178.0,
        "99": 222.0
    }
    """

    # print(f"Total number of tokens in the 'pos' field: {sum(pos_token_list)}")
    # print(f"Average number of tokens per pos: {sum(pos_token_list) / len(dataset):.2f}")

    # print the percentiles
    percentiles = np.percentile(pos_token_list, [25, 50, 75, 90, 95, 99])
    print("Percentiles:")
    print(percentiles)


if __name__ == "__main__":
    main()

    # example usage:
    # python max_lengths.py --dataset="orionweller/instruction-msmarco-passage-aug-50-fixed-standard" --split="train"