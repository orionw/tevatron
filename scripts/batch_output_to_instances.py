import argparse
import json
import glob
from datasets import load_dataset
import tqdm
from collections import defaultdict


def convert_batch_output_to_instances(args):
    batch_output = []
    with open(args.batch_output, "r") as f:
        for line in tqdm.tqdm(f):
            item = line.replace("\n", "").replace("```json", "").replace("```", "").replace("```", "")
            try:
                batch_output.append(json.loads(item))
            except json.JSONDecodeError:
                print(item)
                breakpoint()

    # breakpoint()
    # go through and extract the output using json
    batch_instances_map = defaultdict(list)
    failed = 0
    for idx, batch in tqdm.tqdm(enumerate(batch_output)):
        for choice in batch["response"]["body"]["choices"]:
            if choice["message"]["role"] == "assistant":
                output = choice["message"]["content"]
                try:
                    output = json.loads(output)
                except json.JSONDecodeError:
                    print(output)
                    print("n failed is ", failed)
                    failed += 1
                    continue

                cur_custom_id = batch["custom_id"]
                for passage in output:
                    batch_instances_map[cur_custom_id].append(passage)

    query_ids_to_use = set(batch_instances_map.keys())
    # load the dataset
    print(f"Loading the instruction dataset")
    dataset = load_dataset("orionweller/instruction-msmarco-passage-aug-instructions")["train"]

    final_output = []
    # go through the dataset and add every positive passage to 
    for inst in tqdm.tqdm(dataset):
        if inst["query_id"] not in query_ids_to_use:
            continue

        if not inst["has_instruction"]:
            continue

        final_output.append({
            "passage": inst["positive_passages"][0],
            "matches_both": True,
            "query": inst["only_query"],
            "instruction": inst["only_instruction"],
            "is_real": True,
            "query_id": inst["query_id"],
            "explanation": "real",
            "joint_id": f"{inst['query_id']}"
        })
        # now go through all the generated ones
        for idx, passage in enumerate(batch_instances_map[inst["query_id"]]):
            # print(passage.keys())
            if "matches_both" not in passage:
                if passage["explanation"].strip().lower() != "none":
                    passage["matches_both"] = False
                elif passage["explanation"].strip().lower() == "none":
                    passage["matches_both"] = True
                else:
                    breakpoint()
                    continue
            final_output.append({
                "passage": {"text": passage["passage"], "title": passage["title"] if "title" in passage else "-", "docid": f"{inst['query_id']}_{idx}"},
                "matches_both": passage["matches_both"],
                "query": inst["only_query"],
                "instruction": inst["only_instruction"],
                "is_real": False,
                "query_id": inst["query_id"],
                "explanation": passage["explanation"],
                "joint_id": f"{inst['query_id']}_{idx}"
            })


    print(f"have {len(final_output)} instances (or {len(final_output) / 5} unique queries)")
    # write each of these out for filtering
    with open(args.output_file, "w") as f:
        for inst in final_output:
            f.write(json.dumps(inst) + "\n")
    print(f"Written to {args.output_file}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_output", type=str, required=True)
    parser.add_argument("-o", "--output_file", type=str, required=True)
    args = parser.parse_args()
    convert_batch_output_to_instances(args)

    # example usage
    #   python -m instruction_retrieval.generation.batch_output_to_instances -b batch_50000_output.jsonl -i nfs/prompt_output/msmarco-passage/openai-gpt-4o-2024-05-13/mapping--hard_positives--None--5.json -o batch_instances_50000.jsonl

    # python scripts/batch_output_to_instances.py -b batch_outputs/batch_Y57xfvrFKYSyxp0SSXIaJXUa_output.jsonl -o batch_outputs/batch_instances_Y57xfvrFKYSyxp0SSXIaJXUa.jsonl