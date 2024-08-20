import argparse
import json
import os

DEFAULT_INPUT_DIR = os.path.expanduser("~/tevatron/msmarco-passage-aug")

def read_prompt(prompt_file):
    with open(prompt_file, 'r') as file:
        return file.read().strip()

def process_file(input_file, output_file, prompt, ensure_question_mark):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            query = data['query']
            
            if ensure_question_mark and not query.strip().endswith('?'):
                query = query.strip() + '?'
            
            data['query'] = f"{query} {prompt}".strip()
            json.dump(data, outfile)
            outfile.write('\n')

def main():
    parser = argparse.ArgumentParser(description="Process datasets with added prompt")
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR, help="Input directory containing the datasets")
    parser.add_argument("--prompt", help="Prompt to append to each query")
    parser.add_argument("--prompt_file", help="File containing the prompt to append to each query")
    parser.add_argument("--ensure_question_mark", action="store_true", help="Ensure query ends with a question mark before appending prompt")
    parser.add_argument("output_suffix", help="Suffix to append to the output directory name")
    
    args = parser.parse_args()
    
    if args.prompt and args.prompt_file:
        raise ValueError("Please provide either --prompt or --prompt_file, not both")
    elif args.prompt_file:
        prompt = read_prompt(args.prompt_file)
    elif args.prompt:
        prompt = args.prompt
    else:
        raise ValueError("Please provide either --prompt or --prompt_file")

    input_dir = args.input_dir
    output_dir = f"{input_dir}-{args.output_suffix}"
    
    if os.path.exists(output_dir):
        raise FileExistsError(f"Output directory {output_dir} already exists. Please choose a different output suffix or remove the existing directory.")
    
    os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.jsonl'):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            process_file(input_file, output_file, prompt, args.ensure_question_mark)
    
    print(f"Processing complete. Output directory: {output_dir}")

if __name__ == "__main__":
    main()

    # python prompt_msmarco.py long_train --prompt "Understand the intent behind this query and find passages that best match that intent. Consider both lexical and semantic relevance when ranking the results. Prioritize passages that provide clear, concise, and accurate information related to the query." --ensure_question_mark 
    # python prompt_msmarco.py short_train --prompt "Retrieve passages that answer the users question" --ensure_question_mark
    # python prompt_msmarco.py short_new --prompt "Find passages that are most relevant to the query" --ensure_question_mark
    # python prompt_msmarco.py long_new --prompt "Find passages that are most relevant to the query. Consider both lexical and semantic relevance when ranking the results." --ensure_question_mark
    # python prompt_msmarco.py prompted_new --prompt "Find passages that are most relevant to the query. Be very careful and return the only documents that you are sure answer the query." --ensure_question_mark