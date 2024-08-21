import csv
from collections import defaultdict
import hashlib
import os


PRETTY_NAMES = {
    "arguana": "Arguana",
    "climate-fever": "Climate-FEVER",
    "dbpedia-entity": "DBPedia",
    "fever": "FEVER",
    "fiqa": "FiQA",
    "hotpotqa": "HotpotQA",
    "nfcorpus": "NFCorpus",
    "nq": "NQ",
    "quora": "Quora",
    "scidocs": "SCIDOCS",
    "scifact": "SciFact",
    "trec-covid": "TREC-COVID",
    "webis-touche2020": "Touche-2020"
}

def md5(string):
    return hashlib.md5(string.encode()).hexdigest()

def read_generic_prompts(filename):
    with open(filename, 'r') as f:
        return [md5(line.strip()) for line in f]

def read_domain_prompts(filename):
    domain_prompts = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            dataset, prompt = row
            domain_prompts[dataset.lower()] = md5(prompt)
    return domain_prompts

def read_csv(filename, generic_hashes, domain_hashes):
    data = defaultdict(lambda: defaultdict(dict))
    if not os.path.exists(filename):
        print(f"Warning: file {filename} does not exist")
        return data
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row['Dataset'].lower()
            prompt = row['Prompt']
            ndcg = float(row['NDCG@10']) * 100 if row['NDCG@10'] else None
            if ndcg is None:
                # print a warning, so we can delete it
                print(f"Warning: missing NDCG@10 for {dataset} ({prompt}) in {filename}")
                continue
            
            if prompt == 'no_prompt':
                data[dataset]['None'] = ndcg
            elif prompt in generic_hashes:
                if 'Generic' not in data[dataset] or ndcg > data[dataset]['Generic']:
                    data[dataset]['Generic'] = ndcg
            elif prompt == domain_hashes.get(dataset):
                data[dataset]['Domain'] = ndcg
    return data

def format_value(value):
    return f"{value:.1f}" if value is not None else "-"

def calculate_average(data):
    values = [v for v in data.values() if v is not None]
    return sum(values) / len(values) if values else None

def generate_latex_table(bm25_data, repllama_data, modelname_data):
    datasets = PRETTY_NAMES.keys()
    latex_rows = []
    for dataset in datasets:
        pretty_name = PRETTY_NAMES[dataset]
        
        row = [
            pretty_name,
            format_value(bm25_data[dataset].get('None')),
            format_value(bm25_data[dataset].get('Generic')),
            format_value(bm25_data[dataset].get('Domain')),
            format_value(repllama_data[dataset].get('None')),
            format_value(repllama_data[dataset].get('Generic')),
            format_value(repllama_data[dataset].get('Domain')),
            format_value(modelname_data[dataset].get('None')),
            format_value(modelname_data[dataset].get('Generic')),
            format_value(modelname_data[dataset].get('Domain'))
        ]
        latex_rows.append(" & ".join(row) + " \\\\")
    
    # Calculate averages
    averages = [
        "Average",
        format_value(calculate_average({d: bm25_data[d].get('None') for d in datasets})),
        format_value(calculate_average({d: bm25_data[d].get('Generic') for d in datasets})),
        format_value(calculate_average({d: bm25_data[d].get('Domain') for d in datasets})),
        format_value(calculate_average({d: repllama_data[d].get('None') for d in datasets})),
        format_value(calculate_average({d: repllama_data[d].get('Generic') for d in datasets})),
        format_value(calculate_average({d: repllama_data[d].get('Domain') for d in datasets})),
        format_value(calculate_average({d: modelname_data[d].get('None') for d in datasets})),
        format_value(calculate_average({d: modelname_data[d].get('Generic') for d in datasets})),
        format_value(calculate_average({d: modelname_data[d].get('Domain') for d in datasets}))
    ]
    
    latex_table = r"""
\begin{table*}[t]
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{l|ccc|ccc|ccc}
\toprule
\multirow{2}{*}{Dataset} & \multicolumn{3}{c|}{BM25} & \multicolumn{3}{c|}{RepLLaMA} & \multicolumn{3}{c}{\modelname} \\
\cmidrule(l){2-4} \cmidrule(l){5-7} \cmidrule(l){8-10}
 & None & Generic & Domain & None & Generic & Domain & None & Generic & Domain \\
\midrule
""" + "\n".join(latex_rows) + r"""
\midrule
""" + " & ".join(averages) + r""" \\
\bottomrule
\end{tabular}
}
\caption{Zero-shot effectiveness of BM25, RepLLaMA, and \modelname on BEIR datasets. Results are shown for standard retrieval (None), with generic instructions (Generic), and with domain-specific instructions (Domain). Missing values are indicated by "-".}
\label{tab:beir}
\end{table*}
"""
    return latex_table

# Usage
generic_hashes = read_generic_prompts('generic_prompts.csv')
domain_hashes = read_domain_prompts('domain_prompts.csv')

bm25_data = read_csv('bm25/bm25_aggregate_results.csv', generic_hashes, domain_hashes)
repllama_data = read_csv('reproduced-v2/aggregate_results.csv', generic_hashes, domain_hashes)
modelname_data = read_csv('modelname_results.csv', generic_hashes, domain_hashes)

latex_table = generate_latex_table(bm25_data, repllama_data, modelname_data)
print(latex_table)