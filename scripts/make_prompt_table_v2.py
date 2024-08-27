import csv
from collections import defaultdict
import hashlib
import os
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# BEIR
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

# MSMARCO-based
# PRETTY_NAMES = {
#     "msmarco-dev": "MSMARCO Dev",
#     "msmarco-dl19": "DL19",
#     "msmarco-dl20": "DL20",
# }


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

            # Use MRR for MSMarco dev, NDCG@10 for others
            if dataset == 'msmarco-dev':
                try:
                    score = float(row['MRR']) * 100 if row['MRR'] else None
                except ValueError:
                    print(f"Warning: invalid MRR value for {dataset} ({prompt}) in {filename}")
                    continue
            else:
                try:
                    score = float(row['NDCG@10']) * 100 if row['NDCG@10'] else None
                except ValueError:
                    print(f"Warning: invalid NDCG@10 value for {dataset} ({prompt}) in {filename}")
                    continue

            if score is None:
                print(f"Warning: missing score for {dataset} ({prompt}) in {filename}")
                continue

            if prompt == 'no_prompt':
                data[dataset]['None'] = score
            elif prompt in generic_hashes:
                data[dataset]['Generic'][prompt] = score
            elif prompt == domain_hashes.get(dataset):
                data[dataset]['Domain'] = score

    # Validate that all prompts are present for each dataset
    for dataset in data:
        if 'None' not in data[dataset]:
            print(f"Warning: 'None' prompt missing for {dataset} in {filename}")
        if 'Domain' not in data[dataset]:
            print(f"Warning: Domain prompt missing for {dataset} in {filename}")
        if set(data[dataset]['Generic'].keys()) != set(generic_hashes):
            missing_prompts = set(generic_hashes) - set(data[dataset]['Generic'].keys())
            print(f"Warning: Generic prompts {missing_prompts} missing for {dataset} in {filename}")

    return data

def format_value(value):
    return f"{value:.1f}" if value is not None else "-"

def find_best_generic_prompt(data):
    prompt_averages = defaultdict(list)
    for dataset_data in data.values():
        for prompt, score in dataset_data['Generic'].items():
            prompt_averages[prompt].append(score)
    
    best_prompt = max(prompt_averages, key=lambda p: statistics.mean(prompt_averages[p]))
    return best_prompt

def calculate_average(values, expected_count):
    non_none_values = [v for v in values if v is not None]
    if len(values) != expected_count:
        print(f"Warning: Expected {expected_count} values for average calculation, but got {len(values)}")
    if len(non_none_values) != len(values):
        print(f"Warning: {len(values) - len(non_none_values)} None values excluded from average calculation")
    return statistics.mean(non_none_values) if non_none_values else None

def generate_latex_table(bm25_data, repllama_data, modelname_data):
    datasets = list(PRETTY_NAMES.keys())
    latex_rows = []

    for dataset in datasets:
        pretty_name = PRETTY_NAMES[dataset]

        row = [
            pretty_name,
            format_value(bm25_data[dataset].get('None')),
            format_value(max(bm25_data[dataset]['Generic'].values()) if bm25_data[dataset]['Generic'] else bm25_data[dataset].get('Domain')),
            format_value(repllama_data[dataset].get('None')),
            format_value(max(repllama_data[dataset]['Generic'].values()) if repllama_data[dataset]['Generic'] else repllama_data[dataset].get('Domain')),
            format_value(modelname_data[dataset].get('None')),
            format_value(max(modelname_data[dataset]['Generic'].values()) if modelname_data[dataset]['Generic'] else modelname_data[dataset].get('Domain'))
        ]
        latex_rows.append(" & ".join(row) + " \\\\")

    # Calculate averages
    expected_count = len(datasets)
    averages = [
        "Average",
        format_value(calculate_average([bm25_data[d].get('None') for d in datasets], expected_count)),
        format_value(calculate_average([max(bm25_data[d]['Generic'].values()) if bm25_data[d]['Generic'] else bm25_data[d].get('Domain') for d in datasets], expected_count)),
        format_value(calculate_average([repllama_data[d].get('None') for d in datasets], expected_count)),
        format_value(calculate_average([max(repllama_data[d]['Generic'].values()) if repllama_data[d]['Generic'] else repllama_data[d].get('Domain') for d in datasets], expected_count)),
        format_value(calculate_average([modelname_data[d].get('None') for d in datasets], expected_count)),
        format_value(calculate_average([max(modelname_data[d]['Generic'].values()) if modelname_data[d]['Generic'] else modelname_data[d].get('Domain') for d in datasets], expected_count))
    ]

    latex_table = f"""
\\begin{{table*}}[t]
\\centering
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{l|cc|cc|cc}}
\\toprule
\\multirow{{2}}{{*}}{{Dataset}} & \\multicolumn{{2}}{{c|}}{{BM25}} & \\multicolumn{{2}}{{c|}}{{RepLLaMA}} & \\multicolumn{{2}}{{c}}{{\\modelname}} \\\\
\\cmidrule(l){{2-3}} \\cmidrule(l){{4-5}} \\cmidrule(l){{6-7}}
 & None & Prompted & None & Prompted & None & Prompted \\\\
\\midrule
{chr(10).join(latex_rows)}
\\midrule
{" & ".join(averages)} \\\\
\\bottomrule
\\end{{tabular}}
}}
\\caption{{Effectiveness of BM25, RepLLaMA, and \\modelname on BEIR datasets. Results are shown for standard retrieval (None) and with the best prompt per dataset (Prompted). Missing values are indicated by "-".}}
\\label{{tab:beir-results}}
\\end{{table*}}
"""
    return latex_table

def save_model_csv(data, model_name):
    with open(f"{model_name}_prompts.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Prompt", "Score"])

        datasets = list(PRETTY_NAMES.keys())
        expected_count = len(datasets)

        for dataset in datasets:
            prompt_data = data[dataset]
            if 'None' in prompt_data:
                writer.writerow([dataset, "None", prompt_data['None']])
            if 'Domain' in prompt_data:
                writer.writerow([dataset, "Domain", prompt_data['Domain']])
            for prompt, score in prompt_data.get('Generic', {}).items():
                writer.writerow([dataset, prompt, score])

        # Add average row
        writer.writerow(["Average", "None", format_value(calculate_average([data[d].get('None') for d in datasets], expected_count))])
        writer.writerow(["Average", "Domain", format_value(calculate_average([data[d].get('Domain') for d in datasets], expected_count))])
        
        generic_averages = []
        for prompt in set().union(*[data[d].get('Generic', {}).keys() for d in datasets]):
            avg = calculate_average([data[d]['Generic'].get(prompt) for d in datasets], expected_count)
            generic_averages.append(avg)
            writer.writerow(["Average", prompt, format_value(avg)])

        overall_best_generic = max(generic_averages) if generic_averages else None
        writer.writerow(["Average", "Best Generic", format_value(overall_best_generic)])

def generate_latex_table_domain(bm25_data, repllama_data, modelname_data):
    datasets = list(PRETTY_NAMES.keys())
    latex_rows = []

    for dataset in datasets:
        pretty_name = PRETTY_NAMES[dataset]

        row = [
            pretty_name,
            format_value(bm25_data[dataset].get('None')),
            format_value(bm25_data[dataset].get('Domain')),
            format_value(repllama_data[dataset].get('None')),
            format_value(repllama_data[dataset].get('Domain')),
            format_value(modelname_data[dataset].get('None')),
            format_value(modelname_data[dataset].get('Domain'))
        ]
        latex_rows.append(" & ".join(row) + " \\\\")

    # Calculate averages
    expected_count = len(datasets)
    averages = [
        "Average",
        format_value(calculate_average([bm25_data[d].get('None') for d in datasets], expected_count)),
        format_value(calculate_average([bm25_data[d].get('Domain') for d in datasets], expected_count)),
        format_value(calculate_average([repllama_data[d].get('None') for d in datasets], expected_count)),
        format_value(calculate_average([repllama_data[d].get('Domain') for d in datasets], expected_count)),
        format_value(calculate_average([modelname_data[d].get('None') for d in datasets], expected_count)),
        format_value(calculate_average([modelname_data[d].get('Domain') for d in datasets], expected_count))
    ]

    latex_table = f"""
\\begin{{table*}}[t]
\\centering
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{l|cc|cc|cc}}
\\toprule
\\multirow{{2}}{{*}}{{Dataset}} & \\multicolumn{{2}}{{c|}}{{BM25}} & \\multicolumn{{2}}{{c|}}{{RepLLaMA}} & \\multicolumn{{2}}{{c}}{{\\modelname}} \\\\
\\cmidrule(l){{2-3}} \\cmidrule(l){{4-5}} \\cmidrule(l){{6-7}}
 & None & Prompted & None & Prompted & None & Prompted \\\\
\\midrule
{chr(10).join(latex_rows)}
\\midrule
{" & ".join(averages)} \\\\
\\bottomrule
\\end{{tabular}}
}}
\\caption{{Effectiveness of BM25, RepLLaMA, and \\modelname on BEIR datasets with domain-specific prompts. Results are shown for standard retrieval (None) and with domain-specific prompts (Prompted). Missing values are indicated by "-".}}
\\label{{tab:beir-results-domain}}
\\end{{table*}}
"""
    return latex_table


def create_violin_plots(bm25_data, repllama_data, modelname_data, prompt_type):
    datasets = list(PRETTY_NAMES.keys()) + ['Average']

    for dataset in datasets:
        delta_data = []

        for model_name, data in [('BM25', bm25_data), ('RepLLaMA', repllama_data), ('ModelName', modelname_data)]:
            if dataset == 'Average':
                expected_count = len(PRETTY_NAMES)
                base_score = calculate_average([data[d].get('None') for d in PRETTY_NAMES.keys()], expected_count)
                if prompt_type == 'Generic':
                    prompted_scores = [calculate_average(list(data[d]['Generic'].values()), len(data[d]['Generic'])) for d in PRETTY_NAMES.keys() if data[d]['Generic']]
                else:  # Domain
                    prompted_scores = [data[d].get('Domain') for d in PRETTY_NAMES.keys() if 'Domain' in data[d]]
            else:
                base_score = data[dataset].get('None')
                if prompt_type == 'Generic':
                    prompted_scores = list(data[dataset]['Generic'].values())
                else:  # Domain
                    prompted_scores = [data[dataset].get('Domain')] if 'Domain' in data[dataset] else []

            if base_score is not None and prompted_scores:
                deltas = [score - base_score for score in prompted_scores if score is not None]
                delta_data.extend([(model_name, delta) for delta in deltas])

        if not delta_data:
            print(f"Warning: No valid data for {dataset} in {prompt_type} violin plot")
            continue

        df = pd.DataFrame(delta_data, columns=['Model', 'Delta'])

        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Model', y='Delta', data=df)
        plt.title(f'{PRETTY_NAMES.get(dataset, dataset)}: Delta between Regular and {prompt_type} Prompted Scores')
        plt.ylabel('Delta Score')
        plt.tight_layout()
        plt.axhline(0, color='red', linewidth=0.5)
        plt.savefig(f'plots/{prompt_type.lower()}_{dataset.lower()}_delta_violin.png', dpi=300)
        plt.close()

def check_missing_prompts(data, generic_hashes, domain_hashes, model_name):
    all_generic_hashes = set(generic_hashes)
    all_domain_hashes = set(domain_hashes.values())
    found_generic_hashes = set()
    found_domain_hashes = set()

    missing_generic_details = defaultdict(list)
    missing_domain_details = []

    for dataset, prompt_data in data.items():
        if 'Generic' in prompt_data:
            dataset_generic_hashes = set(prompt_data['Generic'].keys())
            found_generic_hashes.update(dataset_generic_hashes)
            missing_hashes = all_generic_hashes - dataset_generic_hashes
            for hash in missing_hashes:
                missing_generic_details[hash].append(dataset)
        else:
            for hash in all_generic_hashes:
                missing_generic_details[hash].append(dataset)

        if 'Domain' in prompt_data:
            found_domain_hashes.add(domain_hashes[dataset])
        else:
            missing_domain_details.append(dataset)

    missing_domain = all_domain_hashes - found_domain_hashes

    # Prepare data for CSV
    csv_data = []

    # Generic prompts
    for hash, datasets in missing_generic_details.items():
        for dataset in datasets:
            csv_data.append(["Generic", hash, dataset])

    # Domain prompts
    for hash in missing_domain:
        datasets = [dataset for dataset, domain_hash in domain_hashes.items() if domain_hash == hash]
        for dataset in datasets:
            csv_data.append(["Domain", hash, dataset])

    # Missing domain prompts
    for dataset in missing_domain_details:
        csv_data.append(["Domain", "N/A", dataset])

    # Save to CSV
    csv_filename = f"{model_name}_missing_prompts.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Prompt Type", "Hash", "Dataset"])
        csv_writer.writerows(csv_data)

    print(f"Missing prompts information for {model_name} has been saved to {csv_filename}")

    # Print warnings (optional, you can remove if you prefer just the CSV output)
    if missing_generic_details:
        print(f"Warning: Generic prompts are missing from the {model_name} data. Check {csv_filename} for details.")
    if missing_domain or missing_domain_details:
        print(f"Warning: Domain prompts are missing from the {model_name} data. Check {csv_filename} for details.")


def create_average_scores_csv(data, model_name):
    prompt_scores = defaultdict(list)
    
    # Collect scores for each prompt across all datasets
    for dataset, prompt_data in data.items():
        if 'None' in prompt_data:
            prompt_scores['None'].append(prompt_data['None'])
        if 'Domain' in prompt_data:
            prompt_scores['Domain'].append(prompt_data['Domain'])
        for prompt, score in prompt_data.get('Generic', {}).items():
            prompt_scores[prompt].append(score)
    
    # Calculate averages
    average_scores = {prompt: sum(scores) / len(scores) for prompt, scores in prompt_scores.items() if scores}
    
    # Write to CSV
    with open(f"{model_name}_average_scores.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Prompt", "Average Score"])
        for prompt, avg_score in average_scores.items():
            writer.writerow(["all", prompt, f"{avg_score:.2f}"])


# Usage
generic_hashes = read_generic_prompts('generic_prompts.csv')
domain_hashes = read_domain_prompts('domain_prompts.csv')

bm25_data = read_csv('bm25/bm25_aggregate_results.csv', generic_hashes, domain_hashes)
repllama_data = read_csv('reproduced-v2/aggregate_results.csv', generic_hashes, domain_hashes)
modelname_data = read_csv('joint-full/aggregate_results.csv', generic_hashes, domain_hashes)

# Generate and save LaTeX table
latex_table = generate_latex_table(bm25_data, repllama_data, modelname_data)
latex_table_domain = generate_latex_table_domain(bm25_data, repllama_data, modelname_data)

with open('latex_table_results.tex', 'w') as f:
    f.write(latex_table)

with open('latex_table_domain.tex', 'w') as f:
    f.write(latex_table_domain)

# Save CSV files
save_model_csv(bm25_data, 'bm25')
save_model_csv(repllama_data, 'repllama')
save_model_csv(modelname_data, 'modelname')

# Create new average scores CSV files
create_average_scores_csv(bm25_data, 'bm25')
create_average_scores_csv(repllama_data, 'repllama')
create_average_scores_csv(modelname_data, 'modelname')

# Create updated violin plots
create_violin_plots(bm25_data, repllama_data, modelname_data, 'Generic')
create_violin_plots(bm25_data, repllama_data, modelname_data, 'Domain')