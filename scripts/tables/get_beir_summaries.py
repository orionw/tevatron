import csv
from collections import defaultdict
import statistics
import os

SKIP_OLD_HASHES = [
    "0ab0de14665a035b4ce74ea58f0aeb0b", 
    "11c51cdccc21293fad66b37e75bbdc94",
    "476c48e5591c52d8000c65bc88421652"
]

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

def read_csv(filename):
    data = defaultdict(lambda: defaultdict(dict))
    if not os.path.exists(filename):
        print(f"Warning: {filename} does not exist. Skipping this file.")
        return data
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row['dataset'].lower()
            prompt_hash = row['prompt_hash']
            if prompt_hash in SKIP_OLD_HASHES:
                continue
            ndcg = float(row['ndcg@10']) if row['ndcg@10'] else None

            if prompt_hash == 'none':
                data[dataset]['None'] = ndcg
            else:
                data[dataset]['Prompted'][prompt_hash] = ndcg
    return data

def calculate_average(values):
    real_vals = [v for v in values if v is not None]
    try:
        return statistics.mean(real_vals) if real_vals else None
    except Exception as e:
        breakpoint()
        return 0

def get_best_prompt_score(dataset_data):
    if not dataset_data['Prompted']:
        return None
    ret_val = max(dataset_data['Prompted'].values())
    return ret_val

def calculate_beir_averages(data):
    datasets = list(PRETTY_NAMES.keys())
    print(f"Calculating BEIR averages for {len(datasets)} datasets: {', '.join(datasets)}")
    
    baseline_scores = [data[d]['None'] for d in datasets if data[d]['None'] is not None]
    best_prompt_scores = [get_best_prompt_score(data[d]) for d in datasets if get_best_prompt_score(data[d]) is not None]

    # assert they have all the right datasets
    for dataset in datasets:
        if dataset not in data or not len(data[dataset]):
            print(f"Warning: dataset {dataset} not found in data")

    # check for the None baseline
    for dataset in datasets:
        if data[dataset]['None'] in [None, {}]:
            print(f"Warning: dataset {dataset} is missing a None baseline score")
    
    assert len(best_prompt_scores) == len(datasets)
    assert len(baseline_scores) == len(datasets)
    avg_baseline = calculate_average(baseline_scores)
    avg_best_prompt = calculate_average(best_prompt_scores)
    
    return avg_baseline, avg_best_prompt, baseline_scores, best_prompt_scores

# Main execution
bm25_data = read_csv('results/bm25_results.csv')
repllama_data = read_csv('results/reproduced-v2_results.csv')
modelname_data = read_csv('results/joint-full_results.csv')
llama31_instruct = read_csv('results/llama3.1-instruct_results.csv')
llama31 = read_csv('results/llama3.1_results.csv')
mistralv1 = read_csv('results/mistral-v0.1_results.csv')
mistralv3 = read_csv('results/mistral-v0.3_results.csv')

ALL_MODELS = [
    ("BM25", bm25_data),
    ("RepLLaMA", repllama_data),
    ("Llama2", modelname_data),
    ("LLaMA3.1-Instruct", llama31_instruct),
    ("LLaMA3.1", llama31),
    ("Mistral-v1", mistralv1),
    ("Mistral-v3", mistralv3)
]

results = defaultdict(lambda: defaultdict(dict))

for model_name, data in ALL_MODELS:
    avg_baseline, avg_best_prompt, baseline_scores, best_prompt_scores = calculate_beir_averages(data)
    print(f"{model_name} - Average BEIR (Baseline): {avg_baseline:.3f}")
    print(f"{model_name} - Baseline Scores: {','.join([f'{score:.3f}' if score is not None else 'None' for score in baseline_scores])}")
    print(f"{model_name} - Average BEIR (Best Prompt): {avg_best_prompt:.3f}")
    print(f"{model_name} - Best Prompt Scores: {','.join([f'{score:.3f}' if score is not None else 'None' for score in best_prompt_scores])}")
    print()

    # Store results for CSV
    for i, dataset in enumerate(PRETTY_NAMES.keys()):
        results[dataset][model_name]['Baseline'] = f"{baseline_scores[i]:.1f}" if baseline_scores[i] is not None else "N/A"
        results[dataset][model_name]['Prompted'] = f"{best_prompt_scores[i]:.1f}" if best_prompt_scores[i] is not None else "N/A"

# Write results to CSV
with open('beir_scores_summary.csv', 'w', newline='') as csvfile:
    fieldnames = ['Dataset'] + [f"{model} {score_type}" for model, _ in ALL_MODELS for score_type in ['Baseline', 'Prompted']]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for dataset, scores in results.items():
        row = {'Dataset': PRETTY_NAMES[dataset]}
        for model, _ in ALL_MODELS:
            row[f"{model} Baseline"] = scores[model]['Baseline']
            row[f"{model} Prompted"] = scores[model]['Prompted']
        writer.writerow(row)

print("CSV file 'beir_scores_summary.csv' has been created with the compiled results.")