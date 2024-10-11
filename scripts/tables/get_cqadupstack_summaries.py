import csv
from collections import defaultdict
import statistics
import os

CQADUPSTACK_DATASETS = [
    "cqadupstack-android",
    "cqadupstack-english",
    "cqadupstack-gaming",
    "cqadupstack-gis",
    "cqadupstack-wordpress",
    "cqadupstack-physics",
    "cqadupstack-programmers",
    "cqadupstack-stats",
    "cqadupstack-tex",
    "cqadupstack-unix",
    "cqadupstack-webmasters",
    "cqadupstack-wordpress"
]

SKIP_OLD_HASHES = [
    "0ab0de14665a035b4ce74ea58f0aeb0b", 
    "11c51cdccc21293fad66b37e75bbdc94",
    "476c48e5591c52d8000c65bc88421652"
]

def read_csv(filename):
    data = defaultdict(lambda: defaultdict(dict))
    if not os.path.exists(filename):
        print(f"Warning: {filename} does not exist. Skipping this file.")
        return data
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row['dataset'].lower()
            if dataset not in CQADUPSTACK_DATASETS:
                continue
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
    return statistics.mean(real_vals) if real_vals else None

def get_best_prompt_score(dataset_data):
    if isinstance(dataset_data, dict) and 'Prompted' in dataset_data:
        prompted_scores = dataset_data['Prompted']
        return max(prompted_scores.values()) if prompted_scores else None
    return None

def calculate_cqadupstack_averages(data):
    baseline_scores = []
    best_prompt_scores = []
    incomplete_datasets = []
    for dataset in CQADUPSTACK_DATASETS:
        dataset_data = data.get(dataset, {})
        baseline_score = dataset_data.get('None')
        best_prompt_score = get_best_prompt_score(dataset_data)
        
        if baseline_score is None or best_prompt_score is None:
            incomplete_datasets.append({
                'dataset': dataset,
                'baseline': baseline_score,
                'best_prompt': best_prompt_score
            })
        else:
            baseline_scores.append(baseline_score)
            best_prompt_scores.append(best_prompt_score)

    avg_baseline = calculate_average(baseline_scores)
    avg_best_prompt = calculate_average(best_prompt_scores)
    
    return avg_baseline, avg_best_prompt, incomplete_datasets

def format_score(score):
    return f"{score:.3f}" if score is not None else "N/A"

def process_model(model_name, data):
    avg_baseline, avg_best_prompt, incomplete_datasets = calculate_cqadupstack_averages(data)
    
    if incomplete_datasets:
        print(f"Warning: Incomplete data for {model_name}. The following datasets have missing scores:")
        for item in incomplete_datasets:
            print(f"  {item['dataset']}:")
            print(f"    Baseline: {format_score(item['baseline'])}")
            print(f"    Best Prompt: {format_score(item['best_prompt'])}")
        print()
        return None
    
    return {
        'avg_baseline': avg_baseline,
        'avg_best_prompt': avg_best_prompt
    }

# Main execution
if __name__ == "__main__":
    models = [
        ("RepLLaMA", 'results/reproduced-v2_results.csv'),
        ("Llama2", 'results/joint-full_results.csv'),
        ("LLaMA3.1-Instruct", 'results/llama3.1-instruct_results.csv'),
        ("LLaMA3.1", 'results/llama3.1_results.csv'),
        ("Mistral-v1", 'results/mistral-v0.1_results.csv'),
        ("Mistral-v3", 'results/mistral-v0.3_results.csv')
    ]

    results = {}
    for model_name, file_path in models:
        print(f"Processing {model_name}...")
        data = read_csv(file_path)
        model_results = process_model(model_name, data)
        if model_results:
            results[model_name] = model_results
            print(f"{model_name}:")
            print(f"  Average CQADupStack (No Prompt): {format_score(model_results['avg_baseline'])}")
            print(f"  Average CQADupStack (Best Prompt): {format_score(model_results['avg_best_prompt'])}")
            print()

    # Write results to CSV
    with open('cqadupstack_scores_summary.csv', 'w', newline='') as csvfile:
        fieldnames = ['Model', 'No Prompt', 'Best Prompt']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for model, scores in results.items():
            writer.writerow({
                'Model': model,
                'No Prompt': format_score(scores['avg_baseline']),
                'Best Prompt': format_score(scores['avg_best_prompt'])
            })

    print("CSV file 'cqadupstack_scores_summary.csv' has been created with the compiled results.")