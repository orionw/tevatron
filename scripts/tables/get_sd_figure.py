import pandas as pd
import numpy as np
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SKIP_OLD_HASHES = [
    "0ab0de14665a035b4ce74ea58f0aeb0b",
    "11c51cdccc21293fad66b37e75bbdc94",
    "476c48e5591c52d8000c65bc88421652"
]

MODEL_MAPPING = {
    'BM25': 'BM25',
    'Reproduced-v2': 'RepLLaMA',
    'Joint-Full': 'Promptriever'
}

def process_file(file_path):
    df = pd.read_csv(file_path)
    df = df[~df['dataset'].str.contains('msmarco|dev', case=False, na=False)]
    df = df[~df['prompt_hash'].isin(SKIP_OLD_HASHES)]
    
    # Calculate standard deviation for each dataset
    std_devs = df.groupby('dataset')['ndcg@10'].std().reset_index()
    std_devs.columns = ['dataset', 'ndcg@10_std']
    
    return std_devs

def create_violin_plot(dataframes, model_names):
    plt.figure(figsize=(8, 8))
    sns.set_style("white")
    colors = sns.color_palette("Blues")
    colors = [colors[1], colors[3], colors[5]]

    
    all_data = pd.concat([df.assign(Model=name) for df, name in zip(dataframes, model_names)])
    all_data['Model'] = all_data['Model'].map(MODEL_MAPPING)
    
    # Set the order of models
    model_order = ['BM25', 'RepLLaMA', 'Promptriever']
    # reverse it
    # model_order = model_order[::-1]
    
    sns.boxplot(x='Model', y='ndcg@10_std', data=all_data, order=model_order, palette=colors)
    
    # plt.title("SD over 10 Prompts", fontsize=24, pad=20)
    plt.xlabel("Model", fontsize=24, labelpad=10)
    plt.ylabel("Std Dev of Prompts Per Dataset", fontsize=24, labelpad=10)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Create custom legend
    legend_elements = [Patch(facecolor=color, edgecolor='black', label=model)
                       for model, color in zip(model_order, colors)]
    
    # Add legend to the plot
    plt.legend(handles=legend_elements, title='Model', title_fontsize=24, 
               fontsize=20, loc='upper right', bbox_to_anchor=(1, 1))
    
    # Adjust the plot to prevent cutting off labels
    plt.tight_layout()
    
    plt.savefig('ndcg_std_dev_violin_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig('ndcg_std_dev_violin_plot.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main(file_paths):
    dataframes = []
    model_names = []
    
    for file_path in file_paths:
        if "bm25" in file_path.lower():
            model_name = 'BM25'
        elif "reproduced-v2" in file_path.lower():
            model_name = 'Reproduced-v2'
        elif "joint-full" in file_path.lower():
            model_name = 'Joint-Full'
        else:
            continue
        
        df = process_file(file_path)
        dataframes.append(df)
        model_names.append(model_name)
    
    create_violin_plot(dataframes, model_names)
    print("Violin plot has been saved as 'ndcg_std_dev_violin_plot.png'")

# Usage
file_paths = list(glob.glob("results/*.csv"))
main(file_paths)