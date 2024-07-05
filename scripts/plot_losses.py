import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import ast
import glob

def plot_losses(log_file):
    data = []
    with open(log_file, 'r') as f:
        for line in f:
            if "{" not in line:
                continue
            try:
                stripped_line = line.strip()
                data.append(ast.literal_eval(stripped_line))
            except Exception:
                continue

    df = pd.DataFrame([item for item in data if "loss" in item])
    # keep only ones where loss is not NaN
    df = df[df["loss"].notna()]

    # plot loss agaisnt epoch
    sns.set(style="whitegrid")
    plt.figure(figsize=(16, 6))
    ax = sns.lineplot(x="epoch", y="loss", data=df)
    plt.title("Loss vs Epoch")
    plt.savefig(log_file.replace(".log", "_loss.png"))
    plt.close()
    print(f"Saved to {log_file.replace('.log', '_loss.png')}")

    # now do a zoomed in plot from .1 to 1 epochs
    df = df[df["epoch"].apply(lambda x: x > .1 and x < 1)]
    ax = sns.lineplot(x="epoch", y="loss", data=df)
    plt.title("Loss vs Epoch (zoomed in)")
    plt.savefig(log_file.replace(".log", "_loss_zoomed.png"))
    plt.close()
    print(f"Saved to {log_file.replace('.log', '_loss_zoomed.png')}")



if __name__ == "__main__":
    for log_file in glob.glob("*.log"):
        plot_losses(log_file)