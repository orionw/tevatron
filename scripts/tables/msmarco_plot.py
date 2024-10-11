import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data
data = {
    'Model': ['RepLLaMA', 'RepLLaMA', 'RepLLaMA', 'Promptriever', 'Promptriever', 'Promptriever'],
    'Metric': ['DL19 nDCG@10', 'DL20 nDCG@10', 'Dev MRR', 'DL19 nDCG@10', 'DL20 nDCG@10', 'Dev MRR'],
    'Value': [74.5, 71.8, 42.5, 73.2, 72.3, 42.0]
}

# Create DataFrame
df = pd.DataFrame(data)

# Set up the plot
plt.figure(figsize=(10, 12))
sns.set_style("whitegrid")

colors = sns.color_palette("Blues")
colors = [colors[3], colors[5]]

# Create the grouped bar plot
ax = sns.barplot(x='Metric', y='Value', hue='Model', data=df, 
                 palette=colors)  # Light blue for RepLLaMA, darker blue for Promptriever

# y lim 30 to 80
plt.ylim(30, 80)

# Customize the plot
plt.title('MS MARCO (in-domain) Performance', fontsize=20)
plt.xlabel('Metric', fontsize=16)
plt.ylabel('Value', fontsize=16)
plt.legend(title='Model', title_fontsize='16', fontsize='14', loc='upper right')

# Increase tick label size
plt.tick_params(axis='both', which='major', labelsize=14)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add value labels on the bars
for i in ax.containers:
    ax.bar_label(i, fmt='%.1f', padding=3, fontsize=12)

# Adjust layout to prevent cutoff
plt.tight_layout()

# # Show the plot
# plt.show()

# Show the plot
plt.savefig('msmarco_plot.png')
plt.savefig('msmarco_plot.pdf')
print(f"Plot saved to msmarco_plot.png and msmarco_plot.pdf")