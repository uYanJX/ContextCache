import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Adjusting the color palette to a softer tone (e.g., pastel blue)
custom_cmap_soft = sns.light_palette("skyblue", as_cmap=True)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # Adjust width for layout

# MHA-based metrics
mha_confusion_matrix = np.array([[24379, 142], [4823, 19875]])
# Baseline metrics
baseline_confusion_matrix = np.array([[24124, 397], [24257, 441]])

# Titles for the plots
titles = ['MHA-based Method', 'Baseline Method']

# Data for both matrices
matrices = [mha_confusion_matrix, baseline_confusion_matrix]

# Loop to plot each confusion matrix with adjusted style
for ax, matrix, title in zip(axes, matrices, titles):
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap=custom_cmap_soft,  # Use softer color scheme
        ax=ax,
        cbar=False,
        xticklabels=["1", "0"],  # Use 0 and 1 for labels
        yticklabels=["1", "0"],
        annot_kws={"size": 14, "weight": "bold"}  # Adjust font size and bold
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Real Label")

# Tight layout for better spacing
plt.tight_layout()
plt.savefig("/data/home/Jianxin/MyProject/ContextCache/graph/pic/confusion_matrix.png",dpi=300)
