"""

Heatmap.py

This script builds a heatmap of document to patient level IBD detection pearson correlations,

It colours and formats them in an attractive manner and then saves the output.

"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --------------------------------------------------------------------------- #
# 1. SET PROJECT ROOT AND LOAD/MERGE                                          #
# --------------------------------------------------------------------------- #

# Attempt to go up two levels; otherwise use highest available parent
cwd_parents = Path.cwd().parents
if len(cwd_parents) >= 3:
    default_project_root = cwd_parents[2]
else:
    default_project_root = cwd_parents[-1]

sys.path.append(str(default_project_root))

# Paths to the two Excel files (relative to project root)
validation_path = (
    default_project_root
    / "data"
    / "document_validation"
    / "merged_human_validation_redacted.xlsx"
)
train_path = (
    default_project_root
    / "data"
    / "document_validation"
    / "merged_human_train_redacted.xlsx"
)

# Load each sheet into a DataFrame
df_validation = pd.read_excel(validation_path)
df_train = pd.read_excel(train_path)

# Concatenate them (row-wise) into one “df_merged”
df_merged = pd.concat([df_train, df_validation], ignore_index=True)

# --------------------------------------------------------------------------- #
# 2. COMPUTE CORRELATION MATRIX                                               #
# --------------------------------------------------------------------------- #

# Keep only the columns of interest
corr_cols = [
    "Patient_Has_IBD",
    "Histopath_IBD",
    "Endoscopy_IBD",
    "Preceding_Clinic_IBD",
    "Following_Clinic_IBD",
]

# Drop rows with any NaNs in these columns
df_corr = df_merged[corr_cols].dropna()

# Compute Pearson correlation matrix
corr_matrix = df_corr.corr()

# --------------------------------------------------------------------------- #
# 3. PLOT A BLUE-TO-RED HEATMAP WITH FINAL REFINEMENTS                        #
# --------------------------------------------------------------------------- #

# Directory to save the final figure
fig_dir = default_project_root / "data" / "results" / "doc2patient" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
output_path = fig_dir / "ibd_correlation_heatmap.png"

# Seaborn context/style for publication-level output
sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid", {"axes.edgecolor": ".15"})

# Create the figure & axis
fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

# Use the "coolwarm" palette, mapping 0→blue and 1→red
cmap = plt.get_cmap("coolwarm")

# Mask the diagonal so that “1.00” cells are blank
mask = np.eye(len(corr_matrix), dtype=bool)

# Plot the heatmap with cell borders and a slightly larger colorbar
sns.heatmap(
    corr_matrix,
    cmap=cmap,
    vmin=0,
    vmax=1,
    mask=mask,
    annot=False,
    square=True,
    linewidths=1,
    linecolor="black",
    cbar_kws={
        "shrink": 0.65,
        "pad": 0.05,
        "label": "Pearson Correlation",
        "ticks": np.linspace(0, 1, 6),
    },
    ax=ax,
)

# Retrieve the colorbar and adjust its tick and label sizes
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=9)
cbar.ax.yaxis.label.set_size(11)
cbar.ax.yaxis.label.set_weight("bold")

# Annotate each off-diagonal cell with its correlation value in black text
n = len(corr_matrix)
for i in range(n):
    for j in range(n):
        if i == j:
            continue  # skip diagonal
        val = corr_matrix.iloc[i, j]
        ax.text(
            j + 0.5,
            i + 0.5,
            f"{val:.2f}",
            ha="center",
            va="center",
            color="black",
            fontsize=12,
            fontweight="bold",
        )

# Replace underscores with spaces for tick labels and set them bold & slightly smaller
labels_readable = [col.replace("_", " ") for col in corr_matrix.columns.tolist()]
ax.set_xticks(np.arange(n) + 0.5)
ax.set_yticks(np.arange(n) + 0.5)
ax.set_xticklabels(labels_readable, rotation=45, ha="right", fontsize=12, weight="bold")
ax.set_yticklabels(labels_readable, rotation=0, va="center", fontsize=12, weight="bold")

# Add the requested title
ax.set_title(
    "Figure 2: Document to Patient Level Correlations",
    fontsize=18,
    weight="bold",
    pad=20,
)

# Draw a bold outer border around the heatmap
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(2)
    spine.set_color("black")

# Tight layout so nothing overlaps
plt.tight_layout()

# Save to file at high resolution
plt.savefig(output_path, bbox_inches="tight", dpi=300)
plt.close(fig)

print(f"✅ Final heatmap saved to: {output_path}")
