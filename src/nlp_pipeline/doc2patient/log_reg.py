"""

log_reg.py

This script builds a log reg model to try and map document to patient level IBD cohort interactions.

It uses both ridge and lasso penalisation and saves the resulting charts out at the end.

"""

import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------------------------------- #
# 1. SET PROJECT ROOT AND LOAD/MERGE                                          #
# --------------------------------------------------------------------------- #

# Go up exactly two levels to the project root (fallback to highest parent if shorter)
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

# Concatenate into a single DataFrame
df_merged = pd.concat([df_train, df_validation], ignore_index=True)

# --------------------------------------------------------------------------- #
# 2. PREPARE DATA (NO SCALING)                                                #
# --------------------------------------------------------------------------- #

feature_cols = [
    "Histopath_IBD",
    "Endoscopy_IBD",
    "Preceding_Clinic_IBD",
    "Following_Clinic_IBD",
]
target_col = "Patient_Has_IBD"

# Subset and drop any rows with NaNs in these columns
df_model = df_merged[feature_cols + [target_col]].dropna()

# X: (n_samples × 4) matrix of 0/1 flags; y: 0/1 IBD label
X = df_model[feature_cols].astype(int).values
y = df_model[target_col].astype(int).values

# --------------------------------------------------------------------------- #
# 3. FIT LOGISTIC REGRESSION (NO SCALING)                                     #
# --------------------------------------------------------------------------- #

# 3a. L1 (LASSO) logistic regression
logreg_l1 = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=0)
logreg_l1.fit(X, y)
coefs_l1 = logreg_l1.coef_.flatten()
intercept_l1 = logreg_l1.intercept_[0]
odds_l1 = np.exp(coefs_l1)

# 3b. L2 (RIDGE) logistic regression
logreg_l2 = LogisticRegression(penalty="l2", solver="liblinear", C=1.0, random_state=0)
logreg_l2.fit(X, y)
coefs_l2 = logreg_l2.coef_.flatten()
intercept_l2 = logreg_l2.intercept_[0]
odds_l2 = np.exp(coefs_l2)

# Build DataFrames for display
results_l1 = pd.DataFrame(
    {"Feature": feature_cols, "Coefficient": coefs_l1, "Odds_Ratio": odds_l1}
).set_index("Feature")

results_l2 = pd.DataFrame(
    {"Feature": feature_cols, "Coefficient": coefs_l2, "Odds_Ratio": odds_l2}
).set_index("Feature")

# Print L1 and L2 tables
print("\nLogistic Regression (L1) Coefficients and Odds Ratios (No Scaling):\n")
print(results_l1.round(4))

print("\nLogistic Regression (L2) Coefficients and Odds Ratios (No Scaling):\n")
print(results_l2.round(4))

# --------------------------------------------------------------------------- #
# 4. PLOT BAR CHARTS FOR NON-SCALED ODDS RATIOS                               #
# --------------------------------------------------------------------------- #

# Prepare output directory
fig_dir = default_project_root / "data" / "results" / "doc2patient" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

l1_path = fig_dir / "ibd_logistic_odds_ratios_l1_noscale.png"
l2_path = fig_dir / "ibd_logistic_odds_ratios_l2_noscale.png"

def plot_odds_ratios(odds_vals, title_str, save_path, bar_color):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    bars = ax.bar(
        feature_cols, odds_vals, color=bar_color, edgecolor="black", linewidth=1.5
    )
    # Annotate each bar (in black text)
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.02,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="black",
        )
    # Labeling
    ax.set_ylabel("Odds Ratio (exp(coef))", fontsize=12, fontweight="bold")
    ax.set_xticks(np.arange(len(feature_cols)))
    ax.set_xticklabels(
        [f.replace("_", " ") for f in feature_cols],
        rotation=45,
        ha="right",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(title_str, fontsize=14, fontweight="bold", pad=12)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.tight_layout()

    # Immediately save
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"✅ Saved bar chart to: {save_path}")
    print("🧮 File size:", save_path.stat().st_size, "bytes")

    plt.close(fig)

# Plot and save L1 odds ratios
plot_odds_ratios(
    odds_l1,
    "Figure 3a: Logistic Regression (L1) Odds Ratios (No Scaling)",
    l1_path,
    bar_color="skyblue",
)

# Plot and save L2 odds ratios
plot_odds_ratios(
    odds_l2,
    "Figure 3b: Logistic Regression (L2) Odds Ratios (No Scaling)",
    l2_path,
    bar_color="lightcoral",
)

# --------------------------------------------------------------------------- #
# 5. SAVE                                                                     #
# --------------------------------------------------------------------------- #

print("\n📂 Saving figures to:")
print("   L1 (noscale):", l1_path.resolve())
print("   L2 (noscale):", l2_path.resolve())

print(
    "📁 Directory exists:", fig_dir.exists(), "| Writable:", os.access(fig_dir, os.W_OK)
)

# Test writing a dummy text file
try:
    test_path = fig_dir / f"write_test_{datetime.now().strftime('%H%M%S')}.txt"
    with open(test_path, "w") as f:
        f.write("✅ Write test successful.")
    print("🧪 Dummy write test: ✅", test_path.resolve())
except Exception as e:
    print("🧪 Dummy write test: ❌", str(e))

print("\n📄 Directory listing:")
for item in fig_dir.iterdir():
    print(" -", item.name)

# --------------------------------------------------------------------------- #
# 6. DISPLAY SAVED CHARTS INLINE                                              #
# --------------------------------------------------------------------------- #

print("\nDisplaying saved L1 chart:")
if l1_path.exists():
    display(Image(filename=str(l1_path)))

print("\nDisplaying saved L2 chart:")
if l2_path.exists():
    display(Image(filename=str(l2_path)))

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X_vif = add_constant(df_model[feature_cols])
vif_data = pd.DataFrame(
    {
        "Feature": ["Intercept"] + feature_cols,
        "VIF": [
            variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])
        ],
    }
)
print(vif_data)