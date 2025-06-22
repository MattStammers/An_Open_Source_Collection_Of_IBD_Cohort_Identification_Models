"""

Decision_tree.py

This script tries to build a decision tree to map document to patient level relationships.

It then saves and displays this tree.

"""

import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree

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
# 3. FIT DECISION TREE CLASSIFIER                                             #
# --------------------------------------------------------------------------- #

# Initialise Decision Tree Classifier with a fixed random_state for reproducibility
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X, y)

# Predict on the same data to get training accuracy
y_pred = dtc.predict(X)
accuracy = accuracy_score(y, y_pred)

# Print out performance metrics
print(f"\nDecision Tree Training Accuracy: {accuracy:.4f}\n")
print("Classification Report (on training data):\n")
print(classification_report(y, y_pred, target_names=["No IBD", "IBD"], digits=4))

# Print feature importances
feat_importances = pd.DataFrame(
    {"Feature": feature_cols, "Importance": dtc.feature_importances_}
).set_index("Feature")
print("\nFeature Importances:\n")
print(feat_importances.sort_values("Importance", ascending=False).round(4))

# --------------------------------------------------------------------------- #
# 4. PLOT AND SAVE THE TREE                                                   #
# --------------------------------------------------------------------------- #

# Prepare output directory for saving the tree plot
fig_dir = default_project_root / "data" / "results" / "doc2patient" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

tree_output_path = fig_dir / "ibd_decision_tree.png"

# Create figure for the tree
plt.figure(figsize=(8, 6), dpi=300)
plot_tree(
    dtc,
    feature_names=[f.replace("_", " ") for f in feature_cols],
    class_names=["No IBD", "IBD"],
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title(
    "Decision Tree: Predicting IBD from Document Flags",
    fontsize=14,
    weight="bold",
    pad=12,
)
plt.tight_layout()

# Immediately save the figure
plt.savefig(tree_output_path, bbox_inches="tight", dpi=300)
plt.close()

# --------------------------------------------------------------------------- #
# 5. DEBUG SAVE OPERATION                                                     #
# --------------------------------------------------------------------------- #

print(f"\n📂 Saving decision tree figure to: {tree_output_path.resolve()}")
print(
    "📁 Directory exists:", fig_dir.exists(), "| Writable:", os.access(fig_dir, os.W_OK)
)

# Test writing a dummy text file to confirm write access
try:
    test_path = fig_dir / f"write_test_{datetime.now().strftime('%H%M%S')}.txt"
    with open(test_path, "w") as f:
        f.write("✅ Write test successful.")
    print("🧪 Dummy write test: ✅", test_path.resolve())
except Exception as e:
    print("🧪 Dummy write test: ❌", str(e))

# Check file size to ensure the PNG is nonzero
if tree_output_path.exists():
    print("🧮 Decision tree PNG size:", tree_output_path.stat().st_size, "bytes")

# List contents of the target directory (first 10 entries)
print("\n📄 Directory listing (first 10 files):")
for i, item in enumerate(fig_dir.iterdir()):
    if i >= 10:
        break
    print(" -", item.name)

# --------------------------------------------------------------------------- #
# 6. DISPLAY SAVED TREE INLINE                                                #
# --------------------------------------------------------------------------- #

print("\nDisplaying saved decision tree:")
if tree_output_path.exists():
    display(Image(filename=str(tree_output_path)))