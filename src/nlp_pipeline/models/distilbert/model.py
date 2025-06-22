"""
Module: distilbert model.py

Description:
    This module contains functions and classes for training, evaluation, and explainability
    of a DistilBERT model for sequence classification. It includes utilities for reproducibility,
    metric computation, custom dataset handling, as well as multiple explainability methods:
    SHAP, Integrated Gradients, LIME, and attention visualisation.

"""

import logging
import os
import random
import string
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
from captum.attr import IntegratedGradients
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AdamW,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    get_scheduler,
)

# To prevent internet contact
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# --------------------------------------------------------------------------- #
# Set Random Seed                                                             #
# --------------------------------------------------------------------------- #

def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across built-in libraries, NumPy, and Torch.

    Args:
        seed (int): The seed value to set. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_random_seed(42)

# --------------------------------------------------------------------------- #
# Metric Computation                                                          #
# --------------------------------------------------------------------------- #

def compute_bootstrap_metrics(
    true_labels, predictions, n_bootstraps: int = 1000, alpha: float = 0.95
):
    """
    Compute evaluation metrics for classification predictions, returning each as a tuple of
    (mean, lower, upper) bounds. For demonstration purposes, specificity and NPV are set to 0.

    Args:
        true_labels (list or array): Ground truth labels.
        predictions (list or array): Predicted labels.
        n_bootstraps (int): Number of bootstrap iterations. Defaults to 1000.
        alpha (float): Confidence level. Defaults to 0.95.

    Returns:
        dict: Dictionary containing tuples for metrics 'accuracy', 'precision', 'recall',
              'specificity', 'npv', and 'f1'.
    """
    acc = accuracy_score(true_labels, predictions)
    prec = precision_score(true_labels, predictions, zero_division=0)
    rec = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    return {
        "accuracy": (acc, acc, acc),
        "precision": (prec, prec, prec),
        "recall": (rec, rec, rec),
        "specificity": (0.0, 0.0, 0.0),
        "npv": (0.0, 0.0, 0.0),
        "f1": (f1, f1, f1),
    }


def fmt_confint(mean: float, lower: float, upper: float) -> str:
    """
    Format a confidence interval string as:
    "XX.XX% (CI: YY.YY% - ZZ.ZZ%)".

    Args:
        mean (float): Mean metric value.
        lower (float): Lower bound of the confidence interval.
        upper (float): Upper bound of the confidence interval.

    Returns:
        str: Formatted confidence interval string.
    """
    return f"{mean*100:.2f}% (CI: {lower*100:.2f}% - {upper*100:.2f}%)"

# --------------------------------------------------------------------------- #
# Custom Dataset and Utilities                                                #
# --------------------------------------------------------------------------- #

class TextDataset(Dataset):
    """
    Custom Dataset for DistilBERT training and evaluation.

    This dataset tokenises input text, applies truncation and padding to a fixed maximum length,
    and stores the corresponding label for each sample.

    Attributes:
        texts (list): List of input text strings.
        labels (list): List of label values corresponding to each text.
        tokeniser (DistilBertTokenizer): Tokeniser instance for encoding texts.
        max_length (int): Maximum token sequence length. Defaults to 512.
    """

    def __init__(
        self, texts, labels, tokenizer: DistilBertTokenizer, max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int):
        """
        Retrieve and tokenise the sample at the specified index.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Dictionary containing tokenised inputs and the corresponding label tensor.
        """
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def combine_subwords(tokens, attributions):
    """
    Combine subword tokens (those starting with '##') with their preceding tokens,
    and sum their corresponding attribution values.

    Args:
        tokens (list): List of token strings.
        attributions (list or array): Attribution values for each token.

    Returns:
        tuple: (combined_tokens, combined_attributions) where tokens are merged and attributions summed.
    """
    combined_tokens = []
    combined_attributions = []
    current_token = ""
    current_attr = 0.0

    for token, attr in zip(tokens, attributions):
        if token.startswith("##"):
            # Merge subword fragment with the ongoing token.
            current_token += token[2:]
            current_attr += attr
        else:
            # If there's an ongoing token, add it before starting a new one.
            if current_token:
                combined_tokens.append(current_token)
                combined_attributions.append(current_attr)
            current_token = token
            current_attr = attr

    # Add the final accumulated token.
    if current_token:
        combined_tokens.append(current_token)
        combined_attributions.append(current_attr)

    return combined_tokens, combined_attributions

def tokens_with_shap_as_string(tokenizer, input_ids_row, shap_row, top_k=10):
    """
    Convert a row of token IDs and SHAP values into a formatted string listing the top-k tokens
    by absolute SHAP value. This includes:
        - Converting token IDs back to subword tokens.
        - Removing special tokens ([PAD], [CLS], [SEP]).
        - Merging subword fragments.
        - Optionally filtering out very short tokens.
        - Sorting tokens by absolute SHAP value.

    Args:
        tokeniser (DistilBertTokenizer): Tokeniser for decoding token IDs.
        input_ids_row (list or array): Row of token IDs.
        shap_row (list or array): Corresponding SHAP attribution values.
        top_k (int): Number of top tokens to include. Defaults to 10.

    Returns:
        str: A formatted string with tokens and their SHAP values (e.g., "word(+0.12) | word(-0.05) | ...").
    """
    # Convert token IDs back to subword tokens.
    subword_tokens = tokenizer.convert_ids_to_tokens(input_ids_row)

    # Truncate both token list and SHAP values to the same length.
    n_tokens = min(len(subword_tokens), len(shap_row))
    subword_tokens = subword_tokens[:n_tokens]
    shap_row = shap_row[:n_tokens]

    # Remove special tokens.
    special_tokens = {"[PAD]", "[CLS]", "[SEP]"}
    valid_indices = [i for i, t in enumerate(subword_tokens) if t not in special_tokens]
    filtered_tokens = [subword_tokens[i] for i in valid_indices]
    filtered_values = shap_row[valid_indices]

    # Merge subword fragments.
    merged_tokens, merged_attrs = combine_subwords(filtered_tokens, filtered_values)

    # Optionally filter out very short tokens.
    def is_micro_token(t):
        return len(t) <= 3

    final_pairs = [
        (tok, attr)
        for tok, attr in zip(merged_tokens, merged_attrs)
        if not is_micro_token(tok)
    ]

    # Sort tokens by absolute SHAP value and format the result.
    pairs_sorted = sorted(final_pairs, key=lambda x: abs(x[1]), reverse=True)[:top_k]
    return " | ".join([f"{tok}({val:+.2f})" for tok, val in pairs_sorted])

# --------------------------------------------------------------------------- #
# Shap Explainability                                                         #
# --------------------------------------------------------------------------- #

def run_shap_analysis(
    model: torch.nn.Module,
    tokenizer: DistilBertTokenizer,
    train_texts: list,
    val_texts: list,
    output_dir: str,
    report_col: str,
    device,
    background_size: int = 10,
    shap_nsamples: int = 10,
    shap_chunk_size: int = 5,
) -> None:
    """
    Perform SHAP analysis for model explainability using Kernel SHAP.

    The process includes:
        1. Defining a prediction function compatible with SHAP.
        2. Building a background dataset from training texts.
        3. Preparing a subset of validation texts.
        4. Computing SHAP values in chunks.
        5. Aligning and filtering SHAP values.
        6. Generating summary and bar plots.
        7. Saving raw SHAP values and top token outputs for each sample.

    Args:
        model (torch.nn.Module): Trained DistilBERT model.
        tokeniser (DistilBertTokenizer): Tokeniser for encoding texts.
        train_texts (list): List of training text samples.
        val_texts (list): List of validation text samples.
        output_dir (str): Directory for saving outputs.
        report_col (str): Report column identifier used in filenames.
        device: Torch device for computations.
        background_size (int): Number of training samples to use as background. Defaults to 10.
        shap_nsamples (int): Number of samples for SHAP approximation. Defaults to 10.
        shap_chunk_size (int): Chunk size for processing validation samples. Defaults to 5.
    """
    try:
        logging.info(
            f"[SHAP] Starting SHAP for '{report_col}' with "
            f"background_size={background_size}, nsamples={shap_nsamples}."
        )
        torch.cuda.empty_cache()

        shap_folder = os.path.join(output_dir, "shap")
        os.makedirs(shap_folder, exist_ok=True)

        # 1) Define model_predict for shap.KernelExplainer.
        def model_predict(input_ids_np):
            input_ids_t = torch.tensor(input_ids_np, dtype=torch.long).to(device)
            attention_mask_t = (input_ids_t != tokenizer.pad_token_id).long().to(device)
            with torch.no_grad():
                if hasattr(model, "module"):
                    outputs = model.module(
                        input_ids=input_ids_t, attention_mask=attention_mask_t
                    )
                else:
                    outputs = model(
                        input_ids=input_ids_t, attention_mask=attention_mask_t
                    )
                probs = (
                    torch.nn.functional.softmax(outputs.logits, dim=-1)[:, 1]
                    .cpu()
                    .numpy()
                )
            return probs

        # 2) Build background from training texts.
        background_texts = train_texts[:background_size]
        background_enc = tokenizer(
            background_texts,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        background_input_ids = background_enc["input_ids"]
        logging.info(f"[SHAP] Background input shape: {background_input_ids.shape}")

        explainer = shap.KernelExplainer(
            model_predict, background_input_ids, link="logit"
        )

        # 3) Prepare validation samples.
        sample_val_texts = val_texts[:20]
        sample_val_enc = tokenizer(
            sample_val_texts,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        sample_input_ids = sample_val_enc["input_ids"]
        if sample_input_ids.size == 0:
            logging.warning("[SHAP] No validation samples for SHAP analysis.")
            return

        logging.info(f"[SHAP] Initial validation input shape: {sample_input_ids.shape}")

        # Align sequence lengths with the background.
        expected_seq_len = background_input_ids.shape[1]
        if sample_input_ids.shape[1] < expected_seq_len:
            pad_width = expected_seq_len - sample_input_ids.shape[1]
            sample_input_ids = np.pad(
                sample_input_ids,
                pad_width=((0, 0), (0, pad_width)),
                constant_values=tokenizer.pad_token_id,
            )
        elif sample_input_ids.shape[1] > expected_seq_len:
            sample_input_ids = sample_input_ids[:, :expected_seq_len]
        logging.info(f"[SHAP] Aligned validation input shape: {sample_input_ids.shape}")

        # 4) Compute SHAP values in chunks.
        num_val = sample_input_ids.shape[0]
        all_shap_arrays = []
        for start_idx in range(0, num_val, shap_chunk_size):
            end_idx = min(start_idx + shap_chunk_size, num_val)
            chunk_input = sample_input_ids[start_idx:end_idx]
            if chunk_input.shape[0] == 0:
                break

            chunk_values = explainer.shap_values(chunk_input, nsamples=shap_nsamples)
            if isinstance(chunk_values, list):
                chunk_values = chunk_values[1]
            all_shap_arrays.append(chunk_values)

        if not all_shap_arrays:
            logging.warning("[SHAP] No SHAP values computed.")
            return

        shap_values = np.concatenate(all_shap_arrays, axis=0)
        logging.info(f"[SHAP] SHAP array shape: {shap_values.shape}")

        # 5) Align rows of shap_values and sample_input_ids.
        n_shap = shap_values.shape[0]
        n_inp = sample_input_ids.shape[0]
        if n_shap < n_inp:
            sample_input_ids = sample_input_ids[:n_shap]
        elif n_shap > n_inp:
            shap_values = shap_values[:n_inp]

        # 6) Align token dimension.
        if shap_values.shape[1] != sample_input_ids.shape[1]:
            common_dim = min(shap_values.shape[1], sample_input_ids.shape[1])
            shap_values = shap_values[:, :common_dim]
            sample_input_ids = sample_input_ids[:, :common_dim]

        logging.info(
            f"[SHAP] Final aligned shapes: sample_input_ids: {sample_input_ids.shape}, shap_values: {shap_values.shape}"
        )

        # 7) Prepare tokens for plotting.
        X_for_plot = sample_input_ids.copy()
        raw_tokens = tokenizer.convert_ids_to_tokens(sample_input_ids[0])
        if len(raw_tokens) > shap_values.shape[1]:
            raw_tokens = raw_tokens[: shap_values.shape[1]]

        # Aggressive token filtering.
        def get_keep_mask(tokens):
            special_tokens = {"[PAD]", "[CLS]", "[SEP]"}
            stopwords = {"a", "an", "the", "of", "in", "and", "el"}
            mask = []
            for tok in tokens:
                if tok in special_tokens:
                    mask.append(False)
                    continue
                clean_tok = tok[2:] if tok.startswith("##") else tok
                clean_tok = clean_tok.strip()
                if not clean_tok:
                    mask.append(False)
                    continue
                if any(ch in string.punctuation for ch in clean_tok):
                    mask.append(False)
                    continue
                if len(clean_tok) < 3:
                    mask.append(False)
                    continue
                if clean_tok.lower() in stopwords:
                    mask.append(False)
                    continue
                mask.append(True)
            return mask

        mask = get_keep_mask(raw_tokens)
        keep_idx = [i for i, m in enumerate(mask) if m]
        keep_idx = [i for i in keep_idx if i < shap_values.shape[1]]
        shap_values = shap_values[:, keep_idx]
        X_for_plot = X_for_plot[:, keep_idx]

        feature_names = []
        for i in keep_idx:
            tok = raw_tokens[i]
            if tok.startswith("##"):
                tok = tok[2:]
            feature_names.append(tok)

        # 8) Generate SHAP summary plots.
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, X_for_plot, feature_names=feature_names, show=False
        )
        plt.title(f"SHAP Summary for {report_col}")
        summary_path = os.path.join(shap_folder, f"{report_col}_shap_summary.png")
        plt.savefig(summary_path, bbox_inches="tight")
        plt.close()
        logging.info(f"[SHAP] Saved summary plot => {summary_path}")

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X_for_plot,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
        )
        plt.title(f"SHAP Feature Importance for {report_col}")
        bar_path = os.path.join(
            shap_folder, f"{report_col}_shap_feature_importance.png"
        )
        plt.savefig(bar_path, bbox_inches="tight")
        plt.close()
        logging.info(f"[SHAP] Saved bar plot => {bar_path}")

        # 9) Save raw SHAP values.
        shap_values_path = os.path.join(shap_folder, f"{report_col}_shap_values.pkl")
        joblib.dump(shap_values, shap_values_path)
        logging.info(f"[SHAP] Dumped shap_values => {shap_values_path}")

        # 10) Generate top tokens per sample.
        shap_out_folder = os.path.join(shap_folder, "shap_token_outputs")
        os.makedirs(shap_out_folder, exist_ok=True)
        n_samples = min(shap_values.shape[0], X_for_plot.shape[0])
        for i in range(n_samples):
            top_str = tokens_with_shap_as_string(
                tokenizer, X_for_plot[i], shap_values[i], top_k=10
            )
            txt_path = os.path.join(shap_out_folder, f"sample_{i}_top_tokens.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"=== Sample {i} for {report_col} (Top SHAP tokens) ===\n")
                f.write(top_str + "\n")
    except Exception as e:
        logging.error(f"[SHAP] Error during SHAP analysis for '{report_col}': {e}")

# --------------------------------------------------------------------------- #
# Integrated Gradients                                                        #
# --------------------------------------------------------------------------- #

def run_integrated_gradients(
    model: torch.nn.Module,
    tokenizer: DistilBertTokenizer,
    val_texts: list,
    output_dir: str,
    report_col: str,
    device,
    ig_chunk_size: int = 2,
    n_steps_ig: int = 10,
) -> None:
    """
    Perform Integrated Gradients analysis on a subset of validation texts and generate bar plots.

    This function computes attributions using Captum's IntegratedGradients by:
        - Preparing inputs and obtaining embeddings.
        - Building baseline embeddings.
        - Computing and aggregating attributions.
        - Plotting and saving attribution bar plots for each sample.

    Args:
        model (torch.nn.Module): Trained DistilBERT model.
        tokeniser (DistilBertTokenizer): Tokeniser for encoding texts.
        val_texts (list): List of validation text samples.
        output_dir (str): Directory for saving output plots.
        report_col (str): Report column identifier used in filenames.
        device: Torch device for computations.
        ig_chunk_size (int): Number of samples to process per chunk. Defaults to 2.
        n_steps_ig (int): Number of steps for Integrated Gradients approximation. Defaults to 10.
    """
    try:
        logging.info(f"Starting Integrated Gradients analysis for '{report_col}'.")
        ig_folder = os.path.join(output_dir, "integrated_gradients")
        os.makedirs(ig_folder, exist_ok=True)

        # If model is wrapped in DataParallel, extract the underlying model.
        if hasattr(model, "module"):
            model_for_ig = model.module
        else:
            model_for_ig = model

        def forward_func_embeds(embeds, attention_mask):
            distilbert = model_for_ig.distilbert
            pre_classifier = model_for_ig.pre_classifier
            dropout = model_for_ig.dropout
            classifier = model_for_ig.classifier

            # Perform the forward pass.
            outputs = distilbert(inputs_embeds=embeds, attention_mask=attention_mask)
            hidden_state = outputs.last_hidden_state
            pooled_output = hidden_state[:, 0]
            pooled_output = pre_classifier(pooled_output)
            pooled_output = torch.nn.ReLU()(pooled_output)
            pooled_output = dropout(pooled_output)
            logits = classifier(pooled_output)
            probs = torch.nn.functional.softmax(logits, dim=-1)[:, 1]
            return probs

        ig = IntegratedGradients(forward_func_embeds)

        sample_val_texts = val_texts[:10]  # Limit for demonstration.

        for start_idx in range(0, len(sample_val_texts), ig_chunk_size):
            end_idx = start_idx + ig_chunk_size
            sub_texts = sample_val_texts[start_idx:end_idx]

            # Tokenise and prepare inputs.
            sample_encodings = tokenizer(
                sub_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            input_ids = sample_encodings["input_ids"].to(device)
            attention_mask = sample_encodings["attention_mask"].to(device)

            # Obtain embeddings.
            embeddings = model_for_ig.distilbert.embeddings(input_ids)

            # Build baseline embeddings.
            pad_token_id = tokenizer.pad_token_id
            baseline_input_ids = torch.full_like(input_ids, pad_token_id)
            baseline_embeddings = model_for_ig.distilbert.embeddings(baseline_input_ids)

            # Compute attributions.
            attributions = ig.attribute(
                embeddings,
                baselines=baseline_embeddings,
                additional_forward_args=(attention_mask,),
                n_steps=n_steps_ig,
            )
            attributions_sum = attributions.sum(dim=-1).detach().cpu().numpy()

            # Generate and save bar plots.
            for i, txt in enumerate(sub_texts):
                tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
                merged_tokens, merged_attrs = combine_subwords(
                    tokens, attributions_sum[i]
                )

                plt.figure(figsize=(12, 0.5 + 0.3 * len(merged_tokens)))
                sns.barplot(x=merged_attrs, y=merged_tokens)
                plt.xlabel("Attribution")
                plt.ylabel("Word")
                plt.title(f'Integrated Gradients for "{txt[:50]}..."')
                plt.tight_layout()

                plot_path = os.path.join(
                    ig_folder, f"{report_col}_ig_sample_{start_idx + i}.png"
                )
                plt.savefig(plot_path, bbox_inches="tight")
                plt.close()
                logging.info(f"[IG] Saved plot => {plot_path}")

    except Exception as e:
        logging.error(f"Error during Integrated Gradients for '{report_col}': {e}")

# --------------------------------------------------------------------------- #
# Lime Explainability                                                         #
# --------------------------------------------------------------------------- #

def run_lime_analysis(
    model: torch.nn.Module,
    tokenizer: DistilBertTokenizer,
    val_texts: list,
    output_dir: str,
    report_col: str,
    device,
    num_samples: int = 5,
    num_features: int = 10,
) -> None:
    """
    Perform LIME analysis on a subset of validation texts, saving the explanation outputs as HTML files.

    The process includes:
        - Moving the model to CPU to avoid GPU out-of-memory issues.
        - Defining a predict_proba function compatible with LIME.
        - Generating and saving explanation files.

    Args:
        model (torch.nn.Module): Trained DistilBERT model.
        tokeniser (DistilBertTokenizer): Tokeniser for encoding texts.
        val_texts (list): List of validation text samples.
        output_dir (str): Directory for saving LIME outputs.
        report_col (str): Report column identifier used in filenames.
        device: Torch device (model is moved to CPU for LIME analysis).
        num_samples (int): Number of samples to explain. Defaults to 5.
        num_features (int): Number of features in the explanation. Defaults to 10.
    """
    try:
        logging.info(f"Starting LIME analysis for '{report_col}'.")
        lime_folder = os.path.join(output_dir, "lime")
        os.makedirs(lime_folder, exist_ok=True)
        torch.cuda.empty_cache()

        # Move model to CPU for LIME analysis.
        if torch.cuda.is_available():
            logging.info("[LIME] Moving model to CPU for LIME analysis to prevent OOM.")
            model = model.module if hasattr(model, "module") else model
            model.to("cpu")

        # Define a scikit-learn style predict_proba function.
        def predict_proba(texts):
            inputs = tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            inputs = {k: v.to(torch.device("cpu")) for k, v in inputs.items()}
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                probs = (
                    torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
                )
            return probs  # Shape: [N, 2]

        explainer = LimeTextExplainer(class_names=["Class_0", "Class_1"])

        # Subset of validation texts.
        sample_texts = val_texts[:num_samples]

        for i, text in enumerate(sample_texts):
            explanation = explainer.explain_instance(
                text_instance=text,
                classifier_fn=predict_proba,
                labels=(0, 1),
                top_labels=1,
                num_features=num_features,
            )
            pred_label = explanation.top_labels[0]
            html_path = os.path.join(lime_folder, f"{report_col}_sample_{i}_lime.html")
            explanation.save_to_file(html_path)
            logging.info(
                f"[LIME] Explanation for sample {i}, predicted class {pred_label}, saved => {html_path}"
            )

    except Exception as e:
        logging.error(f"Error during LIME analysis for '{report_col}': {e}")

# --------------------------------------------------------------------------- #
# Attention Visusalisation                                                    #
# --------------------------------------------------------------------------- #

def run_attention_visualization(
    model: torch.nn.Module,
    tokenizer: DistilBertTokenizer,
    val_texts: list,
    output_dir: str,
    report_col: str,
    device,
    attn_chunk_size: int = 2,
) -> None:
    """
    Visualise attention weights for the [CLS] token across layers and heads, saving the results as heatmaps.

    The process involves:
        - Tokenising and batching validation texts.
        - Computing attention weights from the model.
        - Averaging the attention weights across layers and heads.
        - Extracting and plotting the attention from the [CLS] token.
        - Saving the heatmap plots.

    Args:
        model (torch.nn.Module): Trained DistilBERT model.
        tokeniser (DistilBertTokenizer): Tokeniser for encoding texts.
        val_texts (list): List of validation text samples.
        output_dir (str): Directory for saving attention plots.
        report_col (str): Report column identifier used in filenames.
        device: Torch device for computations.
        attn_chunk_size (int): Number of samples per chunk. Defaults to 2.
    """
    try:
        logging.info(f"Starting Attention Visualisation for '{report_col}'.")
        attention_folder = os.path.join(output_dir, "attention_weights")
        os.makedirs(attention_folder, exist_ok=True)
        torch.cuda.empty_cache()

        sample_val_texts = val_texts[:10]

        for start_idx in range(0, len(sample_val_texts), attn_chunk_size):
            end_idx = start_idx + attn_chunk_size
            sub_texts = sample_val_texts[start_idx:end_idx]

            enc = tokenizer(
                sub_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            input_ids_attn = enc["input_ids"].to(device)
            attention_mask_attn = enc["attention_mask"].to(device)

            with torch.no_grad():
                if hasattr(model, "module"):
                    outputs_attn = model.module(
                        input_ids=input_ids_attn,
                        attention_mask=attention_mask_attn,
                        output_attentions=True,
                    )
                else:
                    outputs_attn = model(
                        input_ids=input_ids_attn,
                        attention_mask=attention_mask_attn,
                        output_attentions=True,
                    )
                attentions = outputs_attn.attentions  # List of [batch, heads, seq, seq]

            avg_layer = torch.stack(attentions).mean(dim=0)  # Average across layers.
            avg_attn = avg_layer.mean(dim=1).cpu().numpy()  # Average across heads.

            for i, text in enumerate(sub_texts):
                tokens = tokenizer.convert_ids_to_tokens(input_ids_attn[i])
                # Extract attention from [CLS] token.
                cls_attn = avg_attn[i][0, :]
                non_pad_indices = [
                    j for j, t in enumerate(tokens) if t != tokenizer.pad_token
                ]
                tokens_nonpad = [tokens[j] for j in non_pad_indices]
                cls_attn_nonpad = cls_attn[non_pad_indices]

                attn_2d = np.expand_dims(cls_attn_nonpad, axis=0)

                plt.figure(figsize=(12, 3))
                sns.heatmap(
                    attn_2d,
                    annot=[tokens_nonpad],
                    fmt="",
                    cmap="viridis",
                    xticklabels=False,
                    yticklabels=False,
                )
                plt.xlabel("Tokens")
                plt.title(f'Attention of [CLS] => tokens for "{text[:50]}..."')
                plt.tight_layout()

                plot_path = os.path.join(
                    attention_folder,
                    f"{report_col}_attention_sample_{start_idx + i}.png",
                )
                plt.savefig(plot_path, bbox_inches="tight")
                plt.close()
                logging.info(f"[ATTN] Saved attention plot => {plot_path}")

    except Exception as e:
        logging.error(f"Error during Attention Visualisation for '{report_col}': {e}")

# --------------------------------------------------------------------------- #
# Training and Prediction Functions                                           #
# --------------------------------------------------------------------------- #

def train_distilbert_for_column(
    train_df,
    val_df,
    report_col: str,
    label_col: str,
    tokenizer: DistilBertTokenizer,
    output_dir: str,
    num_epochs: int = 20,
    batch_size: int = 8,
    lr: float = 5e-5,
    perform_shap: bool = False,
    perform_lime: bool = False,
):
    """
    Train a DistilBERT model on a specific report column to predict a given label.

    The process includes:
        - Preprocessing the training and validation data.
        - Setting up datasets and dataloaders.
        - Initialising and training the DistilBERT model.
        - Evaluating the model on validation data.
        - Saving the trained model and tokeniser.
        - Optionally running explainability analyses (SHAP, Integrated Gradients, LIME, and Attention Visualisation).

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        report_col (str): Name of the text column to use as input.
        label_col (str): Name of the label column.
        tokeniser (DistilBertTokenizer): Tokeniser for the model.
        output_dir (str): Directory to save the trained model and outputs.
        num_epochs (int): Number of training epochs. Defaults to 20.
        batch_size (int): Batch size for training. Defaults to 8.
        lr (float): Learning rate. Defaults to 5e-5.
        perform_shap (bool): If True, perform SHAP analysis. Defaults to False.
        perform_lime (bool): If True, perform LIME analysis. Defaults to False.

    Returns:
        tuple: (model, tokeniser, val_preds, formatted_metrics) where:
            - model: The trained DistilBERT model.
            - tokenizer: The tokeniser used.
            - val_preds: Numpy array of predictions on the validation set.
            - formatted_metrics: Dictionary of formatted evaluation metrics.
    """
    # Filter missing text/label.
    train_df = train_df.dropna(subset=[report_col, label_col])
    val_df = val_df.dropna(subset=[report_col, label_col])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Prepare data.
    train_texts = train_df[report_col].fillna("").astype(str).tolist()
    train_labels = train_df[label_col].astype(int).tolist()
    val_texts = val_df[report_col].fillna("").astype(str).tolist()
    val_labels = val_df[label_col].astype(int).tolist()

    # Datasets and Dataloaders.
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialise DistilBERT.
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    model.to(device)

    # Wrap model in DataParallel if multiple GPUs are available.
    if torch.cuda.device_count() > 1:
        logging.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # ----------------------- Training Loop -----------------------
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            if loss.dim() > 0:
                loss = loss.mean()

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        logging.info(
            f"Epoch {epoch+1}/{num_epochs} - {report_col} - Loss: {avg_loss:.4f}"
        )

    # ----------------------- Validation Evaluation -----------------------
    model.eval()
    val_preds = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            val_preds.extend(preds.cpu().numpy())

    metrics = compute_bootstrap_metrics(val_labels, val_preds)
    formatted_metrics = {
        "Accuracy": fmt_confint(*metrics["accuracy"]),
        "Precision": fmt_confint(*metrics["precision"]),
        "Recall": fmt_confint(*metrics["recall"]),
        "Specificity": fmt_confint(*metrics["specificity"]),
        "NPV": fmt_confint(*metrics["npv"]),
        "F1_Score": fmt_confint(*metrics["f1"]),
    }

    # Save the trained model and tokeniser.
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Model and tokeniser saved to '{output_dir}' for '{report_col}'.")

    # ----------------------- Explanation Analyses -----------------------
    if perform_shap:
        del optimizer
        torch.cuda.empty_cache()

        run_shap_analysis(
            model,
            tokenizer,
            train_texts,
            val_texts,
            output_dir,
            report_col,
            device,
            background_size=10,
            shap_nsamples=10,
            shap_chunk_size=5,
        )
        run_integrated_gradients(
            model,
            tokenizer,
            val_texts,
            output_dir,
            report_col,
            device,
            ig_chunk_size=2,
            n_steps_ig=10,
        )
        run_attention_visualization(
            model,
            tokenizer,
            val_texts,
            output_dir,
            report_col,
            device,
            attn_chunk_size=2,
        )

    if perform_lime:
        run_lime_analysis(
            model,
            tokenizer,
            val_texts,
            output_dir,
            report_col,
            device,
            num_samples=5,
            num_features=10,
        )

    torch.cuda.empty_cache()

    return model, tokenizer, np.array(val_preds), formatted_metrics


def predict_distilbert(
    model: torch.nn.Module,
    tokenizer: DistilBertTokenizer,
    df,
    report_col: str,
    batch_size: int = 8,
    proba: bool = False,
):
    """
    Generate predictions for a given report column using the trained DistilBERT model.

    Depending on the 'proba' flag, this function returns either class predictions (0/1) or
    the probability for the positive class.

    Args:
        model (torch.nn.Module): Trained DistilBERT model.
        tokeniser (DistilBertTokenizer): Tokeniser for encoding texts.
        df (pd.DataFrame): DataFrame containing the input text data.
        report_col (str): Name of the text column.
        batch_size (int): Batch size for prediction. Defaults to 8.
        proba (bool): If True, returns probabilities for the positive class; otherwise, class labels.

    Returns:
        np.array: Array of predictions or probabilities.
    """
    texts = df[report_col].fillna("").astype(str).tolist()
    dummy_labels = [0] * len(texts)
    dataset = TextDataset(texts, dummy_labels, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if proba:
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[:, 1]
                predictions.extend(probs.cpu().numpy())
            else:
                preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(preds.cpu().numpy())

    return np.array(predictions)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("This module contains DistilBERT training + explainability functions.")
    logging.info(
        "Import and use train_distilbert_for_column(...) / predict_distilbert(...) in your pipeline."
    )

# ------------------------------------------------------------------#
# PATIENT-LEVEL META-MODEL (Not used in Production)                 #
# ------------------------------------------------------------------#
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _build_meta_clf(random_state: int = 42):
    """
    Scaler → (balanced) logistic regression.  Drop-in replaceable with GBM/XGB later.
    """
    return make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        ),
    )


def train_patient_meta_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    patient_feature_cols: list[str],
    label_col: str = "Patient_Has_IBD",
    output_dir: str | None = None,
    random_state: int = 42,
):
    """
    Train a meta-classifier on patient-level aggregated features (0/1 or probabilities).

    Returns
    -------
    meta_clf      : fitted scikit-learn pipeline
    val_pred      : hard 0/1 predictions on `val_df`
    val_pred_proba: positive-class probabilities (for calibration / PR-AUC)
    """
    X_train = train_df[patient_feature_cols].fillna(0)
    y_train = train_df[label_col].astype(int)

    X_val = val_df[patient_feature_cols].fillna(0)
    y_val = val_df[label_col].astype(int)

    clf = _build_meta_clf(random_state)
    clf.fit(X_train, y_train)

    val_pred_proba = clf.predict_proba(X_val)[:, 1]
    val_pred = (val_pred_proba >= 0.5).astype(int)

    # Optional: persist artefacts
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(clf, os.path.join(output_dir, "meta_model.joblib"))
        np.save(os.path.join(output_dir, "val_pred_proba.npy"), val_pred_proba)
        np.save(os.path.join(output_dir, "val_pred.npy"), val_pred)

    return clf, val_pred, val_pred_proba
