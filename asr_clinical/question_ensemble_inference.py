#!/usr/bin/env python3
"""
Inference script for the per‑question ensemble + meta‑model pipeline.
Assumes the training output directory contains:
  - best_hyperparams_all_questions.json
  - selected_questions.csv
  - selected_embedding_features.csv
  - meta_model.joblib (or final_cv_model.joblib)
  - question_models/Q*/model/  (per‑question Hugging Face models)
  - metadata.json (optional, but used to verify task/columns)
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# ---------- reuse utilities from the original script ----------
def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def load_examples_inference(asr_file, demo_file, target_column, task,
                            text_mode="question", min_text_chars=1,
                            filter_questions=None, delimiter=";"):
    """
    Minimal version of the original load_examples.
    Returns a DataFrame with columns: speaker_id, session_id, utterance_id,
    question_id, text, label (if target_column exists in demo file).
    """
    from .data import load_examples   # if running from package; else copy the function
    # For standalone, copy the full load_examples function from your original code.
    # Here we assume you have it available. We'll include a simplified version below.
    # For brevity, I'll just call the original function (assuming this script lives in the same package).
    # If you want a truly standalone script, copy the load_examples implementation.
    return load_examples(
        asr_file, demo_file, target_column, task,
        text_mode=text_mode,
        min_text_chars=min_text_chars,
        filter_questions=filter_questions,
        delimiter=delimiter
    )

def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_tokenizer(model_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_dir)

def extract_embeddings_inference(model_dir, df, tokenizer, model, max_length, batch_size):
    """Extract pooled embeddings for a single question."""
    device = next(model.parameters()).device
    rows = []
    texts = df["text"].tolist()
    for start in range(0, len(texts), batch_size):
        batch_df = df.iloc[start:start+batch_size].reset_index(drop=True)
        enc = tokenizer(
            batch_df["text"].tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)
        embeddings = mean_pool(outputs.hidden_states[-1], enc["attention_mask"]).cpu().numpy()

        for row_idx, emb in enumerate(embeddings):
            meta = batch_df.iloc[row_idx]
            row = {
                "speaker_id": meta["speaker_id"],
                "session_id": meta["session_id"],
                "utterance_id": meta["utterance_id"],
                "question_id": meta["question_id"],
                "y_true": meta.get("label", None),  # may be missing for new data
            }
            row.update({f"emb_{i}": float(v) for i, v in enumerate(emb)})
            rows.append(row)
    return pd.DataFrame(rows)
# -------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained per‑question + meta‑model")
    parser.add_argument("--model-dir", required=True,
                        help="Path to training output directory (contains question_models/, meta_model.joblib, etc.)")
    parser.add_argument("--asr-file", required=True)
    parser.add_argument("--demo-file", required=True)
    parser.add_argument("--target-column", default=None,
                        help="Optional target column (if you want to compute metrics; otherwise ignored)")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification",
                        help="Task type (must match training)")
    parser.add_argument("--output", default="predictions.csv",
                        help="Output CSV file with predictions")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for embedding extraction")
    parser.add_argument("--speaker-col", default="speaker_id",
                        help="Column name for speaker identifier")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory {model_dir} not found")

    # ---- load training artifacts ----
    # Best hyperparameters
    hparams_path = model_dir / "best_hyperparams_all_questions.json"
    if not hparams_path.exists():
        raise FileNotFoundError("best_hyperparams_all_questions.json not found")
    with open(hparams_path) as f:
        best_hparams = json.load(f)
    max_length = best_hparams.get("max_length", 256)
    batch_size = args.batch_size

    # Selected questions and features
    selected_qs_df = pd.read_csv(model_dir / "selected_questions.csv")
    selected_questions = selected_qs_df["question_id"].tolist()
    selected_features_df = pd.read_csv(model_dir / "selected_embedding_features.csv")
    selected_features = selected_features_df["feature"].tolist()
    # Map each feature to its question (prefix before "__")
    feature_to_question = {f: f.split("__", 1)[0] for f in selected_features}

    # Load meta‑model
    meta_model_path = model_dir / "meta_model.joblib"
    if not meta_model_path.exists():
        # fallback to CV model if no test set was used
        meta_model_path = model_dir / "final_cv_model.joblib"
    if not meta_model_path.exists():
        raise FileNotFoundError("meta_model.joblib or final_cv_model.joblib not found")
    meta_model = joblib.load(meta_model_path)

    # ---- load new data ----
    # We need to load all data (all questions) because we will filter per question.
    from .data import load_examples   # adjust if you copied the function
    df, metadata = load_examples(
        args.asr_file, args.demo_file, args.target_column, args.task,
        text_mode="question",
        min_text_chars=1,
        filter_questions=None,        # load all questions present
        delimiter=";"
    )
    # If target_column is not given, we can't compute metrics, but we still need labels for the columns?
    # For inference we don't need labels, but we keep them for compatibility.
    # The df will have 'label' column only if target_column existed.

    # ---- process each selected question to build speaker‑level embeddings ----
    device = choose_device()
    speaker_embeddings = {}  # speaker_id -> dict of question->embeddings (list of feature values)

    # We'll load per‑question models on demand and cache them
    models_cache = {}

    for q in selected_questions:
        q_dir = model_dir / "question_models" / q / "model"
        if not q_dir.exists():
            print(f"Warning: Model for question {q} not found. Skipping (zeros will be filled).")
            continue

        # Load model and tokenizer (cached)
        if q not in models_cache:
            tokenizer = load_tokenizer(str(q_dir))
            model = AutoModelForSequenceClassification.from_pretrained(str(q_dir)).to(device)
            model.eval()
            models_cache[q] = (tokenizer, model)
        else:
            tokenizer, model = models_cache[q]

        # Filter data for this question
        q_df = df[df["question_id"] == q].copy()
        if q_df.empty:
            print(f"No data for question {q}, skipping.")
            continue

        # Extract embeddings for this question
        emb_df = extract_embeddings_inference(
            q_dir, q_df, tokenizer, model, max_length, batch_size
        )
        # Aggregate by speaker (mean pooling over utterances)
        emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
        grouped = emb_df.groupby("speaker_id", as_index=True).agg(
            {col: "mean" for col in emb_cols}
        )
        # Rename columns to include question prefix
        grouped = grouped.rename(columns={col: f"{q}__{col}" for col in emb_cols})
        # Mark that this question is present
        grouped[f"{q}__present"] = 1.0

        # Store in dictionary
        for speaker, row in grouped.iterrows():
            if speaker not in speaker_embeddings:
                speaker_embeddings[speaker] = {}
            # store the features for this question
            for col in grouped.columns:
                speaker_embeddings[speaker][col] = row[col]

    # ---- build speaker‑level feature table (only selected features) ----
    speakers = list(speaker_embeddings.keys())
    if not speakers:
        raise RuntimeError("No speakers processed. Check that the new data contains at least one of the selected questions.")

    # Initialize a DataFrame with all selected features, filled with 0
    feature_df = pd.DataFrame(0.0, index=speakers, columns=selected_features)
    feature_df.index.name = "speaker_id"
    # Also need to add a column for y_true if available (optional)
    y_true_dict = {}
    for sp in speakers:
        # get label from original df if present (first occurrence)
        label_val = df[df["speaker_id"] == sp]["label"].iloc[0] if not df[df["speaker_id"] == sp].empty else None
        y_true_dict[sp] = label_val

    # Fill in the actual features
    for sp, emb_dict in speaker_embeddings.items():
        for col, val in emb_dict.items():
            if col in feature_df.columns:
                feature_df.loc[sp, col] = val
        # also store the present marker for each question (if missing, it stays 0)
        # we already set 0 for absent questions.

    # Add y_true column (if available)
    feature_df["y_true"] = feature_df.index.map(y_true_dict)

    # Reorder columns to match training: features first, then y_true
    feature_cols = selected_features  # these are the features the meta‑model was trained on
    # Ensure we have all columns; missing ones are already 0
    X = feature_df[feature_cols].to_numpy()

    # ---- run meta‑model ----
    preds = meta_model.predict(X)
    if args.task == "classification" and hasattr(meta_model, "predict_proba"):
        probs = meta_model.predict_proba(X)
        classes = meta_model.classes_
    else:
        probs = None
        classes = None

    # ---- save predictions ----
    out_df = feature_df[["y_true"]].copy()
    out_df["y_pred"] = preds
    if probs is not None:
        for i, cls in enumerate(classes):
            out_df[f"prob_{cls}"] = probs[:, i]

    # Add speaker_id as a column (reset index)
    out_df = out_df.reset_index()
    out_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

    # Optionally print performance if we have true labels
    if not out_df["y_true"].isna().all():
        from sklearn.metrics import f1_score, mean_squared_error
        if args.task == "classification":
            # Use macro F1 (like training)
            y_true = out_df["y_true"].dropna().astype(int)
            y_pred = out_df.loc[y_true.index, "y_pred"].astype(int)
            score = f1_score(y_true, y_pred, average="macro")
            print(f"Macro F1 on provided labels: {score:.4f}")
        else:
            y_true = out_df["y_true"].dropna().astype(float)
            y_pred = out_df.loc[y_true.index, "y_pred"].astype(float)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            print(f"RMSE on provided labels: {rmse:.4f}")

if __name__ == "__main__":
    main()