from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def extract_embeddings(
    model_dir: Path,
    texts: list[str],
    tokenizer,
    device: torch.device,
    max_length: int,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract mean‑pooled embeddings for a list of texts using a saved model."""
    model = AutoModel.from_pretrained(model_dir).to(device)
    model.eval()

    all_embeds = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc, output_hidden_states=True)
        embeds = mean_pool(outputs.hidden_states[-1], enc["attention_mask"]).cpu().numpy()
        all_embeds.append(embeds)

    return np.vstack(all_embeds)


def load_question_model(question: str, ensemble_dir: Path, device: torch.device):
    """Load tokenizer and model for a given question."""
    model_dir = ensemble_dir / "question_models" / question / "model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # The model is loaded only when needed; we return a loader function
    return tokenizer, model_dir


def main():
    parser = argparse.ArgumentParser(description="Inference with trained top‑k ensemble")
    parser.add_argument("--ensemble-dir", required=True, help="Output directory from training")
    parser.add_argument("--input-csv", required=True, help="CSV with columns: speaker_id, question_id, text (or audio_path)")
    parser.add_argument("--output-csv", required=True, help="Where to save predictions")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding extraction")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ensemble_dir = Path(args.ensemble_dir)
    device = torch.device(args.device)

    # 1. Load configuration and meta‑model
    with open(ensemble_dir / "question_ensemble_config.json") as f:
        cfg = json.load(f)
    max_length = cfg["max_length"]

    # Selected questions (top‑k)
    selected_df = pd.read_csv(ensemble_dir / "selected_questions.csv")
    selected_questions = selected_df["question_id"].tolist()
    print(f"Loaded {len(selected_questions)} selected questions: {selected_questions}")

    meta_model = joblib.load(ensemble_dir / "meta_model.joblib")

    # 2. Read input data
    df = pd.read_csv(args.input_csv)
    # Ensure required columns
    for col in ["speaker_id", "question_id"]:
        if col not in df.columns:
            raise ValueError(f"Input CSV must contain column '{col}'")
    if "text" not in df.columns:
        # If you have audio paths, you would transcribe them here.
        # For simplicity, we assume transcriptions are already available.
        raise ValueError("Input CSV must contain a 'text' column with transcripts")

    # Keep only questions that are in the selected set (others are ignored)
    df = df[df["question_id"].isin(selected_questions)].copy()

    if df.empty:
        raise ValueError("No data found for the selected questions")

    # 3. For each selected question, extract embeddings per utterance,
    #    then aggregate per speaker (mean across utterances for that speaker).
    speaker_feature_rows = []

    for question in selected_questions:
        q_df = df[df["question_id"] == question].copy()
        if q_df.empty:
            # No data for this question – later we will fill with zeros
            continue

        tokenizer, model_dir = load_question_model(question, ensemble_dir, device)
        texts = q_df["text"].tolist()

        # Extract embeddings for each utterance
        embeds = extract_embeddings(
            model_dir, texts, tokenizer, device, max_length, args.batch_size
        )
        # embeds shape: (n_utterances, hidden_dim)

        # Add embeddings to dataframe
        for i, col in enumerate([f"emb_{j}" for j in range(embeds.shape[1])]):
            q_df[col] = embeds[:, i]

        # Aggregate per speaker (mean)
        speaker_agg = q_df.groupby("speaker_id").agg(
            {col: "mean" for col in q_df.columns if col.startswith("emb_")}
        )
        # Rename columns to match training: QX__emb_i
        speaker_agg = speaker_agg.rename(columns={col: f"{question}__{col}" for col in speaker_agg.columns})
        speaker_agg[f"{question}__present"] = 1.0
        speaker_feature_rows.append(speaker_agg)

    # 4. Combine feature tables from all questions
    if not speaker_feature_rows:
        raise ValueError("No embeddings extracted for any selected question")

    merged = speaker_feature_rows[0]
    for other in speaker_feature_rows[1:]:
        merged = merged.join(other, how="outer")

    # Fill missing questions with zeros (if a speaker didn't have a particular question)
    merged = merged.fillna(0.0)
    merged = merged.reset_index()  # speaker_id becomes a column

    # 5. Ensure the feature columns are in the same order as during training
    #    (the order does not matter for tree models, but for linear models it does)
    #    Load training feature list from saved file.
    feature_cols_path = ensemble_dir / "selected_embedding_features.csv"
    if feature_cols_path.exists():
        expected_features = pd.read_csv(feature_cols_path)["feature"].tolist()
        # Add missing columns with zeros
        for col in expected_features:
            if col not in merged.columns:
                merged[col] = 0.0
        # Keep only expected features and speaker_id
        merged = merged[["speaker_id"] + expected_features]
    else:
        # If not saved, use all columns with "__"
        feature_cols = [c for c in merged.columns if "__" in c]
        merged = merged[["speaker_id"] + feature_cols]

    # 6. Predict with meta‑model
    X = merged[[c for c in merged.columns if c != "speaker_id"]].to_numpy()
    predictions = meta_model.predict(X)

    # If classification and we have proba, get them
    probs = None
    if hasattr(meta_model, "predict_proba"):
        probs = meta_model.predict_proba(X)

    # 7. Save results
    output_df = pd.DataFrame({"speaker_id": merged["speaker_id"], "prediction": predictions})
    if probs is not None:
        classes = meta_model.classes_
        for i, cls in enumerate(classes):
            output_df[f"prob_{cls}"] = probs[:, i]

    output_df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")


if __name__ == "__main__":
    main()
'''

How to Use the Script
Prepare your input CSV
For each utterance, you need a row with:

speaker_id (e.g., "speaker_001")

question_id (e.g., "Q1", "Q2", … must match the question identifiers used during training – case‑sensitive)

text (the transcribed text of the audio)

Example:

csv
speaker_id,question_id,text
spk001,Q1,"I feel very happy today."
spk001,Q2,"My pain level is about 3 out of 10."
spk002,Q1,"I am not feeling well."
Run inference

bash
python infer_ensemble.py \
    --ensemble-dir /path/to/your/training/output \
    --input-csv new_data.csv \
    --output-csv predictions.csv
Output
predictions.csv will contain:

speaker_id

prediction – the final prediction per speaker (class label or numeric value)

For classification: probabilistic predictions per class (prob_0, prob_1, …)
'''