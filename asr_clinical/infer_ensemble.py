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
    return tokenizer, model_dir


def predict_with_ensemble(model, X, task: str):
    """
    Make predictions with ensemble model.
    Handles both single models and VotingClassifier/VotingRegressor.
    """
    # For ensemble models, predict_proba might work differently
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)
            predictions = model.predict(X)
            return predictions, probs
        except:
            # Fallback for some ensemble configurations
            predictions = model.predict(X)
            return predictions, None
    else:
        predictions = model.predict(X)
        return predictions, None


def get_ensemble_info(ensemble_dir: Path) -> dict:
    """Load ensemble configuration if available."""
    ensemble_config_path = ensemble_dir / "ensemble_config.json"
    if ensemble_config_path.exists():
        with open(ensemble_config_path) as f:
            return json.load(f)
    return {"use_ensemble": False}


def main():
    parser = argparse.ArgumentParser(description="Inference with trained top‑k ensemble (supports single models and ensembles)")
    parser.add_argument("--ensemble-dir", required=True, help="Output directory from training")
    parser.add_argument("--input-csv", required=True, help="CSV with columns: speaker_id, question_id, text")
    parser.add_argument("--output-csv", required=True, help="Where to save predictions")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding extraction")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--return-probabilities", action="store_true", 
                        help="Return probability/probability distributions (for classification)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed inference information")
    args = parser.parse_args()

    ensemble_dir = Path(args.ensemble_dir)
    device = torch.device(args.device)

    # 1. Load configuration
    with open(ensemble_dir / "question_ensemble_config.json") as f:
        cfg = json.load(f)
    max_length = cfg.get("max_length", 256)
    task = cfg.get("task", "classification")
    
    # Load ensemble info if available
    ensemble_info = get_ensemble_info(ensemble_dir)
    
    if args.verbose:
        print("=" * 60)
        print("INFERENCE CONFIGURATION")
        print("=" * 60)
        print(f"Ensemble directory: {ensemble_dir}")
        print(f"Task: {task}")
        print(f"Max length: {max_length}")
        print(f"Device: {device}")
        if ensemble_info.get("use_ensemble"):
            print(f"Using ensemble with models: {ensemble_info.get('ensemble_models', [])}")
            print(f"Voting type: {ensemble_info.get('voting_type', 'unknown')}")
        else:
            print(f"Using single meta-model")
        print("=" * 60)

    # 2. Load selected questions (top‑k)
    selected_df = pd.read_csv(ensemble_dir / "selected_questions.csv")
    selected_questions = selected_df["question_id"].tolist()
    print(f"Loaded {len(selected_questions)} selected questions: {selected_questions}")

    # 3. Load meta-model (could be single model or ensemble)
    meta_model = joblib.load(ensemble_dir / "meta_model.joblib")
    
    if args.verbose:
        print(f"\nMeta-model type: {type(meta_model).__name__}")
        if hasattr(meta_model, 'named_estimators_'):
            print(f"Ensemble members: {list(meta_model.named_estimators_.keys())}")

    # 4. Read input data
    df = pd.read_csv(args.input_csv)
    
    # Ensure required columns
    required_cols = ["speaker_id", "question_id", "text"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Input CSV must contain column '{col}'")
    
    # Keep only questions that are in the selected set
    original_questions = df["question_id"].nunique()
    df = df[df["question_id"].isin(selected_questions)].copy()
    
    if args.verbose:
        print(f"\nInput data: {len(df)} rows from {original_questions} questions")
        print(f"Keeping {df['question_id'].nunique()} questions that match selected set")

    if df.empty:
        raise ValueError("No data found for the selected questions")

    # 5. Extract embeddings per question and aggregate per speaker
    speaker_feature_rows = []

    for question in selected_questions:
        q_df = df[df["question_id"] == question].copy()
        if q_df.empty:
            if args.verbose:
                print(f"  Question {question}: no data, will fill with zeros")
            continue

        if args.verbose:
            print(f"  Processing {question}: {len(q_df)} utterances")

        tokenizer, model_dir = load_question_model(question, ensemble_dir, device)
        texts = q_df["text"].tolist()

        # Extract embeddings for each utterance
        embeds = extract_embeddings(
            model_dir, texts, tokenizer, device, max_length, args.batch_size
        )
        
        # Add embeddings to dataframe
        for i in range(embeds.shape[1]):
            q_df[f"emb_{i}"] = embeds[:, i]

        # Aggregate per speaker (mean of embeddings)
        emb_cols = [f"emb_{i}" for i in range(embeds.shape[1])]
        speaker_agg = q_df.groupby("speaker_id").agg(
            {col: "mean" for col in emb_cols}
        )
        
        # Rename columns to match training: QX__emb_i
        speaker_agg = speaker_agg.rename(columns={col: f"{question}__{col}" for col in emb_cols})
        speaker_agg[f"{question}__present"] = 1.0
        speaker_feature_rows.append(speaker_agg)

    # 6. Combine feature tables from all questions
    if not speaker_feature_rows:
        raise ValueError("No embeddings extracted for any selected question")

    merged = speaker_feature_rows[0]
    for other in speaker_feature_rows[1:]:
        merged = merged.join(other, how="outer")

    # Fill missing questions with zeros
    merged = merged.fillna(0.0)
    merged = merged.reset_index()  # speaker_id becomes a column

    # 7. Ensure feature columns match training
    feature_cols_path = ensemble_dir / "selected_embedding_features.csv"
    if feature_cols_path.exists():
        expected_features = pd.read_csv(feature_cols_path)["feature"].tolist()
        
        # Add missing columns with zeros
        for col in expected_features:
            if col not in merged.columns:
                merged[col] = 0.0
        
        # Keep only expected features and speaker_id
        merged = merged[["speaker_id"] + expected_features]
        
        if args.verbose:
            print(f"\nUsing {len(expected_features)} expected features from training")
    else:
        # Fallback: use all embedding columns
        feature_cols = [c for c in merged.columns if "__" in c]
        merged = merged[["speaker_id"] + feature_cols]
        if args.verbose:
            print(f"\nWarning: No feature list found, using {len(feature_cols)} inferred features")

    # 8. Predict with meta-model
    X = merged[[c for c in merged.columns if c != "speaker_id"]].to_numpy()
    
    if args.verbose:
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Number of speakers: {len(merged)}")
    
    # Handle predictions (works for both single models and ensembles)
    if hasattr(meta_model, "predict_proba") and (task == "classification" or args.return_probabilities):
        try:
            probs = meta_model.predict_proba(X)
            predictions = meta_model.predict(X)
            if args.verbose:
                print(f"Using predict_proba for predictions")
        except Exception as e:
            if args.verbose:
                print(f"Warning: predict_proba failed ({e}), falling back to predict")
            predictions = meta_model.predict(X)
            probs = None
    else:
        predictions = meta_model.predict(X)
        probs = None

    # 9. Prepare output
    output_df = pd.DataFrame({"speaker_id": merged["speaker_id"], "prediction": predictions})
    
    # Add probabilities if available
    if probs is not None:
        if hasattr(meta_model, "classes_"):
            classes = meta_model.classes_
        elif hasattr(meta_model, "named_estimators_"):
            # For voting classifier, try to get classes from the first estimator
            first_est = list(meta_model.named_estimators_.values())[0]
            classes = getattr(first_est, "classes_", [f"class_{i}" for i in range(probs.shape[1])])
        else:
            classes = [f"class_{i}" for i in range(probs.shape[1])]
        
        for i, cls in enumerate(classes):
            output_df[f"prob_{cls}"] = probs[:, i]
        
        # For classification, add the predicted class name if available
        if task == "classification" and len(classes) > 0:
            # Map numeric predictions to class names if needed
            if isinstance(predictions[0], (int, np.integer)):
                output_df["predicted_class"] = [classes[int(p)] for p in predictions]
            else:
                output_df["predicted_class"] = predictions
    
    # Add confidence scores for classification
    if task == "classification" and probs is not None:
        output_df["confidence"] = np.max(probs, axis=1)
    
    # For regression, add additional metrics if needed
    if task == "regression" and args.return_probabilities:
        # For regression ensembles, we can also get std dev if available
        if hasattr(meta_model, "predict_proba"):
            # Some regression models can provide uncertainty estimates
            pass
    
    # 10. Save results
    output_df.to_csv(args.output_csv, index=False)
    
    # 11. Print summary
    print(f"\nPredictions saved to {args.output_csv}")
    print(f"Processed {len(output_df)} speakers")
    
    if task == "classification":
        # Show distribution of predictions
        pred_counts = output_df["prediction"].value_counts()
        print("\nPrediction distribution:")
        for pred, count in pred_counts.items():
            pct = (count / len(output_df)) * 100
            print(f"  {pred}: {count} ({pct:.1f}%)")
        
        if "confidence" in output_df.columns:
            print(f"\nAverage confidence: {output_df['confidence'].mean():.3f}")
    
    elif task == "regression":
        print(f"\nPrediction range: [{output_df['prediction'].min():.3f}, {output_df['prediction'].max():.3f}]")
        print(f"Prediction mean: {output_df['prediction'].mean():.3f}")
        print(f"Prediction std: {output_df['prediction'].std():.3f}")
    
    # 12. Print ensemble info if available
    if ensemble_info.get("use_ensemble") and args.verbose:
        print("\n" + "=" * 60)
        print("ENSEMBLE INFORMATION")
        print("=" * 60)
        print(f"Models used: {ensemble_info.get('ensemble_models', [])}")
        print(f"Voting type: {ensemble_info.get('voting_type', 'unknown')}")
        if hasattr(meta_model, 'weights_'):
            print(f"Ensemble weights: {meta_model.weights_}")


if __name__ == "__main__":
    main()
'''

1. Supports Both Single and Ensemble Models:
Automatically detects if using ensemble (checks for ensemble_config.json)

Handles VotingClassifier and VotingRegressor seamlessly

Falls back gracefully if certain methods aren't available

2. Enhanced Output:
python
# For classification:
# - prediction: numeric class
# - predicted_class: class name (if available)
# - prob_*: probabilities for each class
# - confidence: max probability

# For regression:
# - prediction: numeric value
3. Verbose Mode:
bash
python inference.py --ensemble-dir ./output --input-csv test.csv --output-csv results.csv --verbose
4. Usage Examples:
Basic inference (single model or ensemble):

bash
python inference.py \
  --ensemble-dir ./trained_model \
  --input-csv test_data.csv \
  --output-csv predictions.csv
With probabilities and verbose output:

bash
python inference.py \
  --ensemble-dir ./trained_model \
  --input-csv test_data.csv \
  --output-csv predictions.csv \
  --return-probabilities \
  --verbose
5. Input CSV Format:
csv
speaker_id,question_id,text
SPEAKER_001,Q1,The patient reports...
SPEAKER_001,Q2,Memory issues started...
SPEAKER_002,Q1,No significant complaints...
6. Output CSV Format:
For Classification:

csv
speaker_id,prediction,prob_Dementia,prob_MCI,prob_HC,predicted_class,confidence
SPEAKER_001,2,0.05,0.15,0.80,HC,0.80
SPEAKER_002,0,0.70,0.20,0.10,Dementia,0.70
For Regression:

csv
speaker_id,prediction
SPEAKER_001,24.5
SPEAKER_002,18.3
7. Handling Missing Data:
Automatically fills missing questions with zeros

Only processes questions in the top-k selected set

Warns if questions are missing (in verbose mode)

8. Compatibility:
Works with models trained using the updated training script

Handles both --use-ensemble and single model training

Backward compatible with older training outputs

The inference script now fully supports your ensemble approach with majority voting for classification and averaging for regression!


'''