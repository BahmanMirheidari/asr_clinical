from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification

from .config import TrainConfig
from .data import load_examples, make_final_test_split
from .model import load_tokenizer
from .train import choose_device, load_saved_model, saved_model_exists, train_one_fold


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune one LLM model per question, export embeddings, select important "
            "question embeddings, and train a final meta-model."
        )
    )
    parser.add_argument("--asr-file", required=True)
    parser.add_argument("--demo-file", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="distilroberta-base")
    parser.add_argument(
        "--questions",
        nargs="+",
        default=[f"Q{i}" for i in range(1, 14)],
        help="Questions to train separately. Default: Q1 Q2 ... Q13",
    )
    parser.add_argument("--top-k", type=int, default=0, help="0 means use all questions.")
    parser.add_argument(
        "--evaluate-all-top-k",
        action="store_true",
        default=True,
        help="Evaluate top-k question subsets from k=1 to all questions.",
    )
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--final-dev-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--class-weights", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--min-text-chars", type=int, default=1)
    parser.add_argument("--meta-model", choices=["linear", "random_forest"], default="linear")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument(
        "--force-embeddings",
        action="store_true",
        help="Regenerate embedding CSV files even if they already exist.",
    )
    return parser


def make_question_cfg(args, question: str, question_dir: Path) -> TrainConfig:
    return TrainConfig(
        asr_file=args.asr_file,
        demo_file=args.demo_file,
        target_column=args.target_column,
        task=args.task,
        output_dir=str(question_dir),
        model_name=args.model_name,
        text_mode="question",
        aggregate_level="speaker",
        num_folds=1,
        test_size=args.test_size,
        final_dev_size=args.final_dev_size,
        seed=args.seed,
        max_length=args.max_length,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        patience=args.patience,
        class_weights=args.class_weights,
        loss=args.loss,
        focal_gamma=args.focal_gamma,
        filter_questions=[question],
        min_text_chars=args.min_text_chars,
    )


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def extract_embeddings(model_dir: Path, df: pd.DataFrame, args, output_csv: Path):
    if output_csv.exists() and not args.force_embeddings:
        return pd.read_csv(output_csv)

    device = choose_device()
    tokenizer = load_tokenizer(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    rows = []
    texts = df["text"].tolist()
    for start in range(0, len(texts), args.embedding_batch_size):
        batch_df = df.iloc[start : start + args.embedding_batch_size].reset_index(drop=True)
        enc = tokenizer(
            batch_df["text"].tolist(),
            truncation=True,
            padding=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        enc = {key: value.to(device) for key, value in enc.items()}
        outputs = model(**enc, output_hidden_states=True)
        embeddings = mean_pool(outputs.hidden_states[-1], enc["attention_mask"]).cpu().numpy()

        for row_idx, embedding in enumerate(embeddings):
            meta = batch_df.iloc[row_idx]
            row = {
                "speaker_id": meta["speaker_id"],
                "session_id": meta["session_id"],
                "utterance_id": meta["utterance_id"],
                "question_id": meta["question_id"],
                "y_true": meta["label"],
            }
            row.update({f"emb_{i}": float(value) for i, value in enumerate(embedding)})
            rows.append(row)

    emb_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    emb_df.to_csv(output_csv, index=False)
    return emb_df


def train_question_models(df, trainval_df, test_df, metadata, args, out_dir: Path):
    question_dirs = {}
    embedding_files = {"trainval": {}, "test": {}}
    question_summaries = []

    for question in [q.upper() for q in args.questions]:
        q_trainval = trainval_df[trainval_df["question_id"] == question].reset_index(drop=True)
        q_test = test_df[test_df["question_id"] == question].reset_index(drop=True)
        if q_trainval.empty:
            print(f"{question}: skipping, no trainval examples")
            continue

        q_dir = out_dir / "question_models" / question
        q_cfg = make_question_cfg(args, question, q_dir)
        q_train_idx, q_dev_idx = make_final_test_split(
            q_trainval, args.task, args.final_dev_size, args.seed + int(question[1:])
        )
        q_train = q_trainval.iloc[q_train_idx].reset_index(drop=True)
        q_dev = q_trainval.iloc[q_dev_idx].reset_index(drop=True)
        print(f"{question}: train={len(q_train)} dev={len(q_dev)} test={len(q_test)}")

        train_one_fold(q_train, q_dev, q_cfg, metadata, q_dir)
        model_dir = q_dir / "model"
        if not saved_model_exists(model_dir):
            raise FileNotFoundError(f"Expected saved model at {model_dir}")

        train_emb_csv = q_dir / "embeddings_trainval.csv"
        test_emb_csv = q_dir / "embeddings_test.csv"
        extract_embeddings(model_dir, q_trainval, args, train_emb_csv)
        if not q_test.empty:
            extract_embeddings(model_dir, q_test, args, test_emb_csv)

        question_dirs[question] = q_dir
        embedding_files["trainval"][question] = train_emb_csv
        if q_test.empty:
            embedding_files["test"][question] = None
        else:
            embedding_files["test"][question] = test_emb_csv
        question_summaries.append(
            {
                "question_id": question,
                "trainval_examples": len(q_trainval),
                "test_examples": len(q_test),
                "model_dir": str(model_dir),
                "trainval_embeddings": str(train_emb_csv),
                "test_embeddings": str(test_emb_csv) if not q_test.empty else None,
            }
        )

    pd.DataFrame(question_summaries).to_csv(out_dir / "question_model_summary.csv", index=False)
    return embedding_files


def build_feature_table(embedding_paths: dict[str, Path | None], questions: list[str]):
    tables = []
    for question in questions:
        path = embedding_paths.get(question)
        if path is None or not Path(path).exists():
            continue
        emb_df = pd.read_csv(path)
        emb_cols = [col for col in emb_df.columns if col.startswith("emb_")]
        if not emb_cols:
            continue
        grouped = emb_df.groupby("speaker_id", as_index=True).agg(
            y_true=("y_true", "first"),
            **{col: (col, "mean") for col in emb_cols},
        )
        grouped = grouped.rename(columns={col: f"{question}__{col}" for col in emb_cols})
        grouped[f"{question}__present"] = 1.0
        tables.append(grouped)

    if not tables:
        raise ValueError("No embedding tables were available for meta-model training.")

    merged = tables[0]
    for table in tables[1:]:
        merged = merged.join(table.drop(columns=["y_true"]), how="outer")
        merged["y_true"] = merged["y_true"].combine_first(table["y_true"])

    merged = merged.reset_index()
    feature_cols = [col for col in merged.columns if "__" in col]
    merged[feature_cols] = merged[feature_cols].fillna(0.0)
    return merged, feature_cols


def align_feature_tables(train_df, test_df, feature_cols):
    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = 0.0
    extra_cols = [col for col in test_df.columns if "__" in col and col not in feature_cols]
    if extra_cols:
        test_df = test_df.drop(columns=extra_cols)
    return train_df, test_df


def make_meta_model(args):
    if args.task == "classification":
        if args.meta_model == "random_forest":
            return RandomForestClassifier(
                n_estimators=args.n_estimators,
                random_state=args.seed,
                class_weight="balanced",
                min_samples_leaf=2,
                n_jobs=-1,
            )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=args.seed,
                    ),
                ),
            ]
        )
    if args.meta_model == "random_forest":
        return RandomForestRegressor(
            n_estimators=args.n_estimators,
            random_state=args.seed,
            min_samples_leaf=2,
            n_jobs=-1,
        )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def score_meta_model(model, x, y, task):
    pred = model.predict(x)
    if task == "classification":
        return {
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "classification_report": classification_report(
                y, pred, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y, pred).tolist(),
        }
    return {
        "mae": mean_absolute_error(y, pred),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": r2_score(y, pred),
    }


def primary_score(metrics: dict, task: str) -> float:
    return metrics["macro_f1"] if task == "classification" else -metrics["mae"]


def question_groups(feature_cols):
    groups = {}
    for col in feature_cols:
        question = col.split("__", 1)[0]
        groups.setdefault(question, []).append(col)
    return groups


def permutation_question_importance(model, val_df, feature_cols, args):
    x_val = val_df[feature_cols].to_numpy()
    y_val = val_df["y_true"].to_numpy()
    base_metrics = score_meta_model(model, x_val, y_val, args.task)
    base_score = primary_score(base_metrics, args.task)
    groups = question_groups(feature_cols)
    rng = np.random.RandomState(args.seed)
    rows = []

    col_to_idx = {col: idx for idx, col in enumerate(feature_cols)}
    for question, cols in groups.items():
        drops = []
        indices = [col_to_idx[col] for col in cols]
        for _ in range(args.permutation_repeats):
            x_perm = x_val.copy()
            shuffled = x_perm[:, indices].copy()
            rng.shuffle(shuffled)
            x_perm[:, indices] = shuffled
            metrics = score_meta_model(model, x_perm, y_val, args.task)
            drops.append(base_score - primary_score(metrics, args.task))
        rows.append(
            {
                "question_id": question,
                "importance": float(np.mean(drops)),
                "importance_std": float(np.std(drops)),
                "base_score": float(base_score),
            }
        )
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def train_meta_model(trainval_features, test_features, feature_cols, args, out_dir: Path):
    speaker_df = trainval_features[["speaker_id", "y_true"]].copy()
    train_idx, dev_idx = make_final_test_split(
        speaker_df, args.task, args.final_dev_size, args.seed + 9999
    )
    meta_train = trainval_features.iloc[train_idx].reset_index(drop=True)
    meta_dev = trainval_features.iloc[dev_idx].reset_index(drop=True)

    base_model = make_meta_model(args)
    base_model.fit(meta_train[feature_cols].to_numpy(), meta_train["y_true"].to_numpy())
    importance_df = permutation_question_importance(base_model, meta_dev, feature_cols, args)
    importance_df.to_csv(out_dir / "question_embedding_importance.csv", index=False)

    questions_ranked = importance_df["question_id"].tolist()
    if not questions_ranked:
        raise ValueError("No ranked questions available for meta-model training.")

    requested_ks = list(range(1, len(questions_ranked) + 1))
    if args.top_k and args.top_k > 0 and args.top_k not in requested_ks:
        requested_ks.append(min(args.top_k, len(questions_ranked)))
    requested_ks = sorted(set(requested_ks))

    topk_dir = out_dir / "topk_meta_models"
    topk_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    all_metrics = {}

    for k in requested_ks:
        selected_questions = questions_ranked[:k]
        selected_feature_cols = [
            col for col in feature_cols if col.split("__", 1)[0] in set(selected_questions)
        ]
        run_name = f"top_{k}"
        model_path = topk_dir / f"{run_name}_meta_model.joblib"
        metrics_path = topk_dir / f"{run_name}_metrics.json"
        predictions_path = topk_dir / f"{run_name}_predictions.csv"
        features_path = topk_dir / f"{run_name}_features.csv"
        questions_path = topk_dir / f"{run_name}_questions.csv"

        if model_path.exists() and metrics_path.exists() and predictions_path.exists():
            print(f"skipping existing {run_name} meta-model")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            final_model = make_meta_model(args)
            final_model.fit(
                trainval_features[selected_feature_cols].to_numpy(),
                trainval_features["y_true"].to_numpy(),
            )
            metrics = score_meta_model(
                final_model,
                test_features[selected_feature_cols].to_numpy(),
                test_features["y_true"].to_numpy(),
                args.task,
            )
            predictions = test_features[["speaker_id", "y_true"]].copy()
            x_test = test_features[selected_feature_cols].to_numpy()
            predictions["y_pred"] = final_model.predict(x_test)
            if args.task == "classification" and hasattr(final_model, "predict_proba"):
                probs = final_model.predict_proba(x_test)
                classes = list(final_model.classes_) if hasattr(final_model, "classes_") else []
                if not classes and hasattr(final_model, "named_steps"):
                    classes = list(final_model.named_steps["model"].classes_)
                for class_idx, class_id in enumerate(classes):
                    predictions[f"prob_{class_id}"] = probs[:, class_idx]

            joblib.dump(final_model, model_path)
            predictions.to_csv(predictions_path, index=False)
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            pd.DataFrame({"feature": selected_feature_cols}).to_csv(features_path, index=False)
            pd.DataFrame({"question_id": selected_questions}).to_csv(questions_path, index=False)

        summary_metric = metrics["macro_f1"] if args.task == "classification" else metrics["mae"]
        rows.append(
            {
                "top_k": k,
                "questions": ",".join(selected_questions),
                "n_features": len(selected_feature_cols),
                "primary_metric": summary_metric,
                **{
                    key: value
                    for key, value in metrics.items()
                    if isinstance(value, (int, float, str))
                },
            }
        )
        all_metrics[run_name] = metrics

    summary_df = pd.DataFrame(rows)
    if args.task == "classification":
        best_idx = summary_df["macro_f1"].idxmax()
        all_idx = summary_df["top_k"].idxmax()
        all_score = summary_df.loc[all_idx, "macro_f1"]
        summary_df["matches_or_beats_all_questions"] = summary_df["macro_f1"] >= all_score
    else:
        best_idx = summary_df["mae"].idxmin()
        all_idx = summary_df["top_k"].idxmax()
        all_score = summary_df.loc[all_idx, "mae"]
        summary_df["matches_or_beats_all_questions"] = summary_df["mae"] <= all_score

    best_row = summary_df.loc[best_idx].to_dict()
    all_row = summary_df.loc[all_idx].to_dict()
    summary_df.to_csv(out_dir / "topk_meta_metrics.csv", index=False)
    (out_dir / "topk_meta_metrics.json").write_text(
        json.dumps(all_metrics, indent=2), encoding="utf-8"
    )

    best_k = int(best_row["top_k"])
    best_prefix = topk_dir / f"top_{best_k}"
    for source, target in [
        (Path(str(best_prefix) + "_meta_model.joblib"), out_dir / "meta_model.joblib"),
        (Path(str(best_prefix) + "_predictions.csv"), out_dir / "meta_test_predictions.csv"),
        (Path(str(best_prefix) + "_metrics.json"), out_dir / "meta_test_metrics.json"),
        (Path(str(best_prefix) + "_features.csv"), out_dir / "selected_embedding_features.csv"),
        (Path(str(best_prefix) + "_questions.csv"), out_dir / "selected_questions.csv"),
    ]:
        if source.exists():
            target.write_bytes(source.read_bytes())

    best_summary = {
        "best_top_k": best_k,
        "best": best_row,
        "all_questions": all_row,
    }
    (out_dir / "best_topk_summary.json").write_text(
        json.dumps(best_summary, indent=2), encoding="utf-8"
    )
    return best_summary


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "question_ensemble_config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )

    questions = [q.upper() for q in args.questions]
    df, metadata = load_examples(
        args.asr_file,
        args.demo_file,
        args.target_column,
        args.task,
        text_mode="question",
        min_text_chars=args.min_text_chars,
        filter_questions=questions,
    )
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    train_idx, test_idx = make_final_test_split(df, args.task, args.test_size, args.seed)
    trainval_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    trainval_df.to_csv(out_dir / "trainval_question_examples.csv", index=False)
    test_df.to_csv(out_dir / "final_test_question_examples.csv", index=False)

    embedding_files = train_question_models(df, trainval_df, test_df, metadata, args, out_dir)
    available_questions = list(embedding_files["trainval"].keys())
    train_features, feature_cols = build_feature_table(
        embedding_files["trainval"], available_questions
    )
    test_features, _ = build_feature_table(embedding_files["test"], available_questions)
    train_features, test_features = align_feature_tables(train_features, test_features, feature_cols)
    train_features.to_csv(out_dir / "meta_trainval_features.csv", index=False)
    test_features.to_csv(out_dir / "meta_test_features.csv", index=False)

    metrics = train_meta_model(train_features, test_features, feature_cols, args, out_dir)
    print(json.dumps({"meta_test": metrics}, indent=2))


if __name__ == "__main__":
    main()
