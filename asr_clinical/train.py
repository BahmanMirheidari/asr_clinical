from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .config import TrainConfig, parse_args
from .data import (
    load_examples,
    make_cv_splits,
    make_final_test_split,
    read_fold_file,
    read_splits_folder,
)
from .importance import question_ablation_importance
from .losses import make_loss
from .metrics import score_predictions
from .model import TextDataset, load_model, load_tokenizer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(df, tokenizer, cfg: TrainConfig, shuffle: bool):
    ds = TextDataset(
        df["text"].tolist(),
        df["label"].tolist(),
        tokenizer,
        cfg.max_length,
        cfg.task,
    )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size if shuffle else cfg.eval_batch_size,
        shuffle=shuffle,
    )


def class_weights_for(labels, num_labels: int, device, cfg: TrainConfig):
    if cfg.task != "classification" or cfg.class_weights == "none":
        return None
    classes = np.arange(num_labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=np.asarray(labels, dtype=int),
    )
    return torch.tensor(weights, dtype=torch.float, device=device)


def train_one_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: TrainConfig,
    metadata: dict,
    fold_dir: Path,
):
    device = choose_device()
    tokenizer = load_tokenizer(cfg.model_name)
    num_labels = len(metadata["labels"]) if cfg.task == "classification" else 1
    model = load_model(cfg.model_name, cfg.task, num_labels, metadata).to(device)

    train_loader = make_loader(train_df, tokenizer, cfg, shuffle=True)
    val_loader = make_loader(val_df, tokenizer, cfg, shuffle=False)

    class_weight_tensor = class_weights_for(train_df["label"], num_labels, device, cfg)
    loss_fn = make_loss(cfg.task, cfg.loss, class_weight_tensor, cfg.focal_gamma)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = max(1, len(train_loader) * cfg.epochs)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_score = -math.inf
    best_state = None
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            logits = outputs.logits.squeeze(-1) if cfg.task == "regression" else outputs.logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))

        pred_df = predict(model, tokenizer, val_df, cfg, metadata, device)
        metrics = score_predictions(
            pred_df,
            cfg.task,
            cfg.aggregate_level,
            metadata.get("labels"),
        )
        score = metrics["macro_f1"] if cfg.task == "classification" else -metrics["mae"]
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "val_score": score,
                    "metrics": metrics,
                },
                indent=2,
            )
        )

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    fold_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(fold_dir / "model")
    tokenizer.save_pretrained(fold_dir / "model")
    pred_df = predict(model, tokenizer, val_df, cfg, metadata, device)
    pred_df.to_csv(fold_dir / "predictions.csv", index=False)
    metrics = score_predictions(pred_df, cfg.task, cfg.aggregate_level, metadata.get("labels"))
    (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics, pred_df, model, tokenizer, device


@torch.no_grad()
def predict(model, tokenizer, df: pd.DataFrame, cfg: TrainConfig, metadata: dict, device):
    model.eval()
    loader = make_loader(df, tokenizer, cfg, shuffle=False)
    rows = []
    offset = 0
    for batch in loader:
        batch_size = len(batch["labels"])
        labels = batch.pop("labels").numpy()
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits.detach().cpu()
        batch_meta = df.iloc[offset : offset + batch_size].reset_index(drop=True)
        offset += batch_size

        if cfg.task == "classification":
            probs = torch.softmax(logits, dim=-1).numpy()
            preds = probs.argmax(axis=1)
            for i in range(batch_size):
                row = {
                    "utterance_id": batch_meta.loc[i, "utterance_id"],
                    "speaker_id": batch_meta.loc[i, "speaker_id"],
                    "session_id": batch_meta.loc[i, "session_id"],
                    "question_id": batch_meta.loc[i, "question_id"],
                    "y_true": int(labels[i]),
                    "y_pred": int(preds[i]),
                }
                for label_idx, label_name in enumerate(metadata["labels"]):
                    row[f"prob_{label_name}"] = float(probs[i, label_idx])
                rows.append(row)
        else:
            values = logits.squeeze(-1).numpy()
            for i in range(batch_size):
                rows.append(
                    {
                        "utterance_id": batch_meta.loc[i, "utterance_id"],
                        "speaker_id": batch_meta.loc[i, "speaker_id"],
                        "session_id": batch_meta.loc[i, "session_id"],
                        "question_id": batch_meta.loc[i, "question_id"],
                        "y_true": float(labels[i]),
                        "y_pred": float(values[i]),
                    }
                )
    return pd.DataFrame(rows)


def train_final_model(train_df, test_df, cfg: TrainConfig, metadata: dict, out_dir: Path):
    final_dir = out_dir / "final_model"
    subtrain_idx, dev_idx = make_final_test_split(
        train_df, cfg.task, cfg.final_dev_size, cfg.seed + 1000
    )
    _, _, model, tokenizer, device = train_one_fold(
        train_df.iloc[subtrain_idx].reset_index(drop=True),
        train_df.iloc[dev_idx].reset_index(drop=True),
        cfg,
        metadata,
        final_dir,
    )
    pred_df = predict(model, tokenizer, test_df, cfg, metadata, device)
    pred_df.to_csv(out_dir / "final_test_predictions.csv", index=False)
    metrics = score_predictions(pred_df, cfg.task, cfg.aggregate_level, metadata.get("labels"))
    (out_dir / "final_test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    if cfg.question_importance and cfg.text_mode == "question":
        imp = question_ablation_importance(
            pred_df, cfg.task, cfg.aggregate_level, metadata.get("labels")
        )
        imp.to_csv(out_dir / "question_importance.csv", index=False)
    return metrics


def run_external_splits(df, cfg: TrainConfig, metadata: dict, out_dir: Path):
    folds = read_splits_folder(cfg.splits_folder, df)
    split_metrics = []
    split_predictions = []

    for fold in folds:
        fold_idx = fold["fold"]
        fold_dir = out_dir / f"fold_{fold_idx}"
        train_df = df.iloc[fold["train_idx"]].reset_index(drop=True)
        val_df = df.iloc[fold["val_idx"]].reset_index(drop=True)
        eval_idx = fold["test_idx"] if fold["test_idx"] is not None else fold["val_idx"]
        eval_name = "test" if fold["test_idx"] is not None else "val"
        eval_df = df.iloc[eval_idx].reset_index(drop=True)

        print(
            f"fold {fold_idx}: train={len(train_df)} val={len(val_df)} "
            f"{eval_name}={len(eval_df)}"
        )
        val_metrics, val_pred_df, model, tokenizer, device = train_one_fold(
            train_df,
            val_df,
            cfg,
            metadata,
            fold_dir,
        )
        val_metrics["fold"] = fold_idx
        val_metrics["split"] = "val"
        split_metrics.append(val_metrics)
        val_pred_df["fold"] = fold_idx
        val_pred_df["split"] = "val"
        val_pred_df.to_csv(fold_dir / "val_predictions.csv", index=False)

        if eval_name == "test":
            test_pred_df = predict(model, tokenizer, eval_df, cfg, metadata, device)
            test_metrics = score_predictions(
                test_pred_df, cfg.task, cfg.aggregate_level, metadata.get("labels")
            )
            test_metrics["fold"] = fold_idx
            test_metrics["split"] = "test"
            split_metrics.append(test_metrics)
            test_pred_df["fold"] = fold_idx
            test_pred_df["split"] = "test"
            test_pred_df.to_csv(fold_dir / "test_predictions.csv", index=False)
            (fold_dir / "test_metrics.json").write_text(
                json.dumps(test_metrics, indent=2), encoding="utf-8"
            )
            split_predictions.append(test_pred_df)
        else:
            split_predictions.append(val_pred_df)

    pd.DataFrame(split_metrics).to_json(
        out_dir / "split_metrics.json", orient="records", indent=2
    )
    all_preds = pd.concat(split_predictions, ignore_index=True)
    all_preds.to_csv(out_dir / "cv_predictions.csv", index=False)
    if cfg.question_importance and cfg.text_mode == "question":
        imp = question_ablation_importance(
            all_preds, cfg.task, cfg.aggregate_level, metadata.get("labels")
        )
        imp.to_csv(out_dir / "cv_question_importance.csv", index=False)


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(cfg.__dict__, indent=2), encoding="utf-8"
    )

    df, metadata = load_examples(
        cfg.asr_file,
        cfg.demo_file,
        cfg.target_column,
        cfg.task,
        cfg.text_mode,
        cfg.min_text_chars,
    )
    metadata["n_examples"] = len(df)
    metadata["n_speakers"] = int(df["speaker_id"].nunique())
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if cfg.splits_folder:
        if cfg.folds_file:
            raise ValueError("Use either --splits-folder or --folds-file, not both.")
        run_external_splits(df, cfg, metadata, out_dir)
        return

    train_idx, test_idx = make_final_test_split(df, cfg.task, cfg.test_size, cfg.seed)
    trainval_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    trainval_df.to_csv(out_dir / "trainval_examples.csv", index=False)
    test_df.to_csv(out_dir / "final_test_examples.csv", index=False)

    if cfg.folds_file:
        splits = read_fold_file(cfg.folds_file, trainval_df)
    else:
        splits = make_cv_splits(trainval_df, cfg.task, cfg.num_folds, cfg.seed)

    cv_metrics = []
    cv_predictions = []
    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        print(f"fold {fold_idx}: train={len(tr_idx)} val={len(val_idx)}")
        metrics, pred_df, _, _, _ = train_one_fold(
            trainval_df.iloc[tr_idx].reset_index(drop=True),
            trainval_df.iloc[val_idx].reset_index(drop=True),
            cfg,
            metadata,
            out_dir / f"fold_{fold_idx}",
        )
        metrics["fold"] = fold_idx
        cv_metrics.append(metrics)
        pred_df["fold"] = fold_idx
        cv_predictions.append(pred_df)

    pd.DataFrame(cv_metrics).to_json(out_dir / "cv_metrics.json", orient="records", indent=2)
    all_cv_preds = pd.concat(cv_predictions, ignore_index=True)
    all_cv_preds.to_csv(out_dir / "cv_predictions.csv", index=False)
    if cfg.question_importance and cfg.text_mode == "question":
        imp = question_ablation_importance(
            all_cv_preds, cfg.task, cfg.aggregate_level, metadata.get("labels")
        )
        imp.to_csv(out_dir / "cv_question_importance.csv", index=False)

    print("training final model on trainval and evaluating final held-out test")
    final_metrics = train_final_model(trainval_df, test_df, cfg, metadata, out_dir)
    print(json.dumps({"final_test": final_metrics}, indent=2))


if __name__ == "__main__":
    main()
