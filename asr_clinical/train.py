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
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from .config import TrainConfig, parse_args
from .data import (
    load_examples,
    make_cv_splits,
    make_final_test_split,
    read_fold_file,
    read_splits_folder,
)
from .importance import question_ablation_importance
from .losses import build_y_grid, decode_expected_value, make_loss, raft_expected_value_loss
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


def saved_model_exists(model_dir: Path) -> bool:
    return (model_dir / "config.json").exists()


def load_saved_model(model_dir: Path, device):
    if (model_dir / "tract_config.json").exists():
        from .tract import load_tract_model
        return load_tract_model(model_dir, device)
    tokenizer = load_tokenizer(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return model, tokenizer


def maybe_load_trained_fold(
    fold_dir: Path,
    val_df: pd.DataFrame,
    cfg: TrainConfig,
    metadata: dict,
):
    model_dir = fold_dir / "model"
    pred_path = fold_dir / "predictions.csv"
    metrics_path = fold_dir / "metrics.json"
    if not saved_model_exists(model_dir):
        return None
    if cfg.use_tract and not (model_dir / "tract_config.json").exists():
        return None
    if cfg.task == "regression" and cfg.use_raft and not (model_dir / "raft_config.json").exists():
        # A scalar-head checkpoint cannot satisfy a requested RAFT run.
        return None

    print(f"skipping existing model in {fold_dir}")
    device = choose_device()
    if cfg.use_tract:
        from .tract import load_tract_model
        model, tokenizer = load_tract_model(model_dir, device)
        cfg.raft_y_grid = model.tract_y_grid
        cfg.raft_temperature = model.tract_temperature
    else:
        model, tokenizer = load_saved_model(model_dir, device)
    raft_path = model_dir / "raft_config.json"
    if cfg.task == "regression" and raft_path.exists():
        raft_cfg = json.loads(raft_path.read_text(encoding="utf-8"))
        cfg.use_raft = True
        cfg.raft_y_grid = raft_cfg["y_grid"]
        cfg.raft_temperature = float(raft_cfg.get("temperature", cfg.raft_temperature))
    if pred_path.exists() and metrics_path.exists():
        pred_df = pd.read_csv(pred_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        pred_df = predict(model, tokenizer, val_df, cfg, metadata, device)
        pred_df.to_csv(pred_path, index=False)
        metrics = score_predictions(
            pred_df, cfg.task, cfg.aggregate_level, metadata.get("labels")
        )
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics, pred_df, model, tokenizer, device


def train_one_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: TrainConfig,
    metadata: dict,
    fold_dir: Path,
):
    # ── regression: normalise labels if not already normalised ──────────
    if cfg.task == "regression":
        _ymean = float(getattr(cfg, "y_mean", 0.0))
        _ystd  = float(getattr(cfg, "y_std", 1.0))
        normalize_targets = getattr(cfg, "normalize_regression_targets", False)
        # Preserve the legacy heuristic for callers that do not provide an
        # explicit target-scaling policy.
        if (normalize_targets or
                (_ystd > 0 and abs(float(train_df["label"].mean())) > 5.0)):
            train_df = train_df.copy()
            val_df = val_df.copy()
            train_df["label"] = (train_df["label"] - _ymean) / _ystd
            val_df["label"]   = (val_df["label"]   - _ymean) / _ystd
            print(f"[train_one_fold] normalised labels: "
                  f"train mean={train_df['label'].mean():.3f} "
                  f"std={train_df['label'].std():.3f}, "
                  f"val mean={val_df['label'].mean():.3f} "
                  f"std={val_df['label'].std():.3f}")
        else:
            print(f"[train_one_fold] labels already normalised "
                  f"(train mean={train_df['label'].mean():.3f}, "
                  f"std={train_df['label'].std():.3f})")
    # ────────────────────────────────────────────────────────────────────

    cached = (None if getattr(cfg, "force_retrain", False) else
              maybe_load_trained_fold(fold_dir, val_df, cfg, metadata))
    if cached is not None:
        return cached

    if cfg.use_tract:
        if cfg.task != "regression":
            raise ValueError("--use-tract is regression-only")
        if "rationale" not in train_df.columns:
            raise ValueError("--use-tract requires --rationale-file with reviewed rationales")
        return train_one_tract_fold(train_df, val_df, cfg, metadata, fold_dir)

    device = choose_device()
    tokenizer = load_tokenizer(cfg.model_name)
    num_labels = len(metadata["labels"]) if cfg.task == "classification" else 1
    if cfg.task == "regression" and cfg.use_raft:
        cfg.raft_y_grid = build_y_grid(
            train_df["label"].to_numpy(), cfg.y_grid_size,
            cfg.y_grid_min, cfg.y_grid_max).tolist()
        num_labels = len(cfg.raft_y_grid)
    model = load_model(cfg.model_name, cfg.task, cfg.dropout_rate, num_labels,
                       metadata, use_raft=(cfg.task == "regression" and cfg.use_raft)).to(device)
    if cfg.task == "regression" and cfg.use_raft:
        model.config.raft_y_grid = cfg.raft_y_grid
        model.config.raft_temperature = cfg.raft_temperature

    train_loader = make_loader(train_df, tokenizer, cfg, shuffle=True)
    val_loader = make_loader(val_df, tokenizer, cfg, shuffle=False)

    class_weight_tensor = class_weights_for(train_df["label"], num_labels, device, cfg)
    # NEW: Use label smoothing if classification and loss is 'ce'
    if cfg.task == "classification" and cfg.loss == "ce" and cfg.label_smoothing > 0:
        # Override loss function with built-in label smoothing
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=class_weight_tensor,
            label_smoothing=cfg.label_smoothing
        )
    else:
        # Use the custom loss (focal or ce without smoothing)
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
            logits = outputs.logits if (cfg.task == "regression" and cfg.use_raft) else (
                outputs.logits.squeeze(-1) if cfg.task == "regression" else outputs.logits)
            if cfg.task == "regression" and cfg.use_raft:
                loss = cfg.raft_weight * raft_expected_value_loss(
                    logits, labels, torch.tensor(cfg.raft_y_grid, device=device),
                    cfg.raft_temperature)
            else:
                loss = loss_fn(logits, labels)
            loss.backward()
            # NEW: Use cfg.gradient_clip_val instead of hardcoded 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_val)
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
    if cfg.task == "regression" and cfg.use_raft:
        (fold_dir / "model" / "raft_config.json").write_text(json.dumps({
            "y_grid": cfg.raft_y_grid,
            "temperature": cfg.raft_temperature,
            "weight": cfg.raft_weight,
        }, indent=2), encoding="utf-8")
    pred_df = predict(model, tokenizer, val_df, cfg, metadata, device)
    pred_df.to_csv(fold_dir / "predictions.csv", index=False)
    metrics = score_predictions(pred_df, cfg.task, cfg.aggregate_level, metadata.get("labels"))
    (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics, pred_df, model, tokenizer, device


def train_one_tract_fold(train_df, val_df, cfg, metadata, fold_dir):
    from .tract import fit_two_stage_tract

    device = choose_device()
    model, tokenizer, grid = fit_two_stage_tract(
        train_df, cfg, fold_dir, device)
    cfg.raft_y_grid = grid
    fold_dir.mkdir(parents=True, exist_ok=True)
    head_state = model.tract_score_head.state_dict()
    delattr(model, "tract_score_head")
    model.save_pretrained(fold_dir / "model")
    tokenizer.save_pretrained(fold_dir / "model")
    torch.save(head_state, fold_dir / "model" / "tract_score_head.pt")
    (fold_dir / "model" / "tract_config.json").write_text(json.dumps({
        "y_grid": grid,
        "temperature": cfg.raft_temperature,
        "generation_tokens": cfg.tract_generation_tokens,
    }, indent=2), encoding="utf-8")
    from .tract import ExpectedScoreHead, _hidden_size
    model.tract_score_head = ExpectedScoreHead(
        _hidden_size(model.config), len(grid)).to(device)
    model.tract_score_head.load_state_dict(head_state)
    pred_df = predict(model, tokenizer, val_df, cfg, metadata, device)
    pred_df.to_csv(fold_dir / "predictions.csv", index=False)
    metrics = score_predictions(pred_df, cfg.task, cfg.aggregate_level,
                                metadata.get("labels"))
    (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2),
                                           encoding="utf-8")
    return metrics, pred_df, model, tokenizer, device

@torch.no_grad()
def predict(model, tokenizer, df: pd.DataFrame, cfg: TrainConfig, metadata: dict, device):
    model.eval()
    if cfg.use_tract:
        from .tract import predict_tract
        values, rationales = predict_tract(
            model, tokenizer, df["text"].astype(str).tolist(),
            cfg.eval_batch_size, cfg.max_length, device,
            y_grid=cfg.raft_y_grid, temperature=cfg.raft_temperature,
            generation_tokens=cfg.tract_generation_tokens,
        )
        return pd.DataFrame({
            "utterance_id": df["utterance_id"].values,
            "speaker_id": df["speaker_id"].values,
            "session_id": df["session_id"].values,
            "question_id": df["question_id"].values,
            "y_true": df["label"].astype(float).values,
            "y_pred": values,
            "rationale_pred": rationales,
        })
    if cfg.task == "regression" and cfg.use_raft and cfg.raft_y_grid is None:
        saved_grid = getattr(model.config, "raft_y_grid", None)
        if saved_grid is not None:
            cfg.raft_y_grid = saved_grid
            cfg.raft_temperature = float(getattr(
                model.config, "raft_temperature", cfg.raft_temperature))
    if cfg.task == "regression" and cfg.use_raft and cfg.raft_y_grid is None:
        raft_path = Path(getattr(model.config, "_name_or_path", "")) / "raft_config.json"
        if raft_path.exists():
            raft_cfg = json.loads(raft_path.read_text(encoding="utf-8"))
            cfg.raft_y_grid = raft_cfg["y_grid"]
            cfg.raft_temperature = float(raft_cfg.get("temperature", cfg.raft_temperature))
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
            if cfg.use_raft:
                grid = cfg.raft_y_grid
                if grid is None:
                    raise ValueError("RAFT predictions require the saved y_grid")
                values = decode_expected_value(
                    logits, torch.tensor(grid), cfg.raft_temperature).numpy()
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
    final_pred_path = out_dir / "final_test_predictions.csv"
    final_metrics_path = out_dir / "final_test_metrics.json"
    if saved_model_exists(final_dir / "model"):
        print("skipping existing final_model")
        if final_pred_path.exists() and final_metrics_path.exists():
            return json.loads(final_metrics_path.read_text(encoding="utf-8"))
        device = choose_device()
        model, tokenizer = load_saved_model(final_dir / "model", device)
        pred_df = predict(model, tokenizer, test_df, cfg, metadata, device)
        pred_df.to_csv(final_pred_path, index=False)
        metrics = score_predictions(
            pred_df, cfg.task, cfg.aggregate_level, metadata.get("labels")
        )
        final_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics

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
    missing_records = []
    speaker_status = metadata.get("speaker_status", {})

    for fold in folds:
        fold_idx = fold["fold"]
        fold_dir = out_dir / f"fold_{fold_idx}"
        train_df = df.iloc[fold["train_idx"]].reset_index(drop=True)
        val_df = df.iloc[fold["val_idx"]].reset_index(drop=True)
        eval_idx = fold["test_idx"] if fold["test_idx"] is not None else fold["val_idx"]
        eval_name = "test" if fold["test_idx"] is not None else "val"
        eval_df = df.iloc[eval_idx].reset_index(drop=True)
        removed = fold.get("removed_from_train") or []
        missing_speakers = fold.get("missing_speakers") or {}

        print(
            f"fold {fold_idx}: train={len(train_df)} val={len(val_df)} "
            f"{eval_name}={len(eval_df)}"
        )
        for split_name, missing in missing_speakers.items():
            if missing:
                print(
                    f"fold {fold_idx}: warning: ignored {len(missing)} {split_name} "
                    f"speaker(s) not found after ASR/demo merge: {missing[:10]}"
                )
                for speaker_id in missing:
                    missing_records.append(
                        {
                            "fold": fold_idx,
                            "split": split_name,
                            "speaker_id": speaker_id,
                            "reason": speaker_status.get(
                                str(speaker_id), "not_in_asr_or_demo_or_id_mismatch"
                            ),
                        }
                    )
        if removed:
            print(
                f"fold {fold_idx}: removed {len(removed)} leaked speaker(s) "
                f"from train: {removed[:10]}"
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
            test_pred_path = fold_dir / "test_predictions.csv"
            test_metrics_path = fold_dir / "test_metrics.json"
            if test_pred_path.exists() and test_metrics_path.exists():
                print(f"fold {fold_idx}: skipping existing test predictions")
                test_pred_df = pd.read_csv(test_pred_path)
                test_metrics = json.loads(test_metrics_path.read_text(encoding="utf-8"))
            else:
                test_pred_df = predict(model, tokenizer, eval_df, cfg, metadata, device)
                test_metrics = score_predictions(
                    test_pred_df, cfg.task, cfg.aggregate_level, metadata.get("labels")
                )
                test_pred_df.to_csv(test_pred_path, index=False)
                test_metrics_path.write_text(
                    json.dumps(test_metrics, indent=2), encoding="utf-8"
                )
            test_metrics["fold"] = fold_idx
            test_metrics["split"] = "test"
            split_metrics.append(test_metrics)
            test_pred_df["fold"] = fold_idx
            test_pred_df["split"] = "test"
            split_predictions.append(test_pred_df)
        else:
            split_predictions.append(val_pred_df)

    pd.DataFrame(split_metrics).to_json(
        out_dir / "split_metrics.json", orient="records", indent=2
    )
    if missing_records:
        pd.DataFrame(missing_records).to_csv(
            out_dir / "external_split_missing_speakers.csv", index=False
        )
    all_preds = pd.concat(split_predictions, ignore_index=True)
    all_preds.to_csv(out_dir / "cv_predictions.csv", index=False)
    if cfg.question_importance and cfg.text_mode == "question":
        imp = question_ablation_importance(
            all_preds, cfg.task, cfg.aggregate_level, metadata.get("labels")
        )
        imp.to_csv(out_dir / "cv_question_importance.csv", index=False)

    final_train_idx, final_test_idx = make_final_test_split(
        df, cfg.task, cfg.test_size, cfg.seed
    )
    final_train_df = df.iloc[final_train_idx].reset_index(drop=True)
    final_test_df = df.iloc[final_test_idx].reset_index(drop=True)
    final_train_df.to_csv(out_dir / "trainval_examples.csv", index=False)
    final_test_df.to_csv(out_dir / "final_test_examples.csv", index=False)
    print("training final model on trainval and evaluating final held-out test")
    final_metrics = train_final_model(final_train_df, final_test_df, cfg, metadata, out_dir)
    print(json.dumps({"final_test": final_metrics}, indent=2))


def main():
    cfg = parse_args()
    if cfg.use_tract and cfg.task != "regression":
        raise ValueError("--use-tract is regression-only")
    if cfg.use_tract and not cfg.rationale_file:
        raise ValueError("--use-tract requires --rationale-file with reviewed rationales")
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
        cfg.filter_questions,
        rationale_file=cfg.rationale_file if cfg.use_tract else None,
        rationale_column=cfg.rationale_column,
    )
    metadata["n_examples"] = len(df)
    metadata["n_speakers"] = int(df["speaker_id"].nunique())
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if "speaker_status" in metadata:
        pd.DataFrame(
            [
                {"speaker_id": speaker_id, "status": status}
                for speaker_id, status in metadata["speaker_status"].items()
            ]
        ).to_csv(out_dir / "speaker_diagnostics.csv", index=False)

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
