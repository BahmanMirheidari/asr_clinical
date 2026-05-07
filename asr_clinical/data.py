from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedGroupKFold,
    GroupKFold,
    GroupShuffleSplit,
    train_test_split,
)


QUESTION_RE = re.compile(r"_(Q\d+)$", re.IGNORECASE)
SPLIT_FILE_RE = re.compile(r"fold(\d+)_(train|val|test)\.csv$", re.IGNORECASE)


def parse_utterance_id(utterance_id: str) -> tuple[str, str, str]:
    if '_' in utterance_id: 
        parts = utterance_id.strip().split("_")
        if len(parts) < 3:
            raise ValueError(f"Cannot parse utterance id: {utterance_id}")
        speaker_id = "_".join(parts[:2])
        question_id = parts[-1].upper()
        if not QUESTION_RE.search(utterance_id):
            question_id = "UNK"
        session_id = "_".join(parts[:-1])
    else:
        parts = utterance_id.strip().split("-")
        if len(parts) < 3:
            raise ValueError(f"Cannot parse utterance id: {utterance_id}")
        speaker_id = parts[0].split(".")[0]
        question_id = parts[-1].upper()
        if not QUESTION_RE.search(utterance_id):
            question_id = "UNK"
        session_id = "-".join(parts[:-1])

    return speaker_id, session_id, question_id


def read_asr_file(path: str | Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    first = True
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if not row or first:
                first = False
                continue
            if len(row) < 2:
                continue
            utt_id = row[0].strip()
            text = row[-1].strip()
            if utt_id.lower() in {"utt_id", "utterance_id"}:
                continue
            speaker_id, session_id, question_id = parse_utterance_id(utt_id)
            rows.append(
                {
                    "utterance_id": utt_id,
                    "speaker_id": speaker_id,
                    "session_id": session_id,
                    "question_id": question_id,
                    "text": text,
                }
            )
    return pd.DataFrame(rows)


def load_examples(
    asr_file: str,
    demo_file: str,
    target_column: str,
    task: str,
    text_mode: str,
    min_text_chars: int = 1,
    filter_questions: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    asr = read_asr_file(asr_file)
    if filter_questions:
        keep_questions = {q.strip().upper() for q in filter_questions if q.strip()}
        asr = asr[asr["question_id"].isin(keep_questions)].copy()
        if asr.empty:
            raise ValueError(
                "No ASR rows remained after --filter-questions. "
                f"Requested questions: {sorted(keep_questions)}"
            )

    demo = pd.read_csv(demo_file)
    if "speaker_id" not in demo.columns:
        raise ValueError("demo CSV must contain a speaker_id column")
    if target_column not in demo.columns:
        raise ValueError(f"demo CSV does not contain target column: {target_column}")

    merged = asr.merge(demo[["speaker_id", target_column]], on="speaker_id", how="left")
    merged = merged[merged["text"].fillna("").str.len() >= min_text_chars].copy()
    merged = merged[merged[target_column].notna()].copy()

    metadata: dict = {
        "filter_questions": sorted(keep_questions) if filter_questions else "all"
    }
    if task == "classification":
        labels = sorted(merged[target_column].astype(str).unique().tolist())
        label2id = {label: idx for idx, label in enumerate(labels)}
        id2label = {idx: label for label, idx in label2id.items()}
        merged["label"] = merged[target_column].astype(str).map(label2id).astype(int)
        metadata.update({"labels": labels, "label2id": label2id, "id2label": id2label})
    else:
        merged["label"] = pd.to_numeric(merged[target_column], errors="coerce")
        merged = merged[merged["label"].notna()].copy()
        merged["label"] = merged["label"].astype(float)

    if text_mode == "session_concat":
        merged = make_session_examples(merged)

    merged = merged.reset_index(drop=True)
    if merged.empty:
        raise ValueError("No usable examples after merging ASR and demographics.")
    return merged, metadata


def make_session_examples(df: pd.DataFrame) -> pd.DataFrame:
    def join_questions(group: pd.DataFrame) -> str:
        group = group.sort_values("question_id")
        return "\n".join(
            f"{row.question_id}: {row.text}" for row in group.itertuples(index=False)
        )

    rows = []
    for (speaker_id, session_id), group in df.groupby(["speaker_id", "session_id"]):
        rows.append(
            {
                "speaker_id": speaker_id,
                "session_id": session_id,
                "utterance_id": session_id,
                "question_id": "SESSION",
                "text": join_questions(group),
                "label": group["label"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def make_final_test_split(df: pd.DataFrame, task: str, test_size: float, seed: int):
    groups = df["speaker_id"].to_numpy()
    y = df["label"].to_numpy()
    if task == "classification":
        speaker_labels = df.groupby("speaker_id")["label"].first().reset_index()
        counts = speaker_labels["label"].value_counts()
        stratify = speaker_labels["label"] if counts.min() >= 2 else None
        train_speakers, test_speakers = train_test_split(
            speaker_labels["speaker_id"],
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
        train_mask = df["speaker_id"].isin(set(train_speakers))
        test_mask = df["speaker_id"].isin(set(test_speakers))
        return np.flatnonzero(train_mask.to_numpy()), np.flatnonzero(test_mask.to_numpy())
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, y, groups))
    return train_idx, test_idx


def make_cv_splits(df: pd.DataFrame, task: str, num_folds: int, seed: int):
    groups = df["speaker_id"].to_numpy()
    y = df["label"].to_numpy()
    if task == "classification":
        splitter = StratifiedGroupKFold(
            n_splits=num_folds, shuffle=True, random_state=seed
        )
        return list(splitter.split(df, y, groups))
    splitter = GroupKFold(n_splits=num_folds)
    return list(splitter.split(df, y, groups))


def read_fold_file(path: str | Path, df: pd.DataFrame):
    folds = pd.read_csv(path)
    required = {"speaker_id", "fold"}
    if not required.issubset(folds.columns):
        raise ValueError("folds file must contain speaker_id and fold columns")
    speaker_to_fold = dict(zip(folds["speaker_id"], folds["fold"]))
    example_folds = df["speaker_id"].map(speaker_to_fold)
    if example_folds.isna().any():
        missing = df.loc[example_folds.isna(), "speaker_id"].unique()[:10]
        raise ValueError(f"Speakers missing from fold file, examples: {missing}")
    splits = []
    for fold in sorted(example_folds.unique()):
        val_idx = np.flatnonzero(example_folds.to_numpy() == fold)
        train_idx = np.flatnonzero(example_folds.to_numpy() != fold)
        splits.append((train_idx, val_idx))
    return splits


def _speaker_indices(df: pd.DataFrame, speakers: set[str], split_name: str):
    known = set(df["speaker_id"].astype(str).unique())
    missing = sorted(speakers - known)
    if missing:
        raise ValueError(
            f"{split_name} contains speakers not found after ASR/demo merge: {missing[:10]}"
        )
    mask = df["speaker_id"].astype(str).isin(speakers)
    return np.flatnonzero(mask.to_numpy())


def _read_split_speakers(path: Path) -> set[str]:
    split_df = pd.read_csv(path)
    if "speaker_id" not in split_df.columns:
        raise ValueError(f"{path} must contain a speaker_id column")
    speakers = split_df["speaker_id"].dropna().astype(str).unique().tolist()
    if not speakers:
        raise ValueError(f"{path} does not contain any speakers")
    return set(speakers)


def read_splits_folder(path: str | Path, df: pd.DataFrame):
    folder = Path(path)
    if not folder.exists():
        raise FileNotFoundError(f"splits folder does not exist: {folder}")

    files_by_fold: dict[int, dict[str, Path]] = {}
    for split_file in folder.glob("fold*_*.csv"):
        match = SPLIT_FILE_RE.match(split_file.name)
        if not match:
            continue
        fold_idx = int(match.group(1))
        split_name = match.group(2).lower()
        files_by_fold.setdefault(fold_idx, {})[split_name] = split_file

    if not files_by_fold:
        raise ValueError(
            f"No files matching foldN_train.csv/foldN_val.csv/foldN_test.csv in {folder}"
        )

    folds = []
    for fold_idx in sorted(files_by_fold):
        split_files = files_by_fold[fold_idx]
        if "train" not in split_files or "val" not in split_files:
            raise ValueError(
                f"fold{fold_idx} must include fold{fold_idx}_train.csv and "
                f"fold{fold_idx}_val.csv"
            )

        speakers = {
            split_name: _read_split_speakers(split_path)
            for split_name, split_path in split_files.items()
        }
        overlap_checks = [("train", "val"), ("train", "test"), ("val", "test")]
        for left, right in overlap_checks:
            if left in speakers and right in speakers:
                overlap = speakers[left] & speakers[right]
                if overlap:
                    raise ValueError(
                        f"fold{fold_idx} has speaker leakage between {left} and {right}: "
                        f"{sorted(overlap)[:10]}"
                    )

        fold = {
            "fold": fold_idx,
            "train_idx": _speaker_indices(df, speakers["train"], f"fold{fold_idx}_train"),
            "val_idx": _speaker_indices(df, speakers["val"], f"fold{fold_idx}_val"),
            "test_idx": None,
        }
        if "test" in speakers:
            fold["test_idx"] = _speaker_indices(df, speakers["test"], f"fold{fold_idx}_test")
        folds.append(fold)
    return folds
