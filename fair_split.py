from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def clean_data(df, args):
    df = df.dropna(subset=[args.speaker_id, args.age, args.gender, args.label]).copy()
    df[args.speaker_id] = df[args.speaker_id].astype(str).str.strip()
    df[args.gender] = df[args.gender].astype(str).str.strip().str.lower()
    df[args.label] = df[args.label].astype(str).str.strip()
    df = df[df[args.gender].isin(["male", "female"])].copy()
    df[args.age] = pd.to_numeric(df[args.age], errors="coerce")
    df = df.dropna(subset=[args.age]).copy()
    df = df[df[args.age] >= args.age_bins[0]].copy()
    return df[[args.speaker_id, args.label, args.age, args.gender]].copy()


def collapse_to_speaker_level(df, args):
    rows = []
    dropped = []
    for speaker_id, group in df.groupby(args.speaker_id):
        labels = group[args.label].dropna().astype(str).unique()
        genders = group[args.gender].dropna().astype(str).unique()
        if len(labels) != 1 or len(genders) != 1:
            dropped.append(speaker_id)
            continue
        rows.append(
            {
                args.speaker_id: speaker_id,
                args.label: labels[0],
                args.gender: genders[0],
                args.age: float(group[args.age].median()),
                "source_rows": len(group),
            }
        )

    if dropped:
        print(
            f"  Dropped {len(dropped)} speaker(s) with inconsistent label/gender: "
            f"{dropped[:10]}"
        )
    speaker_df = pd.DataFrame(rows)
    if speaker_df.empty:
        raise ValueError("No speakers remained after speaker-level collapse.")
    return speaker_df


def create_age_bins(df, age_col, bins):
    df = df.copy()
    df["age_bin"] = pd.cut(df[age_col], bins=bins, labels=False, include_lowest=True)
    return df.dropna(subset=["age_bin"]).copy()


def create_strata(df, label_col, gender_col):
    df = df.copy()
    df["strata"] = (
        df[label_col].astype(str)
        + "_"
        + df[gender_col].astype(str)
        + "_"
        + df["age_bin"].astype(int).astype(str)
    )
    return df


def remove_rare_strata(df, args):
    df = df.copy()
    min_count = max(2, args.n_splits)
    counts = df["strata"].value_counts()
    rare = counts[counts < min_count]
    if not rare.empty:
        print(f"  Removing {len(rare)} strata with <{min_count} speakers:")
        for stratum, count in rare.items():
            print(f"    - {stratum}: {count} speaker(s)")
        df = df[df["strata"].isin(counts[counts >= min_count].index)].copy()
    if df.empty:
        raise ValueError("No speakers remained after rare-strata filtering.")
    return df


def split_validation(train_df, val_ratio, seed):
    if val_ratio <= 0 or len(train_df) < 2:
        return train_df.copy(), pd.DataFrame(columns=train_df.columns)

    n_val = max(1, int(round(len(train_df) * val_ratio)))
    n_classes = train_df["strata"].nunique()

    try:
        if n_val < n_classes:
            raise ValueError(
                f"validation size {n_val} is smaller than number of strata {n_classes}"
            )
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_ratio,
            random_state=seed,
        )
        train_idx, val_idx = next(splitter.split(train_df, train_df["strata"]))
    except ValueError as exc:
        print(f"  Stratified speaker-level val split failed: {exc}; using random split")
        rng = np.random.RandomState(seed)
        val_idx = rng.choice(len(train_df), size=n_val, replace=False)
        train_idx = np.array([i for i in range(len(train_df)) if i not in set(val_idx)])

    return train_df.iloc[train_idx].copy(), train_df.iloc[val_idx].copy()


def assert_no_speaker_overlap(train_df, val_df, test_df, speaker_col, fold):
    train = set(train_df[speaker_col].astype(str))
    val = set(val_df[speaker_col].astype(str)) if not val_df.empty else set()
    test = set(test_df[speaker_col].astype(str))
    overlaps = {
        "train_val": train & val,
        "train_test": train & test,
        "val_test": val & test,
    }
    bad = {name: sorted(values) for name, values in overlaps.items() if values}
    if bad:
        raise AssertionError(f"Fold {fold} speaker leakage detected: {bad}")


def generate_summary(train_df, val_df, test_df, fold, args):
    summary = {
        "fold": fold,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_unique_speakers": train_df[args.speaker_id].nunique(),
        "val_unique_speakers": val_df[args.speaker_id].nunique() if not val_df.empty else 0,
        "test_unique_speakers": test_df[args.speaker_id].nunique(),
        "train_unique_strata": train_df["strata"].nunique(),
        "val_unique_strata": val_df["strata"].nunique() if not val_df.empty else 0,
        "test_unique_strata": test_df["strata"].nunique(),
    }

    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        if split_df.empty:
            continue
        for label, count in split_df[args.label].value_counts().items():
            summary[f"{split_name}_label_{label}"] = count
        for gender, count in split_df[args.gender].value_counts().items():
            summary[f"{split_name}_gender_{gender}"] = count
        for age_bin, count in split_df["age_bin"].value_counts().items():
            summary[f"{split_name}_age_bin_{age_bin}"] = count
    return summary


def save_summary_files(summaries, output_dir, args):
    summary_df = pd.DataFrame(summaries)
    csv_path = os.path.join(output_dir, "splits_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSummary CSV saved to: {csv_path}")

    txt_path = os.path.join(output_dir, "splits_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("SPEAKER-LEVEL CROSS-VALIDATION SPLITS SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Number of folds: {args.n_splits}\n")
        f.write(f"Validation percentage from train: {args.val_percentage}\n")
        f.write(f"Age bins: {args.age_bins}\n")
        f.write(f"Random seed: {args.seed}\n\n")

        for summary in summaries:
            f.write(f"Fold {summary['fold']}:\n")
            f.write(
                f"  Train: {summary['train_rows']} speaker(s), "
                f"{summary['train_unique_strata']} strata\n"
            )
            f.write(
                f"  Val:   {summary['val_rows']} speaker(s), "
                f"{summary['val_unique_strata']} strata\n"
            )
            f.write(
                f"  Test:  {summary['test_rows']} speaker(s), "
                f"{summary['test_unique_strata']} strata\n\n"
            )
    print(f"Summary text file saved to: {txt_path}")


def save_split(df, path):
    df.to_csv(path, index=False)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    raw_df = pd.read_csv(args.csv)

    print("Cleaning data...")
    cleaned = clean_data(raw_df, args)
    print(f"  After cleaning: {len(cleaned)} row(s)")

    print("Collapsing to one row per speaker...")
    df = collapse_to_speaker_level(cleaned, args)
    print(f"  Speaker-level rows: {len(df)}")

    df = create_age_bins(df, args.age, args.age_bins)
    df = create_strata(df, args.label, args.gender)
    df = remove_rare_strata(df, args)
    print(f"Final speaker dataset: {len(df)} speaker(s), {df['strata'].nunique()} strata")

    min_stratum_count = df["strata"].value_counts().min()
    if min_stratum_count < args.n_splits:
        raise ValueError(
            f"Smallest stratum has {min_stratum_count} speakers, but n_splits="
            f"{args.n_splits}. Lower --n_splits or use wider --age_bins."
        )

    splitter = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )

    summaries = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(df, df["strata"])):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        train_df, val_df = split_validation(train_df, args.val_percentage, args.seed + fold)

        if train_df.empty or test_df.empty:
            print(f"  Fold {fold}: skipping empty split")
            continue

        assert_no_speaker_overlap(train_df, val_df, test_df, args.speaker_id, fold)

        print(f"\n===== Fold {fold} =====")
        print(f"  Train: {len(train_df)} speakers, {train_df['strata'].nunique()} strata")
        print(
            f"  Val:   {len(val_df)} speakers, "
            f"{val_df['strata'].nunique() if not val_df.empty else 0} strata"
        )
        print(f"  Test:  {len(test_df)} speakers, {test_df['strata'].nunique()} strata")

        save_split(train_df, os.path.join(args.output_dir, f"fold{fold}_train.csv"))
        if not val_df.empty:
            save_split(val_df, os.path.join(args.output_dir, f"fold{fold}_val.csv"))
        save_split(test_df, os.path.join(args.output_dir, f"fold{fold}_test.csv"))

        summaries.append(generate_summary(train_df, val_df, test_df, fold, args))

    if summaries:
        save_summary_files(summaries, args.output_dir, args)
    print(f"\nSuccessfully created {len(summaries)}/{args.n_splits} folds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create speaker-level stratified train/val/test CV splits."
    )
    parser.add_argument("--csv", type=str, required=True, help="Input CSV file")
    parser.add_argument(
        "--speaker_id", type=str, default="speaker_id", help="Speaker ID column name"
    )
    parser.add_argument("--age", type=str, default="Age", help="Age column name")
    parser.add_argument("--gender", type=str, default="Gender", help="Gender column name")
    parser.add_argument("--label", type=str, required=True, help="Label/target column name")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument(
        "--val_percentage",
        type=float,
        default=0.2,
        help="Percentage of training speakers to use for validation.",
    )
    parser.add_argument(
        "--age_bins",
        type=float,
        nargs="+",
        required=True,
        help="Age bins for stratification, e.g. 0 30 50 70 100",
    )
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=42)

    parsed_args = parser.parse_args()
    if not 0 <= parsed_args.val_percentage < 1:
        raise ValueError("val_percentage must be between 0 and 1")
    main(parsed_args)
