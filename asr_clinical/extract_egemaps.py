#!/usr/bin/env python3
"""
Extract eGeMAPS features from pXXX_Qy.wav files.

Produces three artifacts in <output_folder>:
  1. Qy.csv                       per-question, one row per file, with speaker_id
  2. audio_features_concat.csv    352 features per speaker (Qp__..Qr__ prefixed)
  3. audio_features_stats.csv     176 features per speaker (mean+std across Qs)
"""

import argparse
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import opensmile

QUESTIONS_DEFAULT = ["Qp", "Qc", "Qk", "Qr"]


def parse_arguments():
    p = argparse.ArgumentParser(description="Extract eGeMAPS features per question.")
    p.add_argument("input_folder", type=str)
    p.add_argument("output_folder", type=str)
    p.add_argument("--feature_set", default="eGeMAPSv02",
                   choices=["eGeMAPSv01b", "eGeMAPSv02"])
    p.add_argument("--feature_level", default="Functionals",
                   choices=["Functionals", "LowLevelDescriptors",
                            "LowLevelDescriptors_Deltas"])
    p.add_argument("--questions", nargs="+", default=QUESTIONS_DEFAULT,
                   help="Question IDs to expect in filenames (default: Qp Qc Qk Qr)")
    return p.parse_args()


def parse_filename(filename: str):
    """Return (speaker_id, question_id) or (None, None)."""
    m = re.match(r"^(p\d+)_(Q[\w]+)\.wav$", filename, re.IGNORECASE)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def extract_features(smile, audio_path: Path) -> pd.DataFrame:
    return smile.process_file(str(audio_path))


def main():
    args = parse_arguments()
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)

    if not input_folder.is_dir():
        print(f"Error: '{input_folder}' is not a directory.", file=sys.stderr)
        sys.exit(1)
    output_folder.mkdir(parents=True, exist_ok=True)

    smile = opensmile.Smile(
        feature_set=getattr(opensmile.FeatureSet, args.feature_set),
        feature_level=getattr(opensmile.FeatureLevel, args.feature_level),
    )

    wav_files = sorted(input_folder.glob("*.wav"))
    if not wav_files:
        print(f"No WAV files in '{input_folder}'.", file=sys.stderr)
        sys.exit(0)
    print(f"Found {len(wav_files)} WAV file(s).")

    # Accumulate per-question rows: {q: [ {speaker_id, feat_0..feat_87}, ... ]}
    per_q_rows: dict[str, list[dict]] = {q: [] for q in args.questions}
    feature_names: list[str] | None = None
    processed, skipped = 0, 0

    for wav_file in wav_files:
        speaker_id, question_id = parse_filename(wav_file.name)
        if speaker_id is None:
            print(f"  [SKIP] '{wav_file.name}': does not match pXXX_Qy.wav")
            skipped += 1
            continue
        if question_id not in per_q_rows:
            print(f"  [SKIP] '{wav_file.name}': question '{question_id}' "
                  f"not in --questions {args.questions}")
            skipped += 1
            continue

        try:
            feats = extract_features(smile, wav_file)     # (1, 88) DataFrame
            row = feats.iloc[0].to_dict()
            row["speaker_id"] = speaker_id
            per_q_rows[question_id].append(row)
            if feature_names is None:
                feature_names = [c for c in feats.columns]
            processed += 1
            print(f"  [OK]   {wav_file.name} -> {question_id} / {speaker_id}")
        except Exception as e:
            print(f"  [FAIL] {wav_file.name}: {e}", file=sys.stderr)
            skipped += 1

    if processed == 0 or feature_names is None:
        print("No features extracted.", file=sys.stderr)
        sys.exit(1)

    # ---------- 1) Per-question CSVs ----------
    print("\nWriting per-question CSVs...")
    for q in args.questions:
        rows = per_q_rows[q]
        if not rows:
            print(f"  ⚠️ {q}: no files, writing empty CSV")
            pd.DataFrame(columns=["speaker_id"] + feature_names).to_csv(
                output_folder / f"{q}.csv", index=False)
            continue

        df = pd.DataFrame(rows)
        # Ensure column order: speaker_id first, then features
        cols = ["speaker_id"] + [c for c in feature_names if c in df.columns]
        df = df[cols]

        # One speaker may appear twice (retakes) -> average
        if df.duplicated(subset=["speaker_id"]).any():
            n_dup = int(df.duplicated(subset=["speaker_id"]).sum())
            print(f"  ⚠️ {q}: {n_dup} duplicate speaker rows, averaging")
            df = df.groupby("speaker_id", as_index=False).mean(numeric_only=True)

        df.to_csv(output_folder / f"{q}.csv", index=False)
        print(f"  ✓ {q}.csv  ({len(df)} speakers, {len(feature_names)} features)")

    # ---------- Build a wide per-speaker table ----------
    # Each per-question block gets columns Qy__<feat>
    blocks = {}
    for q in args.questions:
        path = output_folder / f"{q}.csv"
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = df.set_index("speaker_id")
        df.columns = [f"{q}__{c}" for c in df.columns]
        blocks[q] = df

    if not blocks:
        print("No non-empty per-question tables; skipping aggregation.",
              file=sys.stderr)
        sys.exit(0)

    # Outer join so speakers missing some questions are retained
    wide = blocks[next(iter(blocks))]
    for q, b in blocks.items():
        if q == next(iter(blocks)):
            continue
        wide = wide.join(b, how="outer")

    # ---------- 2) 352-feature concatenation ----------
    # Fill missing per-question blocks with the column mean (not 0 —
    # eGeMAPS feature scales vary widely and 0 is a bad prior).
    wide_concat = wide.copy()
    wide_concat = wide_concat.fillna(wide_concat.mean(numeric_only=True))
    wide_concat = wide_concat.fillna(0.0)          # fallback if all-NaN col

    wide_concat = wide_concat.reset_index()
    wide_concat.to_csv(output_folder / "audio_features_concat.csv", index=False)
    print(f"\n✓ audio_features_concat.csv "
          f"({wide_concat.shape[0]} speakers, {wide_concat.shape[1]-1} features)")

    # ---------- 3) 176-feature mean+std across questions ----------
    # We build (S, Q, F) then reduce across Q. Only use questions that are
    # present for this speaker; otherwise std gets polluted by imputed cells.
    ordered_qs = [q for q in args.questions if q in blocks]
    F = len(feature_names)
    speakers = wide.index.to_numpy()

    # Extract the per-question (S, F) blocks in a common speaker order.
    # Missing speaker entries stay NaN here; we handle them per row below.
    per_q_matrix = np.full((len(speakers), len(ordered_qs), F), np.nan)
    for qi, q in enumerate(ordered_qs):
        b = blocks[q].reindex(speakers)
        per_q_matrix[:, qi, :] = b.to_numpy(dtype=float)

    mean_across_q = np.nanmean(per_q_matrix, axis=1)                  # (S, F)
    with np.errstate(invalid="ignore"):
        std_across_q = np.nanstd(per_q_matrix, axis=1, ddof=1)        # (S, F)
    std_across_q = np.where(np.isnan(std_across_q), 0.0, std_across_q)
    mean_across_q = np.where(np.isnan(mean_across_q), 0.0, mean_across_q)

    stats_df = pd.DataFrame(
        np.hstack([mean_across_q, std_across_q]),
        index=speakers,
        columns=[f"{c}_mean" for c in feature_names] +
                [f"{c}_std"  for c in feature_names],
    ).reset_index().rename(columns={"index": "speaker_id"})

    stats_df.to_csv(output_folder / "audio_features_stats.csv", index=False)
    print(f"✓ audio_features_stats.csv "
          f"({stats_df.shape[0]} speakers, {stats_df.shape[1]-1} features)")

    # ---------- Report ----------
    print(f"\nDone. Processed {processed}, skipped {skipped}.")
    for q in args.questions:
        n = len(per_q_rows[q])
        print(f"  {q}: {n} files")


if __name__ == "__main__":
    main()

''' 
python ~/asr_clinical/extract_egemaps.py /mnt/parscratch/users/ac1bm/MND-expr/MND-Audio/wav /mnt/parscratch/users/ac1bm/MND-expr/eGeMAPSv01b-feats --feature_set eGeMAPSv01b --feature_level Functionals
'''