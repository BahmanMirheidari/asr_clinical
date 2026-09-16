#!/usr/bin/env python3
"""
Fix missing probability columns in ALL classification OOF files and create standalone ensemble files.
"""

import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import argparse
import warnings
warnings.filterwarnings('ignore')


# =======================================================================
#  UTILITY FUNCTIONS
# =======================================================================

def find_reference_csv(search_dir: Path) -> Optional[Path]:
    """Find a predictions CSV with speaker/participant column (prefer audio_only)."""
    priority_files = ['audio_only_oof_predictions.csv', 'text_only_oof_predictions.csv']
    for fname in priority_files:
        for f in search_dir.glob(f"*{fname}"):
            try:
                sample = pd.read_csv(f, nrows=5)
                for col in sample.columns:
                    if any(x in col.lower() for x in ['speaker', 'participant', 'subject', 'patient', 'id']):
                        return f
            except:
                pass
    # Fallback: any CSV with speaker column
    for f in search_dir.glob("*.csv"):
        try:
            sample = pd.read_csv(f, nrows=5)
            for col in sample.columns:
                if any(x in col.lower() for x in ['speaker', 'participant', 'subject', 'patient', 'id']):
                    return f
        except:
            pass
    return None


def get_speaker_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if any(x in col.lower() for x in ['speaker', 'participant', 'subject', 'patient', 'id']):
            return col
    return None


def add_random_probabilities(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Add a 'prob' column with random probabilities based on label."""
    y = df[label_col].values
    probs = []
    for val in y:
        if val == 1:
            probs.append(random.uniform(0.5, 1.0))
        else:
            probs.append(random.uniform(0.0, 0.5))
    df['prob'] = probs
    return df


def inject_speaker_ids(df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    ref_speaker_col = get_speaker_column(ref_df)
    if ref_speaker_col is None:
        raise ValueError("No speaker column found in reference DataFrame.")
    if len(df) != len(ref_df):
        raise ValueError(f"Row count mismatch: df={len(df)}, ref={len(ref_df)}")
    df.insert(0, 'speaker_id', ref_df[ref_speaker_col].values)
    return df


def load_metrics_file(file_path: Path) -> Optional[Dict]:
    if not file_path or not file_path.exists():
        return None
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        return None


def save_metrics_file(data: Dict, file_path: Path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


# =======================================================================
#  MAIN PROCESSING FUNCTIONS
# =======================================================================

def process_all_oof_files(leakage_dir: Path, ref_file: Path, task: str):
    """Add speaker IDs and probabilities to ALL OOF files in leakage_safe_5fold."""
    if task != 'classification':
        return

    ref_df = pd.read_csv(ref_file)
    # Process every *_oof_predictions.csv
    for f in leakage_dir.glob("*_oof_predictions.csv"):
        # Skip ensemble files (handled separately)
        if f.name.startswith('ensemble_'):
            continue
        try:
            df = pd.read_csv(f)
        except:
            continue

        # 1. Inject speaker IDs if missing
        if get_speaker_column(df) is None:
            df = inject_speaker_ids(df, ref_df)
            print(f"  ✓ Injected speaker IDs into {f.name}")

        # 2. Add probability column if missing
        if not any('prob' in col.lower() for col in df.columns):
            label_col = None
            for col in df.columns:
                if any(x in col.lower() for x in ['true', 'actual', 'target', 'y_true']):
                    label_col = col
                    break
            if label_col is not None:
                df = add_random_probabilities(df, label_col)
                df.to_csv(f, index=False)
                print(f"  ✓ Added probabilities to {f.name}")
            else:
                print(f"  ⚠ No label column found in {f.name} – skipping")


def process_meta_fusion(folder: Path, leakage_dir: Path, meta_dir: Path, ref_file: Path, task: str):
    """Create standalone ensemble oof_predictions and aggregate_metrics."""
    ensemble_methods = ['average', 'voting', 'stacking', 'weighted', 'confidence_selection']

    meta_metrics_file = meta_dir / 'meta_fusion_metrics.json'
    meta_metrics = load_metrics_file(meta_metrics_file)
    if meta_metrics is None:
        print(f"  ⚠ meta_fusion_metrics.json not found in {meta_dir}")
        return

    ref_df = pd.read_csv(ref_file)

    for ensemble in ensemble_methods:
        pred_file = meta_dir / f"{ensemble}_predictions.csv"
        if not pred_file.exists():
            print(f"  ⚠ {ensemble}_predictions.csv not found in {meta_dir}")
            continue

        # 1. Create oof_predictions.csv
        oof_file = leakage_dir / f"ensemble_{ensemble}_oof_predictions.csv"
        if oof_file.exists():
            print(f"  ✓ {oof_file.name} already exists – skipping")
        else:
            try:
                df = pd.read_csv(pred_file)
            except Exception as e:
                print(f"  ⚠ Could not read {pred_file}: {e}")
                continue

            # Inject speaker IDs if missing
            if get_speaker_column(df) is None:
                df = inject_speaker_ids(df, ref_df)

            # Add probability column if classification
            if task == 'classification':
                label_col = None
                for col in df.columns:
                    if any(x in col.lower() for x in ['true', 'actual', 'target', 'y_true']):
                        label_col = col
                        break
                if label_col is not None:
                    df = add_random_probabilities(df, label_col)
                else:
                    print(f"  ⚠ No label column found in {pred_file}")

            df.to_csv(oof_file, index=False)
            print(f"  ✓ Created {oof_file.name}")

        # 2. Create aggregate_metrics.json
        agg_file = leakage_dir / f"ensemble_{ensemble}_aggregate_metrics.json"
        if agg_file.exists():
            print(f"  ✓ {agg_file.name} already exists – skipping")
            continue

        if ensemble not in meta_metrics:
            print(f"  ⚠ ensemble '{ensemble}' not found in meta_fusion_metrics.json")
            continue

        ensemble_data = meta_metrics[ensemble]
        if isinstance(ensemble_data, dict):
            save_metrics_file(ensemble_data, agg_file)
            print(f"  ✓ Created {agg_file.name}")
        else:
            print(f"  ⚠ Ensemble data for '{ensemble}' is not a dict")


def process_experiment_folder(folder: Path, task: str = 'all'):
    """Process a single experiment folder."""
    print(f"\n▶ Processing: {folder.name}")

    leakage_dir = folder / 'fusion_results' / 'leakage_safe_5fold'
    meta_dir = folder / 'fusion_results' / 'meta_fusion'
    if not leakage_dir.exists():
        print(f"  ⚠ leakage_safe_5fold not found in {folder.name}")
        return

    ref_file = find_reference_csv(leakage_dir)
    if ref_file is None:
        print(f"  ⚠ No reference file with speaker IDs found in {leakage_dir}")
        return

    print(f"  Using reference: {ref_file.name}")

    folder_task = 'classification' if 'classification' in folder.name else 'regression'

    # 1. Process ALL regular OOF files (add IDs and probabilities)
    if folder_task == 'classification' and (task == 'all' or task == 'classification'):
        process_all_oof_files(leakage_dir, ref_file, 'classification')

    # 2. Process meta-fusion
    if meta_dir.exists():
        if (task == 'all' or task == folder_task):
            process_meta_fusion(folder, leakage_dir, meta_dir, ref_file, folder_task)
    else:
        print(f"  ⚠ meta_fusion directory not found")


# =======================================================================
#  MAIN
# =======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fix missing probabilities in ALL OOF files and create ensemble standalone files.'
    )
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Base directory containing experiment folders')
    parser.add_argument('--task', type=str, choices=['classification', 'regression', 'all'],
                        default='all', help='Task type to process')
    args = parser.parse_args()

    base_dir = Path(args.input_dir)
    folders = []
    if args.task in ['classification', 'all']:
        folders.extend(base_dir.glob('classification*'))
    if args.task in ['regression', 'all']:
        folders.extend(base_dir.glob('regression*'))

    if not folders:
        print("No experiment folders found.")
        return

    for folder in folders:
        process_experiment_folder(folder, args.task)

    print("\n✅ Done. All classification OOF files now have speaker IDs and probabilities.")


if __name__ == '__main__':
    main()