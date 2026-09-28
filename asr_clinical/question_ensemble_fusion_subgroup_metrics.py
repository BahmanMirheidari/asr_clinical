#!/usr/bin/env python3
"""
Compute classification metrics on a predictions CSV, split by subgroup.

Inputs
------
--csv    : CSV with columns speaker_id, y_true, y_pred, agreement, prob
           - speaker_id : string identifier
           - y_true     : binary ground-truth label (0/1)
           - y_pred     : hard prediction (0/1)
           - agreement  : optional; ignored unless you want it
           - prob       : predicted probability for class 1
--ids    : text file, one speaker_id per line (e.g. 'p001')
--out    : optional output CSV path for the metrics table

Output
------
Prints a metrics table for:
    Combined     (all rows)
    Subgroup     (rows whose speaker_id is in --ids)
    Non-subgroup (rows whose speaker_id is NOT in --ids)

Metrics per split:
    n, positives, negatives,
    AUC-ROC, sensitivity, specificity, macro-F1, balanced accuracy,
    precision, NPV, accuracy.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata


# ----------------------------------------------------------------------
#  Metric primitives (no sklearn dependency)
# ----------------------------------------------------------------------

def _auc_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney U AUC-ROC (tie-safe). NaN if a class is missing."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    ranks = rankdata(y_score)
    sum_pos = ranks[y_true == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: Optional[np.ndarray] = None) -> dict:
    """Binary classification metrics. Positive class = 1."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)[m]

    n = len(y_true)
    out = {
        'n': n,
        'positives': int((y_true == 1).sum()) if n else 0,
        'negatives': int((y_true == 0).sum()) if n else 0,
        'auc': float('nan'),
        'sensitivity': float('nan'),
        'specificity': float('nan'),
        'macro_f1': float('nan'),
        'balanced_accuracy': float('nan'),
        'precision': float('nan'),
        'npv': float('nan'),
        'accuracy': float('nan'),
    }
    if n == 0:
        return out

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())

    sens = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    spec = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
    prec_pos = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    prec_neg = tn / (tn + fn) if (tn + fn) > 0 else float('nan')
    sens_neg = tn / (tn + fp) if (tn + fp) > 0 else float('nan')

    f1_pos = (2 * prec_pos * sens / (prec_pos + sens)
              if (prec_pos and sens and (prec_pos + sens) > 0) else float('nan'))
    f1_neg = (2 * prec_neg * sens_neg / (prec_neg + sens_neg)
              if (prec_neg and sens_neg and (prec_neg + sens_neg) > 0) else float('nan'))

    if not np.isnan(sens):
        out['sensitivity'] = float(sens)
    if not np.isnan(spec):
        out['specificity'] = float(spec)
    if not (np.isnan(f1_pos) or np.isnan(f1_neg)):
        out['macro_f1'] = float((f1_pos + f1_neg) / 2.0)
    if not (np.isnan(sens) or np.isnan(spec)):
        out['balanced_accuracy'] = float((sens + spec) / 2.0)
    if not np.isnan(prec_pos):
        out['precision'] = float(prec_pos)
    if not np.isnan(prec_neg):
        out['npv'] = float(prec_neg)
    out['accuracy'] = float((tp + tn) / n)

    if y_prob is not None and len(y_prob) == n:
        out['auc'] = _auc_rank(y_true, y_prob)

    return out


# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', required=True,
                    help='CSV with speaker_id,y_true,y_pred,agreement,prob')
    ap.add_argument('--ids', required=True,
                    help='Text file with one subgroup speaker_id per line')
    ap.add_argument('--out', default=None,
                    help='Optional path to write the metrics table (CSV)')
    ap.add_argument('--id-col', default='speaker_id',
                    help='Name of the speaker ID column (default: speaker_id)')
    ap.add_argument('--true-col', default='y_true',
                    help='Name of the ground-truth column (default: y_true)')
    ap.add_argument('--pred-col', default='y_pred',
                    help='Name of the hard prediction column (default: y_pred)')
    ap.add_argument('--prob-col', default='prob',
                    help='Name of the probability column (default: prob)')
    args = ap.parse_args()

    # ---- Load predictions ----
    df = pd.read_csv(args.csv)
    needed = [args.id_col, args.true_col, args.pred_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: missing column(s) in {args.csv}: {missing}\n"
                 f"       available: {list(df.columns)}")

    prob_available = args.prob_col in df.columns
    if not prob_available:
        print(f"Warning: '{args.prob_col}' column not found — "
              f"AUC will be NaN.")

    # ---- Load subgroup IDs ----
    with open(args.ids) as f:
        subgroup_ids = {line.strip() for line in f if line.strip()}
    if not subgroup_ids:
        sys.exit(f"ERROR: no speaker IDs found in {args.ids}")

    print(f"Loaded {len(df)} prediction rows from {args.csv}")
    print(f"Loaded {len(subgroup_ids)} subgroup speaker IDs from {args.ids}")

    # ---- Align types ----
    df[args.id_col] = df[args.id_col].astype(str).str.strip()

    in_sub = df[args.id_col].isin(subgroup_ids)
    splits = {
        'Combined':     df,
        'Subgroup':     df[in_sub],
        'Non-subgroup': df[~in_sub],
    }

    # ---- Compute ----
    rows = []
    for name, sub in splits.items():
        y_true = pd.to_numeric(sub[args.true_col], errors='coerce').values
        y_pred = pd.to_numeric(sub[args.pred_col], errors='coerce').values
        y_prob = (pd.to_numeric(sub[args.prob_col], errors='coerce').values
                  if prob_available else None)

        m = compute_metrics(y_true, y_pred, y_prob)
        m = {'split': name, **m}
        rows.append(m)

    table = pd.DataFrame(rows)[[
        'split', 'n', 'positives', 'negatives',
        'auc', 'sensitivity', 'specificity',
        'macro_f1', 'balanced_accuracy',
        'precision', 'npv', 'accuracy',
    ]]

    # ---- Pretty print ----
    pd.set_option('display.float_format', lambda v: f'{v:.4f}')
    print()
    print(table.to_string(index=False))

    # ---- Speaker coverage check ----
    found = set(df[args.id_col]) & subgroup_ids
    not_found = subgroup_ids - set(df[args.id_col])
    if not_found:
        print(f"\n⚠ {len(not_found)} subgroup ID(s) not present in CSV "
              f"(first 5): {sorted(not_found)[:5]}")
    print(f"Subgroup IDs matched in CSV: {len(found)} / {len(subgroup_ids)}")

    # ---- Optional CSV output ----
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_path, index=False)
        print(f"\n✓ Metrics table written to: {out_path}")


if __name__ == '__main__':
    main()

'''
c=/mnt/parscratch/users/ac1bm/MND-expr/outputs-ensemble/classification-fusion-ecas_105-roberta-large/fusion_results/leakage_safe_5fold/ensemble_voting_oof_predictions.csv
a=dysarthria-list.txt
o=vote-ens-dys.csv

python ~/asr_clinical/question_ensemble_fusion_subgroup_metrics.py --csv $c --ids $a --out $o 


Loaded 51 prediction rows from /mnt/parscratch/users/ac1bm/MND-expr/outputs-ensemble/classification-fusion-ecas_105-roberta-large/fusion_results/leakage_safe_5fold/ensemble_voting_oof_predictions.csv
Loaded 25 subgroup speaker IDs from dysarthria-list.txt

       split  n  positives  negatives    auc  sensitivity  specificity  macro_f1  balanced_accuracy  precision    npv  accuracy
    Combined 51         15         36 1.0000       1.0000       0.8611    0.8913             0.9306     0.7500 1.0000    0.9020
    Subgroup 25          8         17 1.0000       1.0000       0.7647    0.8333             0.8824     0.6667 1.0000    0.8400
Non-subgroup 26          7         19 1.0000       1.0000       0.9474    0.9532             0.9737     0.8750 1.0000    0.9615
Subgroup IDs matched in CSV: 25 / 25

✓ Metrics table written to: vote-ens-dys.csv

c=/mnt/parscratch/users/ac1bm/MND-expr/outputs-ensemble/classification-fusion-ecas_105-roberta-large/fusion_results/leakage_safe_5fold/audio_only_oof_predictions2.csv
a=dysarthria-list.txt
o=vote-clinc-dys.csv

python ~/asr_clinical/question_ensemble_fusion_subgroup_metrics.py --csv $c --ids $a --out $o 



Loaded 51 prediction rows from /mnt/parscratch/users/ac1bm/MND-expr/outputs-ensemble/classification-fusion-ecas_105-roberta-large/fusion_results/leakage_safe_5fold/audio_only_oof_predictions2.csv
Loaded 25 subgroup speaker IDs from dysarthria-list.txt

       split  n  positives  negatives    auc  sensitivity  specificity  macro_f1  balanced_accuracy  precision    npv  accuracy
    Combined 51         15         36 0.8556       0.9333       0.8056    0.8283             0.8694     0.6667 0.9667    0.8431
    Subgroup 25          8         17 0.9118       1.0000       0.7647    0.8333             0.8824     0.6667 1.0000    0.8400
Non-subgroup 26          7         19 0.8271       0.8571       0.8421    0.8194             0.8496     0.6667 0.9412    0.8462
Subgroup IDs matched in CSV: 25 / 25

✓ Metrics table written to: vote-clinc-dys.csv 

'''