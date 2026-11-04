"""Audio-features-only experiment — split generation + CV that matches
Pasted text.txt's audio-only path when starting from the same labels.

Split resolution (returns trainval sorted by speaker_id):
  A. final_train.csv + final_val.csv exist → load (Script 2 format)
  B. fold_*_train.csv exist → reconstruct trainval from union
  C. neither → generate via Script 2's _speaker_split; write fold_k_*.csv + test.csv
  --force-regenerate-splits forces Case C.

CV (identical regardless of source):
  * normalize trainval via groupby("speaker_id")["label"].first().reset_index()
  * StratifiedKFold(n_cv_folds, shuffle=True, random_state=seed)
  * raw sklearn ensemble members unless --use-smote, then
    ImbPipeline(to_float → imputer → scaler → SMOTE(k=3) → model)
  * np.nan_to_num(X, nan=0, posinf=0, neginf=0)
  * F1-optimal threshold over np.linspace(0, 1, 101) per fold
  * OOF aggregate; no test evaluation
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    VotingClassifier, VotingRegressor,
    RandomForestClassifier, RandomForestRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import (
    StratifiedKFold, KFold, StratifiedShuffleSplit, ShuffleSplit,
)
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix,
)

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


def to_float(X):
    return X.astype(float)


# =============================================================================
#  Ensemble factories — must match Pasted text.txt's create_* exactly
# =============================================================================
CLF_FACTORIES = {
    "linear":                 lambda s, a: LogisticRegression(
        max_iter=5000, class_weight="balanced", random_state=s, C=1.0),
    "random_forest":          lambda s, a: RandomForestClassifier(
        n_estimators=a.n_estimators, random_state=s,
        class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=-1,
        max_depth=a.max_depth),
    "hist_gradient_boosting": lambda s, a: HistGradientBoostingClassifier(
        max_iter=a.n_estimators, learning_rate=0.1, max_depth=a.max_depth,
        random_state=s, verbose=0, class_weight="balanced"),
    "svm":                    lambda s, a: SVC(
        kernel="rbf", C=1.0, gamma="scale", probability=True,
        class_weight="balanced", random_state=s),
    "knn":                    lambda s, a: KNeighborsClassifier(
        n_neighbors=5, weights="distance"),
}
REG_FACTORIES = {
    "linear":                 lambda s, a: Ridge(alpha=1.0),
    "random_forest":          lambda s, a: RandomForestRegressor(
        n_estimators=a.n_estimators, min_samples_leaf=2, n_jobs=-1,
        random_state=s, max_depth=a.max_depth),
    "hist_gradient_boosting": lambda s, a: HistGradientBoostingRegressor(
        max_iter=a.n_estimators, learning_rate=0.1, max_depth=a.max_depth,
        random_state=s, verbose=0),
    "svm":                    lambda s, a: SVR(kernel="rbf", C=1.0),
    "knn":                    lambda s, a: KNeighborsRegressor(
        n_neighbors=5, weights="distance"),
}


def _wrap_clf(model, seed, use_smote):
    """Mirror create_balanced_pipeline: raw unless use_smote, then wrap."""
    if not use_smote:
        return model
    if not HAS_IMBLEARN:
        print("⚠️  imblearn not installed; ignoring --use-smote")
        return model
    return ImbPipeline([
        ("to_float",  FunctionTransformer(to_float, validate=False)),
        ("imputer",   SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler",    StandardScaler()),
        ("imbalance", SMOTE(random_state=seed, k_neighbors=3)),
        ("model",     model),
    ])


def make_ensemble(task, names, seed, use_smote, args):
    if task == "classification":
        ests = [(n, _wrap_clf(CLF_FACTORIES[n](seed, args), seed, use_smote))
                for n in names if n in CLF_FACTORIES]
        if not ests:
            raise ValueError("No valid ensemble estimators")
        return VotingClassifier(estimators=ests, voting="soft", n_jobs=-1)

    ests = [(n, Pipeline([
                ("to_float", FunctionTransformer(to_float, validate=False)),
                ("imputer",  SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler",   StandardScaler()),
                ("model",    REG_FACTORIES[n](seed, args)),
            ]))
            for n in names if n in REG_FACTORIES]
    if not ests:
        raise ValueError("No valid ensemble estimators")
    return VotingRegressor(estimators=ests, n_jobs=-1)


# =============================================================================
#  Speaker split — mirror SplitManager._speaker_split
# =============================================================================
def _speaker_split(speakers_df, test_size, seed, task):
    if test_size <= 0:
        return np.arange(len(speakers_df)), np.array([], dtype=int)
    if test_size >= 1:
        return np.array([], dtype=int), np.arange(len(speakers_df))
    if task == "classification":
        sp = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                    random_state=seed)
        return next(sp.split(speakers_df, speakers_df["label"]))
    sp = ShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(sp.split(speakers_df))


# =============================================================================
#  Split resolution
# =============================================================================
def resolve_splits(sd, labels_df, args):
    """Return (trainval, test_df). trainval is sorted by speaker_id."""
    sd = Path(sd)
    sd.mkdir(parents=True, exist_ok=True)
    test_df = None

    # ---- Case A: final_train / final_val ----
    if not args.force_regenerate_splits:
        ft = sd / "final_train.csv"
        fv = sd / "final_val.csv"
        if ft.exists() and fv.exists():
            print(f"[splits] Case A: loading final_train + final_val from {sd}")
            tv = pd.concat([pd.read_csv(ft), pd.read_csv(fv)], ignore_index=True)
            ftest = sd / "final_test.csv"
            if ftest.exists():
                test_df = pd.read_csv(ftest)
            tv = (tv[["speaker_id", "label"]]
                  .groupby("speaker_id")["label"].first()
                  .reset_index())
            return tv, test_df

    # ---- Case B: fold_*_train / fold_*_val ----
    if not args.force_regenerate_splits:
        fold_trains = sorted(sd.glob("fold_*_train.csv"))
        if fold_trains:
            print(f"[splits] Case B: reconstructing from {len(fold_trains)} fold files in {sd}")
            parts = []
            for f in fold_trains:
                k = int(f.stem.split("_")[1])
                parts.append(pd.read_csv(f))
                va = sd / f"fold_{k}_val.csv"
                if va.exists():
                    parts.append(pd.read_csv(va))
            tv = pd.concat(parts, ignore_index=True)
            ttest = sd / "test.csv"
            if ttest.exists():
                test_df = pd.read_csv(ttest)
            tv = (tv[["speaker_id", "label"]]
                  .groupby("speaker_id")["label"].first()
                  .reset_index())
            return tv, test_df

    # ---- Case C: generate ----
    if labels_df is None:
        print("❌ splits-dir is empty and no labels source.")
        print("   Pass --labels-csv, OR ensure the audio CSV has a label column "
              "via --label-col, OR point --splits-dir at existing split files.")
        sys.exit(1)

    print(f"[splits] Case C: generating splits at {sd} "
          f"(seed={args.seed}, k={args.n_cv_folds}, "
          f"test_fraction={args.test_fraction})")

    # Sort by speaker_id (matches Script 2's groupby order)
    all_speakers = (labels_df[["speaker_id", "label"]]
                    .groupby("speaker_id")["label"].first()
                    .reset_index())

    tv_i, te_i = _speaker_split(all_speakers, args.test_fraction, args.seed, args.task)
    trainval_sorted = all_speakers.iloc[tv_i].reset_index(drop=True)
    test_df = all_speakers.iloc[te_i].reset_index(drop=True)

    # Normalize trainval order (sorted by speaker_id) for consistent fold assignment
    trainval_sorted = (trainval_sorted[["speaker_id", "label"]]
                       .groupby("speaker_id")["label"].first()
                       .reset_index())

    # Save fold files (Script 1's original output format)
    if args.task == "classification":
        split_iter = StratifiedKFold(args.n_cv_folds, shuffle=True,
                                     random_state=args.seed).split(
            trainval_sorted, trainval_sorted["label"])
    else:
        split_iter = KFold(args.n_cv_folds, shuffle=True,
                           random_state=args.seed).split(trainval_sorted)

    for k, (tr_i, va_i) in enumerate(split_iter):
        trainval_sorted.iloc[tr_i][["speaker_id", "label"]].reset_index(drop=True).to_csv(
            sd / f"fold_{k}_train.csv", index=False)
        trainval_sorted.iloc[va_i][["speaker_id", "label"]].reset_index(drop=True).to_csv(
            sd / f"fold_{k}_val.csv", index=False)
    test_df[["speaker_id", "label"]].to_csv(sd / "test.csv", index=False)
    print(f"[splits] wrote {args.n_cv_folds} fold pairs + test "
          f"({len(test_df)} speakers) to {sd}")

    return trainval_sorted, test_df


# =============================================================================
#  Fold construction — mirror make_outer_folds
# =============================================================================
def build_outer_folds(trainval_sorted, args):
    if args.task == "classification":
        sp = StratifiedKFold(n_splits=args.n_cv_folds, shuffle=True,
                             random_state=args.seed)
        splits = sp.split(trainval_sorted, trainval_sorted["label"])
    else:
        sp = KFold(n_splits=args.n_cv_folds, shuffle=True,
                   random_state=args.seed)
        splits = sp.split(trainval_sorted)

    folds = []
    for k, (tr_i, va_i) in enumerate(splits):
        tr_sp = set(trainval_sorted.iloc[tr_i]["speaker_id"])
        va_sp = set(trainval_sorted.iloc[va_i]["speaker_id"])
        if tr_sp & va_sp:
            raise RuntimeError(f"Speaker leakage in fold {k}")
        folds.append((tr_sp, va_sp))
    return folds


# =============================================================================
#  Metrics — mirror score_meta_model
# =============================================================================
def best_f1_threshold(y_true, y_proba):
    best_th, best_s = 0.5, -np.inf
    for th in np.linspace(0.0, 1.0, 101):
        s = f1_score(y_true, (y_proba >= th).astype(int), zero_division=0)
        if s > best_s:
            best_s, best_th = s, th
    return best_th, best_s


def clf_metrics(y_true, y_pred, y_proba=None, threshold_used=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    m = {
        "macro_f1":          float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
        "weighted_f1":       float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix":  confusion_matrix(y_true, y_pred).tolist(),
        "threshold_used":    float(threshold_used) if threshold_used is not None else 0.5,
    }
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    if len(labels) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        acc  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        f1v  = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        m.update({
            "sensitivity": float(rec), "specificity": float(spec),
            "precision":   float(prec), "npv":         float(npv),
            "accuracy":    float(acc),  "f1":          float(f1v),
        })
        if y_proba is not None:
            try:
                m["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            except Exception:
                pass
    return m


def reg_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


# =============================================================================
#  Audio loading — mirror load_audio_features
# =============================================================================
def load_audio_csv(path):
    df = pd.read_csv(path)
    if "speaker_id" not in df.columns:
        print("❌ audio CSV must contain a 'speaker_id' column")
        sys.exit(1)
    exclude = {"speaker_id", "session_id", "utterance_id", "question_id",
               "label", "y_true", "target"}
    feats = [c for c in df.select_dtypes(include=[np.number]).columns
             if c not in exclude]
    if not feats:
        print("❌ No numeric audio feature columns found")
        sys.exit(1)
    audio = df[["speaker_id"] + feats].copy()
    if audio.duplicated(subset=["speaker_id"]).any():
        audio = audio.groupby("speaker_id").mean().reset_index()
    return audio, feats, df


# =============================================================================
#  CLI
# =============================================================================
def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir", required=True,
                   help="Contains or will contain split CSVs")
    p.add_argument("--audio-features-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--task", choices=["classification", "regression"],
                   default="classification")

    p.add_argument("--n-cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--max-depth", type=int, default=None)

    p.add_argument("--ensemble-models", nargs="+", default=[
        "linear", "random_forest", "hist_gradient_boosting", "svm", "knn",
    ])
    p.add_argument("--use-smote", action="store_true")

    # Split generation (only used in Case C)
    p.add_argument("--test-fraction", type=float, default=0.1,
                   help="Held-out test fraction when generating splits")
    p.add_argument("--force-regenerate-splits", action="store_true",
                   help="Ignore existing split files and regenerate")
    p.add_argument("--labels-csv", default=None,
                   help="Speaker-level CSV for split generation "
                        "(needs --speaker-col + --label-col)")
    p.add_argument("--speaker-col", default="speaker_id")
    p.add_argument("--label-col",   default="label")

    return p


# =============================================================================
#  Main
# =============================================================================
def main():
    args = build_parser().parse_args()
    np.random.seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    audio, feats, audio_raw = load_audio_csv(args.audio_features_csv)
    print(f"Audio features: {len(feats)} columns, {len(audio)} speakers")

    # ---- Labels source (only needed if generating splits) ----
    labels_df = None
    if args.labels_csv:
        ldf = pd.read_csv(args.labels_csv)
        missing = [c for c in (args.speaker_col, args.label_col)
                   if c not in ldf.columns]
        if missing:
            print(f"❌ --labels-csv missing {missing}; found: {list(ldf.columns)}")
            sys.exit(1)
        labels_df = (ldf.rename(columns={args.speaker_col: "speaker_id",
                                         args.label_col:   "label"})
                        [["speaker_id", "label"]])
        print(f"[labels] from {args.labels_csv}: {len(labels_df)} speakers")
    elif args.label_col in audio_raw.columns:
        labels_df = (audio_raw.rename(
                        columns={args.speaker_col: "speaker_id",
                                 args.label_col:   "label"})
                     [["speaker_id", "label"]])
        print(f"[labels] from audio CSV '{args.label_col}': "
              f"{len(labels_df)} speakers")

    # ---- Split resolution ----
    trainval, test_df = resolve_splits(args.splits_dir, labels_df, args)
    print(f"[splits] trainval: {len(trainval)} speakers, "
          f"test: {len(test_df) if test_df is not None and len(test_df) > 0 else 0}")

    # ---- Attach labels to audio ----
    audio_labels = audio.merge(trainval, on="speaker_id", how="inner")
    if args.task == "classification":
        audio_labels["label"] = audio_labels["label"].astype(int)
        cc = audio_labels["label"].value_counts()
        print(f"Class distribution (audio ∩ trainval): {dict(cc)}")
        if len(cc) < 2:
            print("❌ Only one class present in audio data")
            sys.exit(1)
    if len(audio_labels) < 10:
        print(f"❌ Not enough audio speakers with labels ({len(audio_labels)})")
        sys.exit(1)

    # ---- Build CV folds ----
    folds = build_outer_folds(trainval, args)
    print(f"Built {len(folds)} speaker-disjoint folds")

    # ---- CV loop ----
    oof_rows, fold_rows = [], []

    for fold_idx, (tr_sp, va_sp) in enumerate(folds):
        print(f"\n--- Fold {fold_idx + 1}/{len(folds)} "
              f"(train={len(tr_sp)} spk, val={len(va_sp)} spk) ---")

        atr = audio_labels[audio_labels.speaker_id.isin(tr_sp)].copy()
        ava = audio_labels[audio_labels.speaker_id.isin(va_sp)].copy()
        if len(atr) < 3 or len(ava) < 2:
            print(f"  ⚠️ skipped: train={len(atr)}, val={len(ava)}")
            continue

        Xtr = np.nan_to_num(atr[feats].to_numpy(float),
                            nan=0.0, posinf=0.0, neginf=0.0)
        Xva = np.nan_to_num(ava[feats].to_numpy(float),
                            nan=0.0, posinf=0.0, neginf=0.0)
        ytr = atr["label"].to_numpy()
        yva = ava["label"].to_numpy()
        if args.task == "classification":
            ytr = ytr.astype(int)
            yva = yva.astype(int)

        mdl = make_ensemble(args.task, args.ensemble_models,
                            args.seed, args.use_smote, args)
        mdl.fit(Xtr, ytr)

        if args.task == "classification":
            proba = None
            if hasattr(mdl, "predict_proba"):
                try:
                    proba = mdl.predict_proba(Xva)
                except Exception as e:
                    print(f"  ⚠️ predict_proba failed: {e}")

            if proba is not None and proba.ndim == 2 and proba.shape[1] == 2:
                best_th, _ = best_f1_threshold(yva, proba[:, 1])
                pred = (proba[:, 1] >= best_th).astype(int)
                y_proba_col = proba[:, 1]
            elif proba is not None:
                pred = np.argmax(proba, axis=1)
                best_th, y_proba_col = 0.5, None
            else:
                pred = mdl.predict(Xva)
                best_th, y_proba_col = 0.5, None

            met = clf_metrics(yva, pred, y_proba_col, threshold_used=best_th)
            print(f"  F1={met.get('f1', 0):.4f}  "
                  f"macro_F1={met.get('macro_f1', 0):.4f}  "
                  f"AUC={met.get('roc_auc', 0):.4f}  th={best_th:.3f}")

            row = pd.DataFrame({
                "speaker_id": ava.speaker_id.values,
                "y_true": yva, "y_pred": pred, "fold": fold_idx,
            })
            if y_proba_col is not None:
                row["y_proba"] = y_proba_col
        else:
            pred = mdl.predict(Xva)
            met = reg_metrics(yva, pred)
            print(f"  RMSE={met['rmse']:.4f}  R²={met['r2']:.4f}")
            row = pd.DataFrame({
                "speaker_id": ava.speaker_id.values,
                "y_true": yva, "y_pred": pred, "fold": fold_idx,
            })

        oof_rows.append(row)
        fold_rows.append({"fold": fold_idx, **{
            k: v for k, v in met.items()
            if isinstance(v, (int, float, np.integer, np.floating))
        }})

    if not oof_rows:
        print("❌ No fold produced predictions")
        sys.exit(1)

    # ---- Aggregate OOF (mirror compute_subgroup_metrics) ----
    oof = pd.concat(oof_rows, ignore_index=True)
    if not (oof.groupby("speaker_id").size() == 1).all():
        print("⚠️  Deduplicating speakers (keep first)")
        oof = oof.drop_duplicates(subset=["speaker_id"], keep="first")

    if args.task == "classification":
        agg = clf_metrics(
            oof["y_true"].values.astype(int),
            oof["y_pred"].values.astype(int),
            oof["y_proba"].values if "y_proba" in oof.columns else None,
        )
    else:
        agg = reg_metrics(oof["y_true"].values, oof["y_pred"].values)

    # ---- Output ----
    oof.to_csv(out / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "fold_metrics.csv", index=False)
    (out / "aggregate_metrics.json").write_text(json.dumps(agg, indent=2))

    print("\n=== AGGREGATE OOF METRICS ===")
    print(json.dumps(agg, indent=2))
    print(f"\nDone → {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)