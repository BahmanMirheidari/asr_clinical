"""Audio-features-only experiment.

If --splits-dir does NOT exist (or is empty, or --force-regenerate-splits):
    Generates speaker-level K-Fold stratified splits using --seed and writes:
        <splits-dir>/fold_{k}_train.csv   (speaker_id, label)
        <splits-dir>/fold_{k}_val.csv     (speaker_id, label)
        <splits-dir>/test.csv             (speaker_id, label)   # held-out
If the splits dir already exists:
    Loads splits from disk (k-fold files, or the fusion pipeline's
    final_train/val/test.csv for backward compat) and reuses them.
"""
import argparse, json, sys, traceback
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import (
    VotingClassifier, VotingRegressor,
    RandomForestClassifier, RandomForestRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score,
    recall_score, precision_score, confusion_matrix, matthews_corrcoef,
)
try:
    from sklearn.metrics import mean_absolute_percentage_error
except Exception:
    mean_absolute_percentage_error = None


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def _base(name, task, seed, n_est, max_depth):
    if task == "classification":
        return {
            "linear":                 LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed),
            "random_forest":          RandomForestClassifier(n_estimators=n_est, class_weight="balanced_subsample",
                                                              min_samples_leaf=2, n_jobs=-1, random_state=seed, max_depth=max_depth),
            "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=n_est, learning_rate=0.1,
                                                                     max_depth=max_depth, class_weight="balanced", random_state=seed),
            "gradient_boosting":      GradientBoostingClassifier(n_estimators=n_est, learning_rate=0.1,
                                                                 max_depth=3, random_state=seed),
            "svm":                    SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                                          class_weight="balanced", random_state=seed),
            "knn":                    KNeighborsClassifier(n_neighbors=5, weights="distance"),
        }[name]
    return {
        "linear":                 Ridge(alpha=1.0),
        "random_forest":          RandomForestRegressor(n_estimators=n_est, min_samples_leaf=2, n_jobs=-1,
                                                        random_state=seed, max_depth=max_depth),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=n_est, learning_rate=0.1,
                                                                max_depth=max_depth, random_state=seed),
        "gradient_boosting":      GradientBoostingRegressor(n_estimators=n_est, learning_rate=0.1,
                                                            max_depth=3, random_state=seed),
        "svm":                    SVR(kernel="rbf", C=1.0),
        "knn":                    KNeighborsRegressor(n_neighbors=5, weights="distance"),
    }[name]


def _pipe(name, task, seed, n_est, max_depth, use_smote):
    base = _base(name, task, seed, n_est, max_depth)
    steps = [("imp", SimpleImputer(strategy="constant", fill_value=0.0)),
             ("sc",  StandardScaler())]
    if use_smote and task == "classification":
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        return ImbPipeline(steps + [("smote", SMOTE(random_state=seed, k_neighbors=3)), ("m", base)])
    return Pipeline(steps + [("m", base)])


def make_ensemble(task, names, seed, n_est, max_depth, use_smote):
    ests = [(n, _pipe(n, task, seed, n_est, max_depth, use_smote)) for n in names]
    return (VotingClassifier(estimators=ests, voting="soft", n_jobs=-1)
            if task == "classification" else
            VotingRegressor(estimators=ests, n_jobs=-1))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def best_threshold(proba, y):
    best_th, best = 0.5, -1.0
    for th in np.linspace(0, 1, 101):
        s = f1_score(y, (proba >= th).astype(int), zero_division=0)
        if s > best:
            best, best_th = s, th
    return best_th


def clf_metrics(y, pred, proba=None):
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    labels = sorted(np.unique(np.concatenate([y, pred])))
    binary = (len(labels) == 2)

    m = {
        "accuracy":          float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1":                float(f1_score(y, pred, average="binary", zero_division=0)) if binary else float("nan"),
        "macro_f1":          float(f1_score(y, pred, average="macro",  zero_division=0)),
        "weighted_f1":       float(f1_score(y, pred, average="weighted", zero_division=0)),
        "precision":         float(precision_score(y, pred, average="binary", zero_division=0)) if binary else float("nan"),
        "sensitivity":       float(recall_score(y, pred, average="binary", zero_division=0)) if binary else float("nan"),
        "mcc":               float(matthews_corrcoef(y, pred)) if len(labels) > 1 else float("nan"),
    }

    if binary:
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        m["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
        m["npv"]         = float(tn / (tn + fn)) if (tn + fn) > 0 else float("nan")
        m.update({"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)})
    else:
        m["specificity"] = float("nan")
        m["npv"]         = float("nan")

    if proba is not None:
        try:
            m["roc_auc"] = float(roc_auc_score(y, proba))
        except Exception:
            pass
    return m


def reg_metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = {
        "rmse":               float(np.sqrt(mean_squared_error(y, pred))),
        "mae":                float(mean_absolute_error(y, pred)),
        "r2":                 float(r2_score(y, pred)),
        "explained_variance": float(explained_variance_score(y, pred)),
    }
    if mean_absolute_percentage_error is not None:
        try:
            m["mape"] = float(mean_absolute_percentage_error(y, pred))
        except Exception:
            pass
    try:
        from scipy.stats import pearsonr, spearmanr
        if len(y) > 1 and np.std(pred) > 0:
            m["pearson"]  = float(pearsonr(y, pred)[0])
            m["spearman"] = float(spearmanr(y, pred)[0])
    except Exception:
        pass
    return m


# ---------------------------------------------------------------------------
# Split resolution / generation
# ---------------------------------------------------------------------------
def _write_folds(sd, trainval_df, test_df, n_folds, seed, is_clf):
    sd.mkdir(parents=True, exist_ok=True)
    if is_clf:
        split_iter = list(StratifiedKFold(n_folds, shuffle=True, random_state=seed)
                          .split(trainval_df, trainval_df["label"]))
    else:
        split_iter = list(KFold(n_folds, shuffle=True, random_state=seed)
                          .split(trainval_df))

    folds = []
    for k, (tri, vai) in enumerate(split_iter):
        tr = trainval_df.iloc[tri][["speaker_id", "label"]].reset_index(drop=True)
        va = trainval_df.iloc[vai][["speaker_id", "label"]].reset_index(drop=True)
        tr.to_csv(sd / f"fold_{k}_train.csv", index=False)
        va.to_csv(sd / f"fold_{k}_val.csv", index=False)
        folds.append((tr, va))

    if test_df is not None and len(test_df) > 0:
        test_df[["speaker_id", "label"]].reset_index(drop=True).to_csv(sd / "test.csv", index=False)
    return folds


def resolve_splits(args, labels_df, is_clf):
    """Return (folds, test_df). folds: list of (train_df, val_df) with speaker_id,label."""
    sd = Path(args.splits_dir)

    kfold_files = sorted(sd.glob("fold_*_train.csv")) if sd.exists() else []
    has_final   = sd.exists() and (sd / "final_train.csv").exists() and (sd / "final_val.csv").exists()

    # ---- Case A: existing k-fold files ----
    if kfold_files and not args.force_regenerate_splits:
        print(f"[splits] loading existing k-fold files from {sd}")
        folds = []
        for f in kfold_files:
            k = int(f.stem.split("_")[1])
            tr = pd.read_csv(f)
            va = pd.read_csv(sd / f"fold_{k}_val.csv")
            folds.append((tr, va))
        test_path = sd / "test.csv"
        test_df = pd.read_csv(test_path) if test_path.exists() else None
        return folds, test_df

    # ---- Case B: backward-compat final_train/val/test (fusion pipeline) ----
    if has_final and not args.force_regenerate_splits:
        print(f"[splits] loading final_train/val/test from {sd}")
        tr = pd.read_csv(sd / "final_train.csv")
        va = pd.read_csv(sd / "final_val.csv")
        te_path = sd / "final_test.csv"
        te = pd.read_csv(te_path) if te_path.exists() else None

        trainval = pd.concat([tr, va], ignore_index=True)
        trainval = trainval[["speaker_id", "label"]].drop_duplicates(subset="speaker_id").reset_index(drop=True)
        test_df = (te[["speaker_id", "label"]].drop_duplicates(subset="speaker_id").reset_index(drop=True)
                   if te is not None and not te.empty else None)

        # internal k-fold on train+val (identical to original script)
        if is_clf:
            split_iter = list(StratifiedKFold(args.n_cv_folds, shuffle=True, random_state=args.seed)
                              .split(trainval, trainval["label"]))
        else:
            split_iter = list(KFold(args.n_cv_folds, shuffle=True, random_state=args.seed)
                              .split(trainval))
        folds = []
        for tri, vai in split_iter:
            folds.append((trainval.iloc[tri].reset_index(drop=True),
                          trainval.iloc[vai].reset_index(drop=True)))
        return folds, test_df

    # ---- Case C: generate new splits ----
    if labels_df is None or "label" not in labels_df.columns:
        print("❌ Splits dir is missing and no label source found.")
        print("   Provide --labels-csv OR a 'label' column in --audio-features-csv.")
        sys.exit(1)

    print(f"[splits] generating new splits at {sd}  (seed={args.seed}, k={args.n_cv_folds}, task={args.task})")
    spk = labels_df.groupby("speaker_id")["label"].first().reset_index()

    if is_clf:
        trainval_df, test_df = train_test_split(
            spk, test_size=args.test_fraction, stratify=spk["label"], random_state=args.seed)
    else:
        trainval_df, test_df = train_test_split(
            spk, test_size=args.test_fraction, random_state=args.seed)
    trainval_df = trainval_df.reset_index(drop=True)
    test_df     = test_df.reset_index(drop=True)

    folds = _write_folds(sd, trainval_df, test_df, args.n_cv_folds, args.seed, is_clf)
    print(f"[splits] wrote {len(folds)} fold pairs + test to {sd}")
    return folds, test_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir", required=True,
                   help="dir with split CSVs; created + populated if missing")
    p.add_argument("--audio-features-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--task", choices=["classification", "regression"], required=True)
    p.add_argument("--n-cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--use-smote", action="store_true")
    p.add_argument("--force-recompute", action="store_true")
    p.add_argument("--force-regenerate-splits", action="store_true",
                   help="ignore existing split files and regenerate from labels")
    p.add_argument("--test-fraction", type=float, default=0.15,
                   help="held-out test fraction when generating splits")
    p.add_argument("--ensemble-models", nargs="+", default=[
        "linear", "random_forest", "hist_gradient_boosting", "svm", "knn"])
    p.add_argument("--labels-csv", default=None,
               help="optional speaker-level CSV used to generate splits when needed")
    p.add_argument("--speaker-col", default="speaker_id",
                   help="column name for speaker id (default: speaker_id)")
    p.add_argument("--label-col", default="label",
                   help="column name for the label/target (default: label)")
    args = p.parse_args()

    np.random.seed(args.seed)
    is_clf = (args.task == "classification")
    out = Path(args.output_dir) / "audio_only"
    out.mkdir(parents=True, exist_ok=True)

    # -- preflight --------------------------------------------------------
    old = out / "fold_0" / "model.joblib"
    if old.exists() and not args.force_recompute:
        try:
            m_old = joblib.load(old)
            names = [n for n, _ in m_old.estimators] if hasattr(m_old, "estimators") else None
            smote = ("imblearn" in type(m_old.estimators[0][1]).__module__) if names else None
            print(f"[preflight] existing model → {type(m_old).__name__} {names} (smote={smote})")
        except Exception:
            pass

    # -- audio features ---------------------------------------------------
    audio_raw = pd.read_csv(args.audio_features_csv)
    if "speaker_id" not in audio_raw.columns:
        print("❌ audio CSV needs a 'speaker_id' column"); sys.exit(1)

    exclude = {"speaker_id", "session_id", "utterance_id", "question_id",
               "label", "y_true", "target"}
    feats = [c for c in audio_raw.select_dtypes("number").columns if c not in exclude]
    if not feats:
        print("❌ No numeric audio feature columns"); sys.exit(1)

    audio = (audio_raw[["speaker_id"] + feats]
             .groupby("speaker_id").mean().reset_index())

    exclude = {"speaker_id", "session_id", "utterance_id", "question_id",
           "label", "y_true", "target", args.speaker_col, args.label_col}

    # -- labels source (only needed when generating splits) ---------------
    labels_df = None
    if args.labels_csv:
        ldf = pd.read_csv(args.labels_csv)
        missing = [c for c in (args.speaker_col, args.label_col) if c not in ldf.columns]
        if missing:
            print(f"❌ --labels-csv missing {missing}; found columns: {list(ldf.columns)}")
            sys.exit(1)
        labels_df = (ldf.rename(columns={args.speaker_col: "speaker_id",
                                         args.label_col:   "label"})
                        [["speaker_id", "label"]]
                        .groupby("speaker_id")["label"].first()
                        .reset_index())
    elif args.label_col in audio_raw.columns:
        if args.speaker_col not in audio_raw.columns:
            print(f"❌ audio CSV has no '{args.speaker_col}' column"); sys.exit(1)
        labels_df = (audio_raw.rename(columns={args.speaker_col: "speaker_id",
                                                args.label_col:   "label"})
                              [["speaker_id", "label"]]
                              .groupby("speaker_id")["label"].first()
                              .reset_index())

    print(f"Audio features: {len(feats)}   ensemble: {args.ensemble_models}")

    # -- splits -----------------------------------------------------------
    folds, test_df = resolve_splits(args, labels_df, is_clf)
    print(f"Folds: {len(folds)}   test speakers: {len(test_df) if test_df is not None else 0}")

    # -- cross-validation ------------------------------------------------
    oof_rows, fold_rows = [], []
    for k, (tr_df, va_df) in enumerate(folds):
        tr_k = audio.merge(tr_df[["speaker_id", "label"]], on="speaker_id", how="inner").fillna(0.0)
        va_k = audio.merge(va_df[["speaker_id", "label"]], on="speaker_id", how="inner").fillna(0.0)
        if len(tr_k) == 0 or len(va_k) == 0:
            print(f"  fold {k}: skipped (no overlapping speakers with audio features)")
            continue

        Xtr = tr_k[feats].to_numpy(float); ytr = tr_k["label"].to_numpy()
        Xva = va_k[feats].to_numpy(float); yva = va_k["label"].to_numpy()
        if is_clf:
            ytr, yva = ytr.astype(int), yva.astype(int)

        mdl = make_ensemble(args.task, args.ensemble_models, args.seed,
                            args.n_estimators, args.max_depth, args.use_smote).fit(Xtr, ytr)

        if is_clf:
            proba = mdl.predict_proba(Xva)[:, 1] if hasattr(mdl, "predict_proba") else None
            th    = best_threshold(proba, yva) if proba is not None else 0.5
            pred  = (proba >= th).astype(int) if proba is not None else mdl.predict(Xva)
            met   = clf_metrics(yva, pred, proba)
            row   = pd.DataFrame({"speaker_id": va_k.speaker_id.values,
                                  "y_true": yva, "y_pred": pred, "fold": k})
            if proba is not None:
                row["y_proba"] = proba
        else:
            pred = mdl.predict(Xva)
            met  = reg_metrics(yva, pred)
            row  = pd.DataFrame({"speaker_id": va_k.speaker_id.values,
                                 "y_true": yva, "y_pred": pred, "fold": k})

        fold_rows.append({"fold": k, **met})
        oof_rows.append(row)

        pretty = ", ".join(
            f"{kk}={vv:.4f}" for kk, vv in met.items()
            if isinstance(vv, (int, float)) and not (isinstance(vv, float) and np.isnan(vv))
        )
        print(f"  fold {k}: {pretty}")

    if not oof_rows:
        print("❌ No fold produced predictions; check that speaker_ids match audio features.")
        sys.exit(1)

    # -- aggregate OOF ----------------------------------------------------
    oof = pd.concat(oof_rows, ignore_index=True)
    if is_clf:
        agg = clf_metrics(oof.y_true.astype(int), oof.y_pred.astype(int),
                          oof["y_proba"].values if "y_proba" in oof else None)
    else:
        agg = reg_metrics(oof.y_true.values, oof.y_pred.values)

    oof.to_csv(out / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "fold_metrics.csv", index=False)
    (out / "aggregate_metrics.json").write_text(json.dumps(agg, indent=2))
    print("\nOOF aggregate:", json.dumps(agg, indent=2))

    # -- refit on all train+val, eval on held-out test --------------------
    if test_df is not None and len(test_df) > 0:
        # union of fold train/val speakers → training pool
        all_tv = pd.concat(
            [tr for tr, _ in folds] + [va for _, va in folds],
            ignore_index=True,
        )[["speaker_id", "label"]].drop_duplicates(subset="speaker_id").reset_index(drop=True)

        tv = audio.merge(all_tv, on="speaker_id", how="inner").fillna(0.0)
        Xtr = tv[feats].to_numpy(float); ytr = tv["label"].to_numpy()
        if is_clf:
            ytr = ytr.astype(int)

        mdl = make_ensemble(args.task, args.ensemble_models, args.seed,
                            args.n_estimators, args.max_depth, args.use_smote).fit(Xtr, ytr)
        joblib.dump(mdl, out / "model.joblib")

        te = audio.merge(test_df[["speaker_id", "label"]], on="speaker_id", how="inner").fillna(0.0)
        if len(te) == 0:
            print("\n⚠ Test split has no speakers overlapping the audio features; skipping test eval.")
        else:
            Xte = te[feats].to_numpy(float); yte = te["label"].to_numpy()
            if is_clf:
                yte   = yte.astype(int)
                proba = mdl.predict_proba(Xte)[:, 1] if hasattr(mdl, "predict_proba") else None
                th    = best_threshold(proba, yte) if proba is not None else 0.5
                pred  = (proba >= th).astype(int) if proba is not None else mdl.predict(Xte)
                met   = clf_metrics(yte, pred, proba)
            else:
                pred = mdl.predict(Xte); met = reg_metrics(yte, pred)

            pd.DataFrame({"speaker_id": te.speaker_id.values,
                          "y_true": yte, "y_pred": pred}) \
              .to_csv(out / "test_predictions.csv", index=False)
            (out / "test_metrics.json").write_text(json.dumps(met, indent=2))
            print("\nTEST:", json.dumps(met, indent=2))

    print(f"\nDone → {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)