from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split


def build_parser():
    parser = argparse.ArgumentParser(
        description="Estimate question importance with SHAP from saved question predictions."
    )
    parser.add_argument("--predictions-file", required=True)
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument(
        "--group-level",
        choices=["session", "speaker"],
        default="speaker",
        help="Prediction unit to explain.",
    )
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    return parser


def make_feature_table(pred_df: pd.DataFrame, task: str, group_level: str):
    group_col = "session_id" if group_level == "session" else "speaker_id"
    id_cols = [group_col, "question_id"]

    if task == "classification":
        value_cols = [c for c in pred_df.columns if c.startswith("prob_")]
    else:
        value_cols = ["y_pred"]

    parts = []
    for value_col in value_cols:
        pivot = pred_df.pivot_table(
            index=group_col,
            columns="question_id",
            values=value_col,
            aggfunc="mean",
        )
        pivot.columns = [f"{q}__{value_col}" for q in pivot.columns]
        parts.append(pivot)

    x = pd.concat(parts, axis=1)
    y = pred_df.groupby(group_col)["y_true"].first().loc[x.index]
    return x, y


def aggregate_shap_by_question(shap_values, feature_names):
    values = np.asarray(shap_values)
    if isinstance(shap_values, list):
        values = np.stack(shap_values, axis=0)
    if values.ndim == 3:
        values = np.mean(np.abs(values), axis=(0, 1))
    else:
        values = np.mean(np.abs(values), axis=0)

    rows = []
    for feature_name, value in zip(feature_names, values):
        question_id = feature_name.split("__", 1)[0]
        rows.append({"question_id": question_id, "feature": feature_name, "mean_abs_shap": value})
    feature_df = pd.DataFrame(rows)
    question_df = (
        feature_df.groupby("question_id", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )
    return question_df, feature_df.sort_values("mean_abs_shap", ascending=False)


def main():
    args = build_parser().parse_args()
    try:
        import shap
    except ImportError as exc:
        raise SystemExit(
            "SHAP is not installed. Install dependencies with `pip install -r requirements.txt` "
            "or run `pip install shap`."
        ) from exc

    pred_df = pd.read_csv(args.predictions_file)
    x, y = make_feature_table(pred_df, args.task, args.group_level)
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    x_values = imputer.fit_transform(x)

    stratify = y if args.task == "classification" and y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x_values,
        y,
        test_size=0.25,
        random_state=args.seed,
        stratify=stratify,
    )

    if args.task == "classification":
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.seed,
            class_weight="balanced",
            min_samples_leaf=2,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            random_state=args.seed,
            min_samples_leaf=2,
        )

    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    if args.task == "classification":
        print({"heldout_macro_f1": f1_score(y_test, preds, average="macro", zero_division=0)})
    else:
        print({"heldout_mae": mean_absolute_error(y_test, preds)})

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_values)
    question_df, feature_df = aggregate_shap_by_question(shap_values, x.columns.tolist())

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    question_df.to_csv(output_path, index=False)
    feature_df.to_csv(output_path.with_name(output_path.stem + "_features.csv"), index=False)


if __name__ == "__main__":
    main()
