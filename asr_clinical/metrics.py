from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def aggregate_predictions(
    pred_df: pd.DataFrame,
    task: str,
    aggregate_level: str,
    label_names: list[str] | None = None,
) -> pd.DataFrame:
    if aggregate_level == "question":
        return pred_df.copy()

    group_col = "session_id" if aggregate_level == "session" else "speaker_id"
    if task == "classification":
        prob_cols = [c for c in pred_df.columns if c.startswith("prob_")]
        agg = pred_df.groupby(group_col, as_index=False).agg(
            y_true=("y_true", "first"),
            **{c: (c, "mean") for c in prob_cols},
        )
        probs = agg[prob_cols].to_numpy()
        agg["y_pred"] = probs.argmax(axis=1)
        if label_names:
            agg["y_true_name"] = agg["y_true"].map(lambda x: label_names[int(x)])
            agg["y_pred_name"] = agg["y_pred"].map(lambda x: label_names[int(x)])
        return agg

    return pred_df.groupby(group_col, as_index=False).agg(
        y_true=("y_true", "first"),
        y_pred=("y_pred", "mean"),
    )


def classification_metrics(y_true, y_pred, label_names: list[str]) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


def score_predictions(pred_df, task: str, aggregate_level: str, label_names=None) -> dict:
    agg = aggregate_predictions(pred_df, task, aggregate_level, label_names)
    if task == "classification":
        metrics = classification_metrics(agg["y_true"], agg["y_pred"], label_names)
    else:
        metrics = regression_metrics(agg["y_true"], agg["y_pred"])
    metrics["n_units"] = len(agg)
    metrics["aggregate_level"] = aggregate_level
    return metrics
