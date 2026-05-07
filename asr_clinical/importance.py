from __future__ import annotations

import pandas as pd

from .metrics import aggregate_predictions, classification_metrics, regression_metrics


def question_ablation_importance(
    pred_df: pd.DataFrame,
    task: str,
    aggregate_level: str,
    label_names: list[str] | None = None,
) -> pd.DataFrame:
    if "question_id" not in pred_df.columns:
        raise ValueError("pred_df must contain question_id for question importance")
    if aggregate_level == "question":
        raise ValueError("question importance needs session or speaker aggregation")

    base_agg = aggregate_predictions(pred_df, task, aggregate_level, label_names)
    if task == "classification":
        base_score = classification_metrics(
            base_agg["y_true"], base_agg["y_pred"], label_names
        )["macro_f1"]
        metric_name = "macro_f1"
    else:
        base_score = -regression_metrics(base_agg["y_true"], base_agg["y_pred"])["mae"]
        metric_name = "negative_mae"

    rows = []
    for question_id in sorted(pred_df["question_id"].dropna().unique()):
        masked = pred_df[pred_df["question_id"] != question_id].copy()
        if masked.empty:
            continue
        agg = aggregate_predictions(masked, task, aggregate_level, label_names)
        if task == "classification":
            score = classification_metrics(agg["y_true"], agg["y_pred"], label_names)[
                "macro_f1"
            ]
        else:
            score = -regression_metrics(agg["y_true"], agg["y_pred"])["mae"]
        rows.append(
            {
                "question_id": question_id,
                "metric": metric_name,
                "base_score": base_score,
                "masked_score": score,
                "importance": base_score - score,
            }
        )
    return pd.DataFrame(rows).sort_values("importance", ascending=False)
