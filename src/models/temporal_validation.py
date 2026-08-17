"""Expanding-window temporal cross-validation for churn models.

Instead of a random train/test split, we simulate production conditions:
train on historical data, test on future data. This proves the model
generalises forward in time, not just across a random holdout.
"""
from __future__ import annotations

import json
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import settings
from src.data.preprocessor import build_preprocessor, _clean, DROP_COLS
from src.models.churn_predictor import train_logistic, train_random_forest
from src.utils.logger import get_logger

log = get_logger(__name__)


def expanding_window_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    model_fn=None,
) -> dict:
    """Run expanding-window temporal CV and return per-fold + aggregate metrics.

    Splits the dataset into (n_splits + 1) cohorts by signup_date.
    For each fold k (1..n_splits): train on cohorts 0..k-1, test on cohort k.
    """
    if model_fn is None:
        model_fn = train_logistic

    df = _clean(df)
    if "signup_date" not in df.columns:
        raise ValueError("DataFrame needs 'signup_date' for temporal CV.")

    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df = df.sort_values("signup_date").reset_index(drop=True)

    cohort_edges = pd.qcut(
        df["signup_date"].rank(method="first"),
        q=n_splits + 1,
        labels=False,
    )
    df["_cohort"] = cohort_edges

    y_col = settings.target_col
    drop_cols = [y_col, *DROP_COLS, "signup_date", "_cohort"]

    fold_results: List[dict] = []

    for k in range(1, n_splits + 1):
        train_df = df[df["_cohort"] < k]
        test_df = df[df["_cohort"] == k]

        if len(test_df) < 20 or train_df[y_col].nunique() < 2:
            log.warning("Fold %d skipped (too few samples).", k)
            continue

        feat_train = train_df.drop(columns=drop_cols)
        feat_test = test_df.drop(columns=drop_cols)
        y_train = train_df[y_col].to_numpy()
        y_test = test_df[y_col].to_numpy()

        pre = build_preprocessor()
        X_train = pre.fit_transform(feat_train)
        X_test = pre.transform(feat_test)

        result = model_fn(X_train, y_train, X_test, y_test)

        fold_results.append({
            "fold": k,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "roc_auc": result.metrics["roc_auc"],
            "f1": result.metrics["f1"],
            "precision": result.metrics["precision"],
            "recall": result.metrics["recall"],
        })
        log.info(
            "Fold %d: train=%d, test=%d, ROC-AUC=%.4f",
            k, len(train_df), len(test_df), result.metrics["roc_auc"],
        )

    if not fold_results:
        return {"folds": [], "mean_roc_auc": 0.0, "std_roc_auc": 0.0}

    aucs = [f["roc_auc"] for f in fold_results]
    summary = {
        "folds": fold_results,
        "mean_roc_auc": round(float(np.mean(aucs)), 4),
        "std_roc_auc": round(float(np.std(aucs)), 4),
        "min_roc_auc": round(float(np.min(aucs)), 4),
        "max_roc_auc": round(float(np.max(aucs)), 4),
        "n_splits": n_splits,
    }

    out_path = settings.report_dir / "temporal_cv.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    log.info(
        "Temporal CV complete: mean ROC-AUC=%.4f (±%.4f) across %d folds → %s",
        summary["mean_roc_auc"], summary["std_roc_auc"], len(fold_results), out_path,
    )
    return summary
