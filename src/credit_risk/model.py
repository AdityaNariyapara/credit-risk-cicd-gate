"""Model factory + metrics (Topic 12).

Model families are chosen from YAML; nothing is hard-coded here beyond the
mapping from a family name to an sklearn estimator.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_estimator(family: str, params: dict, random_seed: int):
    """Return an sklearn pipeline for one model family with concrete params."""
    if family == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=random_seed, **params)
        # Scale features so regularised logistic regression behaves.
        return Pipeline([("scale", StandardScaler()), ("clf", clf)])
    if family == "random_forest":
        clf = RandomForestClassifier(random_state=random_seed, n_jobs=-1, **params)
        return Pipeline([("clf", clf)])
    raise ValueError(f"Unknown model family: {family}")


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    """Metrics for an imbalanced binary target. PR-AUC (average precision) and
    ROC-AUC are threshold-free; the rest use a 0.5 decision threshold."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((y_pred == np.asarray(y_true)).mean()),
    }
