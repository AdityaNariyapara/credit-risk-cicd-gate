"""Data-change drill: v1 vs v2 (Task 7 / Topics 2, 4).

Trains the best model family on both data versions across several random seeds,
reports mean +/- std of the gate metric, and judges whether any v2 improvement
is real or inside normal run-to-run variation. Data version identifiers are
logged with every run.

Usage:
    python scripts/data_change_drill.py --config configs/full.yaml \
        --v1 data/credit_card_customers_v1.csv \
        --v2 data/credit_card_customers_v2.csv --seeds 42 7 123
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mlflow  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from credit_risk.config import load_config  # noqa: E402
from credit_risk.data import load_dataset, split_X_y  # noqa: E402
from credit_risk.features import FeatureEngineer  # noqa: E402
from credit_risk.model import compute_metrics, make_estimator  # noqa: E402


def run_version(cfg, path, version_label, seeds, metric):
    df = load_dataset(path)
    X_raw, y = split_X_y(df, cfg.data.target, cfg.data.id_column)
    # Use the best family found in tuning: random forest at a solid depth.
    family, params = "random_forest", {"n_estimators": 300, "max_depth": 12,
                                        "class_weight": "balanced"}
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment("credit_risk_drill")
    scores = []
    for seed in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_raw, y, test_size=cfg.data.test_size, stratify=y, random_state=seed)
        fe = FeatureEngineer(cfg.features).fit(X_tr)
        est = make_estimator(family, params, seed)
        est.fit(fe.transform(X_tr), y_tr)
        prob = est.predict_proba(fe.transform(X_te))[:, 1]
        m = compute_metrics(y_te, prob)
        scores.append(m[metric])
        with mlflow.start_run(run_name=f"drill_{version_label}_seed{seed}"):
            mlflow.log_param("data_version", version_label)
            mlflow.log_param("seed", seed)
            mlflow.set_tag("data_version", version_label)
            mlflow.log_metrics(m)
    return np.array(scores)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--v1", default="data/credit_card_customers_v1.csv")
    ap.add_argument("--v2", default="data/credit_card_customers_v2.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    args = ap.parse_args()

    cfg = load_config(args.config)
    metric = cfg.gate.metric

    s1 = run_version(cfg, args.v1, "v1", args.seeds, metric)
    s2 = run_version(cfg, args.v2, "v2", args.seeds, metric)

    frame = pd.DataFrame({
        "data_version": ["v1", "v2"],
        f"mean_{metric}": [s1.mean(), s2.mean()],
        f"std_{metric}": [s1.std(ddof=1), s2.std(ddof=1)],
        "n_seeds": [len(s1), len(s2)],
    })
    delta = s2.mean() - s1.mean()
    pooled = np.sqrt(s1.var(ddof=1) / len(s1) + s2.var(ddof=1) / len(s2))
    print(frame.to_string(index=False))
    print(f"\nv2 - v1 mean {metric} delta = {delta:+.4f}")
    print(f"Approx std error of the difference = {pooled:.4f}")
    real = abs(delta) > 2 * pooled
    print(f"Delta is {'LARGER' if real else 'WITHIN'} ~2x run-to-run noise "
          f"-> {'likely real' if real else 'not distinguishable from noise'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
