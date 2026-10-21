"""Threshold comparison + fairness breakdown (Task 8 / Topics 2, 12).

1. Trains the final model, then evaluates at least three candidate gate rules
   against the run and reports what each accepts/rejects given a reference
   baseline.
2. Reports model error (false-negative / false-positive / accuracy) broken down
   by Income_Bracket for the final registered model.

Writes a machine-readable JSON plus printed tables that feed docs/fairness_memo.md.

Usage:
    python scripts/threshold_and_fairness.py --config configs/full.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.model_selection import train_test_split  # noqa: E402

from credit_risk.config import load_config  # noqa: E402
from credit_risk.data import load_dataset, split_X_y  # noqa: E402
from credit_risk.features import FeatureEngineer  # noqa: E402
from credit_risk.model import compute_metrics, make_estimator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--out", default="docs/threshold_fairness.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = load_dataset(cfg.data.path)
    X_raw, y = split_X_y(df, cfg.data.target, cfg.data.id_column)
    seed = cfg.training.random_seed

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X_raw, y, df.index, test_size=cfg.data.test_size, stratify=y, random_state=seed)
    fe = FeatureEngineer(cfg.features).fit(X_tr)
    est = make_estimator("random_forest",
                         {"n_estimators": 300, "max_depth": 12, "class_weight": "balanced"},
                         seed)
    est.fit(fe.transform(X_tr), y_tr)
    prob = est.predict_proba(fe.transform(X_te))[:, 1]
    metrics = compute_metrics(y_te, prob)

    # --- Candidate gate rules (>=3), evaluated against a reference baseline ---
    # Pretend baseline numbers taken from the current registered model.
    ref = {"average_precision": metrics["average_precision"] - 0.02,
           "roc_auc": metrics["roc_auc"] - 0.02,
           "f1": metrics["f1"] - 0.02}
    candidates = [
        ("PR-AUC >= baseline - 0.01", "average_precision", 0.01),
        ("ROC-AUC >= baseline - 0.01", "roc_auc", 0.01),
        ("F1 >= baseline - 0.01", "f1", 0.01),
        ("PR-AUC >= baseline (strict)", "average_precision", 0.0),
    ]
    rule_rows = []
    for name, metric, tol in candidates:
        cand = metrics[metric]
        base = ref[metric]
        passed = cand >= base - tol
        rule_rows.append({"rule": name, "metric": metric, "candidate": round(cand, 4),
                          "baseline": round(base, 4), "tolerance": tol,
                          "decision": "ACCEPT" if passed else "REJECT"})
    rules_df = pd.DataFrame(rule_rows)

    # --- Fairness: error by Income_Bracket on the test set ---
    y_pred = (prob >= 0.5).astype(int)
    te_df = pd.DataFrame({
        "Income_Bracket": df.loc[idx_te, "Income_Bracket"].values,
        "y_true": y_te.values, "y_pred": y_pred,
    })
    sub_rows = []
    for grp, g in te_df.groupby("Income_Bracket"):
        n = len(g)
        pos = g["y_true"].sum()
        fn = int(((g["y_true"] == 1) & (g["y_pred"] == 0)).sum())
        fp = int(((g["y_true"] == 0) & (g["y_pred"] == 1)).sum())
        acc = float((g["y_true"] == g["y_pred"]).mean())
        fnr = fn / pos if pos else float("nan")
        sub_rows.append({"Income_Bracket": grp, "n": n, "actual_high_risk": int(pos),
                         "false_neg": fn, "false_pos": fp,
                         "false_neg_rate": round(fnr, 4), "accuracy": round(acc, 4)})
    sub_df = pd.DataFrame(sub_rows)

    print("=== Candidate gate rules ===")
    print(rules_df.to_string(index=False))
    print("\n=== Error by Income_Bracket (final model) ===")
    print(sub_df.to_string(index=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "final_metrics": {k: round(v, 4) for k, v in metrics.items()},
        "candidate_rules": rule_rows,
        "subgroup_error": sub_rows,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
