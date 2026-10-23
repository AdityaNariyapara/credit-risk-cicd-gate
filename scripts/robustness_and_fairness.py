"""Robustness of the v1 vs v2 drill and subgroup-aware fairness mitigation.

Two analyses that strengthen Tasks 7 and 8:

1. Robust data-change drill: train the final model on v1 and v2 across FIVE
   paired seeds, run a paired t-test on the per-seed PR-AUC differences, and
   report a bootstrap 95% confidence interval for the mean difference. This
   replaces the rough "within two times the noise" statement with a proper test.

2. Fairness mitigation: instead of only reporting the higher false-negative rate
   for Low-income customers, we CHOOSE a decision threshold per Income_Bracket
   group (an equal-opportunity policy) so that every group reaches a target
   false-negative rate, and report the before/after plus the precision cost.

Usage:
    python scripts/robustness_and_fairness.py --config configs/full.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.metrics import average_precision_score, precision_score, recall_score  # noqa: E402

from credit_risk.config import load_config  # noqa: E402
from credit_risk.data import load_dataset, split_X_y  # noqa: E402
from credit_risk.features import FeatureEngineer  # noqa: E402
from credit_risk.model import make_estimator  # noqa: E402

RF = ("random_forest", {"n_estimators": 300, "max_depth": 12, "class_weight": "balanced"})


def pr_auc_for(cfg, path, seed):
    df = load_dataset(path)
    X, y = split_X_y(df, cfg.data.target, cfg.data.id_column)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cfg.data.test_size,
                                              stratify=y, random_state=seed)
    fe = FeatureEngineer(cfg.features).fit(X_tr)
    est = make_estimator(RF[0], RF[1], seed)
    est.fit(fe.transform(X_tr), y_tr)
    prob = est.predict_proba(fe.transform(X_te))[:, 1]
    return average_precision_score(y_te, prob)


def robust_drill(cfg, seeds):
    v1 = np.array([pr_auc_for(cfg, "data/credit_card_customers_v1.csv", s) for s in seeds])
    v2 = np.array([pr_auc_for(cfg, "data/credit_card_customers_v2.csv", s) for s in seeds])
    diff = v2 - v1
    t_stat, p_val = stats.ttest_rel(v2, v1)
    # bootstrap 95% CI of the mean difference
    rng = np.random.default_rng(0)
    boots = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "seeds": seeds, "v1": v1.round(4).tolist(), "v2": v2.round(4).tolist(),
        "mean_v1": round(v1.mean(), 4), "mean_v2": round(v2.mean(), 4),
        "mean_diff": round(diff.mean(), 4), "paired_t": round(float(t_stat), 3),
        "p_value": round(float(p_val), 4), "ci95": [round(lo, 4), round(hi, 4)],
    }


def fairness_mitigation(cfg, target_fnr=0.20):
    df = load_dataset(cfg.data.path)
    X, y = split_X_y(df, cfg.data.target, cfg.data.id_column)
    seed = cfg.training.random_seed
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=cfg.data.test_size, stratify=y, random_state=seed)
    fe = FeatureEngineer(cfg.features).fit(X_tr)
    est = make_estimator(RF[0], RF[1], seed)
    est.fit(fe.transform(X_tr), y_tr)
    prob = est.predict_proba(fe.transform(X_te))[:, 1]
    g = pd.DataFrame({"grp": df.loc[idx_te, "Income_Bracket"].values,
                      "y": y_te.values, "p": prob})

    def rates(mask_pred, sub):
        tp = int(((sub.y == 1) & mask_pred & (sub.y == 1)).sum())
        pos = int((sub.y == 1).sum())
        fn = int(((sub.y == 1) & ~mask_pred).sum())
        fp = int(((sub.y == 0) & mask_pred).sum())
        prec = precision_score(sub.y, mask_pred.astype(int), zero_division=0)
        rec = recall_score(sub.y, mask_pred.astype(int), zero_division=0)
        return fn, fp, (fn / pos if pos else float("nan")), round(prec, 3), round(rec, 3)

    rows = []
    thresholds = {}
    for grp, sub in g.groupby("grp"):
        # before: global 0.5
        b_fn, b_fp, b_fnr, b_prec, _ = rates(sub.p >= 0.5, sub)
        # after: lower this group's threshold until its FN rate <= target
        t = 0.5
        for cand in np.round(np.arange(0.50, 0.04, -0.01), 2):
            _, _, fnr, _, _ = rates(sub.p >= cand, sub)
            t = cand
            if fnr <= target_fnr:
                break
        a_fn, a_fp, a_fnr, a_prec, _ = rates(sub.p >= t, sub)
        thresholds[grp] = float(t)
        rows.append({"group": grp, "threshold": t,
                     "fnr_before": round(b_fnr, 3), "fnr_after": round(a_fnr, 3),
                     "fp_before": b_fp, "fp_after": a_fp,
                     "precision_before": b_prec, "precision_after": a_prec})
    return {"target_fnr": target_fnr, "group_thresholds": thresholds, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123, 2024, 99])
    ap.add_argument("--out", default="docs/robustness_fairness.json")
    args = ap.parse_args()
    cfg = load_config(args.config)

    print("=== Robust v1 vs v2 drill (5 paired seeds) ===")
    drill = robust_drill(cfg, args.seeds)
    for k in ("mean_v1", "mean_v2", "mean_diff", "paired_t", "p_value", "ci95"):
        print(f"  {k}: {drill[k]}")
    verdict = ("statistically significant (p < 0.05)" if drill["p_value"] < 0.05
               else "NOT statistically significant (p >= 0.05)")
    print(f"  verdict: v2 improvement is {verdict}")

    print("\n=== Fairness mitigation: per-group thresholds (equal opportunity) ===")
    fair = fairness_mitigation(cfg)
    fdf = pd.DataFrame(fair["rows"])
    print(fdf.to_string(index=False))

    Path(args.out).write_text(json.dumps({"drill": drill, "fairness": fair}, indent=2),
                              encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
