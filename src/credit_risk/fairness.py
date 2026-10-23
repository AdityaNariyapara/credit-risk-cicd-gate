"""Subgroup fairness check used by the quality gate (Task 8).

Computes the false-negative rate of a model across Income_Bracket groups and the
largest gap between groups. The gate reports this on every run and, if
gate.max_fairness_gap is configured, fails when the gap is too large.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config
from .data import load_dataset, split_X_y
from .features import FeatureEngineer
from .model import compute_metrics, make_estimator


@dataclass
class FairnessResult:
    group_fnr: Dict[str, float]
    max_gap: float

    def summary_rows(self):
        return [(g, round(v, 3)) for g, v in sorted(self.group_fnr.items())]


def subgroup_fn_gap(cfg: Config, threshold: float = 0.5) -> FairnessResult:
    """Train the gate's best family on one split and measure the false-negative
    rate per Income_Bracket group."""
    df = load_dataset(cfg.data.path)
    X, y = split_X_y(df, cfg.data.target, cfg.data.id_column)
    seed = cfg.training.random_seed
    X_tr, X_te, y_tr, y_te, _, idx_te = train_test_split(
        X, y, df.index, test_size=cfg.data.test_size, stratify=y, random_state=seed)
    fe = FeatureEngineer(cfg.features).fit(X_tr)
    Xt_tr, Xt_te = fe.transform(X_tr), fe.transform(X_te)

    # Pick the better family on this split, among those in the config.
    best = None
    for m in cfg.training.models:
        params = {k: v[0] for k, v in m.params.items()}
        est = make_estimator(m.family, params, seed)
        est.fit(Xt_tr, y_tr)
        prob = est.predict_proba(Xt_te)[:, 1]
        ap = compute_metrics(y_te, prob)["average_precision"]
        if best is None or ap > best[0]:
            best = (ap, prob)

    pred = (best[1] >= threshold).astype(int)
    g = pd.DataFrame({"grp": df.loc[idx_te, "Income_Bracket"].values,
                      "y": y_te.values, "pred": pred})
    fnr = {}
    for grp, sub in g.groupby("grp"):
        pos = int((sub.y == 1).sum())
        fn = int(((sub.y == 1) & (sub.pred == 0)).sum())
        fnr[str(grp)] = (fn / pos) if pos else 0.0
    gap = (max(fnr.values()) - min(fnr.values())) if fnr else 0.0
    return FairnessResult(group_fnr=fnr, max_gap=gap)
