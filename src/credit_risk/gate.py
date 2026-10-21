"""Quality gate (Task 6 / Topic 2).

Compares a candidate model's metric against the baseline stored in the repo and
decides pass/fail. The rule is intentionally simple and inspectable:

    PASS  if  candidate_metric >= baseline_metric - tolerance

PR-AUC (average_precision) is the gate metric: it tracks ranking quality of the
minority (high-risk) class and ignores the easy majority, unlike plain accuracy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class GateResult:
    passed: bool
    metric: str
    candidate: float
    baseline: Optional[float]
    tolerance: float
    reason: str

    def as_dict(self):
        return asdict(self)


def load_baseline(path: str | Path) -> Optional[dict]:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def evaluate_gate(candidate_metrics: dict, baseline: Optional[dict],
                  metric: str, tolerance: float) -> GateResult:
    cand = float(candidate_metrics[metric])
    if baseline is None:
        return GateResult(
            passed=True, metric=metric, candidate=cand, baseline=None,
            tolerance=tolerance,
            reason="No baseline present; accepting candidate as the first baseline.",
        )
    base = float(baseline["metrics"][metric])
    threshold = base - tolerance
    passed = cand >= threshold
    reason = (
        f"candidate {metric}={cand:.4f} "
        f"{'>=' if passed else '<'} baseline-tolerance={threshold:.4f} "
        f"(baseline={base:.4f}, tolerance={tolerance})"
    )
    return GateResult(passed, metric, cand, base, tolerance, reason)
