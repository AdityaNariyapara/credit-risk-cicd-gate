"""CI quality gate (Task 6).

Trains with the given config, selects the best model, compares it to the
baseline stored in the repo, writes a short Markdown summary and exits non-zero
if the gate fails so the pull request is blocked.

Usage:
    python scripts/run_gate.py --config configs/ci.yaml --summary gate_summary.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make stdout UTF-8 so the summary (which contains ✅/❌) prints on Windows
# consoles too, not only the UTF-8 Linux CI runner.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.config import load_config  # noqa: E402
from credit_risk.gate import evaluate_gate, load_baseline  # noqa: E402
from credit_risk.pipeline import train_and_track, best_record  # noqa: E402
from credit_risk.data import load_dataset  # noqa: E402
from credit_risk.validation import validate_dataset  # noqa: E402
from credit_risk.fairness import subgroup_fn_gap  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--summary", default="gate_summary.md")
    args = ap.parse_args()

    cfg = load_config(args.config)
    metric = cfg.gate.metric

    # --- Data gate: reject a bad batch before any training happens. ---
    data_issues = []
    if cfg.data_validation.enabled:
        df = load_dataset(cfg.data.path)
        vres = validate_dataset(df, cfg.data, cfg.data_validation)
        data_issues = vres.issues
        if not vres.ok:
            summary = ("## Credit Risk Quality Gate\n\n**Result:** ❌ FAIL (data gate)\n\n"
                       + vres.summary() + "\n")
            Path(args.summary).write_text(summary, encoding="utf-8")
            print(summary)
            gh = os.environ.get("GITHUB_STEP_SUMMARY")
            if gh:
                open(gh, "a", encoding="utf-8").write(summary)
            return 1

    # --- Model gate: train, select best, compare to baseline. ---
    records = train_and_track(cfg, register=False)
    best = best_record(records, metric)

    baseline = load_baseline(cfg.gate.baseline_path)
    result = evaluate_gate(best.metrics, baseline, metric, cfg.gate.tolerance)

    # --- Fairness check: report subgroup FN gap; enforce if configured. ---
    fair = subgroup_fn_gap(cfg)
    fairness_passed = True
    if cfg.gate.max_fairness_gap is not None and fair.max_gap > cfg.gate.max_fairness_gap:
        fairness_passed = False
    overall_passed = result.passed and fairness_passed

    base_str = f"{result.baseline:.4f}" if result.baseline is not None else "none"
    gap_limit = ("none" if cfg.gate.max_fairness_gap is None
                 else f"{cfg.gate.max_fairness_gap}")
    lines = [
        "## Credit Risk Quality Gate",
        "",
        f"**Result:** {'✅ PASS' if overall_passed else '❌ FAIL'}",
        "",
        f"- Config: `{args.config}` (environment `{cfg.environment}`, "
        f"data `{cfg.data.version}`)",
        "- Data gate: ✅ passed" if cfg.data_validation.enabled else "- Data gate: (disabled)",
        f"- Gate metric: **{metric}**",
        f"- Best candidate run: `{best.run_name}`",
        f"- Candidate {metric}: **{result.candidate:.4f}**",
        f"- Baseline {metric}: **{base_str}** (tolerance {cfg.gate.tolerance})",
        f"- Model gate: {'✅ pass' if result.passed else '❌ fail'}",
        f"- Fairness (subgroup FN gap): **{fair.max_gap:.3f}** "
        f"(limit {gap_limit}) -> {'✅ ok' if fairness_passed else '❌ too large'}",
        f"- Runs trained: {len(records)}",
        "",
        f"> {result.reason}",
        "",
        "| metric | candidate |",
        "| --- | --- |",
    ]
    for k, v in best.metrics.items():
        lines.append(f"| {k} | {v:.4f} |")
    lines += ["", "| Income_Bracket | false-negative rate |", "| --- | --- |"]
    for grp, v in fair.summary_rows():
        lines.append(f"| {grp} | {v:.3f} |")
    summary = "\n".join(lines) + "\n"

    Path(args.summary).write_text(summary, encoding="utf-8")
    print(summary)

    # Also append to the GitHub Actions job summary if available.
    gh_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh_summary:
        with open(gh_summary, "a", encoding="utf-8") as fh:
            fh.write(summary)

    out = result.as_dict()
    out["fairness"] = {"group_fnr": fair.group_fnr, "max_gap": fair.max_gap,
                       "passed": fairness_passed}
    out["data_gate_issues"] = data_issues
    Path("gate_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
