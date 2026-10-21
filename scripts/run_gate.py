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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--summary", default="gate_summary.md")
    args = ap.parse_args()

    cfg = load_config(args.config)
    records = train_and_track(cfg, register=False)
    metric = cfg.gate.metric
    best = best_record(records, metric)

    baseline = load_baseline(cfg.gate.baseline_path)
    result = evaluate_gate(best.metrics, baseline, metric, cfg.gate.tolerance)

    base_str = f"{result.baseline:.4f}" if result.baseline is not None else "none"
    lines = [
        "## Credit Risk Quality Gate",
        "",
        f"**Result:** {'✅ PASS' if result.passed else '❌ FAIL'}",
        "",
        f"- Config: `{args.config}` (environment `{cfg.environment}`, "
        f"data `{cfg.data.version}`)",
        f"- Gate metric: **{metric}**",
        f"- Best candidate run: `{best.run_name}`",
        f"- Candidate {metric}: **{result.candidate:.4f}**",
        f"- Baseline {metric}: **{base_str}** (tolerance {cfg.gate.tolerance})",
        f"- Runs trained: {len(records)}",
        "",
        f"> {result.reason}",
        "",
        "| metric | candidate |",
        "| --- | --- |",
    ]
    for k, v in best.metrics.items():
        lines.append(f"| {k} | {v:.4f} |")
    summary = "\n".join(lines) + "\n"

    Path(args.summary).write_text(summary, encoding="utf-8")
    print(summary)

    # Also append to the GitHub Actions job summary if available.
    gh_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh_summary:
        with open(gh_summary, "a", encoding="utf-8") as fh:
            fh.write(summary)

    Path("gate_result.json").write_text(json.dumps(result.as_dict(), indent=2),
                                        encoding="utf-8")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
