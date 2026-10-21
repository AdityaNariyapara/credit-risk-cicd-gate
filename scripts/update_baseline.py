"""Update the stored baseline (Task 6 baseline rule).

The baseline is NOT updated automatically by CI. A maintainer runs this
explicitly, on the main branch, only after a model has passed the gate and been
reviewed. This keeps the baseline from silently drifting down.

Usage:
    python scripts/update_baseline.py --config configs/ci.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.config import load_config  # noqa: E402
from credit_risk.pipeline import train_and_track, best_record  # noqa: E402


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    records = train_and_track(cfg, register=False)
    best = best_record(records, cfg.gate.metric)

    payload = {
        "metric_for_gate": cfg.gate.metric,
        "metrics": best.metrics,
        "data_version": cfg.data.version,
        "config": args.config,
        "run_name": best.run_name,
        "git_commit": _git_commit(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    out = Path(cfg.gate.baseline_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Baseline written to {out}:\n{json.dumps(payload, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
