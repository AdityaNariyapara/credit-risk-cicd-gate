"""Run a tracked training sweep and optionally register the best model.

Usage:
    python scripts/run_training.py --config configs/full.yaml --register
    python scripts/run_training.py --config configs/dev.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.config import load_config  # noqa: E402
from credit_risk.pipeline import train_and_track, records_to_frame, best_record  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--register", action="store_true",
                    help="Register the best model in the MLflow registry.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    records = train_and_track(cfg, register=args.register)

    frame = records_to_frame(records).sort_values(cfg.training.scoring, ascending=False)
    print(f"\n{len(records)} runs logged to {cfg.mlflow.tracking_uri} "
          f"(experiment={cfg.mlflow.experiment_name})\n")
    cols = ["run_name", "family", "seed", cfg.training.scoring, "roc_auc", "f1", "accuracy"]
    print(frame[cols].to_string(index=False))

    best = best_record(records, cfg.training.scoring)
    print(f"\nBest by {cfg.training.scoring}: {best.run_name} "
          f"= {best.metrics[cfg.training.scoring]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
