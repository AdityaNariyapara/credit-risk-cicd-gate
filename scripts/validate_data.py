"""Standalone data gate. Validates a dataset against the config and exits
non-zero if the batch is unacceptable.

Usage:
    python scripts/validate_data.py --config configs/ci.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.config import load_config  # noqa: E402
from credit_risk.data import load_dataset  # noqa: E402
from credit_risk.validation import validate_dataset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    df = load_dataset(cfg.data.path)
    res = validate_dataset(df, cfg.data, cfg.data_validation)
    print(res.summary())
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
