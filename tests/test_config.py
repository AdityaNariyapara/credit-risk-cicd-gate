"""Config validation tests (Task 3): good configs load, broken ones fail loudly."""
from pathlib import Path

import pytest
import yaml

from credit_risk.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("name", ["dev.yaml", "ci.yaml", "full.yaml"])
def test_real_configs_load(name):
    cfg = load_config(CONFIGS / name)
    assert cfg.environment
    assert cfg.training.models


def test_broken_config_fails_fast(tmp_path):
    bad = {"environment": "x", "data": {"version": "v1", "path": "p",
           "test_size": 5.0},  # test_size must be < 1
           "features": {}, "training": {"models": []}, "mlflow": {}, "gate": {}}
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_unknown_key_rejected(tmp_path):
    cfg = yaml.safe_load((CONFIGS / "dev.yaml").read_text())
    cfg["surprise_key"] = 123
    p = tmp_path / "extra.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_age_labels_must_match_bins(tmp_path):
    cfg = yaml.safe_load((CONFIGS / "dev.yaml").read_text())
    cfg["features"]["age_labels"] = ["only", "two"]  # bins imply 4 labels
    p = tmp_path / "mismatch.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")
