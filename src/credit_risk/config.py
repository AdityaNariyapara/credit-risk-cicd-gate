"""Typed, validated configuration loaded from YAML (Task 3, Topic 10).

Every tunable value — paths, feature thresholds, the hyperparameter search
space, the random seed, MLflow settings and the quality-gate threshold — lives
in YAML. Nothing here is hard-coded in the modelling code. Configs are
validated with pydantic the moment they are loaded, so a broken config fails
immediately with a clear message instead of midway through a run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


class DataConfig(BaseModel):
    version: str = Field(..., description="Dataset version label, e.g. v1 / v2")
    path: str = Field(..., description="Path to the CSV (DVC-tracked working copy)")
    target: str = "High_Risk_Customer"
    id_column: str = "Customer_ID"
    test_size: float = Field(0.25, gt=0.0, lt=1.0)


class FeatureConfig(BaseModel):
    # The utilisation risk cut-off is a feature threshold, not a magic number.
    utilisation_high_cutoff: float = Field(0.5, ge=0.0, le=5.0)
    payment_delay_flag_days: int = Field(5, ge=0)
    age_bins: List[int] = Field(default_factory=lambda: [0, 25, 40, 60, 200])
    age_labels: List[str] = Field(
        default_factory=lambda: ["young", "early_career", "established", "senior"]
    )

    @field_validator("age_labels")
    @classmethod
    def _labels_match_bins(cls, v: List[str], info):
        bins = info.data.get("age_bins")
        if bins is not None and len(v) != len(bins) - 1:
            raise ValueError(
                f"age_labels must have len(age_bins)-1 entries "
                f"(got {len(v)} labels for {len(bins)} bins)"
            )
        return v


class ModelSpace(BaseModel):
    name: str
    family: Literal["logistic_regression", "random_forest"]
    params: Dict[str, List[Any]] = Field(default_factory=dict)


class TrainingConfig(BaseModel):
    random_seed: int = 42
    seeds: List[int] = Field(default_factory=lambda: [42])
    cv_folds: int = Field(3, ge=2)
    scoring: str = "average_precision"
    models: List[ModelSpace]

    @field_validator("models")
    @classmethod
    def _at_least_one_model(cls, v):
        if not v:
            raise ValueError("training.models must list at least one model family")
        return v


class MLflowConfig(BaseModel):
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "credit_risk"
    registered_model_name: str = "credit_risk_gate_model"


class GateConfig(BaseModel):
    metric: str = "average_precision"
    # The model must be at least baseline - tolerance to pass the gate.
    min_delta: float = Field(0.0, description="Allowed drop vs baseline (<=0 strict)")
    tolerance: float = Field(0.0, ge=0.0, description="Noise band around baseline")
    baseline_path: str = "baseline/baseline_metrics.json"


class Config(BaseModel):
    environment: str
    data: DataConfig
    features: FeatureConfig
    training: TrainingConfig
    mlflow: MLflowConfig
    gate: GateConfig

    # Reject unknown keys so typos in YAML are caught, not silently ignored.
    model_config = {"extra": "forbid"}


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config. Raises a clear error on any problem."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} did not parse to a mapping (got {type(raw)})")
    try:
        return Config(**raw)
    except ValidationError as exc:
        raise ValueError(
            f"Invalid configuration in {path}:\n{exc}"
        ) from exc


def config_to_dict(cfg: Config) -> Dict[str, Any]:
    """Flat-ish dict suitable for logging the exact config with an MLflow run."""
    return cfg.model_dump()
