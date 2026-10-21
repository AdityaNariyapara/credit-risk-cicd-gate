"""End-to-end tracked training pipeline (Tasks 5 & 6 / Topic 12).

Loads a validated config, versions-aware loads the data, engineers features,
sweeps the configured model families / hyperparameters / seeds, logs every run
to MLflow with its params, metrics, dataset version, full config and model
artefact, then selects and (optionally) registers the best model.
"""
from __future__ import annotations

import itertools
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config, config_to_dict
from .data import load_dataset, split_X_y
from .features import FeatureEngineer
from .model import compute_metrics, make_estimator


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


@dataclass
class RunRecord:
    run_name: str
    family: str
    seed: int
    params: dict
    metrics: Dict[str, float]
    run_id: str


def _param_grid(params: Dict[str, list]) -> List[dict]:
    if not params:
        return [{}]
    keys = list(params.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*params.values())]


def train_and_track(cfg: Config, select_metric: Optional[str] = None,
                    register: bool = False) -> List[RunRecord]:
    """Run the full sweep, log to MLflow, return all run records."""
    select_metric = select_metric or cfg.training.scoring
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    df = load_dataset(cfg.data.path)
    X_raw, y = split_X_y(df, cfg.data.target, cfg.data.id_column)
    commit = _git_commit()

    records: List[RunRecord] = []

    for seed in cfg.training.seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_raw, y, test_size=cfg.data.test_size,
            stratify=y, random_state=seed,
        )
        fe = FeatureEngineer(cfg.features).fit(X_tr)
        Xt_tr, Xt_te = fe.transform(X_tr), fe.transform(X_te)

        for model in cfg.training.models:
            for params in _param_grid(model.params):
                run_name = f"{model.name}_seed{seed}_" + "_".join(
                    f"{k}{v}" for k, v in params.items()
                )
                with mlflow.start_run(run_name=run_name) as run:
                    est = make_estimator(model.family, params, seed)
                    est.fit(Xt_tr, y_tr)
                    y_prob = est.predict_proba(Xt_te)[:, 1]
                    metrics = compute_metrics(y_te, y_prob)

                    # Params, data version, config and provenance.
                    mlflow.log_params({f"hp_{k}": v for k, v in params.items()})
                    mlflow.log_param("family", model.family)
                    mlflow.log_param("seed", seed)
                    mlflow.log_param("data_version", cfg.data.version)
                    mlflow.log_param("environment", cfg.environment)
                    mlflow.set_tag("git_commit", commit)
                    mlflow.set_tag("data_version", cfg.data.version)
                    mlflow.log_metrics(metrics)

                    # Store the exact config used for this run.
                    cfg_path = Path("run_config.json")
                    cfg_path.write_text(json.dumps(config_to_dict(cfg), indent=2,
                                                   default=str), encoding="utf-8")
                    mlflow.log_artifact(str(cfg_path), artifact_path="config")
                    cfg_path.unlink(missing_ok=True)

                    mlflow.sklearn.log_model(
                        est, name="model", serialization_format="cloudpickle"
                    )

                    records.append(RunRecord(
                        run_name, model.family, seed, params, metrics, run.info.run_id
                    ))

    if register and records:
        best = max(records, key=lambda r: r.metrics[select_metric])
        model_uri = f"runs:/{best.run_id}/model"
        mlflow.register_model(model_uri, cfg.mlflow.registered_model_name)
    return records


def records_to_frame(records: List[RunRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {"run_name": r.run_name, "family": r.family, "seed": r.seed,
               "run_id": r.run_id}
        row.update(r.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def best_record(records: List[RunRecord], metric: str) -> RunRecord:
    return max(records, key=lambda r: r.metrics[metric])
