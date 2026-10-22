# Task 1 - Lifecycle Map (Topic 1)

**Team:** Aditya Nariyapara (B009), Shreya Dutt (B016)

This project is mapped onto the classic DevOps lifecycle and the MLOps
lifecycle. Every stage points at a concrete artefact or workflow step in this
repository.

## DevOps ↔ MLOps ↔ this repository

| DevOps stage | MLOps activity | Concrete artefact / step in this repo |
|--------------|----------------|----------------------------------------|
| **Plan** | Problem framing, metric choice, gate design | `docs/metric_justification.md`, `docs/report.md`, this map |
| **Code** | Pipeline, feature and model code | `src/credit_risk/` (`config.py`, `features.py`, `model.py`, `pipeline.py`, `gate.py`) |
| **Build** | Reproducible environment + data materialisation | `requirements.txt`, `dvc pull` of DVC-tracked data |
| **Test** | Unit tests + data/feature validation | `tests/` (feature edge cases, config validation, gate logic) |
| **Release** | Model selection, registry, baseline promotion | MLflow registry (`credit_risk_gate_model`), `scripts/update_baseline.py`, `baseline/baseline_metrics.json` |
| **Deploy** | Gating the candidate before it replaces the baseline | `.github/workflows/ci.yml` + `scripts/run_gate.py` (the merge is the "deploy to baseline") |
| **Operate** | Retraining on new data through the same pipeline | Task 7 data-change drill: `scripts/data_change_drill.py` with v2 |
| **Monitor** | Metric + subgroup/fairness tracking across runs | MLflow experiment history, `scripts/threshold_and_fairness.py`, `docs/fairness_memo.md` |

## Where ML delivery differs from ordinary software delivery

1. **Correctness is statistical, not binary.** A traditional unit test passes or
 fails deterministically. A model is "good enough" only relative to a
 baseline and within run-to-run noise, so our **test** stage needs a
 *quality gate* (`run_gate.py`) comparing a metric to a stored baseline, plus
 a noise analysis across seeds (Task 7), not just assertions.

2. **Data is a versioned input, not just configuration.** In software the code
 fully determines behaviour. Here the same code on `v1` vs `v2` data produces
 different models, so data must be versioned alongside code (**DVC**) and the
 data version recorded with every run. Reproducing a result means restoring
 *code + config + data* together.

3. **Experiments are first-class artefacts.** Software builds are
 interchangeable; ML runs each carry parameters, metrics and a model artefact
 worth keeping and comparing. We log every run to **MLflow** and select the
 model from that history rather than from a single build.

4. **Metric and fairness choices are design decisions that must be defended.**
 Accuracy is misleading under a 1-in-5 class imbalance; the gate metric
 (PR-AUC) and its threshold are justified and compared against alternatives
 (Task 8), and error is checked across Income_Bracket subgroups - concerns
 that have no analogue in ordinary CI.

## Stages intentionally left out, with reasons

- **Serving / real-time deployment infrastructure.** The brief's "deploy" is the
 acceptance of a model past the gate and its registration, not a live
 prediction endpoint. We stop at the registry: a hosted API adds no marks
 against the rubric and needs paid infrastructure the brief forbids.
- **Production monitoring / drift alerting dashboards.** Simulated here only as
 the v1→v2 data-change drill. Continuous drift monitoring belongs to a later
 project (churn/drift) and would require a running service.
- **Infrastructure-as-code / environment provisioning.** Replaced by a single
 `requirements.txt` + GitHub-hosted runners; no cloud account is used.
