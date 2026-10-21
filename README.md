# Credit Risk Model with a CI/CD Quality Gate

**Elective II — MLOps · Project 1**
**Team:** Aditya Nariyapara (B009) · Shreya Dutt (B016)

A retail-bank credit-default model where **every change to code, config, features
or data is versioned, tested, tracked, and accepted only if the model is at
least as good as the stored baseline** — enforced automatically on every pull
request, and with the gate's threshold *justified*, not just implemented.

---

## Repository layout

```
credit-risk-cicd-gate/
├── .github/
│   ├── workflows/ci.yml        # CI quality gate (tests + train + gate vs baseline)
│   └── CODEOWNERS              # protects the baseline from silent changes
├── configs/
│   ├── dev.yaml               # fast local config
│   ├── ci.yaml                # config used by the CI gate
│   └── full.yaml              # wide sweep (generates the tracked runs)
├── data/
│   ├── credit_card_customers_v1.csv.dvc   # DVC pointer (CSV is gitignored)
│   └── credit_card_customers_v2.csv.dvc   # enters as a tracked change (Task 7)
├── dvcstore/                  # in-repo DVC remote (content-addressed blobs)
├── src/credit_risk/           # package: config, data, features, model, pipeline, gate
├── scripts/                   # run_training, run_gate, update_baseline, drill, threshold+fairness
├── tests/                     # pytest: features (edge cases), config validation, gate logic
├── baseline/baseline_metrics.json   # the quality-gate baseline (governed)
├── docs/                      # lifecycle map, feature table, metric justification, memos, report
└── requirements.txt
```

## Quick start (reproducible from a fresh clone)

```bash
# 1. Install (Python 3.11)
pip install -r requirements.txt

# 2. Materialise the DVC-tracked data (local in-repo remote; no cloud needed)
dvc pull

# 3. Run the tests
pytest tests/ -q

# 4. Train the full tracked sweep (>=10 MLflow runs) and register the best model
python scripts/run_training.py --config configs/full.yaml --register

# 5. Inspect runs in the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000

# 6. Run the quality gate locally exactly as CI does
python scripts/run_gate.py --config configs/ci.yaml
```

## Data versioning (Task 2)

- Datasets are tracked with **DVC**, never committed to plain git history
  (`data/*.csv` is gitignored; only `*.csv.dvc` pointers are in git).
- The DVC remote is `./dvcstore`, a content-addressed store **committed inside
  the repo**, so a fresh clone and the CI runner can `dvc pull` with **no cloud
  account and no secrets**.
- **Restore code + data together** (demo):
  ```bash
  git checkout <old-commit>   # restores code AND the *.csv.dvc pointers
  dvc checkout                # restores the exact CSV that commit used
  # ... run, get the same result ... then return:
  git checkout main && dvc checkout
  ```

## Configuration (Task 3)

Every tunable — paths, the utilisation cut-off, hyperparameter space, random
seed, MLflow settings and the gate threshold — lives in `configs/*.yaml` and is
**validated with pydantic on load**. A broken config fails immediately:

```bash
python -c "from pathlib import Path; import sys; sys.path.insert(0,'src'); \
from credit_risk.config import load_config; load_config('configs/dev.yaml')"
```
The exact config is stored as an artefact with every MLflow run.

## The CI quality gate (Task 6)

On every pull request, `.github/workflows/ci.yml`:
1. installs deps and `dvc pull`s the data,
2. runs `pytest`,
3. trains with `configs/ci.yaml` and compares the best model's **PR-AUC** to
   `baseline/baseline_metrics.json`,
4. **blocks the merge** if a test fails or the model is worse, and
5. publishes a short Markdown summary (`gate_summary.md`, also to the job summary).

The baseline-update rule and its protection are in `docs/baseline_rule.md`.

## Where each deliverable lives

| Deliverable | Location |
|-------------|----------|
| Lifecycle mapping | `docs/lifecycle_map.md` |
| Feature table | `docs/feature_table.md` |
| Metric justification | `docs/metric_justification.md` |
| Runs comparison (≥10) | `docs/runs_comparison.md` + MLflow UI |
| CI 3-PR history | GitHub Pull Requests tab (see `docs/report.md`) |
| v1 vs v2 drill | `docs/v1_vs_v2_report.md` |
| Threshold comparison + fairness memo | `docs/fairness_memo.md`, `docs/threshold_fairness.json` |
| Short report (≤4 pages) | `docs/report.md` |

## Rules compliance

- Only free / open-source tools; no paid cloud, no secrets committed.
- Reproducible from a fresh clone via this README.
- Every reported result traces to a tracked MLflow run (git commit + data
  version logged on each run).
