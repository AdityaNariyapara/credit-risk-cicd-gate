# Credit Risk Model with a CI/CD Quality Gate

**Course:** Elective II, Machine Learning Operations (MLOps Lab)
**Project 1:** Credit Risk Model with a CI/CD Quality Gate
**Team:** Aditya Nariyapara (B009) and Shreya Dutt (B016)
**Repository:** https://github.com/AdityaNariyapara/credit-risk-cicd-gate

This project builds the workflow a retail bank's risk team asked for. Every
change to code, configuration, features or data is versioned, tested
automatically, tracked in an experiment log, and accepted only if the new model
is at least as good as the current baseline. The quality gate runs on every pull
request and blocks the merge when a test fails or the model gets worse. The gate
metric and its threshold are chosen with evidence from our own runs, not picked
at random.

This README is the technical documentation. It explains the layout, how to run
every part from a fresh clone, and where each item the brief asks for lives.

---

## 1. What problem this solves

A bank retrains its credit-card default model every month. At one point an
untested change lowered model quality and nobody noticed for two weeks, and
afterwards nobody could say which code, configuration or data version produced
the bad model. The workflow here prevents that in four ways:

1. Code, configuration and data are versioned together, so any past result can
 be rebuilt exactly.
2. Every training run is logged with its parameters, metrics, data version and
 configuration.
3. A gate compares each candidate against a stored baseline and stops anything
 worse from merging.
4. The choice of gate metric and threshold is defended with numbers, and model
 error is checked across customer income groups for fairness.

### The same idea in plain words

Think of the model as a product that is rebuilt every month. Before the new
version is allowed to replace the old one, it has to pass a checkpoint. The
checkpoint runs the tests and retrains the model, then asks one question: is the
new model at least as good as the one we already trust. If yes, it is allowed in.
If it is worse, or if a test breaks, the checkpoint stops it and nobody can merge
the change by accident. We also keep a labelled logbook of every model we ever
trained, so we can always point to the exact code, data and settings behind any
result. Finally, we do not just trust the first score we see. We check that our
pass or fail rule is the right one, and we check that the model is not quietly
worse for one group of customers than another.

### Why some choices were made (short version)

- We measure success with PR-AUC, not plain accuracy, because only about one
  customer in five is high risk. A lazy model that calls everyone safe already
  looks 81 percent accurate while catching no defaulters, so accuracy hides
  failure. PR-AUC looks only at how well we find the risky customers.
- We allow a small tolerance (0.01) at the gate because the score moves by about
  that much just from the random train and test split. Without a tolerance the
  gate would block good models for no real reason.
- We keep data out of git and in DVC so the repository stays small and any commit
  can still pull back the exact dataset it used.

## 2. Tools used and why

| Purpose | Tool | Reason |
| --- | --- | --- |
| Source versioning | Git and GitHub | Standard, free, integrates with Actions. |
| Data versioning | DVC | Keeps large CSVs out of ordinary git history while still letting any commit restore its exact data. |
| Configuration | YAML with PyYAML, validated by Pydantic | Human-readable settings, validated on load so a broken config fails fast. |
| Modelling | pandas, scikit-learn | Free, well understood, fast enough for CI. |
| Experiment tracking and registry | MLflow | Logs runs and holds a model registry locally with no paid account. |
| CI and testing | GitHub Actions, pytest | Runs the gate on every pull request. |

All tools are free and open source. No paid cloud account is used and no secrets
are stored in the repository.

## 3. Repository layout

```
credit-risk-cicd-gate/
 .github/
 workflows/ci.yml CI quality gate: tests, train, compare to baseline
 CODEOWNERS marks the baseline and gate config as review-protected
 configs/
 dev.yaml fast local settings
 ci.yaml settings used by the CI gate
 full.yaml wide sweep that produces the tracked runs
 data/
 credit_card_customers_v1.csv.dvc DVC pointer for dataset v1 (CSV is gitignored)
 credit_card_customers_v2.csv.dvc DVC pointer for dataset v2
 dvcstore/ in-repo DVC remote (content addressed blobs)
 src/credit_risk/
 config.py load and validate YAML into typed objects
 data.py load a dataset, split features and target
 features.py cleaning and feature engineering (FeatureEngineer)
 model.py model factory and metrics
 pipeline.py training sweep with MLflow logging
 gate.py pass or fail decision against the baseline
 scripts/
 run_training.py run a tracked sweep, optionally register the best model
 run_gate.py CI entry point: train, compare, write summary, set exit code
 update_baseline.py governed baseline update (not run by CI)
 data_change_drill.py v1 versus v2 comparison across seeds
 threshold_and_fairness.py gate rule comparison and income subgroup error
 tests/ pytest: features (with edge cases), config, gate
 baseline/baseline_metrics.json the stored quality gate baseline
 docs/ lifecycle map, feature table, metric justification,
 runs comparison, reports and the fairness memo
 requirements.txt exact pinned versions
```

## 4. Quick start from a fresh clone

Python 3.11 is required. Every step below works offline after the clone.

```bash
# 1. Install the exact pinned dependencies
pip install -r requirements.txt

# 2. Restore the DVC tracked datasets from the in-repo remote (no cloud needed)
dvc pull

# 3. Run the test suite
pytest tests/ -q

# 4. Train the full tracked sweep and register the best model
python scripts/run_training.py --config configs/full.yaml --register

# 5. Open the MLflow UI to inspect runs and the registry
mlflow ui --backend-store-uri sqlite:///mlflow.db
# then open http://127.0.0.1:5000

# 6. Run the quality gate locally, exactly as CI does
python scripts/run_gate.py --config configs/ci.yaml
```

## 5. How each task is met

### Task 1, Lifecycle map
`docs/lifecycle_map.md` maps every project step onto the DevOps lifecycle (plan,
code, build, test, release, deploy, operate, monitor) and the MLOps lifecycle,
points each stage at a real file or workflow step, explains three ways ML
delivery differs from ordinary software delivery, and names the stages we leave
out with reasons.

### Task 2, Repository and data versioning
Datasets are tracked with DVC. The root `.gitignore` keeps `data/*.csv` out of
git, so only the small `.dvc` pointer files are committed. The DVC remote is the
folder `dvcstore/`, a content addressed store committed inside the repository,
so a fresh clone and the CI runner both restore the data with `dvc pull` and no
cloud account. To restore an old result together with its data:

```bash
git checkout <old-commit> # restores code and the .dvc pointers
dvc checkout # restores the exact CSV that commit used
# run, confirm the same result, then return
git checkout main && dvc checkout
```

Dataset v2 entered the project later as a tracked DVC change, which you can see
in the commit history.

### Task 3, Configuration management
No values are hard coded in the modelling code. File paths, the utilisation risk
cut-off, the hyperparameter search space, the random seed, the MLflow settings
and the quality gate threshold are all set in `configs/*.yaml`. Three
environments are provided: `dev` (fast local), `ci` (used by the gate) and
`full` (wide sweep). Configs are validated by Pydantic on load. A broken config
fails at once with a clear message, and unknown keys are rejected so typos do not
pass silently. The exact config is saved as an artefact with every MLflow run.

### Task 4, Cleaning and feature engineering
`src/credit_risk/features.py` holds `FeatureEngineer`, a fit and transform
component. Everything it needs for new data (imputation medians, output columns)
is learned on the training split in `fit` and reapplied in `transform`, so
training and inference use identical logic and there is no train and serve skew.
Missing values in Credit_Limit, Current_Balance and Payment_Delay_Days are filled
with the training median, which is robust to the skew in these money columns and
cannot be influenced by the test set. Five features are engineered: a ratio
(credit utilisation), two binary flags (high utilisation and late payer, both
from configurable thresholds), a binned categorical feature (age band) and one of
our own design (average transaction value). `docs/feature_table.md` lists the
name, formula and reason for each. Customer_ID is dropped and never used as a
predictor, the target is never read during feature building, and unit tests cover
the logic including the zero credit limit edge case.

### Task 5, Training, tuning and tracking
`scripts/run_training.py` establishes a baseline and tunes two model families,
Logistic Regression and Random Forest, over several hyperparameters and three
random seeds. This produces 21 tracked runs, well above the ten required. Each
run logs its parameters, metrics, dataset version, full configuration and model
artefact to MLflow. The final model is selected by average precision (PR-AUC) and
registered as `credit_risk_gate_model`, traceable back to its git commit and data
version through run tags. `docs/runs_comparison.md` holds the full comparison and
`docs/metric_justification.md` explains why plain accuracy is not enough here.

### Task 6, Automated quality gate
`.github/workflows/ci.yml` runs on every pull request into main. It installs
dependencies, runs `dvc pull`, runs pytest, trains with `configs/ci.yaml`,
compares the best model's PR-AUC to `baseline/baseline_metrics.json`, and blocks
the merge if a test fails or the model is worse. It writes a short Markdown
summary (uploaded as a build artefact and added to the Actions job summary) that
a reviewer can read without opening logs. The rule for updating the baseline is in
`docs/baseline_rule.md`: CI never writes the baseline, only a reviewed manual run
of `scripts/update_baseline.py` does, and `.github/CODEOWNERS` marks the baseline
and gate config as review protected. Branch protection on main requires the
`quality-gate` check to pass before any merge.

### Task 7, Data-change drill
`scripts/data_change_drill.py` retrains the same model on v1 and v2 across three
random seeds each and logs every run with a data version tag. The written
conclusion is in `docs/v1_vs_v2_report.md`.

### Task 8, Quality-gate threshold and fairness defense
`scripts/threshold_and_fairness.py` compares three candidate gate rules against
our real run history and reports model error by Income_Bracket for the registered
model. The defense memo in `docs/fairness_memo.md` recommends one rule, states the
trade-off, reports the subgroup breakdown honestly, and says what new evidence
would change the recommendation.

## 6. Results at a glance

- 21 tracked runs, PR-AUC between 0.690 and 0.869, driven mostly by the random
 seed. Accuracy stays near 0.80 to 0.92 for every run, which is why it cannot be
 the gate metric.
- Stored baseline PR-AUC is 0.7948 (the CI model, Logistic Regression, seed 42).
- v1 versus v2 drill: mean PR-AUC rises from 0.788 to 0.843, but the gain of
 0.055 is within about two times the seed noise, so it is not a proven
 improvement. v2 is accepted because it does not regress and its variance is
 lower.
- Fairness: the model misses a high-risk Low-income customer almost twice as
 often (false negative rate 0.35) as Medium or High income customers (0.19 to
 0.20). This is reported as a known limitation.

## 7. The three pull requests

| Scenario | Branch | CI result |
| --- | --- | --- |
| Passes | docs branch (added reports and docs) | tests pass, gate model equals baseline |
| Fails, model worse | experiment/shallow-fast-model | gate step fails, PR-AUC below baseline minus 0.01 |
| Fails, test fails | feature/simplify-utilisation | unit test fails before the gate runs |

The pull requests are visible in the repository Pull requests tab and their runs
in the Actions tab.

## 8. Reproducibility and rules compliance

- Everything runs from a fresh clone by following Section 4.
- No secrets, passwords or tokens are committed.
- Every number reported traces to a tracked MLflow run, which records the git
 commit and data version.
- `requirements.txt` pins the exact versions the baseline was generated with, so
 the CI runner reproduces the same metrics.

## 9. Who did what

Both members can explain and modify any part of the submission. Aditya Nariyapara
set up the repository, DVC data versioning, the configuration system, feature
engineering, the training and tracking pipeline, the quality gate and the CI
workflow. Shreya Dutt contributed the analysis write-ups (the fairness and
threshold memo, the v1 versus v2 drill report and the contributing guide) and
raised the passing pull request. The experiment evidence and the data-change
drill were reviewed together.
