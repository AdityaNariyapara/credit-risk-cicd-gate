# Project 1 — Credit Risk CI/CD Gate: Short Report

**Team:** Aditya Nariyapara (B009) · Shreya Dutt (B016)
**Repo:** https://github.com/AdityaNariyapara/credit-risk-cicd-gate

## 1. Design decisions

**Problem.** Build a workflow where every change to code, config, features or
data is versioned, tested, tracked, and accepted only if the model is at least
as good as the current baseline — and defend *why* the gate is right.

**Data versioning (Task 2).** Datasets are tracked with **DVC**, never in plain
git (`data/*.csv` is gitignored; only `.dvc` pointers are committed). The DVC
remote is `./dvcstore`, a content-addressed store **committed inside the repo**,
so a fresh clone and GitHub Actions can `dvc pull` with **no cloud account and
no secrets**. Restoring an old commit + `dvc checkout` restores code *and* the
exact data together.

**Configuration (Task 3).** All tunables live in `configs/*.yaml` (paths,
utilisation cut-off, hyperparameter space, seed, MLflow settings, gate
threshold), validated by **pydantic** on load — a broken config fails
immediately with a clear message, and unknown keys are rejected. Three
environments: `dev` (fast), `ci` (gate), `full` (wide sweep). The exact config
is logged as an artefact with every run.

**Features (Task 4).** `FeatureEngineer` is a fit/transform component so training
and inference share one code path (no train/serve skew). Missing values in
`Credit_Limit`, `Current_Balance`, `Payment_Delay_Days` are median-imputed using
**train-split medians only** (robust to skew, no leakage). Five engineered
features: `credit_utilisation` (ratio), `high_utilisation_flag` and
`late_payer_flag` (binary, config thresholds), `age_band` (binned, one-hot),
and `avg_transaction_value` (own design). `Customer_ID` is dropped; edge cases
(zero credit limit, zero transactions) are unit-tested.

**Tracking & selection (Task 5).** 21 runs over Logistic Regression and Random
Forest × hyperparameters × seeds, all logged to **MLflow** with params, metrics,
data version, config and model artefact. The final model is selected by
**PR-AUC** and registered as `credit_risk_gate_model`, traceable to its git
commit and data version via run tags.

**Why PR-AUC, not accuracy.** With 18.9% positives, a trivial "all-safe" model
scores ~81% accuracy yet catches zero defaulters. Across our 21 runs accuracy
sits at 0.80–0.92 for *every* model, so it cannot discriminate. PR-AUC measures
ranking of the minority (high-risk) class and is threshold-free. (Full argument:
`docs/metric_justification.md`.)

**The gate (Task 6).** On every PR, CI installs deps, `dvc pull`s data, runs
`pytest`, trains with `configs/ci.yaml`, and compares the best model's PR-AUC to
`baseline/baseline_metrics.json` (**0.7948**). It **blocks the merge** if a test
fails or PR-AUC < baseline − 0.01, and publishes a Markdown summary. The baseline
is **never** updated by CI — only by a reviewed manual `update_baseline.py` run,
protected by CODEOWNERS (`docs/baseline_rule.md`).

**Threshold & fairness defense (Task 8).** We compared three gate rules against
our real run history: the PR-AUC rule accepts 13/21 runs, while ROC-AUC and F1
rules each accept 20/21 — waving through 7 weak RF models (PR-AUC down to 0.69).
We adopt **PR-AUC ≥ baseline − 0.01**. Subgroup error by Income_Bracket shows the
model misses high-risk **Low-income** customers almost twice as often (FN rate
0.35 vs 0.19–0.20) — flagged as a known limitation, not hidden. (`docs/fairness_memo.md`.)

## 2. v1 vs v2 data-change drill (Task 7)

Retraining through the same pipeline on v2 (1,100 rows) vs v1 (800), 3 seeds each:

| version | mean PR-AUC | std | 
|---------|-------------|-----|
| v1 | 0.7878 | 0.0769 |
| v2 | 0.8427 | 0.0432 |

Delta **+0.055** is **within ~2× the seed noise** (SE ≈ 0.051) — *not*
distinguishable from run-to-run variation. **Conclusion:** accept the v2 model
(it does not regress and variance shrinks with more data), but do **not** claim a
real improvement without more seeds. This is why the gate uses a tolerance band
rather than trusting single-number comparisons.

## 3. CI history — three pull requests

| PR | Change | CI result | Why |
|----|--------|-----------|-----|
| #1 | Add reports/docs (Shreya) | ✅ pass | tests pass; gate model unchanged = baseline |
| #2 | Weaken model in `ci.yaml` (shallow trees, no class weighting) | ❌ fail | PR-AUC drops below baseline − 0.01 |
| #3 | Break feature logic | ❌ fail | a unit test fails before the gate runs |

See the repo's Pull Requests tab for the live checks.

## 4. Reproducibility & integrity

Reproducible from a fresh clone via the README (`pip install` → `dvc pull` →
`pytest` → `run_training`). No secrets committed. Every reported number traces to
a tracked MLflow run (git commit + data version on each). Pins in
`requirements.txt` fix the exact versions the baseline was generated with.
