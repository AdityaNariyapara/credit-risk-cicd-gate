# Baseline Update Rule (Task 6)

The quality gate compares every candidate model against
`baseline/baseline_metrics.json`. The integrity of the gate depends on that file
not drifting downward by accident, so its update is **governed, not automatic**.

## The rule

1. **CI never writes the baseline.** `scripts/run_gate.py` only *reads* it. A
   passing PR does **not** update the baseline.
2. The baseline is updated **only** by a maintainer running
   `python scripts/update_baseline.py --config configs/ci.yaml` on `main`,
   **after** a model has passed the gate and been reviewed, and committing the
   result in a dedicated commit.
3. The gate metric and tolerance (`gate.metric`, `gate.tolerance`) are set in
   YAML and reviewed like any other code change.

## Protection against accidental changes

- `baseline/baseline_metrics.json` should be covered by a **CODEOWNERS** entry
  and branch protection requiring review (see `.github/CODEOWNERS`), so no PR
  can silently lower it.
- The baseline file records `git_commit`, `data_version`, `config` and
  `updated_at`, so any change is auditable.
- Because the update is a separate manual step, a model that merely *passes*
  cannot ratchet the bar up or down without a human decision.

## Why not auto-update to the best model?

Auto-promoting the best run would let a lucky high-variance seed raise the bar
and then block honest future models, or let a slow downward drift pass unnoticed
(exactly the failure in the project scenario). A human-reviewed, explicitly
committed baseline keeps the gate meaningful.
