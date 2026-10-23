# Task 5 - One-Page Metric Justification (Topic 12)

**Why plain accuracy is not sufficient, and why PR-AUC (average precision) is the gate metric.**

## The problem with accuracy here

The target is imbalanced: only **~18.9%** of customers are high risk. A model
that predicts "not high risk" for **everyone** therefore scores **~81%
accuracy** while catching **zero** defaulters - useless to a risk team, yet it
would sail past an accuracy gate. Accuracy rewards the easy majority class and
is blind to the minority class we actually care about.

For a credit-default model the costly error is a **false negative** (a
high-risk customer approved as safe). Accuracy weights that error the same as
the harmless false positive, so it cannot express the bank's real preference.

## What we use instead

| Metric | What it measures | Why it fits |
|--------|------------------|-------------|
| **Average precision (PR-AUC)** - *gate metric* | Area under the precision-recall curve; summarises how well the model ranks the minority (high-risk) class across all thresholds | Focuses entirely on the positive class, is threshold-free (so the gate does not depend on an arbitrary 0.5 cut-off), and degrades honestly when the model stops finding defaulters. |
| ROC-AUC | Ranking quality over both classes | Reported as a secondary check; can look optimistic under heavy imbalance, so not the gate. |
| Recall / Precision / F1 @0.5 | Operating-point behaviour | Reported for interpretability; depend on the chosen threshold. |
| Accuracy | Overall correctness | Reported only to demonstrate *why it is misleading* (high even for a trivial model). |

PR-AUC is the gate metric because it most directly answers "does this model
rank high-risk customers better than the baseline?", which is the decision the
gate must make. The full three-rule comparison that justifies this choice over
ROC-AUC and F1 - with concrete numbers from our own runs - is in
`docs/fairness_memo.md` (Task 8).

## Model selection

The final model is selected as the run with the highest **average_precision**
across the tracked sweep (≥10 runs over Logistic Regression and Random Forest,
multiple hyperparameters and seeds), then registered in MLflow as
`credit_risk_gate_model`. The comparison table of all runs is exported to
`docs/runs_comparison.md`.
