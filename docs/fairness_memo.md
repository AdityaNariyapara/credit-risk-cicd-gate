# Task 8 — Quality-Gate Threshold & Fairness Defense Memo

**To:** Risk team **From:** MLOps (Aditya Nariyapara B009, Shreya Dutt B016)
**Re:** Which gate rule to adopt, and the fairness trade-off we accept.

## 1. We did not accept the first metric. Three rules compared on real runs.

We took our **21-run history** (v1 data) and asked: against a stored baseline of
the CI model (**PR-AUC 0.7948, ROC-AUC 0.9189, F1 0.6535**, seed 42), how many of
the 21 runs would each candidate gate rule **accept**?

| Candidate gate rule | Runs accepted | Behaviour |
|---------------------|---------------|-----------|
| **A. PR-AUC ≥ baseline − 0.01** | **13 / 21** | Discriminating — rejects the genuinely weaker models. |
| B. ROC-AUC ≥ baseline − 0.01 | 20 / 21 | Too lenient — stays high even when minority-class ranking collapses. |
| C. F1 ≥ baseline − 0.01 | 20 / 21 | Lenient and threshold-bound (depends on the 0.5 cut-off). |

**The disagreement is the whole point.** Seven runs that rules B and C would
**accept** are **rejected** by rule A — these are random-forest models whose
PR-AUC falls to as low as **0.6898** while their ROC-AUC still reads ~0.92:

| run | PR-AUC | ROC-AUC | F1 | A (PR-AUC) | B (ROC) | C (F1) |
|-----|--------|---------|----|-----------|---------|--------|
| rf_seed42_n100_depth12 | 0.6898 | 0.9232 | 0.6835 | ❌ reject | ✅ accept | ✅ accept |
| rf_seed42_n300_depth12 | 0.7090 | 0.9239 | 0.7000 | ❌ reject | ✅ accept | ✅ accept |
| rf_seed7_n100_depth6 | 0.7190 | 0.9321 | 0.6933 | ❌ reject | ✅ accept | ✅ accept |

If we gated on ROC-AUC or F1, a model **0.10 PR-AUC worse** at ranking high-risk
customers would merge unnoticed — the exact failure in the project scenario.

## 2. Recommendation

Adopt **Rule A: PR-AUC ≥ baseline − 0.01** as the quality gate
(`gate.metric = average_precision`, `gate.tolerance = 0.01` in `configs/ci.yaml`).

- **Why PR-AUC:** with 18.9% positives, it measures ranking of the minority
  (high-risk) class directly and is threshold-free.
- **Why −0.01 and not strict:** our run history shows ~±0.05 PR-AUC seed noise
  (Task 7). A 0.01 band absorbs trivial noise while still rejecting the real
  0.07–0.10 drops above. A strict (0.0) rule would flap on noise; a loose
  (−0.05) rule would let real regressions through.

## 3. Fairness: error by Income_Bracket (final registered model)

Test-set error of the registered model, broken down by subgroup:

| Income_Bracket | n | actual high-risk | false neg | false pos | **false-neg rate** | accuracy |
|----------------|---|------------------|-----------|-----------|--------------------|----------|
| High | 42 | 5 | 1 | 2 | 0.200 | 0.929 |
| Medium | 75 | 16 | 3 | 8 | 0.188 | 0.853 |
| **Low** | 83 | 17 | 6 | 4 | **0.353** | 0.880 |

**Disparity found:** the model **misses a high-risk Low-income customer almost
twice as often** (FN rate 0.35) as Medium/High (0.19–0.20). For a credit-default
model the false negative is the costly error (a defaulter approved as safe), so
this disparity falls on the group least able to absorb it.

**Is it acceptable?** Not as-is for deployment. It is acceptable to **register**
this model (it is the best available and the gate is about not regressing), but
the subgroup FN gap must be fixed before the model drives real decisions. We do
**not** hide it behind the high aggregate accuracy. Section 3b shows the fix.

## 3b. Mitigation we implemented (equal-opportunity thresholds)

Rather than only flag the gap, we implemented a subgroup-aware decision
threshold (an equal-opportunity policy): we lower the threshold for the group
that is being under-served until its false-negative rate matches the others.
`scripts/robustness_and_fairness.py` computes this on the registered model's test
split.

| Income group | Threshold | FN rate before | FN rate after | Precision before | Precision after |
| --- | --- | --- | --- | --- | --- |
| High | 0.50 | 0.200 | 0.200 | 0.667 | 0.667 |
| Medium | 0.50 | 0.188 | 0.188 | 0.619 | 0.619 |
| **Low** | **0.31** | **0.353** | **0.176** | 0.733 | 0.583 |

Lowering only the Low-income group's threshold to 0.31 **cuts its false-negative
rate from 0.353 to 0.176**, bringing it in line with (slightly below) the High
and Medium groups. High and Medium are already below the 0.20 target, so their
thresholds are unchanged.

**The trade-off, stated honestly.** Catching more high-risk Low-income customers
costs precision: Low-income false positives rise from 4 to 10 and precision falls
from 0.73 to 0.58. In credit terms, more Low-income customers are flagged as
risky who would not have defaulted. We judge this acceptable because the
false negative (a missed defaulter) is the more expensive error for the bank and
the more harmful outcome for the customer, who is extended credit they cannot
service. The policy is a per-group operating threshold applied at decision time;
it does not change the trained model or the gate metric.

## 4. What new evidence would change our recommendation

- If, across more seeds, the PR-AUC seed noise were shown to be **> 0.01**, we
  would widen the tolerance accordingly (or gate on a mean over seeds).
- If the Low-income FN gap **closed** under a subgroup-aware threshold without
  materially lowering overall PR-AUC, we would add a per-group fairness check to
  the gate rather than treating it as an advisory flag.
- If the bank's cost model showed false *positives* dominate, we would move the
  operating threshold and might gate on recall-at-fixed-precision instead.
