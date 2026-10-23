# Task 7 — Data-Change Drill: v1 vs v2 (Topics 2, 4)

When `credit_card_customers_v2.csv` (1,100 rows = v1's 800 + 300 new customers)
arrived, we retrained through the **same** pipeline and compared it to v1. To
separate a real improvement from luck, we trained the same model
(RandomForest, n=300, depth=12, balanced) on **each data version across three
random seeds (42, 7, 123)** and logged every run to MLflow
(experiment `credit_risk_drill`, with a `data_version` tag/param on each run).

## Results (gate metric: PR-AUC)

| data version | mean PR-AUC | std (3 seeds) | n seeds |
|--------------|-------------|---------------|---------|
| v1 | 0.7878 | 0.0769 | 3 |
| v2 | 0.8427 | 0.0432 | 3 |

- **v2 − v1 mean delta = +0.0550**
- Approx. standard error of the difference ≈ **0.0509**
- The delta is **within ~2× the run-to-run noise**, i.e. **not statistically
  distinguishable from seed variation**.

## Conclusion — should the gate accept the v2 model?

**On this evidence, the apparent +0.055 PR-AUC gain is not trustworthy as a real
improvement** — it is the same size as the noise between seeds. However, v2 is
also clearly **not worse** than v1 (the mean is higher and the variance actually
*shrinks*, as expected with 300 more rows).

**Recommendation:** the gate should **accept the v2-trained model** (it does not
regress — the gate's job is to block *worse* models, and v2 is not worse), but we
should **not advertise a performance improvement**. The honest reading is "more
data, equal-or-slightly-better, lower variance." To *claim* an improvement we
would need more seeds (or a proper paired test) showing the gain exceeds the
noise band — which is exactly the kind of over-claiming the project scenario
warns against.

This is the whole point of the drill: the gate and the tolerance (±0.01) exist
precisely because single-number comparisons like "0.84 > 0.79" are misleading
once you account for seed variance.
