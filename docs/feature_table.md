# Task 4 - Feature Documentation Table (Topic 11)

All transformations live in `src/credit_risk/features.py` (`FeatureEngineer`),
a fit/transform component. State learned on the **training split only**
(imputation medians, output columns) is re-applied verbatim to validation and
to any new data, so training and inference share one code path - no train/serve
skew, no target leakage (the target is never read during feature building, and
`Customer_ID` is dropped before modelling).

## Cleaning / missing-value handling

| Column | Missing in v1 | Treatment | Justification |
|--------|---------------|-----------|---------------|
| Credit_Limit | 24 | Median impute (train median) | Right-skewed money column; median is robust to outliers and cannot be influenced by the test set. |
| Current_Balance | 24 | Median impute (train median) | Same reasoning; balances are skewed. |
| Payment_Delay_Days | 24 | Median impute (train median) | Count-like, skewed; median avoids inventing extreme delays. |

Medians are computed in `fit` and stored, so a fresh clone and later inference
impute with the *same* numbers.

## Engineered features (≥4 required, 5 provided)

| # | Feature | Type | Formula | Reason (domain) |
|---|---------|------|---------|-----------------|
| 1 | `credit_utilisation` | ratio | `Current_Balance / Credit_Limit` | Utilisation is the single strongest consumer-credit risk signal; carrying balance close to the limit indicates stress. Zero/absent limit guarded to 0. |
| 2 | `high_utilisation_flag` | binary | `credit_utilisation >= utilisation_high_cutoff` (config, default 0.5) | Captures a non-linear "danger zone" above a configurable cut-off that the risk team can tune without code changes. |
| 3 | `late_payer_flag` | binary | `Payment_Delay_Days >= payment_delay_flag_days` (config, default 5) | Any consistent lateness is a known default precursor; thresholded so occasional 1-2 day slips are not penalised. |
| 4 | `age_band` | binned / categorical | `cut(Age, [0,25,40,60,200])` → one-hot (`young`/`early_career`/`established`/`senior`) | Risk is non-monotonic in age; bands let the model fit different levels per life stage. Bins are config-driven. |
| 5 | `avg_transaction_value` | own design | `Monthly_Spend / Num_Transactions_Monthly` | Many tiny transactions (low average value) can signal cash-flow stress / reliance on the card for essentials, a different risk profile than few large purchases. Zero-transaction edge case guarded to 0. |

Plus the raw numeric predictors (`Age`, `Credit_Limit`, `Current_Balance`,
`Monthly_Spend`, `Num_Transactions_Monthly`, `Payment_Delay_Days`,
`Tenure_Months`) and the ordinal encoding `Income_Bracket_ord`
(`Low`=0, `Medium`=1, `High`=2 - ordinal, so ordered encoding, not one-hot).

## Leakage & reproducibility checks

- `Customer_ID` is dropped in `data.split_X_y` and asserted absent by a unit test.
- The target is never referenced inside `FeatureEngineer`.
- Imputation statistics come from `fit` on the train split only.
- `test_transform_columns_stable_on_new_data` proves inference data missing a
 whole age band still produces identical feature columns.
- Edge cases unit-tested: zero credit limit, zero transactions, missing values.
