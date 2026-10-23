# Task 5 - Tracked Runs Comparison (21 runs)

All runs logged to MLflow (`sqlite:///mlflow.db`, experiment `credit_risk`),
data version **v1**, each with params, metrics, config artefact and model.
Sorted by the gate metric **average_precision (PR-AUC)**. Final model =
top row, registered as `credit_risk_gate_model`.

| run | family | seed | PR-AUC | ROC-AUC | F1 | accuracy | recall |
|-----|--------|------|--------|---------|----|----------|--------|
| rf_seed123_n100_depth12 | random_forest | 123 | **0.8688** | 0.9570 | 0.7500 | 0.910 | 0.711 |
| logreg_seed123_C0.1 | logistic_regression | 123 | 0.8633 | 0.9599 | 0.7692 | 0.895 | 0.921 |
| rf_seed123_n300_depth12 | random_forest | 123 | 0.8627 | 0.9596 | 0.7429 | 0.910 | 0.684 |
| logreg_seed123_C1.0 | logistic_regression | 123 | 0.8584 | 0.9578 | 0.7609 | 0.890 | 0.921 |
| rf_seed123_n300_depth6 | random_forest | 123 | 0.8565 | 0.9597 | 0.7467 | 0.905 | 0.737 |
| logreg_seed123_C10.0 | logistic_regression | 123 | 0.8518 | 0.9561 | 0.7473 | 0.885 | 0.895 |
| rf_seed123_n100_depth6 | random_forest | 123 | 0.8471 | 0.9579 | 0.7692 | 0.910 | 0.789 |
| logreg_seed7_C0.1 | logistic_regression | 7 | 0.8403 | 0.9378 | 0.7010 | 0.855 | 0.895 |
| logreg_seed7_C1.0 | logistic_regression | 7 | 0.8275 | 0.9405 | 0.6957 | 0.860 | 0.842 |
| logreg_seed7_C10.0 | logistic_regression | 7 | 0.8240 | 0.9418 | 0.7191 | 0.875 | 0.842 |
| logreg_seed42_C1.0 | logistic_regression | 42 | 0.7948 | 0.9189 | 0.6535 | 0.825 | 0.868 |
| logreg_seed42_C10.0 | logistic_regression | 42 | 0.7923 | 0.9189 | 0.6600 | 0.830 | 0.868 |
| rf_seed7_n300_depth12 | random_forest | 7 | 0.7916 | 0.9362 | 0.7463 | 0.915 | 0.658 |
| logreg_seed42_C0.1 | logistic_regression | 42 | 0.7741 | 0.9077 | 0.6422 | 0.805 | 0.921 |
| rf_seed7_n100_depth12 | random_forest | 7 | 0.7729 | 0.9322 | 0.7353 | 0.910 | 0.658 |
| rf_seed42_n100_depth6 | random_forest | 42 | 0.7631 | 0.9297 | 0.6593 | 0.845 | 0.789 |
| rf_seed42_n300_depth6 | random_forest | 42 | 0.7415 | 0.9271 | 0.6742 | 0.855 | 0.789 |
| rf_seed7_n300_depth6 | random_forest | 7 | 0.7368 | 0.9319 | 0.6933 | 0.885 | 0.684 |
| rf_seed7_n100_depth6 | random_forest | 7 | 0.7190 | 0.9321 | 0.6933 | 0.885 | 0.684 |
| rf_seed42_n300_depth12 | random_forest | 42 | 0.7090 | 0.9239 | 0.7000 | 0.880 | 0.737 |
| rf_seed42_n100_depth12 | random_forest | 42 | 0.6898 | 0.9232 | 0.6835 | 0.875 | 0.711 |

## Reading of the table

- **PR-AUC spans 0.690 - 0.869**, driven heavily by the **random seed** (the
 train/test split), not only by the model family - the single biggest source of
 variation. This is exactly why the gate uses a tolerance band and why Task 7
 checks improvements across multiple seeds rather than trusting one number.
- **Accuracy is a poor discriminator**: it sits at 0.80-0.92 across *every* run,
 including the weakest (a trivial "all negative" model would already score
 ~0.81), confirming the metric-justification argument.
- Logistic regression trades higher **recall** (catches more defaulters) for
 lower precision; random forest is the reverse. PR-AUC captures this ranking
 quality without fixing a threshold.
- Every row traces to a git commit and data version via MLflow tags
 (`git_commit`, `data_version`), satisfying the traceability criterion.
