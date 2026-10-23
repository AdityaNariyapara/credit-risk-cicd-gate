# Contributing & Workflow

Two authors: **Aditya Nariyapara (B009)** and **Shreya Dutt (B016)**. All work
flows through pull requests into `main`; the CI quality gate
(`.github/workflows/ci.yml`) must pass before merge.

## Branch + PR flow

```bash
git checkout -b feature/<short-name>
# ... edits ...
git add -p && git commit -m "clear, small message"
git push -u origin feature/<short-name>
# open a PR on GitHub into main; CI runs the gate; merge only when green
```

## Commit hygiene

- Small commits, imperative messages ("add utilisation feature", not "stuff").
- No secrets, tokens or raw CSVs committed (data goes through DVC).
- Each author commits under their **own** GitHub account and git identity:
  ```bash
  git config user.name  "Shreya Dutt"
  git config user.email "<shreya-github-email>"
  ```

## Baseline changes

Never edit `baseline/baseline_metrics.json` by hand. Use
`python scripts/update_baseline.py --config configs/ci.yaml` on `main` after a
reviewed, passing model (see `docs/baseline_rule.md`).
