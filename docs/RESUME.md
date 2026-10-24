# QR / DS resume evidence

Choose one or two bullets appropriate to the role. Evidence: [real report](../reports/real/REPORT.md), [protocol](../ml_validation/datasets/protocol.json), [frozen choice](../reports/real/frozen_selection.json), [audit](../reports/real/audit.json) and [tests](../tests/test_treasury.py).

## Quantitative research / rates research

- Built a reproducible Treasury-yield classification study on 9,005 official public observations, using delayed forecast origins, label-availability purging, expanding development folds and frozen 2020–2025 historical confirmation.
- Benchmarked nine fixed classifier/feature configurations against two training-fitted baselines across 1,479 confirmation decisions; retained baseline outperformance, block-sensitive alternatives and inconclusive accuracy/staleness effects rather than claiming alpha.

## Financial data science

- Engineered keyed, training-fitted pipelines for logistic regression, LightGBM and CatBoost with 25 whitelisted yield-curve/dynamics features, source-licensed offline data and prediction-level timing audits.
- Validated future-data invariance, label-end purging and confirmation-isolated preprocessing in 33 automated cases; published six English figures covering model comparisons, calibration, dependent uncertainty and changing regimes.

## Numerical and phrasing boundaries

9,005 is the raw observed-date count; 8,732 are complete timed samples; 7,232 labels are known before the final confirmation fit; 1,479 are confirmation decisions, not independent monthly trials. Nine configurations are three estimators × three feature variants, plus two separate baselines. The suite includes 22 real-study cases and 11 preserved legacy cases; do not call all 33 new tests.

The best development classifier is CatBoost/dynamics, but the baseline leads development overall. Confirmation log-loss gain is −0.00951 with 63-session 95% interval [−0.01688,−0.00297], so this metric favors the baseline. CatBoost accuracy uncertainty spans zero; logistic gain is block-sensitive, and the four-session stress is inconclusive. A gain is a loss-score difference, not percentage points, returns, alpha or Sharpe. These are current-vintage historical temporal checks, not a prospective unseen holdout. Do not promote a new model using confirmation outcomes.

The separately labeled synthetic accuracy and QRT adapter do not provide additional real-market samples, an official leaderboard score or permission to distribute challenge data. Development was AI-assisted; use these statements only after you can explain the implementation and limits.
