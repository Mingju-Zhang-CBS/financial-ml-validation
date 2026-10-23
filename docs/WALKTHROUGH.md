# Walkthrough: from a panel row to a research conclusion

## 1. Define the observation before choosing a model

One observation represents an allocation on a date. Its predictors contain 20 return lags, 20 signed-volume lags, turnover and an allocation group. The outcome is whether the next-period return is positive. Thus a good classification score alone says nothing about executable position size or net P&L.

`synthetic_panel()` generates 360 dates by 12 allocations: 4,320 observations. Common daily shocks create within-date dependence. Return histories overlap across adjacent dates. The visible date names are shuffled; the generator separately retains the genuine order for testing the known-time protocol.

## 2. A correct join is part of the model

Consider this deliberately reordered example:

| Feature-row order | X row ID | Target-file order | Y row ID | Target sign |
| --- | --- | --- | --- | --- |
| 1 | R002 | 1 | R001 | Positive |
| 2 | R001 | 2 | R003 | Negative |
| 3 | R003 | 3 | R002 | Positive |

Pairing rows by position gives at least one wrong label. The implementation verifies exact unique key sets, joins by `ROW_ID`, sorts into canonical order, and only then defines the binary target. Target columns are rejected in X. Other unlisted fields do not automatically become predictors.

Inspect [`align_training`](../ml_validation/data.py) and the input-permutation tests. The downloadable lowercase `target` spelling is explicitly normalized; receiving both `target` and `TARGET` is ambiguous and rejected.

## 3. Isolate information in time when the order is known

The synthetic protocol allocates 288 dates to the pre-confirmation side and 72 to confirmation. It discards the last 20 dates on the pre-confirmation side, leaving 268 development dates. Three expanding development folds each retain a 20-date gap between training and validation. After selection, the chosen pipeline fits all 268 development dates.

The sample arithmetic is therefore:

```text
Generated:                 360 * 12 = 4,320 rows
Final-fit development:     268 * 12 = 3,216 rows
Unused boundary:            20 * 12 =   240 rows
Confirmation:               72 * 12 =   864 rows
```

Development validation folds do not score every development row: an initial block is needed to train the first fold. The 20-date gap addresses the declared history horizon; it does not certify independence against every possible data-generating process.

Official TS names are anonymous and shuffled. In that branch, the hash order reproducibly assigns whole dates to groups without claiming a calendar. Unknown temporal overlap remains a material limitation.

## 4. Keep preprocessing inside the estimator

Median imputation, scaling and category vocabularies fit on a training fold. Validation and confirmation call `transform` using those fitted values. This matters even when a preprocessing step does not read labels: a global transform can still use future distributions.

All estimators receive the same preprocessing contract. Unknown categories are ignored by the fitted encoder. An all-missing training feature is retained rather than silently changing array positions.

## 5. State what was searched

The study compares three estimators and three feature variants:

| Variant | Added information | Research question |
| --- | --- | --- |
| Raw | Return/volume histories, turnover, group | What does the baseline history support? |
| Summary | Means, standard deviations, missing fractions | Do stable summaries improve the representation? |
| Identity | Allocation category | Does allocation-specific information help on these splits? |

Nine configurations are a model-selection exercise. They do not constitute nine independent confirmations. Parameters and the 0.50 threshold remain fixed; selection uses pooled development accuracy. Confirmation is evaluated afterward with no fallback based on its score.

## 6. Read several measures together

The selected logistic/identity model reaches 52.66% confirmation accuracy versus 49.31% for the training-fitted majority baseline. Its accuracy difference has a paired date-bootstrap interval of -2.08 to +9.03 percentage points. The interval includes zero, while log loss is worse than baseline.

This is an informative inconclusive result. Threshold accuracy and probability quality are different questions. The report retains the unfavorable metric instead of searching confirmation for a better threshold. Date-level resampling keeps same-date observations together, although it does not resolve arbitrary dependence between dates.

## 7. Trace the implementation yourself

Run the compact example:

```text
python examples/explain_validation.py
```

It permutes the input files, verifies keyed alignment, prints split boundaries and trains one illustrative development fold. Its one-fold score is a teaching example; the committed report contains the separate full selection/confirmation study.

Read [the executed report](../reports/demo/REPORT.md), [the full method](METHODS.md) and [the behavioral tests](../tests/test_protocol.py). The public results remain synthetic; prior official-label exposure is disclosed rather than erased by a new split.
