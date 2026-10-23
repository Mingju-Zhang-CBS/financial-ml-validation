# Method and input contract

## Files and alignment

X is a CSV with `ROW_ID,TS,ALLOCATION,GROUP,RET_1..RET_20,SIGNED_VOLUME_1..SIGNED_VOLUME_20,MEDIAN_DAILY_TURNOVER`. Y has `ROW_ID` and either `TARGET` or the downloaded-file spelling `target`; providing both is rejected. CSV row IDs are read as strings. X and Y must have identical unique key sets; targets must be finite. The binary outcome is `TARGET > 0`, following the public task. Zero maps to class 0. Every date/allocation pair must be unique.

The 40 history columns and turnover may contain missing values; infinities are rejected. Identifiers and groups cannot be missing. A TARGET column in X is rejected. Other unlisted columns are dropped, so arbitrary extra fields cannot become features. String dates are opaque categories and are never predictors.

`align_training(X,Y)` returns a canonical ROW_ID-sorted feature frame and aligned binary labels. `feature_table(frame,variant)` builds a label-free feature table. `split_protocol(frame,date_order=None)` returns development folds, final-fit rows and confirmation rows. `evaluate(frame,labels,date_order=None)` runs the fixed comparison. Explicit chronological orders are currently used by the synthetic generator, not inferred for official data.

## Features and estimators

Raw: the 40 histories, turnover and GROUP. Summary: add means and standard deviations over the last 5/10/20 returns, mean signed volume and return/volume missing fractions. Identity: add the allocation category. No same-date targets, row key, date suffix or external financial matching is used.

Numeric values use training-fold median imputation and standardization; all-missing training features are retained. GROUP and optional ALLOCATION use training-fold one-hot encoding, with unknown categories ignored. All three estimators receive the same preprocessing contract.

Logistic regression has C=1 and 600 maximum iterations. LightGBM has 80 trees, depth 4, 15 leaves and learning rate .05. CatBoost has 80 iterations, depth 4 and learning rate .05. Tree models use two CPU threads and seed 42. These are fixed demonstration settings, not tuned winners. The majority comparator is fitted from each corresponding training fold's class frequency.

## Validation and inference

The synthetic generator produces 360 dates with 12 allocations, correlated return histories and missing features. TS names are shuffled; a separate known generating order is supplied only to the splitter. The last 20% of dates form confirmation; the preceding 20 dates are unused. The remaining development dates use three expanding forward folds, each with a 20-date purge. This separates adjacent 20-observation histories and their next-period targets.

For user-supplied data without a known order, a seeded SHA-256 order of unique TS groups chooses the last 20% for confirmation and partitions development into three date folds. This hash has **no temporal meaning**. It prevents within-date mixing, but cannot eliminate unknown cross-date history overlap. It must not be called forward out-of-sample testing.

Nine candidates are ranked by pooled development-row accuracy, breaking ties alphabetically. The decision threshold is fixed at .50. Final preprocessing and the selected model fit all development rows; confirmation scores are calculated afterward. No confirmation-based fallback, promotion, threshold adjustment or reranking occurs. A poor confirmation result remains in the report.

The paired bootstrap resamples complete confirmation dates with replacement and recomputes the difference in row-weighted accuracy. It preserves within-date dependence and unequal cluster sizes. It does not remove serial dependence across dates or quantify the entire researcher's model-selection history. Repeating the workflow on previously seen labels does not create new independent evidence.

## Outputs and resource measurements

Outputs include aggregate metrics, per-development-fold scores, the nine-candidate ranking, group diagnostics, a frozen-choice record, two plots and a report. No original rows, individual official predictions, trained pickles or challenge data are part of the initial public release. Source hashes and an input fingerprint are in the manifest. Fit times exclude report plotting; process RSS is sampled every 10ms and is not a guaranteed native-memory peak.

References: [QRT official task](https://challengedata.ens.fr/challenges/167), [scikit-learn grouped validation](https://scikit-learn.org/stable/modules/cross_validation.html), [LightGBM classifier](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html), [CatBoost classifier](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier).
