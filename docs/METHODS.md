# Treasury probability forecasting: methods and time contract

## Data and units

The package contains `date,y3m,y1,y2,y5,y10`, measured in percent per annum. A change from 4.00 to 4.20 is 0.20 percentage points, or 20 basis points; it is not a 20% bond return. Date order and uniqueness are checked. Ten-year observations must be finite. Infinite values, unexpected columns and implausible units fail closed. Other missing tenors remain missing for training-fitted imputation; the current retained snapshot has none. No forward/backward filling or invented holiday return is used.

An observation session is a retained official date with a 10-year yield, not necessarily every exchange weekday. The source has 9,005 dates over 1990–2025. Eligible feature observations begin in 1991. Constant-maturity yields are indicative curve measurements; no single bond with that constant remaining maturity is assumed to be held.

## Feature and label separation

Nine curve features: the five levels, `y10−y2`, `y10−y3m`, `y5−y2` and `2*y5−y2−y10`. Sixteen dynamic features: 1-, 5- and 20-session changes for each tenor, plus the sample standard deviation of the latest twenty one-session 10-year changes. Combined contains all 25. Every calculation uses observations through t only, without centered windows, future shifts or full-sample normalization. Keys and all label/timing fields are rejected by the feature whitelist.

For observation index i, publication lag L=1, decision delay D=2 and target horizon H=20:

```text
feature observation:       i
assumed availability:     date[i+L], 23:59:59 New York
decision:                 date[i+D], 09:00 New York
target start measurement: date[i+D], later that day
target end measurement:   date[i+D+H]
label availability:       date[i+D+H+L], 23:59:59 New York
target:                   1 if y10[i+D+H] > y10[i+D], else 0
```

Both endpoints are future label fields, excluded from features. A zero change maps to 0 and ties are reported, not dropped. The label does not begin at the feature date; doing so would mix pre-decision movement into the forecast. Rows without a complete target and its publication allowance are censored. There are 8,732 complete timed samples; confirmation ends on 1 December 2025 because later twenty-session outcomes would be unfinished within the snapshot's available clocks.

Exported timestamp columns are timezone-aware UTC; date-only labels and the modeled 09:00 decision refer to New York dates. For example, 31 December at 23:59:59 New York appears as 1 January UTC. That representation is not a new measurement or a future-data import.

## Expanding validation and the selection boundary

Three validation windows: 2008–2011, 2012–2015, 2016–2019. Each training set starts with eligible historical rows and is restricted to labels fully available strictly before the window's first decision. All development-selection labels must also be available before 1 January 2020 00:00 New York. This removes 21 late-2019 boundary rows whose outcomes reach confirmation. A sample's earlier decision date alone does not establish that its target is known.

Past feature windows may overlap across the boundary because past information can legitimately be known at both decisions; that does not justify overlapping unavailable training outcomes. Label-availability purging is performed from timestamps rather than an approximate twenty-row gap. Development outcomes can overlap across neighboring validation folds, so pooled scores and folds are not treated as independent replications.

Confirmation decisions cover 2020–2025, with 1,479 fully labeled rows. A single final model is fitted on 7,232 eligible known-label rows before the first confirmation decision; its feature construction continues to update from available past observations, but its fitted parameters do not update on confirmation labels. Historical market familiarity, current-vintage corrections and missing archival release clocks prevent a prospective independent OOS claim.

## Fixed model comparison

Three estimators × three variants produce nine classifier configurations. Logistic regression uses C=1, maximum 600 iterations. LightGBM uses 80 trees, 15 leaves, maximum depth 4 and learning rate 0.05. CatBoost uses 80 iterations, depth 4 and learning rate 0.05. Seeds are fixed at 42; boosted models use two threads. No early stopping or confirmation evaluation set is passed to fitting.

The shared pipeline fits median imputation and standard scaling inside each training fold. The comparison uses the same valid rows and labels for every configuration. No production probability recalibration is claimed. Predictions are bounded to [1e−6,1−1e−6] for numerical stability and threshold 0.50 is fixed.

Two baselines fit only eligible training labels: `(positive_count+1)/(n+2)` overall; and the same Laplace-smoothed probability conditional on the sign of the past twenty-session 10-year change. The conditional model's state is known from features. It does not copy or observe the future sign.

Selection minimizes pooled development log loss, breaking ties by configuration name. It freezes one variant per family, the best classifier across nine candidates, and the lower-loss baseline. The artifact's `global_champion` field is explicitly classifier-only; `overall_development_leader` includes baselines. In this run the baseline leads development, so the best classifier has not earned promotion over it. All three family representatives are evaluated in confirmation as predeclared comparisons. A different confirmation winner does not replace the selected classifier.

## Metrics and dependence

Accuracy, balanced accuracy, log loss, Brier score and ROC AUC are reported. Undefined single-class AUC is null. Primary gain is the mean paired per-row loss difference `baseline loss − selected-classifier loss`; positive favors the classifier. It is not a probability percentage-point difference or a financial return.

Non-circular overlapping moving blocks of 63 and 126 sessions, 2,000 draws, seed 20261024, yield percentile 95% intervals. The twenty-session labels overlap; purging addresses training information leakage while blocks address local dependence in evaluation. Neither makes daily rows independent nor corrects the full historical search. Family intervals are descriptive comparisons, without a new multiple-testing-based model selection. Block choice, stationarity and changing regimes constrain interpretation.

Fixed five-bin reliability, annual scores, as-of inverted/non-inverted curve slices and every-twentieth-anchor scores are diagnostics. They do not select thresholds or regimes. Sparse anchors reduce overlapping windows but do not prove independence.

## Availability stress

Delay4 uses an older observation for each fixed decision date, while the decision-day target start and twenty-session endpoint stay aligned. The code verifies identical decision-date outcomes across protocols. The base selected model/feature variant stays fixed, then refits only on eligible pre-confirmation labels under the stress clock. The two baselines remain train-fitted. No stress winner or replacement configuration is selected.

## Data and model limits

The Board's reuse policy is documented in [DATA_LICENSE.md](../DATA_LICENSE.md); Treasury's 2021 measurement-method change is retained. Current-vintage data can contain unknown revisions, and the historical 16:15 release schedule is not reconstructed for every date. A conservative full-session allowance is an assumption rather than proof of archived point-in-time availability. No duration, convexity, coupon, carry, financing, transaction-cost or executable-quote model converts these classifications into trading returns.

## Legacy synthetic / anonymous-challenge contract

The following is the preserved prior interface. It is a separate demonstration and does not describe the Treasury schema or its known calendar. Opaque challenge dates do not become a true forward history by sorting their labels.


<details>
<summary>Legacy input and validation contract</summary>

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

</details>


## Worked example and interpretation


## One row, six clocks

Run `python examples/explain_treasury_clock.py`. The example uses clearly artificial yields increasing 0.01 percentage points each supplied weekday, so the arithmetic is visible without pretending it is a market forecast.

The first feature row uses index20. At index21's end, it is assumed available after a full publication session. The decision is index22 at 09:00 New York. The target starts with the later index22 measurement and ends at index42, twenty observed sessions later. Its value is `0.01 × 20 = 0.20` percentage points = 20 basis points, so the label is 1. Using index20 as the target start would include two pre-decision sessions and would be the wrong task.

The outcome becomes an eligible training label only after index43's publication allowance. Before that, its feature row may exist, but its future target must not be used in fitting, preprocessing decisions or model selection. The example prints how many earlier decisions actually have known outcomes at a later boundary.

## A real-data keyed join

`ROW_ID` is the observation's ISO date, used only as an alignment key. `decision_date` is later. Feature rows are selected by key rather than position; shuffling the feature table leaves the aligned inputs unchanged. Adding a `target`, label endpoint or future-change column to the whitelisted table is rejected. None of those identifiers is a predictor.

The report's `sample_clocks.csv` records all 8,732 complete sample clocks. `fold_audit.csv` shows each validation boundary, last available training label and purged count. Twenty-one pre-confirmation rows are excluded from model-selection outcomes. CSV timestamp offsets are UTC; the interpretation of observation/decision dates remains New York time.

## Compare a classifier with a real alternative

The training-climatology probability is `(up_count+1)/(n+2)`. It is fitted to known training labels, not the majority class eventually seen in confirmation. A second baseline conditions that probability on a past twenty-session direction. Both are legitimate probability forecasts with no boosted trees.

Development log loss prefers the simple baseline overall. CatBoost/dynamics is the lowest-loss classifier, so it remains the frozen classifier comparator. Confirmation log loss is 0.71866 versus the baseline's 0.70915: gain is `baseline − classifier = −0.00951`, which favors the baseline. Its 63-session interval is below zero. The accuracy difference, however, has an interval spanning zero; the two metrics answer different questions.

## Explain a tempting result without changing the experiment

The frozen logistic/curve representative has confirmation log loss 0.68488 and accuracy 52.54%. That does not allow a retrospective switch to a new selected model. Its paired gain interval spans zero for 63-session blocks and is positive for 126-session blocks. Both are shown. The availability stress also has an inconclusive gain interval. These are useful limitations, not rows to remove from a presentation.

## Read the evidence in this order

1. [Main report](../reports/real/REPORT.md): what was selected, what failed, and what remains uncertain.
2. [Protocol](../ml_validation/datasets/protocol.json) and [frozen choice](../reports/real/frozen_selection.json): what was fixed before confirmation metrics.
3. [Confirmation predictions](../reports/real/confirmation_predictions.csv): per-date probabilities, outcomes and clocks.
4. [Methods](METHODS.md), [validation](VALIDATION.md) and [data rights](../DATA_LICENSE.md): assumptions and checks.

The yield-direction question does not include a bond position, coupons, costs or market fills. A classification improvement cannot be restated as trading return or Sharpe.
