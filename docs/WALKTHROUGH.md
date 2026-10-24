# Walkthrough: a probability forecast with a future origin

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
