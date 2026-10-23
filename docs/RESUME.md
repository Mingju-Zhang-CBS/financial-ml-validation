# Resume drafts and evidence

Use after reviewing the code, reproducing the study and being able to explain it. Development was AI-assisted. These drafts describe the new portfolio implementation, not an employer assignment, actual market performance or original invention of the method. Counts and numerical statements below refer to the checked-in run; refresh them if the report changes.

## Two English bullets

- Engineered a financial-classification research workflow comparing logistic regression, LightGBM and CatBoost across nine fixed model/feature configurations, with keyed label alignment, training-fold preprocessing and forward validation on a synthetic panel.
- Validated the selected model on 864 synthetic confirmation observations across 72 dates and quantified its accuracy difference versus a majority baseline using 2,000 paired date-bootstrap draws; reported a 95% interval of [-2.08, 9.03] percentage points.

## Evidence and restrictions

- Counts, selected model and uncertainty: [metrics](../reports/demo/metrics.json), [model table](../reports/demo/model_comparison.csv), [report](../reports/demo/REPORT.md).
- Protocol, feature controls and prior official-label exposure: [methods](METHODS.md).
- Neither bullet asserts an official challenge score, real-data alpha or never-seen official holdout. Purged rows are intentionally omitted from development/final-fit counts, so these are not a claim that every generated row was evaluated.
- Relevant tests: key permutation, class-column aliases, date isolation, purge, training-only transforms, unknown categories and paired bootstrap.
