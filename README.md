![Financial ML Validation — Probability forecasts that earn their confidence](assets/header.svg)

# Financial ML Validation

**Real public Treasury yields · time-aware financial ML · probability quality**

Can yield-curve features improve forecasts of the next twenty observed-session direction in the 10-year Treasury reference yield? I compare logistic regression, LightGBM and CatBoost on **9,005 official H.15 observations**, with a delayed forecast origin, label-availability purging and a development freeze. The result connects rates research with practical data-science judgment: a more complex classifier must earn its place against a simple probability baseline.

[Results](reports/real/REPORT.md) · [Walkthrough](docs/WALKTHROUGH.md) · [Methods](docs/METHODS.md) · [Data license](DATA_LICENSE.md) · [Validation](docs/VALIDATION.md) · [Research directory](https://github.com/Mingju-Zhang-CBS/Quant)

## Start with the finding

The **training-climatology baseline already leads development**. Among the nine classifiers, `catboost/dynamics` has the lowest development log loss; it is frozen as the classifier comparator, not promoted over the baseline.

On **1,479 historical confirmation decisions** from January 2020 through December 2025, that CatBoost configuration has log loss **0.71866**, versus **0.70915** for the baseline. Its mean paired log-loss gain is **−0.00951** (positive would favor the classifier), with a 63-session block 95% interval **[−0.01688, −0.00297]**. The selected classifier is worse on this probability loss in the recorded confirmation; that is retained.

Other evidence is less decisive. CatBoost's accuracy difference is **−0.88 percentage points**, with an interval spanning zero. The four-session availability stress also has an interval spanning zero. Logistic regression's confirmation log-loss point estimate is better, but its interval includes zero with 63-session blocks and excludes zero with 126-session blocks. **I do not switch the selected classifier after seeing confirmation.** Neither a favorable slice nor a different block length becomes a new headline winner.

![Frozen confirmation comparisons](reports/real/confirmation_comparison.png)

|Frozen configuration|Accuracy|Balanced accuracy|Log loss ↓|Brier ↓|
|---|---:|---:|---:|---:|
|Logistic / curve|52.54%|52.46%|0.68488|0.24589|
|LightGBM / dynamics|46.72%|49.86%|0.73373|0.26931|
|CatBoost / dynamics — selected classifier|45.03%|48.91%|0.71866|0.26257|
|Training climatology — selected baseline|45.91%|50.00%|0.70915|0.25796|
|Past-direction-conditioned baseline|45.91%|50.00%|0.70887|0.25782|

The baseline predicts the training period's class tendency, rather than learning the confirmation majority. Class balance changes: a classifier's accuracy is not meaningful without that context. All results are historical reference-yield forecasts, not bond returns, executable quotes or a trading record.

## What the implementation demonstrates

|Skill|Inspectable evidence|
|---|---|
|Financial data judgment|Five government yield tenors, public-domain reuse policy, unit checks, source hashes and a current-vintage limitation|
|Leakage-aware research|Separate observation, assumed publication, decision, label start/end and label-availability clocks|
|Model comparison|Nine fixed model/feature configurations, two train-fitted baselines and identical expanding folds|
|Probability evaluation|Log loss, Brier score, accuracy, balanced accuracy, AUC and fixed-bin reliability|
|Research judgment|Development baseline wins, frozen classifier underperforms, and favorable but block-sensitive alternatives remain qualified|
|Engineering|33 automated tests, offline packaged data, keyed predictions, complete timing audits and source-bound artifacts|

I independently implemented the yield schema, trailing feature construction, timed label contract, purged experiment, reporting and validation. Public methods and Python libraries retain attribution. Development and documentation were AI-assisted; original course, employer and challenge files are excluded.

## Read the clock before fitting a model

![Feature, decision and label clocks](assets/workflow.svg)

An observation on date **t** contributes only current and trailing values. A complete intervening observation session is reserved for publication. The modeled decision is at **09:00 New York time on t+2**, after assumed feature availability. The label begins with the later yield measurement on **that decision day**, then ends twenty observed sessions later. It does **not** start at the stale feature-date quote, which would include movement that happened before the decision.

Training labels must be fully available before each validation decision. Development selection also excludes late-2019 labels that become known in confirmation: **21 boundary rows are removed**. Preprocessing fits training rows only. Dates, row IDs, target endpoints and future changes are excluded from the 25-feature combined table. No missing date or yield is forward-filled.

The snapshot is the currently published historical vintage, retrieved in October 2026. Exact historical release timestamps and revision archives are unavailable. The conservative lag therefore supports an explicit timing contract, not a complete point-in-time guarantee. These are chronological historical checks; familiar market regimes cannot become genuinely unseen future evidence through a new split.

## Reproduce everything offline

Use **Python 3.12** and a fresh environment:

```bash
python -m venv .venv
# Activate .venv using the standard command for your shell.
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps -e .
python -m pip check
python -m pytest
python -m ml_validation --real --output local-runs/real
python examples/explain_treasury_clock.py
```

After dependency installation, the real study needs **no network, service, key or GPU**. Choose a new empty output directory for every run. The [checked-in report](reports/real/REPORT.md) contains the complete experiment; its manifest binds code, data, protocol, environment and generated files. A standalone wheel is checked outside the source tree so the packaged data cannot be accidentally supplied by an editable checkout.

## Fixed experiment, visible uncertainty

The source spans 1990–2025; eligible observations begin in 1991. Three expanding development windows cover 2008–2011, 2012–2015 and 2016–2019. Curve features are five yield levels, three slopes and curvature; dynamics are 1/5/20-session changes and trailing yield-change volatility. All hyperparameters, feature variants and threshold 0.50 are fixed. Pooled development log loss chooses one representative per model family, a classifier comparator and a baseline. Their choices are persisted before confirmation metrics.

Twenty-session labels overlap. Purging stops unavailable training outcomes from entering fitting or selection; it does not make confirmation rows independent. Paired moving-block intervals use both **63 and 126 sessions**, with 2,000 deterministic resamples. Yearly, as-of curve-inversion, fixed-bin reliability and every-twentieth-anchor diagnostics are retained without retuning. No confirmation recalibration, threshold optimization or early-stopping feedback is used.

![Paired uncertainty, including block-sensitive findings](reports/real/uncertainty.png)

## Source and interpretation limits

The [Federal Reserve Board H.15 service](https://www.federalreserve.gov/releases/h15/) publishes Treasury constant-maturity reference yields. Its [reuse policy](https://www.federalreserve.gov/disclaimer.htm) permits copying and distribution of public-domain information unless an exception is marked. The selected government yield series have no identified third-party restriction; attribution and exact reuse scope are documented in [DATA_LICENSE.md](DATA_LICENSE.md). No government logos or website assets are distributed.

Yields are percentages per annum, not decimal returns or bond prices. Treasury changed its curve methodology in December 2021; the series preserves actual official measurements. Predicting a yield sign does not model duration, convexity, carry, coupons, financing, costs or executable trades. This study does not claim alpha, Sharpe, a QRT leaderboard score or universally superior models.

The [legacy synthetic study](reports/demo/REPORT.md) remains a separate alignment/validation demonstration. Its anonymous-challenge adapter cannot infer a real calendar or remove unknown overlapping histories; previously inspected labels remain exposed. Run it with `python -m ml_validation --output local-runs/synthetic`. Its old accuracy and date-bootstrap numbers are not transferred to the real study.

## Continue reading

[Methods](docs/METHODS.md) · [Numerical walkthrough](docs/WALKTHROUGH.md) · [Validation record](docs/VALIDATION.md) · [Sources and contribution](docs/PROVENANCE.md) · [Interview guide](docs/INTERVIEW.md) · [QR / DS resume evidence](docs/RESUME.md)

For financial NLP, compare [Financial Text Validation](https://github.com/Mingju-Zhang-CBS/financial-text-validation). For allocation and funding, see [Systematic Strategy Lab](https://github.com/Mingju-Zhang-CBS/systematic-strategy-lab). Together they connect financial data, model validation and investment decisions without equating prediction accuracy with profitability.
