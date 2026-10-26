# Equity signals: information, risk and validation

**Question:** Do transparent price and liquidity characteristics retain cross-sectional information after risk adjustment and temporal isolation?

This extension is newly implemented from public methods. Earlier Quant Investment coursework supplies research context, not public source code, licensed feature tables or inherited performance. The public Treasury study remains independently reproducible; equity inputs require the reader's own license. The [fixed protocol](../ml_validation/equity/protocol.json) precedes the new evaluations.

## Research chain

```mermaid
flowchart LR
    A[Historical identity and month-end size] --> B[Past-only features]
    B --> C[Lagged factor risk and explicit target clock]
    C --> D[Inner expanding selection]
    D --> E[Annual outer evaluation]
    E --> F[Date-dependent inference and basket diagnostics]
```

### Historical universe and units

The intended real universe uses the preceding month's final market session: US-incorporated ordinary ACOR/CORP shares, regular-way active NYSE/NYSE American/Nasdaq listings, largest eligible class per company and then top 500 by market capitalization. Eligibility is dated, not a current constituent list. Exit and delisting records remain in the underlying return history for already selected holdings. An identity conflict is quarantined rather than resolved using future frequency. The [WRDS CIZ conversion macro](https://wrds-www.wharton.upenn.edu/pages/wrds-research/macros/crsp-ciztosiz-macro/) documents the ordinary-share and exchange mappings.

CIZ capitalization is in thousands and is converted to dollars. Volume counts raw daily shares; close times shares is a dollar-volume proxy, not intraday turnover summed over trades. CIZ daily total return incorporates its distribution/delisting conventions. No second dividend or legacy delisting-return series is added. Return-duration flags and missing outcomes require audit before calling a record a one-session label.

### Eight transparent features

|Feature|Definition at observation close t|
|---|---|
|5 / 20 / 60-session return|Compounded total returns over the respective complete window|
|12–1 momentum|231-session compounded return ending 21 sessions before t|
|20-session volatility|Sample standard deviation of daily total returns|
|Log capitalization|Log of contemporaneous dollar capitalization|
|20-session average dollar volume|Mean of absolute daily price times raw shares volume|
|20-session Amihud proxy|Mean absolute total return divided by positive dollar volume|

All windows use a shared session calendar. Missing observations are not forward or backward filled.

### Numerical timing example

Observe Monday's close, assume the selected fields available before Tuesday 09:00 New York, decide Tuesday and implement Tuesday close. The label is the return from Tuesday close to Wednesday close, assumed available Thursday morning. Training before Wednesday morning cannot use that label. A next-observed-row shift could silently turn a missing Wednesday record into a longer horizon; calendar joins avoid that error.

The 252-session five-factor regression fits excess returns through t−1. Its origin coefficients remain fixed when subtracting factor contributions from the t+2 outcome. A separate 252-session residual-risk estimate ends t−1. With a 1.2% raw return, 0.01% risk-free return, beta 1.2, 0.8% market factor and zero other terms, the excess residual is `0.012 - 0.0001 - 1.2*0.008 = 0.0023`, or 23 bps. This differs from a convention subtracting factors directly from raw return.

Factor exposures are sensitivities; realized factor returns are daily outcomes. Residual attribution uses future realized factors only as part of the target, never as predictor features. Current-vintage French factors and a simple availability allowance do not reconstruct historical publication/revision archives.

The target also contains estimation error from the rolling intercept and betas. A model can predict that error without predicting raw stock returns. The fixture generates independent idiosyncratic shocks, not a designed alpha process; positive residual RankIC in that fixture is therefore not evidence of raw-return predictability or a market edge. Read the raw-return basket separately from residual prediction. This benchmark dependence also limits real-data interpretation.

### Models and objectives

Compare zero, training-mean and 20-session-return-only baselines with WLS, weighted Ridge and weighted Lasso. Fit weight caps, feature clips, means and scales using training alone. Normalize clipped training inverse-variance weights to mean one; apply the fitted caps to evaluation. Ridge minimizes weighted squared error plus alpha times squared coefficients; scikit-learn's Lasso uses its normalized weighted squared-error convention plus alpha times absolute coefficients. Their penalty values are not directly comparable. A separate standardized-target sensitivity uses equal weights, avoiding unexplained double risk weighting.

Outer historical evaluations cover 2016–2025, each trained on the prior four years. Five inner expanding validation windows select penalties. Both boundaries purge observations whose labels are not available before evaluation. Inner scores support selection; annual outer scores evaluate the chosen procedure. Original classroom “N5FCV” terminology does not turn reused tuning scores into independent tests.

### Events, multiplicity and baskets

The independent event example uses positive/negative five-session price shocks relative to volatility observable at the same close, with offsets −5…+5. These are price events, not earnings or news. Event-date summaries reduce contemporaneous cross-sectional dependence; overlapping windows and selection on prices remain limitations. Bonferroni/BH operate on a fixed eight-feature family, retaining missing and failed hypotheses. Date-dependent p-values remain approximations; BH's usual FDR guarantee needs appropriate dependence assumptions.

Form equal-weight top/bottom quintiles before looking at outcomes. Each leg has gross weight one. Costs at 0/5/20 bps apply to the full sum of absolute target-weight changes; this approximation does not drift yesterday's holdings with intervening returns. Missing outcomes invalidate a diagnostic rather than change membership or become zero. Raw stock returns, origin-beta residuals and regression slopes remain separately labeled. Financing, borrow, actual fills and a funded NAV are not established by these diagnostics.

## Run and inspect

```bash
pip install ".[test,equity]"
python -m ml_validation.equity.data --raw private-data/crsp.csv.gz --factors private-data/french_factors.zip --output private-data/equity
python -m ml_validation.equity --fixture --output local-runs/equity-fixture
python -m ml_validation.equity --panel private-data/equity_panel.parquet --daily private-data/equity_daily.parquet --output local-runs/equity-real
```

The fixture is synthetic and demonstrates mechanics, not real alpha. Licensed panels and reports remain outside the public distribution; publication rights must be assessed separately from access. Reports record protocol/input/source/output hashes, every annual model, dependent intervals, coefficients, exclusions and cost sensitivity. [Temporal tests](../tests/test_equity.py) check future perturbations, missing-session labels, training-only preprocessing, historical class selection and identity conflicts.

[Main study](../README.md) · [WRDS data policy](https://wrds-www.wharton.upenn.edu/pages/about/data-download-and-analysis-policy/) · [French data and vintage changes](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
