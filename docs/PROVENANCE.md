# Sources and personal contribution

This is an independent new implementation of a Treasury-yield classification experiment. The original synthetic/challenge adapter remains separately labeled. No employer, course framework, recruiting task or downloaded challenge data is republished. Development and documentation were AI-assisted.

## Primary sources

- [Federal Reserve Board H.15](https://www.federalreserve.gov/releases/h15/): nominal Treasury constant-maturity yields and current release schedule. The selected five series are government reference yields, not the commercial-paper or other third-party series appearing elsewhere in H.15.
- [Official data download page](https://www.federalreserve.gov/datadownload/Choose.aspx?rel=H15): Treasury Constant Maturities CSV package. Exact URL, retrieval time, original response hash, identifiers, normalization and dataset hash are in [source.json](../ml_validation/datasets/source.json).
- [Board reuse policy](https://www.federalreserve.gov/disclaimer.htm): public-domain information may be copied/distributed unless an exception is identified, with attribution. [DATA_LICENSE.md](../DATA_LICENSE.md) limits this assessment to the selected source material; no seals, site assets or copyrighted third-party database is distributed.
- [Treasury curve methodology](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/treasury-yield-curve-methodology): indicative curve construction and the December 2021 method change. Historical official observations are not rewritten as a constant-methodology series.

## Implementation and contribution

The implementation covers the selected-series data contract, past-only curve/dynamics features, delayed prediction origins, label-availability rules, expanding purged splits, model comparison, frozen confirmation and dependent uncertainty. Mingju Zhang selected the research direction; code, experiments, tests and writing were AI-assisted. Personal course contributions and independent human review remain separately evidenced. NumPy, pandas, scikit-learn, LightGBM, CatBoost, Matplotlib, SciPy and psutil retain their own authorship and licenses. The new equity extension is an independent implementation; source classroom notebooks, collaborator utilities and vendor records are excluded.

Methods such as logistic regression, gradient boosting, standard scaling and block resampling are established building blocks. No novel alpha, official challenge result or trading track record is claimed. The economic question differs from the anonymous asset-allocation challenge: observed-calendar Treasury yield direction can be audited directly, while anonymous dates remain a limitation of the separate legacy adapter.

## Evidence boundary

The snapshot is a current historical vintage retrieved in October 2026. Exact past release clocks, revisions and a prospective evaluation are unavailable. The label predicts a reference-yield change, not a bond total return. Fitted probability quality, classification accuracy and conditional model rankings are kept separate from portfolio performance, transaction costs and execution. Negative, inconclusive and block-sensitive findings are retained without changing the selected classifier after confirmation.
