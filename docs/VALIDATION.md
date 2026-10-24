# Validation and reproducibility

The suite contains **33 automated cases**: 22 real-study cases and 11 preserved synthetic/anonymous-challenge cases. It checks behavior rather than only reproducing a displayed metric.

## Real-data checks

- Public snapshot fingerprint, date coverage, source-specific reuse policy and units.
- Duplicate dates, missing 10-year labels, infinite yields, extra target columns and invalid input order fail closed; other missing tenors are not filled.
- Future yield perturbations leave earlier feature rows unchanged.
- The target begins after the decision clock, not at the stale feature quote; incomplete future outcomes are censored.
- Label availability, not decision date alone, determines training eligibility; all development labels finish before confirmation.
- Input-key permutation preserves feature alignment; label/timing fields are excluded by schema.
- Logistic, LightGBM and CatBoost fit training-only transforms and ignore changed test labels.
- Mutating confirmation labels cannot change the frozen choice or its predicted probabilities.
- Baselines fit only known training labels; reordered development rankings preserve deterministic selection.
- Moving-block intervals match an independent manual resampling calculation and preserve paired rows; invalid shape, probabilities and duplicate dates are rejected.
- Reliability bins account for all predictions; undefined single-class AUC is represented explicitly.
- Longer availability delay uses older features while preserving aligned decision-date target outcomes.

The legacy tests separately cover keyed challenge joins, unknown categories, anonymous-date limitations and synthetic forward splits. They do not establish that real anonymous challenge labels have an ordered calendar.

## Executed reproduction

The release uses a fresh Python 3.12 environment and the locked dependency closure. Source tests, dependency checks, the real study, the legacy synthetic run and both worked examples are run. A standalone wheel is installed and exercised outside the source directory; numerical outputs and CSVs are compared with the source run. The [current workflow](https://github.com/Mingju-Zhang-CBS/financial-ml-validation/actions/workflows/checks.yml) performs tests, dependency checks, both complete studies and both examples on Ubuntu/Python 3.12.

The run manifest binds the packaged snapshot, protocol, source record, Python source files, environment and every generated artifact. Runtime and sampled process RSS are machine-specific; plotting and final artifact writing are outside the timed experiment block. Dependency deprecation warnings are recorded; they are not silently treated as failed tests.

## What checks cannot establish

No test recovers missing historical release vintages or proves that familiar markets were unseen. Purging protects training information boundaries; twenty-session confirmation labels remain dependent. Bootstrap intervals are conditional historical inference, with visible block sensitivity. No trading-P&L, production deployment or execution-capacity claim follows from this classifier study.
