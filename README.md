![Financial ML Validation](assets/header.svg)

[![Tests and offline demo](https://github.com/Mingju-Zhang-CBS/financial-ml-validation/actions/workflows/checks.yml/badge.svg)](https://github.com/Mingju-Zhang-CBS/financial-ml-validation/actions/workflows/checks.yml)

[Research overview](https://github.com/Mingju-Zhang-CBS/Quant) · [Mingju Zhang](https://github.com/Mingju-Zhang-CBS)

# Financial ML Validation

**Research question:** How much confidence should we place in a financial classifier when observations share dates, feature histories overlap, and several model variants compete for selection?

This project implements the complete route from keyed panel inputs to a documented confirmation result. It is motivated by the [QRT asset-allocation challenge](https://challengedata.ens.fr/challenges/167), with a fully reproducible synthetic example and a separate input adapter for independently obtained challenge files.

**Python 3.12 · scikit-learn · LightGBM · CatBoost · 11 local tests passed · MIT**

## Read the result first

The public experiment contains **4,320 synthetic observations across 360 dates and 12 allocations**. Known generating order is used for forward validation; the last 72 dates form confirmation, with a 20-date gap before them.

| Confirmation measure | Selected model | Training-fitted majority baseline |
| --- | ---: | ---: |
| Accuracy | 52.66% | 49.31% |
| Balanced accuracy | 52.76% | 50.00% |
| Log loss; lower is better | 0.6993 | 0.6935 |

The development-selected model is **logistic regression with allocation identity**. Its accuracy advantage is **3.36 percentage points**, with a paired date-bootstrap 95% interval of **[-2.08, 9.03] percentage points**.

**Research interpretation:** the accuracy point estimate improves, but the interval includes zero and log loss is worse. This run supports a working validation pipeline and a qualified model comparison. It does not establish a dependable forecasting edge. A more complex estimator did not automatically earn selection.

![Nine model and feature comparisons](reports/demo/model_comparison.png)

The chart summarizes development results; the table above summarizes the separately evaluated confirmation stage. They serve different purposes.

## What this project demonstrates

| Skill | Concrete evidence |
| --- | --- |
| Financial data engineering | Unique keyed X/y joins, exact key-set checks, explicit target aliases and a feature whitelist. |
| Statistical validation | Whole-date isolation, forward folds when chronology is known, and explicit limits when it is unknown. |
| Applied machine learning | Comparable pipelines for logistic regression, LightGBM and CatBoost. |
| Feature research | Three fixed ablations distinguish raw history, summary statistics and allocation identity. |
| Uncertainty and judgment | Paired date-cluster bootstrap, group diagnostics and retention of an uncertain result. |
| Python engineering | Behavioral tests, portable packaging, source-bound manifests and measured resource use. |

## The complete research workflow

![Financial ML workflow](assets/workflow.svg)

1. **Define the task.** Predict whether the next-period allocation return is positive. Zero maps to class 0; this is classification rather than a trading-P&L model.
2. **Establish alignment.** Match labels by `ROW_ID`, reject duplicates or incomplete key sets, and remove dates, row keys and labels from the feature space.
3. **Choose a defensible split.** For synthetic data, use the known generator order, three expanding folds and a 20-date purge. For anonymous official dates, use reproducible whole-date groups and disclose unknown chronology.
4. **Fit transformations locally.** Each training fold learns its own imputation, numeric scaling and category vocabulary. Unknown validation categories are handled without refitting.
5. **Compare nine predeclared candidates.** Three estimators receive raw, summary or identity features on the same folds. Hyperparameters and threshold 0.50 remain fixed.
6. **Freeze the selection.** Rank candidates on pooled development accuracy; break ties deterministically. Refit the selected configuration on development rows before evaluating confirmation.
7. **Explain the evidence.** Compare the final candidate with a training-fitted baseline, resample entire dates for uncertainty, and inspect groups without retuning on their outcomes.

The synthetic split uses 268 final-fit development dates, 20 unused boundary dates and 72 confirmation dates. Development fold validation covers a subset of development rows; the model is then refitted on all development rows. See the [walkthrough](docs/WALKTHROUGH.md) for the split arithmetic and a worked keyed join.

## Reproduce the study

```text
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on macOS/Linux. Then:

```text
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps -e .
python -m pytest
python -m ml_validation --output local-runs/demo
```

Use Python 3.12 and a new or empty output directory. After dependencies are installed, the synthetic demonstration needs no network access. The checked-in run is in `reports/demo/`; new runs go to ignored `local-runs/`.

### Use independently obtained challenge data

```text
python -m ml_validation --x private-data/X_train.csv --y private-data/y_train.csv --output local-runs/challenge
```

X requires `ROW_ID`, `TS`, `ALLOCATION`, `GROUP`, 20 return histories, 20 signed-volume histories and turnover. Y contains `ROW_ID` and exactly one `TARGET` or `target` column. Keep downloaded files and user-data runs outside the public release. Full field and missing-value rules are in [METHODS](docs/METHODS.md).

## Engineering map and verification

| Implementation | Responsibility |
| --- | --- |
| [`data.py`](ml_validation/data.py) | Input contracts, label alignment, whitelisted features and synthetic generation. |
| [`research.py`](ml_validation/research.py) | Splits, fold-fitted models, selection, confirmation and bootstrap. |
| [`__main__.py`](ml_validation/__main__.py) | Command-line run, aggregate artifacts, plots and research report. |
| [`test_protocol.py`](tests/test_protocol.py) | Alignment, temporal boundaries, train-only transforms, estimator integration and invalid uncertainty inputs. |

Tests include row permutation, unknown categories, target-name ambiguity, exact prediction-vector shapes, missing date groups and nonfinite probabilities. In particular, a length-one prediction vector is rejected rather than silently broadcast across every label.

The run writes `model_comparison.csv`, `development_folds.csv`, `group_diagnostics.csv`, `frozen_selection.json`, `metrics.json`, `manifest.json` and `REPORT.md`. The manifest records input and source fingerprints, configuration and dependency versions. The badge above links to the live GitHub Actions run; the recorded local demo remains bound to the source fingerprints in its manifest.

## Assumptions and limits

- The checked-in results are synthetic. They are not an official QRT leaderboard score or a live trading record.
- Anonymous date labels cannot establish a calendar. Whole-date grouping alone cannot remove unknown overlap or dependence across dates.
- Previously inspected official labels remain exposed after a new split; a new folder is not fresh out-of-sample evidence.
- Date bootstrap preserves within-date pairing, but does not account for arbitrary serial dependence or the entire historical research search.
- Accuracy does not specify returns, position sizing, costs or execution. A separate trading model is required before an economic-performance claim.

## Explore the evidence

| Read | Purpose |
| --- | --- |
| [Worked walkthrough](docs/WALKTHROUGH.md) | Follow the research logic and a numerical example. |
| [Methods and input contract](docs/METHODS.md) | Check assumptions, conventions and supported interfaces. |
| [Executed research report](reports/demo/REPORT.md) | Inspect the recorded results and limitations. |
| [Interview discussion](docs/INTERVIEW.md) | Practice explaining decisions and failure modes. |
| [Resume evidence](docs/RESUME.md) | Trace draft claims to measured artifacts. |
| [Provenance](docs/PROVENANCE.md) | Understand attribution, AI assistance and data rights. |

New implementation and documentation were developed with AI assistance. The public task and estimator libraries retain their original attribution; no employer or classroom framework was copied into this distribution. [MIT license](LICENSE); external data rights are separate.
