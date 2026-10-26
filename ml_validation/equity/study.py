from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd

from .core import FEATURES, fit_model, predict, purged_training, weighted_mse, block_interval, checked_unique
from .events import multiplicity, dependent_mean_test, shock_events, event_windows


def inner_selection(training, method, candidates, standardized):
    dates = np.sort(training.decision_date.unique())
    boundaries = np.linspace(len(dates) // 2, len(dates), 6, dtype=int)
    records = []
    for alpha in candidates:
        scores = []
        for fold in range(5):
            validation_dates = dates[boundaries[fold]:boundaries[fold + 1]]
            if not len(validation_dates):
                raise ValueError('Insufficient time windows')
            train = purged_training(training, validation_dates[0])
            validation = training[training.decision_date.isin(validation_dates)]
            fitted = fit_model(train, method, alpha, standardized)
            actual = validation.target / validation.risk_scale if standardized else validation.target
            score = weighted_mse(actual, predict(fitted, validation), fitted[0].weights(validation, standardized))
            scores.append(score)
            records.append(dict(model=method, alpha=alpha, fold=fold, train_rows=len(train), validation_rows=len(validation),
                                train_label_max=str(train.label_available.max()), validation_start=str(validation_dates[0]), mse=score))
        records[-1]['average_mse'] = float(np.mean(scores))
    averages = [(item['average_mse'], item['alpha']) for item in records if 'average_mse' in item]
    return min(averages)[1], records


def basket_diagnostics(predictions, column):
    weights, rows, prior = {}, [], {}
    for date, group in predictions.groupby('execution_date', sort=True):
        group = group.sort_values([column, 'permno'])
        count = len(group) // 5
        if count < 1:
            continue
        current = {int(identifier): -1 / count for identifier in group.permno.iloc[:count]}
        current.update({int(identifier): 1 / count for identifier in group.permno.iloc[-count:]})
        turnover = sum(abs(current.get(key, 0.) - prior.get(key, 0.)) for key in set(current) | set(prior))
        selected = group[group.permno.isin(current)].copy()
        selected['weight'] = selected.permno.map(current)
        complete = selected.raw_return.notna().all()
        gross = float(np.sum(selected.weight * selected.raw_return)) if complete else np.nan
        residual = float(np.sum(selected.weight * selected.target)) if selected.target.notna().all() else np.nan
        exposure = {f'exposure_{feature}': float(np.sum(selected.weight * selected[feature])) for feature in FEATURES}
        rows.append(dict(execution_date=date, year=int(group.year.iloc[0]), gross_raw_return=gross, residual_diagnostic=residual,
                         turnover=turnover, gross_exposure=sum(abs(value) for value in current.values()), net_exposure=sum(current.values()),
                         outcome_coverage=float(selected.raw_return.notna().mean()),
                         **{f'net_{cost}bps': gross - turnover * cost / 10000 for cost in [0, 5, 20]}, **exposure))
        weights[date] = current
        prior = current
    return pd.DataFrame(rows)


def annual_evaluation(panel, protocol, standardized=False):
    eligible_features = np.isfinite(panel[FEATURES]).all(axis=1) & panel.risk_scale.gt(0) & panel.label_available.notna()
    usable = panel[eligible_features].copy()
    results, tuning, coefficients, annual = [], [], [], []
    for year in protocol['outer_years']:
        evaluation = usable[usable.year == year].copy()
        if evaluation.empty:
            raise ValueError(f'Missing outer evaluation year {year}')
        candidates = usable[usable.year.between(year - 4, year - 1) & usable.target.notna()]
        training = purged_training(candidates, evaluation.decision_date.min())
        if training.empty:
            raise ValueError(f'Missing training history for {year}')
        evaluation['training_mean'] = predict(fit_model(training, 'training_mean', standardized=standardized), evaluation)
        for method in protocol['models']:
            alpha = 1.
            if method in ['ridge', 'lasso']:
                alpha, scores = inner_selection(training, method, protocol[method + '_alphas'], standardized)
                tuning.extend([dict(row, year=year, standardized=standardized) for row in scores])
            fitted = fit_model(training, method, alpha, standardized)
            evaluation[method] = predict(fitted, evaluation)
            if not isinstance(fitted[1], float):
                coefficients.extend(dict(year=year, model=method, feature=feature, coefficient=float(value), standardized=standardized)
                                    for feature, value in zip(fitted[2], fitted[1].coef_))
            actual = evaluation.target / evaluation.risk_scale if standardized else evaluation.target
            valid = np.isfinite(actual)
            weights = fitted[0].weights(evaluation, standardized)
            error = weighted_mse(actual[valid], evaluation.loc[valid, method], weights[valid])
            baseline = weighted_mse(actual[valid], evaluation.loc[valid, 'training_mean'] if 'training_mean' in evaluation else np.zeros(valid.sum()), weights[valid])
            annual.append(dict(year=year, model=method, standardized=standardized, alpha=alpha, rows=int(valid.sum()), predicted_rows=len(evaluation),
                               weighted_mse=error, skill_vs_training_mean=1 - error / baseline,
                               train_label_max=str(training.label_available.max()), evaluation_start=str(evaluation.decision_date.min())))
        evaluation['evaluation_target'] = evaluation.target / evaluation.risk_scale if standardized else evaluation.target
        evaluation['evaluation_weight'] = fitted[0].weights(evaluation, standardized)
        results.append(evaluation)
        print(json.dumps(dict(year=year, standardized=standardized, train=len(training), predicted=len(evaluation))), flush=True)
    return pd.concat(results, ignore_index=True), pd.DataFrame(annual), pd.DataFrame(tuning), pd.DataFrame(coefficients)


def run(panel_path, output, *, fixture=False, protocol_path=None, daily_path=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty result directory')
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = Path(protocol_path) if protocol_path else Path(__file__).with_name('protocol.json')
    protocol = json.loads(protocol_path.read_text())
    (output / 'protocol.json').write_bytes(protocol_path.read_bytes())
    panel = pd.read_parquet(panel_path)
    checked_unique(panel, ['permno', 'date'])
    started = time.perf_counter()
    predictions, annual, tuning, coefficients = annual_evaluation(panel, protocol)
    _, sensitivity, _, _ = annual_evaluation(panel, protocol, standardized=True)
    daily = []
    for date, group in predictions.groupby('decision_date', sort=True):
        valid = group[group.evaluation_target.notna()]
        for model in protocol['models']:
            predicted = valid[model]
            variable = predicted.nunique() > 1 and len(valid) > 2
            daily.append(dict(date=date, year=int(group.year.iloc[0]), model=model, rows=len(valid),
                              ic=float(predicted.corr(valid.target)) if variable else np.nan,
                              rank_ic=float(predicted.corr(valid.target, method='spearman')) if variable else np.nan,
                              mse=weighted_mse(valid.target, predicted, valid.evaluation_weight),
                              loss_gain=weighted_mse(valid.target, valid.training_mean, valid.evaluation_weight) - weighted_mse(valid.target, predicted, valid.evaluation_weight)))
    daily = pd.DataFrame(daily)
    inference = []
    for model, group in daily.groupby('model'):
        for quantity in ['rank_ic', 'loss_gain']:
            for block in [20, 63]:
                estimate, low, high = block_interval(group[quantity], block=block)
                inference.append(dict(model=model, quantity=quantity, block=block, estimate=estimate, ci_low=low, ci_high=high))
    feature_tests = []
    for feature in FEATURES:
        correlations = predictions.groupby('decision_date').apply(lambda group: group[feature].corr(group.target, method='spearman'), include_groups=False)
        feature_tests.append(dict(feature=feature, dates=int(correlations.notna().sum()), mean_rank_ic=float(correlations.mean()), pvalue=dependent_mean_test(correlations)))
    feature_tests = pd.DataFrame(feature_tests)
    adjusted = multiplicity(feature_tests.pvalue)
    feature_tests = pd.concat([feature_tests.drop(columns='pvalue'), adjusted], axis=1)
    baskets = pd.concat([basket_diagnostics(predictions, model).assign(model=model) for model in ['return_20_only', 'wls', 'ridge', 'lasso']], ignore_index=True)
    for name, frame in [('annual.csv', annual), ('standardized_target_sensitivity.csv', sensitivity), ('inner_selection.csv', tuning),
                        ('coefficients.csv', coefficients), ('daily_metrics.csv', daily), ('inference.csv', pd.DataFrame(inference)),
                        ('feature_family.csv', feature_tests), ('basket_diagnostics.csv', baskets)]:
        frame.to_csv(output / name, index=False)
    predictions.to_parquet(output / 'private_predictions.parquet', index=False)
    if daily_path is not None:
        daily_returns = pd.read_parquet(daily_path, columns=['permno', 'date', 'ret'])
        events = shock_events(panel)
        windows = event_windows(events, daily_returns, pd.DatetimeIndex(sorted(daily_returns.date.unique())))
        windows.to_csv(output / 'price_shock_event_windows.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for model in ['return_20_only', 'wls', 'ridge', 'lasso']:
        group = daily[daily.model == model]
        yearly = group.groupby('year')[['rank_ic', 'loss_gain']].mean()
        axes[0].plot(yearly.index, yearly.rank_ic, 'o-', label=model)
        axes[1].plot(yearly.index, yearly.loss_gain, 'o-', label=model)
    axes[0].set(title='Annual mean daily RankIC', xlabel='Historical outer year', ylabel='Rank correlation')
    axes[1].set(title='Loss improvement vs training mean', xlabel='Historical outer year', ylabel='Weighted squared-return units')
    for ax in axes:
        ax.axhline(0, color='#777777', linewidth=.7)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / 'annual_validation.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for model, group in baskets.groupby('model'):
        values = group.groupby('year')[['net_0bps', 'net_5bps', 'net_20bps', 'turnover']].mean()
        axes[0].plot([0, 5, 20], [group[f'net_{cost}bps'].mean() * 10000 for cost in [0, 5, 20]], 'o-', label=model)
        axes[1].plot(values.index, values.turnover, 'o-', label=model)
    axes[0].set(title='Quintile basket cost sensitivity', xlabel='Cost per dollar traded (bps)', ylabel='Mean daily diagnostic return (bps)')
    axes[1].set(title='One-way traded gross notional', xlabel='Year', ylabel='Sum absolute weight changes')
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / 'basket_costs.png', dpi=160); plt.close(fig)
    metrics = dict(data_kind='synthetic-fixture' if fixture else 'licensed-real-private', outer_years=protocol['outer_years'],
                   predicted_rows=len(predictions), evaluable_rows=int(predictions.target.notna().sum()),
                   dates=int(predictions.decision_date.nunique()), securities=int(predictions.permno.nunique()),
                   elapsed_seconds=time.perf_counter() - started,
                   limitations=['current vintage, no historical factor release/revision archive', 'costed basket diagnostic, no borrow/funding or NAV claim',
                                'historical evaluation; original periods may have been studied', 'vendor-derived aggregates require rights review before publication'])
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    report = f'''# Transparent equity research — executed {'synthetic fixture' if fixture else 'private licensed-data study'}

{metrics['predicted_rows']:,} feature-valid prediction rows; {metrics['evaluable_rows']:,} evaluable labels; {metrics['dates']:,} decision sessions. These counts describe this run and do not establish independent observations or alpha. A synthetic fixture demonstrates behavior only.

![Annual evaluation](annual_validation.png)

## Read the evidence

The annual file retains every model and outer year. Inner selection uses five expanding windows inside the preceding four years; its validation scores are not outer-test scores. Both training and validation use fitted training-only clipping, scaling and inverse-variance weight caps. Selection is purged by label availability. The standardized-target sensitivity uses equal weights, separately from the main raw-residual target.

`daily_metrics.csv` contains IC, RankIC and weighted loss improvements; `inference.csv` reports fixed 20- and 63-session block intervals. Constant baselines have undefined IC, not artificial zero IC. `feature_family.csv` retains all eight hypotheses, including failures; Bonferroni controls familywise error with valid marginal tests, while BH requires independence or suitable positive dependence for its usual FDR interpretation. HAC p-values remain an asymptotic approximation.

![Costs and turnover](basket_costs.png)

## Financial meaning

Stocks are sorted before observing outcomes. Missing realized returns invalidate that basket's outcome rather than becoming zeros or changing its membership. Long and short legs each have unit gross weight; costs use the full sum of absolute weight changes at 0/5/20 bps. This is a diagnostic: no borrow availability, financing, volume participation or executable NAV is established. Raw stock returns, origin-beta residuals and cross-sectional prediction slopes have different units and must not be conflated.

## Information contract and limitations

Observation t close precedes decision t+1 morning and implementation t+1 close. The return runs to t+2 close and is assumed available t+3 morning. Reindexed calendar joins prevent a missing session from becoming a next-observed-row target. The 252-session factor fit ends t-1; its coefficients are held fixed when constructing the future residual. No future volatility normalizes a target. Factor returns are used for ex-post attribution; the current factor vintage and assumed release allowance do not establish historical point-in-time factor availability. CIZ total returns already include their income/delisting conventions; no second dividend or delisting return is added.

Raw records, security predictions and contractually restricted derived data stay private. Publication requires the applicable vendor rights assessment; a fixture report cannot substitute for a real-data result. Source, protocol and output hashes are recorded in `manifest.json`.
'''
    (output / 'REPORT.md').write_text(report, encoding='utf-8')
    manifest = dict(input_sha256=hashlib.sha256(Path(panel_path).read_bytes()).hexdigest(), protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                    source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(__file__).parent.glob('*.py')},
                    outputs_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir() if path.is_file()})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return metrics
