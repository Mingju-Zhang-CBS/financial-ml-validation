import numpy as np
import pandas as pd
import pytest
from ml_validation.equity.core import FEATURES, FACTORS, Transform, asset_panel, monthly_universe, purged_training, fit_model, predict, historical_mapping
from ml_validation.equity.events import multiplicity
from ml_validation.equity.study import basket_diagnostics, annual_evaluation


def fixture():
    rng = np.random.default_rng(912)
    dates = pd.bdate_range('2010-01-01', periods=850, name='date')
    factors = pd.DataFrame(rng.normal(0, .004, (850, 5)), index=dates, columns=FACTORS).assign(rf=.0001)
    asset = pd.DataFrame(dict(date=dates, ret=rng.normal(0, .01, 850), cap=100000., dollar_volume=10000000., permno=1))
    return asset, factors


def test_future_data_cannot_change_past_features_or_origin_risk():
    asset, factors = fixture()
    original = asset_panel(asset, factors)
    changed = asset.copy()
    changed.loc[700:, ['ret', 'cap', 'dollar_volume']] *= 7
    changed_factors = factors.copy()
    changed_factors.iloc[700:] *= 3
    perturbed = asset_panel(changed, changed_factors)
    pd.testing.assert_frame_equal(original.iloc[:700][FEATURES + ['risk_scale']], perturbed.iloc[:700][FEATURES + ['risk_scale']])
    pd.testing.assert_series_equal(original.target.iloc[:698], perturbed.target.iloc[:698])


def test_calendar_target_does_not_bridge_missing_security_sessions():
    asset, factors = fixture()
    asset = asset[asset.date != factors.index[702]]
    panel = asset_panel(asset, factors)
    assert pd.isna(panel.target.iloc[700])
    assert panel.label_end.iloc[700] == factors.index[702]
    assert panel.label_available.iloc[700] == factors.index[703]


def test_training_transform_and_selection_ignore_later_extremes():
    asset, factors = fixture()
    panel = asset_panel(asset, factors).dropna(subset=FEATURES + ['target', 'risk_scale', 'label_available'])
    panel = panel[panel.risk_scale > 0]
    boundary = panel.decision_date.iloc[150]
    train = purged_training(panel, boundary)
    changed = panel.copy()
    changed.loc[changed.decision_date >= boundary, FEATURES] = 1e12
    changed.loc[changed.decision_date >= boundary, 'risk_scale'] = 1e-12
    other = purged_training(changed, boundary)
    first, second = Transform.fit(train), Transform.fit(other)
    assert first.weight_high == second.weight_high
    np.testing.assert_array_equal(first.mean, second.mean)
    for method in ['wls', 'ridge', 'lasso']:
        np.testing.assert_allclose(predict(fit_model(train, method), train), predict(fit_model(other, method), train))
    assert (train.label_available < boundary).all()


def test_monthly_universe_uses_previous_month_and_one_share_class():
    dates = pd.to_datetime(['2020-01-31'] * 3 + ['2020-02-28'] * 3)
    frame = pd.DataFrame(dict(date=dates, permno=[1, 2, 3]*2, permco=[1, 1, 2]*2, cap=[10, 20, 15, 10000, 1, 1],
                              securitytype='EQTY', securitysubtype='COM', sharetype='NS', usincflg='Y', issuertype='CORP', primaryexch='N', conditionaltype='RW', tradingstatusflg='A'))
    selected = monthly_universe(frame, limit=2)
    february = selected[selected.hold_month == pd.Period('2020-02')]
    assert set(february.permno) == {2, 3}
    frame.loc[frame.date.dt.month == 2, 'cap'] = 100000000
    other = monthly_universe(frame, 2)
    pd.testing.assert_frame_equal(february, other[other.hold_month == pd.Period('2020-02')])


def test_identity_ambiguity_is_quarantined_without_future_frequency():
    records = pd.DataFrame(dict(permno=[1], date=pd.to_datetime(['2020-01-02'])))
    mappings = pd.DataFrame(dict(permno=[1, 1], entity=['A', 'B'], start_date=pd.to_datetime(['2020-01-01']*2), end_date=pd.to_datetime(['2020-12-31']*2)))
    valid, ambiguous = historical_mapping(records, mappings)
    assert valid.empty and len(ambiguous) == 2
    future = mappings.iloc[:1].assign(start_date=pd.Timestamp('2021-01-01'), end_date=pd.Timestamp('2022-01-01'))
    other, _ = historical_mapping(records, pd.concat([mappings, future]))
    assert other.empty


def test_missing_return_does_not_reform_quintiles_or_become_zero():
    panel = pd.DataFrame(dict(permno=np.arange(10), execution_date=pd.Timestamp('2020-01-03'), year=2020,
                              ridge=np.arange(10), raw_return=[np.nan] + [.01]*9, target=.001))
    for feature in FEATURES:
        panel[feature] = 1.
    result = basket_diagnostics(panel, 'ridge')
    assert pd.isna(result.gross_raw_return.iloc[0])
    assert result.outcome_coverage.iloc[0] == .75
    assert result.gross_exposure.iloc[0] == 2


def test_multiplicity_keeps_unavailable_family_members():
    result = multiplicity([.01, .02, np.nan, .8])
    assert result.bonferroni.iloc[0] == .04
    assert not result.bh_reject.iloc[2]
    assert pd.isna(result.bh.iloc[2])


def test_outer_labels_cannot_change_model_selection_or_predictions():
    rng = np.random.default_rng(18)
    dates = pd.bdate_range('2012-01-02', '2016-12-30')
    frame = pd.DataFrame(dict(decision_date=np.repeat(dates, 3), target=rng.normal(0, .01, len(dates)*3), risk_scale=.01))
    frame['label_available'] = frame.decision_date + pd.offsets.BDay(2)
    frame['year'] = frame.decision_date.dt.year
    for feature in FEATURES:
        frame[feature] = rng.normal(size=len(frame))
    protocol = dict(outer_years=[2016], models=['zero', 'training_mean', 'ridge'], ridge_alphas=[.01, 100.])
    original, _, selection, _ = annual_evaluation(frame, protocol)
    frame.loc[frame.year == 2016, 'target'] = 1e5
    changed, _, other_selection, _ = annual_evaluation(frame, protocol)
    pd.testing.assert_frame_equal(selection, other_selection)
    np.testing.assert_array_equal(original.ridge, changed.ridge)
