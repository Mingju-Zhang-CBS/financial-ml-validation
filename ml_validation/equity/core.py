from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso

FEATURES = ['return_5', 'return_20', 'return_60', 'momentum_12_1', 'volatility_20', 'log_cap', 'adv_20', 'amihud_20']
FACTORS = ['mktrf', 'smb', 'hml', 'rmw', 'cma']


def checked_unique(frame, keys):
    conflicts = frame[frame.duplicated(keys, keep=False)]
    if len(conflicts):
        raise ValueError(f'Conflicting or duplicate source keys: {keys}; quarantine before research')
    return frame


def historical_mapping(records, mappings):
    checked_unique(records, ['permno', 'date'])
    output = records.merge(mappings, on='permno', how='left')
    valid = output[(output.date >= output.start_date) & (output.date <= output.end_date)]
    ambiguous = valid.duplicated(['permno', 'date'], keep=False)
    return valid[~ambiguous].copy(), valid[ambiguous].copy()


def monthly_universe(daily, limit=500):
    checked_unique(daily, ['permno', 'date'])
    frame = daily.copy()
    frame['month'] = frame.date.dt.to_period('M')
    last = frame.groupby('month').date.transform('max')
    selection = frame[frame.date.eq(last)].copy()
    mask = selection.securitytype.eq('EQTY') & selection.securitysubtype.eq('COM') & selection.sharetype.eq('NS')
    mask &= selection.usincflg.eq('Y') & selection.issuertype.isin(['ACOR', 'CORP'])
    mask &= selection.primaryexch.isin(['N', 'A', 'Q']) & selection.conditionaltype.eq('RW') & selection.tradingstatusflg.eq('A')
    selection = selection[mask & selection.cap.gt(0)]
    selection = selection.sort_values(['month', 'cap', 'permno'], ascending=[True, False, True])
    selection = selection.drop_duplicates(['month', 'permco']).groupby('month').head(limit)
    selection['hold_month'] = selection.month + 1
    return selection[['permno', 'hold_month', 'date', 'cap']].rename(columns={'date': 'selection_date', 'cap': 'selection_cap'})


def rolling_coefficients(returns, factors, window=252):
    design = np.column_stack([np.ones(len(factors)), factors])
    valid = np.isfinite(returns) & np.isfinite(design).all(axis=1)
    design_safe = np.where(valid[:, None], design, 0.)
    response = np.where(valid, returns, 0.)
    gram = np.einsum('ni,nj->nij', design_safe, design_safe)
    cross = design_safe * response[:, None]
    def past_sum(values):
        cumulative = np.concatenate([np.zeros_like(values[:1]), np.cumsum(values, axis=0)])
        result = np.full_like(values, np.nan, dtype=float)
        result[window:] = cumulative[window:-1] - cumulative[:-window-1]
        return result
    counts = past_sum(valid.astype(float))
    grams, crosses = past_sum(gram), past_sum(cross)
    eligible = counts == window
    coefficients = np.full((len(returns), design.shape[1]), np.nan)
    coefficients[eligible] = np.einsum('nij,nj->ni', np.linalg.pinv(grams[eligible]), crosses[eligible])
    return coefficients


def asset_panel(asset, factors):
    dates = factors.index
    frame = asset.set_index('date').reindex(dates).copy()
    returns = frame.ret.where(frame.ret.ge(-1))
    growth = np.log1p(returns.where(returns.gt(-1)))
    for horizon in [5, 20, 60]:
        frame[f'return_{horizon}'] = np.expm1(growth.rolling(horizon, min_periods=horizon).sum())
    frame['momentum_12_1'] = np.expm1(growth.shift(21).rolling(231, min_periods=231).sum())
    frame['volatility_20'] = returns.rolling(20, min_periods=20).std(ddof=1)
    frame['log_cap'] = np.log(frame.cap.where(frame.cap.gt(0)))
    frame['adv_20'] = frame.dollar_volume.rolling(20, min_periods=20).mean()
    frame['amihud_20'] = (returns.abs() / frame.dollar_volume.where(frame.dollar_volume.gt(0))).rolling(20, min_periods=20).mean()
    excess = (returns - factors.rf).to_numpy()
    coefficients = rolling_coefficients(excess, factors[FACTORS].to_numpy())
    design = np.column_stack([np.ones(len(factors)), factors[FACTORS]])
    realized_residual = excess - np.sum(coefficients * design, axis=1)
    frame['risk_scale'] = pd.Series(realized_residual, index=dates).shift(1).rolling(252, min_periods=252).std(ddof=0)
    future_design = pd.DataFrame(design, index=dates).shift(-2).to_numpy()
    frame['target'] = pd.Series(excess, index=dates).shift(-2) - np.sum(coefficients * future_design, axis=1)
    frame['raw_return'] = returns.shift(-2)
    for offset, name in [(1, 'decision_date'), (1, 'execution_date'), (2, 'label_end'), (3, 'label_available')]:
        frame[name] = pd.Series(dates, index=dates).shift(-offset)
    frame['date'] = dates
    return frame


def prepare_panel(daily, factors, universe):
    checked_unique(daily, ['permno', 'date'])
    checked_unique(factors.reset_index(), ['date'])
    frames = []
    for permno, asset in daily.groupby('permno', sort=True):
        part = asset_panel(asset, factors)
        part['permno'] = permno
        part['hold_month'] = part.date.dt.to_period('M')
        frames.append(part.reset_index(drop=True))
    panel = pd.concat(frames, ignore_index=True).merge(universe[['permno', 'hold_month', 'selection_date']], on=['permno', 'hold_month'], validate='many_to_one')
    if not (panel.selection_date < panel.date).all():
        raise ValueError('Universe selected using current or future date')
    panel['year'] = panel.decision_date.dt.year
    return panel


@dataclass
class Transform:
    low: np.ndarray
    high: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    weight_low: float
    weight_high: float
    weight_mean: float

    @classmethod
    def fit(cls, frame, equal_weight=False, features=FEATURES):
        values = frame[features].to_numpy(float)
        low, high = np.quantile(values, [.005, .995], axis=0)
        clipped = np.clip(values, low, high)
        mean, scale = clipped.mean(axis=0), clipped.std(axis=0)
        scale = np.where(scale > 0, scale, 1.)
        weights = np.ones(len(frame)) if equal_weight else 1 / frame.risk_scale.to_numpy()**2
        weight_low, weight_high = np.quantile(weights, [.005, .995])
        return cls(low, high, mean, scale, float(weight_low), float(weight_high), float(np.clip(weights, weight_low, weight_high).mean()))

    def features(self, frame, names=FEATURES):
        return (np.clip(frame[names].to_numpy(float), self.low, self.high) - self.mean) / self.scale

    def weights(self, frame, equal_weight=False):
        values = np.ones(len(frame)) if equal_weight else 1 / frame.risk_scale.to_numpy()**2
        return np.clip(values, self.weight_low, self.weight_high) / self.weight_mean


def purged_training(frame, evaluation_start):
    return frame[(frame.decision_date < evaluation_start) & (frame.label_available < evaluation_start)].copy()


def weighted_mse(actual, predicted, weights):
    return float(np.average((np.asarray(actual) - predicted)**2, weights=weights))


def fit_model(training, method, alpha=1., standardized=False):
    names = ['return_20'] if method == 'return_20_only' else FEATURES
    transform = Transform.fit(training, standardized, names)
    weights = transform.weights(training, standardized)
    target = training.target.to_numpy() / training.risk_scale.to_numpy() if standardized else training.target.to_numpy()
    if method == 'zero':
        estimator = 0.
    elif method == 'training_mean':
        estimator = float(np.average(target, weights=weights))
    else:
        estimator = Ridge(alpha=alpha) if method == 'ridge' else Lasso(alpha=alpha, max_iter=10000, tol=1e-7) if method == 'lasso' else LinearRegression()
        estimator.fit(transform.features(training, names), target, sample_weight=weights)
    return transform, estimator, names


def predict(fitted, frame):
    transform, estimator, names = fitted
    return np.full(len(frame), estimator) if isinstance(estimator, float) else estimator.predict(transform.features(frame, names))


def block_interval(values, block=20, draws=2000, seed=20261026):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return [None, None, None]
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for position in range(draws):
        starts = rng.integers(0, len(values), int(np.ceil(len(values) / block)))
        indices = ((starts[:, None] + np.arange(block)) % len(values)).ravel()[:len(values)]
        estimates[position] = values[indices].mean()
    return [float(values.mean()), *np.quantile(estimates, [.025, .975]).tolist()]
