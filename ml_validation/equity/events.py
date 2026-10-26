import numpy as np
import pandas as pd
from scipy.stats import norm


def multiplicity(pvalues, alpha=.05):
    pvalues = np.asarray(pvalues, float)
    family = len(pvalues)
    bonferroni = np.minimum(1., pvalues * family)
    order = np.argsort(np.where(np.isfinite(pvalues), pvalues, np.inf))
    sorted_values = np.where(np.isfinite(pvalues[order]), pvalues[order], 1.)
    adjusted = np.minimum.accumulate((sorted_values * family / np.arange(1, family + 1))[::-1])[::-1]
    bh = np.empty(family)
    bh[order] = np.minimum(1., adjusted)
    bh[~np.isfinite(pvalues)] = np.nan
    return pd.DataFrame(dict(pvalue=pvalues, bonferroni=bonferroni, bh=bh,
                             bonferroni_reject=bonferroni <= alpha, bh_reject=bh <= alpha))


def dependent_mean_test(values, lag=20):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) <= lag + 2:
        return float('nan')
    centered = values - values.mean()
    variance = np.dot(centered, centered) / len(values)
    for offset in range(1, lag + 1):
        variance += 2 * (1 - offset / (lag + 1)) * np.dot(centered[offset:], centered[:-offset]) / len(values)
    if variance <= 0:
        return float('nan')
    return float(2 * norm.sf(abs(values.mean()) / np.sqrt(variance / len(values))))


def shock_events(panel):
    events = panel[(panel.return_5.abs() >= 2 * panel.volatility_20 * np.sqrt(5)) & panel.volatility_20.gt(0)]
    return events[['permno', 'date', 'return_5']].assign(event_kind=lambda value: np.where(value.return_5 >= 0, 'positive_price_shock', 'negative_price_shock'))


def event_windows(events, realized_returns, calendar):
    positions = {date: index for index, date in enumerate(calendar)}
    rows = []
    keyed = realized_returns.set_index(['permno', 'date']).ret
    if keyed.index.duplicated().any():
        raise ValueError('Unique stock/session returns required')
    for event in events.itertuples():
        origin = positions[event.date]
        for offset in range(-5, 6):
            index = origin + offset
            if 0 <= index < len(calendar):
                value = keyed.get((event.permno, calendar[index]), np.nan)
                rows.append(dict(event_date=event.date, event_kind=event.event_kind, offset=offset, ret=value))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    clusters = frame.groupby(['event_kind', 'offset', 'event_date']).ret.mean().reset_index()
    return clusters.groupby(['event_kind', 'offset']).ret.agg(['mean', 'std', 'count']).reset_index()
