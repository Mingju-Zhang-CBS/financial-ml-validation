"""Public Treasury reference yields; features and future label clocks stay separate."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

DATA=Path(__file__).resolve().parent/'datasets'
TENORS=('y3m','y1','y2','y5','y10')
CURVE=(*TENORS,'slope_10_2','slope_10_3m','slope_5_2','curvature')
DYNAMICS=tuple(f'{tenor}_change_{lag}' for lag in (1,5,20) for tenor in TENORS)+('y10_vol_20',)
VARIANTS={'curve':list(CURVE),'dynamics':list(DYNAMICS),'combined':list(CURVE+DYNAMICS)}


def validate_yields(frame):
    if list(frame.columns)!=['date',*TENORS]:raise ValueError('Expected date and five percent-per-annum Treasury yields only')
    frame=frame.copy();frame['date']=pd.to_datetime(frame.date,errors='raise')
    if frame.empty or frame.date.isna().any() or frame.date.duplicated().any() or not frame.date.is_monotonic_increasing:
        raise ValueError('Dates must be present, unique and chronological')
    if (frame.date!=frame.date.dt.normalize()).any() or (frame.date.dt.dayofweek>4).any():raise ValueError('Expected weekday observation dates')
    values=frame[list(TENORS)].to_numpy(dtype=float)
    if np.isinf(values).any() or not np.isfinite(frame.y10.to_numpy(dtype=float)).all():raise ValueError('Ten-year observations cannot be missing or infinite')
    if (np.abs(values[np.isfinite(values)])>30).any():raise ValueError('Check yield units: expected percent per annum')
    if frame.date.diff().dt.days.max()>7:raise ValueError('Unexpected observation gap')
    return frame.reset_index(drop=True)


def load_treasury():
    source=json.loads((DATA/'source.json').read_text(encoding='utf-8'))
    if hashlib.sha256((DATA/'treasury_yields.csv').read_bytes()).hexdigest()!=source['dataset_sha256']:
        raise ValueError('Treasury snapshot fingerprint mismatch')
    data=validate_yields(pd.read_csv(DATA/'treasury_yields.csv'))
    if len(data)!=source['rows']:raise ValueError('Treasury snapshot row count mismatch')
    return data,source,json.loads((DATA/'protocol.json').read_text(encoding='utf-8'))


def make_features(data):
    """Only contemporaneous and lagged observations. No targets enter this function."""
    data=validate_yields(data)
    x=data[list(TENORS)].copy()
    x['slope_10_2']=data.y10-data.y2
    x['slope_10_3m']=data.y10-data.y3m
    x['slope_5_2']=data.y5-data.y2
    x['curvature']=2*data.y5-data.y2-data.y10
    for lag in (1,5,20):
        for tenor in TENORS:x[f'{tenor}_change_{lag}']=data[tenor].diff(lag)
    x['y10_vol_20']=data.y10.diff().rolling(20,min_periods=20).std(ddof=1)
    x.index=data.date.dt.strftime('%Y-%m-%d');x.index.name='ROW_ID'
    return x


def make_samples(data,*,delay=2,horizon=20,publication_lag=1,start='1991-01-01'):
    """Labels start at the future decision-day measurement, never at stale feature time."""
    if not all(isinstance(v,int) for v in (delay,horizon,publication_lag)) or publication_lag<1 or delay<=publication_lag or horizon<1:
        raise ValueError('Need a positive horizon and a decision after a full publication session')
    data=validate_yields(data);dates=pd.DatetimeIndex(data.date)
    local=dates.tz_localize('America/New_York')
    # t+lag at end of day is a conservative availability assumption, not an archive.
    available=local+pd.Timedelta(hours=23,minutes=59,seconds=59)
    decision=local+pd.Timedelta(hours=9)
    rows=[]
    for i in range(20,len(data)-delay-horizon-publication_lag):
        if dates[i]<pd.Timestamp(start):continue
        d=i+delay;end=d+horizon
        change=float(data.y10.iloc[end]-data.y10.iloc[d])
        rows.append({'ROW_ID':str(dates[i].date()),'observation_date':str(dates[i].date()),
                     'max_feature_date':str(dates[i].date()),
                     'feature_available_at':available[i+publication_lag],
                     'decision_at':decision[d], 'decision_date':str(dates[d].date()),
                     'label_start_date':str(dates[d].date()),'label_end_date':str(dates[end].date()),
                     'label_available_at':available[end+publication_lag],
                     'target':int(change>0),'future_change_pp':change,
                     'past_direction':int(data.y10.iloc[i]-data.y10.iloc[i-20]>0),
                     'curve_inverted':bool(data.y10.iloc[i]-data.y3m.iloc[i]<0)})
    if not rows:raise ValueError('Insufficient data for complete timed labels')
    return validate_samples(pd.DataFrame(rows))


def validate_samples(samples):
    s=samples.copy()
    if s.empty or s.ROW_ID.duplicated().any() or s.ROW_ID.isna().any():raise ValueError('Sample IDs must be nonempty and unique')
    for key in ['feature_available_at','decision_at','label_available_at']:
        s[key]=pd.to_datetime(s[key],utc=True)
    if not s.decision_at.is_monotonic_increasing or s.decision_at.duplicated().any():raise ValueError('Decisions must be chronological and unique')
    if not (s.feature_available_at<s.decision_at).all():raise ValueError('Feature not yet available at decision')
    if not (s.label_available_at>s.decision_at).all():raise ValueError('Future label must not be available at its own decision')
    if not (s.label_end_date>s.label_start_date).all() or not (s.label_start_date==s.decision_date).all():raise ValueError('Future label endpoints are invalid')
    if not (s.max_feature_date<s.label_start_date).all():raise ValueError('Label start must follow the last feature observation')
    if not np.isin(s.target,[0,1]).all() or not np.isfinite(s.future_change_pp).all():raise ValueError('Labels must be complete binary signs')
    return s


def feature_rows(features,samples,variant):
    if variant not in VARIANTS or features.index.duplicated().any():raise ValueError('Invalid feature variant or duplicate key')
    if list(features.columns)!=VARIANTS['combined']:
        raise ValueError('Feature schema is whitelisted; keys, dates and labels are forbidden')
    if samples.ROW_ID.duplicated().any() or not samples.ROW_ID.isin(features.index).all():raise ValueError('Feature/sample key mismatch')
    return features.loc[samples.ROW_ID,VARIANTS[variant]].reset_index(drop=True)


def purged_training(samples,first_decision):
    boundary=pd.Timestamp(first_decision)
    if boundary.tzinfo is None:raise ValueError('Use a timezone-aware first decision')
    mask=(samples.decision_at<boundary)&(samples.label_available_at<boundary)
    return samples.loc[mask].reset_index(drop=True)
