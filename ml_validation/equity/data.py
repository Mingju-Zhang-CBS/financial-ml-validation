import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import pandas as pd
from .core import monthly_universe, prepare_panel


def build(raw, factor_archive, output):
    raw, factor_archive, output = Path(raw), Path(factor_archive), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    required = ['permno', 'permco', 'dlycaldt', 'primaryexch', 'conditionaltype', 'tradingstatusflg', 'usincflg', 'issuertype', 'securitytype', 'securitysubtype', 'sharetype', 'dlycap', 'dlyret', 'dlyretx', 'dlyprc', 'dlyvol', 'dlyretmissflg', 'dlyretdurflg']
    header = pd.read_csv(raw, nrows=0).columns.str.lower()
    if not set(required).issubset(header):
        raise ValueError('Missing CIZ input fields: ' + ', '.join(sorted(set(required) - set(header))))
    snapshots, all_dates, total = [], set(), 0
    for chunk in pd.read_csv(raw, chunksize=500000, low_memory=False, keep_default_na=False):
        chunk.columns = chunk.columns.str.lower()
        chunk = chunk[[column for column in required if column not in ['dlyret', 'dlyretx', 'dlyprc', 'dlyvol', 'dlyretmissflg', 'dlyretdurflg']]].rename(columns={'dlycaldt': 'date', 'dlycap': 'cap'})
        chunk.date = pd.to_datetime(chunk.date)
        chunk.cap = pd.to_numeric(chunk.cap, errors='coerce') * 1000
        total += len(chunk)
        all_dates.update(chunk.date.unique())
        chunk['month'] = chunk.date.dt.to_period('M')
        latest = chunk.groupby(['permno', 'month']).date.transform('max')
        snapshots.append(chunk[chunk.date.eq(latest)])
        print(json.dumps({'first_pass_rows': total}), flush=True)
    snapshots = pd.concat(snapshots, ignore_index=True)
    latest = snapshots.groupby(['permno', 'month']).date.transform('max')
    snapshots = snapshots[snapshots.date.eq(latest)].drop_duplicates()
    calendar = pd.DatetimeIndex(sorted(all_dates), name='date')
    month_ends = pd.Series(calendar, index=calendar).groupby(calendar.to_period('M')).max()
    snapshots = snapshots[snapshots.date.eq(snapshots.month.map(month_ends))].drop(columns='month')
    conflicting_identity = snapshots.duplicated(['permno', 'date'], keep=False)
    snapshots[conflicting_identity].to_csv(output / 'private_identity_conflicts.csv', index=False)
    universe = monthly_universe(snapshots[~conflicting_identity])
    identifiers = set(universe.permno)
    universe.to_parquet(output / 'equity_universe.parquet', index=False)
    frames = []
    for chunk in pd.read_csv(raw, chunksize=500000, low_memory=False, keep_default_na=False):
        chunk.columns = chunk.columns.str.lower()
        frames.append(chunk[chunk.permno.isin(identifiers)].copy())
    daily = pd.concat(frames, ignore_index=True).rename(columns={'dlycaldt': 'date'})
    daily.date = pd.to_datetime(daily.date)
    for column in ['dlyret', 'dlyretx', 'dlycap', 'dlyprc', 'dlyvol']:
        daily[column] = pd.to_numeric(daily[column], errors='coerce')
    exact_duplicates = int(daily.duplicated().sum())
    daily = daily.drop_duplicates()
    conflicting = daily.duplicated(['permno', 'date'], keep=False)
    daily[conflicting].to_parquet(output / 'private_return_conflicts.parquet', index=False)
    daily = daily[~conflicting].copy()
    daily['cap'] = daily.dlycap * 1000
    single_period = daily.dlyretdurflg.isin(['D1', 'D2', 'D3', 'D4', 'DU'])
    daily['ret'] = daily.dlyret.where(single_period)
    daily['recorded_total_return'] = daily.dlyret
    daily['dollar_volume'] = daily.dlyprc.abs() * daily.dlyvol
    daily.groupby(['dlyretmissflg', 'dlyretdurflg'], dropna=False).size().to_csv(output / 'return_flag_audit.csv')
    daily.to_parquet(output / 'equity_daily.parquet', index=False)
    archive = zipfile.ZipFile(factor_archive)
    lines = archive.read(archive.namelist()[0]).decode('utf-8-sig').splitlines()
    header = next(line for line in lines if 'Mkt-RF' in line)
    observations = [line for line in lines if len(line.split(',')[0].strip()) == 8 and line.split(',')[0].strip().isdigit()]
    factors = pd.read_csv(io.StringIO('\n'.join([header] + observations)))
    factors.columns = ['date', 'mktrf', 'smb', 'hml', 'rmw', 'cma', 'rf']
    factors.date = pd.to_datetime(factors.date.astype(str), format='%Y%m%d')
    factors = factors.set_index('date').reindex(calendar) / 100
    if factors.isna().any().any():
        raise ValueError('Factor/calendar mismatch; no forward or backward fill permitted')
    panel = prepare_panel(daily, factors, universe)
    panel = panel[panel.date.between('2012-01-01', '2025-12-31')]
    panel.to_parquet(output / 'equity_panel.parquet', index=False)
    record = dict(data_kind='licensed-real-private', raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(), factor_sha256=hashlib.sha256(factor_archive.read_bytes()).hexdigest(),
                  raw_rows=total, exact_duplicates_collapsed=exact_duplicates, conflicting_return_rows_quarantined=int(conflicting.sum()),
                  conflicting_identity_rows_quarantined=int(conflicting_identity.sum()), non_single_period_returns=int((~single_period & daily.dlyret.notna()).sum()),
                  universe_security_ids=len(identifiers), panel_rows=len(panel), start=str(panel.date.min()), end=str(panel.date.max()),
                  total_return_convention='CIZ total return; no second dividend or delisting return added',
                  source_units='capitalization in thousands converted to dollars; price times raw shares is dollar-volume proxy',
                  vintage='current historical download; factor publication/revision archive unavailable',
                  rights='licensed source and derived outputs private pending applicable publication rights')
    (output / 'equity_source.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description='Build a private equity panel from a user-acquired CIZ extract')
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--factors', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.raw, args.factors, args.output)


if __name__ == '__main__':
    main()
