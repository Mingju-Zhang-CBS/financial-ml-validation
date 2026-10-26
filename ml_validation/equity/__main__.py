import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .core import FACTORS, prepare_panel, monthly_universe
from .study import run


def main():
    parser = argparse.ArgumentParser(description='Equity research: licensed panel or clearly labeled synthetic fixture')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--panel', type=Path)
    modes.add_argument('--fixture', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--daily', type=Path)
    args = parser.parse_args()
    panel_path = args.panel
    if args.fixture:
        rng = np.random.default_rng(20261026)
        dates = pd.bdate_range('2010-01-01', '2025-12-31', name='date')
        factors = pd.DataFrame(rng.normal(0, .004, (len(dates), 5)), index=dates, columns=FACTORS).assign(rf=.0001)
        assets = []
        for identifier in range(24):
            returns = factors.mktrf.to_numpy() + rng.normal(0, .01, len(dates))
            assets.append(pd.DataFrame(dict(date=dates, permno=identifier, permco=identifier, ret=returns,
                                            cap=1e6*np.exp(np.cumsum(returns)), dollar_volume=rng.lognormal(16, .3, len(dates)),
                                            securitytype='EQTY', securitysubtype='COM', sharetype='NS', usincflg='Y',
                                            issuertype='CORP', primaryexch='N', conditionaltype='RW', tradingstatusflg='A')))
        daily = pd.concat(assets, ignore_index=True)
        universe = monthly_universe(daily)
        panel = prepare_panel(daily, factors, universe)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        panel_path = args.output.parent / (args.output.name + '-synthetic-input.parquet')
        panel.to_parquet(panel_path, index=False)
        args.daily = args.output.parent / (args.output.name + '-synthetic-daily.parquet')
        daily[['permno', 'date', 'ret']].to_parquet(args.daily, index=False)
    result = run(panel_path, args.output, fixture=args.fixture, daily_path=args.daily)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
