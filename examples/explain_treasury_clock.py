"""A small clock example: stale features cannot supply either future label endpoint."""
import numpy as np
import pandas as pd
from ml_validation.treasury_data import TENORS,make_samples,purged_training

dates=pd.bdate_range('2020-01-02',periods=100)
values=2+np.arange(100)*.01
data=pd.DataFrame({'date':dates,**{name:values.copy() for name in TENORS}})
samples=make_samples(data,start='2020-01-01')
row=samples.iloc[0]
assert row.label_start_date==str(dates[22].date())
assert row.label_end_date==str(dates[42].date())
assert abs(row.future_change_pp-.2)<1e-12
assert row.feature_available_at<row.decision_at<row.label_available_at
boundary=samples.decision_at.iloc[40]
eligible=purged_training(samples,boundary)
assert (eligible.label_available_at<boundary).all()
print(f"Observe {row.observation_date}; decide {row.decision_at.tz_convert('America/New_York').isoformat()}")
print(f"Target: {row.label_start_date} to {row.label_end_date}; change={row.future_change_pp:.2f} percentage points, class={row.target}")
print(f"At the later boundary, {len(eligible)} of 40 earlier decisions have fully available training labels.")
