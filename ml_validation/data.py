"""Strict keyed targets, an explicit feature whitelist and synthetic panel data."""
from __future__ import annotations
import numpy as np
import pandas as pd

HISTORY=tuple(f"RET_{i}" for i in range(1,21))
VOLUME=tuple(f"SIGNED_VOLUME_{i}" for i in range(1,21))
NUMERIC=HISTORY+VOLUME+("MEDIAN_DAILY_TURNOVER",)
REQUIRED=("ROW_ID","TS","ALLOCATION","GROUP")+NUMERIC

def validate_features(x):
    if {"TARGET","target"}&set(x):raise ValueError("Target column is forbidden in X")
    if not set(REQUIRED).issubset(x):raise ValueError(f"Missing features: {sorted(set(REQUIRED)-set(x))}")
    d=x[list(REQUIRED)].copy()
    for col in ["ROW_ID","TS","ALLOCATION","GROUP"]:
        if d[col].isna().any():raise ValueError(f"Missing {col}")
        d[col]=d[col].astype(str)
    if d.ROW_ID.duplicated().any():raise ValueError("ROW_ID must be unique")
    if d.duplicated(["TS","ALLOCATION"]).any():raise ValueError("Duplicate date/allocation pair")
    d[list(NUMERIC)]=d[list(NUMERIC)].apply(pd.to_numeric,errors="raise")
    if np.isinf(d[list(NUMERIC)].to_numpy()).any():raise ValueError("Infinite feature values rejected")
    return d.sort_values("ROW_ID").reset_index(drop=True)

def align_training(x,y):
    d=validate_features(x)
    target_names=[c for c in ("TARGET","target") if c in y]
    if "ROW_ID" not in y or len(target_names)!=1:raise ValueError("y needs ROW_ID and exactly one TARGET/target column")
    targets=y[["ROW_ID",target_names[0]]].rename(columns={target_names[0]:"TARGET"}).copy()
    if targets.ROW_ID.isna().any():raise ValueError("Missing target ROW_ID")
    targets["ROW_ID"]=targets.ROW_ID.astype(str)
    if targets.ROW_ID.duplicated().any() or set(targets.ROW_ID)!=set(d.ROW_ID):
        raise ValueError("X/y ROW_ID sets must match exactly and be unique")
    targets["TARGET"]=pd.to_numeric(targets.TARGET,errors="raise")
    if not np.isfinite(targets.TARGET).all():raise ValueError("Finite complete targets required")
    values=targets.set_index("ROW_ID").TARGET.reindex(d.ROW_ID).to_numpy()
    binary=(values>0).astype(int)
    if len(np.unique(binary))!=2:raise ValueError("Both target signs required")
    return d,binary

def feature_table(frame,variant):
    if variant not in {"raw","summary","identity"}:raise ValueError("Unknown feature ablation")
    out=frame[list(NUMERIC)+["GROUP"]].copy()
    if variant in {"summary","identity"}:
        for n in [5,10,20]:
            cols=[f"RET_{i}" for i in range(1,n+1)]
            out[f"return_mean_{n}"]=frame[cols].mean(axis=1)
            out[f"return_std_{n}"]=frame[cols].std(axis=1,ddof=0)
        out["volume_mean"]=frame[list(VOLUME)].mean(axis=1)
        out["return_missing_fraction"]=frame[list(HISTORY)].isna().mean(axis=1)
        out["volume_missing_fraction"]=frame[list(VOLUME)].isna().mean(axis=1)
    if variant=="identity":out["ALLOCATION"]=frame.ALLOCATION
    # No label, row key, date suffix, same-date labels or externally matched data.
    return out

def synthetic_panel(dates=360,allocations=12,seed=2026):
    if dates<180 or allocations<2:raise ValueError("Synthetic demo needs >=180 dates and >=2 allocations")
    rng=np.random.default_rng(seed);history=20
    returns=np.zeros((dates+history+1,allocations));volume=rng.normal(size=returns.shape)
    common=rng.normal(0,.006,len(returns))
    for t in range(1,len(returns)):
        returns[t]=.12*returns[t-1]+common[t]+rng.normal(0,.012,allocations)
    labels=rng.permutation(np.arange(dates));xs=[];ys=[];order=[]
    for t in range(dates):
        ts=f"DATE_{labels[t]:04d}";order.append(ts)
        for a in range(allocations):
            key=f"R{t*allocations+a:06d}"
            row={"ROW_ID":key,"TS":ts,"ALLOCATION":f"A{a:02d}","GROUP":f"G{a%3}",
                 "MEDIAN_DAILY_TURNOVER":float(.2+.02*a)}
            row.update({f"RET_{i}":float(returns[t+history-i+1,a]) for i in range(1,21)})
            row.update({f"SIGNED_VOLUME_{i}":float(volume[t+history-i+1,a]) for i in range(1,21)})
            xs.append(row);ys.append({"ROW_ID":key,"TARGET":float(returns[t+history+1,a])})
    x=pd.DataFrame(xs);y=pd.DataFrame(ys)
    mask=rng.random((len(x),40))<.015
    x.loc[:,list(HISTORY+VOLUME)]=x[list(HISTORY+VOLUME)].mask(mask)
    return x,y,order
