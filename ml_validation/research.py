"""Predeclared grouped/forward comparisons with a single confirmation stage."""
from __future__ import annotations
import hashlib
import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,log_loss
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from .data import feature_table


def split_protocol(frame,*,date_order=None,gap=20,folds=3,seed=42):
    if not isinstance(gap,(int,np.integer)) or gap<0 or not isinstance(folds,(int,np.integer)) or folds<2:
        raise ValueError("Need an integer gap >=0 and integer folds >=2")
    groups=frame.TS.to_numpy();unique=sorted(set(groups))
    if len(unique)<max(30,(gap+5)*(folds+1)):raise ValueError("Insufficient distinct dates for protocol")
    if date_order is not None:
        if len(date_order)!=len(set(date_order)) or set(date_order)!=set(unique):
            raise ValueError("Declared date order must contain each observed date exactly once")
        order=np.array(date_order);split=int(.8*len(order));dev=order[:split-gap];confirm=order[split:]
        inner=[]
        for train,valid in TimeSeriesSplit(n_splits=folds,gap=gap).split(dev):
            inner.append((np.flatnonzero(np.isin(groups,dev[train])),np.flatnonzero(np.isin(groups,dev[valid]))))
        label="known synthetic chronology; 20-date purge" if gap==20 else f"declared chronology; {gap}-date purge"
    else:
        # Dates are opaque. Hash order is stable to row permutation, never a calendar.
        order=np.array(sorted(unique,key=lambda s:hashlib.sha256(f"{seed}:{s}".encode()).hexdigest()))
        split=int(.8*len(order));dev=order[:split];confirm=order[split:];inner=[]
        chunks=np.array_split(dev,folds)
        for chunk in chunks:
            inner.append((np.flatnonzero(np.isin(groups,dev[~np.isin(dev,chunk)])),np.flatnonzero(np.isin(groups,chunk))))
        label="opaque date groups; temporal order and overlap remain unknown"
    train_idx=np.flatnonzero(np.isin(groups,dev));confirmation=np.flatnonzero(np.isin(groups,confirm))
    for train,valid in inner:
        if set(groups[train])&set(groups[valid]):raise AssertionError("Date overlap")
        if set(groups[valid])&set(groups[confirmation]):raise AssertionError("Confirmation contamination")
    return inner,train_idx,confirmation,label


def pipeline(table,name):
    categorical=[c for c in ["GROUP","ALLOCATION"] if c in table]
    numeric=[c for c in table if c not in categorical]
    preprocessing=ColumnTransformer([
        ("numeric",Pipeline([("impute",SimpleImputer(strategy="median",keep_empty_features=True)),("scale",StandardScaler())]),numeric),
        ("category",OneHotEncoder(handle_unknown="ignore",sparse_output=False),categorical)],sparse_threshold=0)
    models={"logistic":LogisticRegression(C=1.,max_iter=600,random_state=42),
            "lightgbm":LGBMClassifier(n_estimators=80,num_leaves=15,max_depth=4,learning_rate=.05,
                                       n_jobs=2,random_state=42,verbosity=-1,deterministic=True,force_col_wise=True),
            "catboost":CatBoostClassifier(iterations=80,depth=4,learning_rate=.05,loss_function="Logloss",
                                            thread_count=2,random_seed=42,verbose=False,allow_writing_files=False)}
    if name not in models:raise ValueError("Unknown model")
    return Pipeline([("preprocess",preprocessing),("model",models[name])])


def scores(y,probability):
    y=np.asarray(y);p=np.asarray(probability,dtype=float)
    if y.ndim!=1 or len(y)==0 or y.shape!=p.shape or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError("Probabilities must match labels and lie in [0,1]")
    pred=(p>=.5).astype(int)
    return {"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),
            "log_loss":float(log_loss(y,p,labels=[0,1]))}


def date_bootstrap(y,candidate,baseline,groups,*,repetitions=2000,seed=42):
    """Paired date-cluster bootstrap, weighted by row counts within each draw."""
    y=np.asarray(y);groups=np.asarray(groups)
    candidate=np.asarray(candidate,dtype=float);baseline=np.asarray(baseline,dtype=float)
    if y.ndim!=1 or not (y.shape==groups.shape==candidate.shape==baseline.shape):
        raise ValueError("One-dimensional labels, groups and predictions must align exactly")
    scores(y,candidate);scores(y,baseline)
    if pd.isna(groups).any():raise ValueError("Missing date groups are forbidden")
    if not isinstance(repetitions,(int,np.integer)) or repetitions<100:
        raise ValueError("Need >=100 integer bootstrap repetitions")
    c=(candidate>=.5)==y;b=(baseline>=.5)==y
    tab=pd.DataFrame({"date":groups,"delta":c.astype(int)-b.astype(int)}).groupby("date").agg(total=("delta","sum"),n=("delta","size"))
    if len(tab)<2 or repetitions<100:raise ValueError("Need >=2 dates and >=100 repetitions")
    rng=np.random.default_rng(seed);deltas=np.empty(repetitions);total=tab.total.to_numpy();n=tab.n.to_numpy()
    for i in range(repetitions):
        sample=rng.integers(0,len(tab),len(tab));deltas[i]=total[sample].sum()/n[sample].sum()
    return {"accuracy_difference":float((c.astype(int)-b.astype(int)).mean()),
            "ci_low":float(np.quantile(deltas,.025)),"ci_high":float(np.quantile(deltas,.975)),
            "repetitions":repetitions,"sampling_unit":"date cluster"}


def evaluate(frame,y,*,date_order=None):
    splits,development,confirmation,label=split_protocol(frame,date_order=date_order)
    tables={v:feature_table(frame,v) for v in ["raw","summary","identity"]}
    rows=[];oof={};candidate_fit={}
    for variant in tables:
        x=tables[variant]
        for model in ["logistic","lightgbm","catboost"]:
            key=f"{model}/{variant}";parts=[];fit_seconds=0.
            for fold,(train,valid) in enumerate(splits):
                if len(np.unique(y[train]))<2:raise ValueError("A development training fold has only one class")
                est=pipeline(x,model);start=time.perf_counter();est.fit(x.iloc[train],y[train]);fit_seconds+=time.perf_counter()-start
                probability=est.predict_proba(x.iloc[valid])[:,1]
                base=np.full(len(valid),y[train].mean())
                rows.append({"candidate":key,"fold":fold,"rows":len(valid),**scores(y[valid],probability)})
                parts.append(pd.DataFrame({"row":valid,"probability":probability,"baseline":base}))
            oof[key]=pd.concat(parts).sort_values("row");candidate_fit[key]=fit_seconds
    table=pd.DataFrame(rows)
    ranking=[]
    for key,parts in oof.items():
        ranking.append({"candidate":key,**scores(y[parts.row.to_numpy()],parts.probability.to_numpy()),"development_fit_seconds":candidate_fit[key]})
    ranking=pd.DataFrame(ranking).sort_values(["accuracy","candidate"],ascending=[False,True]).reset_index(drop=True)
    # Freeze before reading confirmation labels for evaluation. No threshold tuning.
    winner=str(ranking.candidate.iloc[0]);model,variant=winner.split("/")
    freeze={"selected":winner,"selection_rule":"maximum row-weighted development accuracy; ties by candidate name",
            "threshold":.5,"configuration_count":9,"confirmation_role":"one evaluation; never used for model choice"}
    freeze["selection_sha256"]=hashlib.sha256(str(sorted(freeze.items())).encode()).hexdigest()
    final=pipeline(tables[variant],model);start=time.perf_counter();final.fit(tables[variant].iloc[development],y[development]);final_seconds=time.perf_counter()-start
    p=final.predict_proba(tables[variant].iloc[confirmation])[:,1];base=np.full(len(confirmation),y[development].mean())
    bootstrap=date_bootstrap(y[confirmation],p,base,frame.TS.iloc[confirmation])
    slices=[]
    cframe=frame.iloc[confirmation].copy();cframe["truth"]=y[confirmation];cframe["probability"]=p
    for group,part in cframe.groupby("GROUP"):
        slices.append({"group":str(group),"rows":len(part),**scores(part.truth.to_numpy(),part.probability.to_numpy())})
    selected_oof=oof[winner]
    return {"protocol":label,"selected":winner,"freeze":freeze,"development_rows":len(development),
            "confirmation_rows":len(confirmation),"development_dates":int(frame.TS.iloc[development].nunique()),
            "confirmation_dates":int(frame.TS.iloc[confirmation].nunique()),
            "confirmation":scores(y[confirmation],p),"baseline":scores(y[confirmation],base),"paired_bootstrap":bootstrap,
            "confirmation_groups":slices,"final_fit_seconds":final_seconds,
            "development_baseline":scores(y[selected_oof.row.to_numpy()],selected_oof.baseline.to_numpy())},table,ranking
