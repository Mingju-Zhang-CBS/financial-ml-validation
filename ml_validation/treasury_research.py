"""Expanding historical comparisons; selection is frozen before confirmation scoring."""
from __future__ import annotations
import hashlib,json,time
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss,roc_auc_score
from .research import pipeline,scores
from .treasury_data import feature_rows,purged_training

MODELS=('logistic','lightgbm','catboost')
BASELINES=('training_climatology','training_fitted_direction_persistence')


def development_subset(samples,protocol):
    freeze_at=pd.Timestamp(protocol['confirmation'][0]).tz_localize('America/New_York')
    return samples.loc[(samples.decision_at<freeze_at)&(samples.label_available_at<freeze_at)].reset_index(drop=True)


def probability_metrics(y,p):
    m=scores(y,p)
    m['brier']=float(brier_score_loss(y,p))
    m['roc_auc']=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None
    return m


def losses(y,p):
    y=np.asarray(y);p=np.asarray(p,dtype=float)
    scores(y,p)
    p=np.clip(p,1e-6,1-1e-6)
    return -(y*np.log(p)+(1-y)*np.log1p(-p))


def baseline_probability(train,test,name):
    if name not in BASELINES:raise ValueError('Unknown baseline')
    y=train.target.to_numpy()
    if len(y)==0 or not np.isin(y,[0,1]).all():raise ValueError('Invalid baseline training labels')
    overall=(y.sum()+1)/(len(y)+2)
    if name=='training_climatology':return np.full(len(test),overall)
    p=np.full(len(test),overall)
    for state in (0,1):
        ys=y[train.past_direction.to_numpy()==state]
        p[test.past_direction.to_numpy()==state]=(ys.sum()+1)/(len(ys)+2)
    return p


def fit_predict(train,test,features,model,variant):
    if train.empty or len(train.target.unique())!=2:raise ValueError('Training must include both classes')
    if not (train.label_available_at<test.decision_at.iloc[0]).all():raise ValueError('Training label crossed prediction boundary')
    x=feature_rows(features,train,variant);v=feature_rows(features,test,variant)
    est=pipeline(x,model)
    start=time.perf_counter();est.fit(x,train.target.to_numpy());seconds=time.perf_counter()-start
    p=np.clip(est.predict_proba(v)[:,1],1e-6,1-1e-6)
    return p,seconds,est


def development_selection(samples,features,protocol):
    # Caller supplies only pre-confirmation rows; confirmation labels are inaccessible here.
    end=protocol['confirmation'][0]
    if (samples.decision_date>=end).any():raise ValueError('Confirmation rows cannot enter development selection')
    cutoff=pd.Timestamp(end).tz_localize('America/New_York')
    if not (samples.label_available_at<cutoff).all():raise ValueError('Development outcomes were not available before confirmation')
    parts=[];audit=[];fits={}
    for fold,bounds in enumerate(protocol['development_folds']):
        test=samples.loc[samples.decision_date.between(*bounds)].reset_index(drop=True)
        if test.empty:raise ValueError('Empty expanding validation fold')
        train=purged_training(samples,test.decision_at.iloc[0])
        if train.empty:raise ValueError('Empty purged training set')
        audit.append({'fold':fold,'training_rows':len(train),'validation_rows':len(test),
                      'first_validation_decision':test.decision_at.iloc[0].isoformat(),
                      'last_training_decision':train.decision_at.iloc[-1].isoformat(),
                      'last_training_label_available':train.label_available_at.max().isoformat(),
                      'purged_boundary_rows':int(((samples.decision_at<test.decision_at.iloc[0])&~samples.ROW_ID.isin(train.ROW_ID)).sum())})
        for name in BASELINES:
            p=baseline_probability(train,test,name)
            parts.append(pd.DataFrame({'ROW_ID':test.ROW_ID,'fold':fold,'candidate':name,'target':test.target,'probability':p}))
        for variant in protocol['feature_variants']:
            for model in MODELS:
                name=f'{model}/{variant}'
                p,seconds,_=fit_predict(train,test,features,model,variant)
                fits[name]=fits.get(name,0)+seconds
                parts.append(pd.DataFrame({'ROW_ID':test.ROW_ID,'fold':fold,'candidate':name,'target':test.target,'probability':p}))
    predictions=pd.concat(parts,ignore_index=True)
    ranking=[]
    for name,group in predictions.groupby('candidate',sort=True):
        ranking.append({'candidate':name,**probability_metrics(group.target.to_numpy(),group.probability.to_numpy()),'development_fit_seconds':fits.get(name,0.)})
    ranking=pd.DataFrame(ranking).sort_values(['log_loss','candidate'],kind='stable').reset_index(drop=True)
    return ranking,predictions,pd.DataFrame(audit)


def freeze_selection(ranking,development,protocol):
    ranking=ranking.sort_values(['log_loss','candidate'],kind='stable').reset_index(drop=True)
    candidates=ranking[~ranking.candidate.isin(BASELINES)]
    expected={f'{m}/{v}' for m in MODELS for v in protocol['feature_variants']}
    if len(candidates)!=9 or set(candidates.candidate)!=expected or ranking.candidate.duplicated().any() or not set(BASELINES).issubset(set(ranking.candidate)):
        raise ValueError('Expected nine unique predeclared model configurations and both baselines')
    family={model:str(candidates[candidates.candidate.str.startswith(model+'/')].iloc[0].candidate) for model in MODELS}
    baseline=str(ranking[ranking.candidate.isin(BASELINES)].iloc[0].candidate)
    training_digest=hashlib.sha256(development[['ROW_ID','target','label_available_at']].to_csv(index=False).encode()).hexdigest()
    frozen={'global_champion':str(candidates.iloc[0].candidate),'family_representatives':family,
            'baseline':baseline,'selection_rule':protocol['selection'],'threshold':.5,
            'overall_development_leader':str(ranking.iloc[0].candidate),
            'champion_scope':'Best of the nine classifiers only; baselines remain separately ranked. No classifier promotion is implied when a baseline has lower development loss.',
            'development_label_fingerprint':training_digest,'protocol_version':protocol['version'],
            'confirmation_policy':'Evaluate all three frozen family representatives; never choose a new champion from confirmation.'}
    frozen['selection_sha256']=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
    return frozen


def confirmation_predictions(samples,features,frozen,protocol):
    confirm=samples.loc[samples.decision_date.between(*protocol['confirmation'])].reset_index(drop=True)
    if confirm.empty:raise ValueError('Empty confirmation')
    train=purged_training(samples,confirm.decision_at.iloc[0])
    result=confirm.copy();times={}
    for name in BASELINES:result[name]=baseline_probability(train,confirm,name)
    for model,name in frozen['family_representatives'].items():
        _,variant=name.split('/')
        result[name],times[name],_=fit_predict(train,confirm,features,model,variant)
    audit={'training_rows':len(train),'confirmation_rows':len(confirm),
           'last_training_label_available':train.label_available_at.max().isoformat(),
           'first_confirmation_decision':confirm.decision_at.iloc[0].isoformat(),
           'first_confirmation_date':confirm.decision_date.iloc[0],'last_confirmation_date':confirm.decision_date.iloc[-1],
           'last_confirmation_label_available':confirm.label_available_at.max().isoformat(),
           'fit_seconds':times}
    return result,audit


def paired_intervals(predictions,candidate,baseline,*,block=63,replications=2000,seed=20261024):
    if predictions.ROW_ID.duplicated().any() or not predictions.decision_date.is_monotonic_increasing:
        raise ValueError('Paired predictions must have unique ordered decision keys')
    y=predictions.target.to_numpy();a=predictions[candidate].to_numpy();b=predictions[baseline].to_numpy()
    gain=losses(y,b)-losses(y,a)
    accuracy=((a>=.5)==y).astype(float)-((b>=.5)==y).astype(float)
    delta=np.column_stack([gain,accuracy]);n=len(delta)
    if not isinstance(block,int) or not 1<=block<=n or replications<100:raise ValueError('Invalid block interval configuration')
    rng=np.random.default_rng(seed)
    blocks=np.lib.stride_tricks.sliding_window_view(delta,block,axis=0)
    draws=np.empty((replications,2))
    for i in range(replications):
        starts=rng.integers(0,len(blocks),int(np.ceil(n/block)))
        resample=blocks[starts].transpose(0,2,1).reshape(-1,2)[:n]
        draws[i]=resample.mean(axis=0)
    low,high=np.quantile(draws,[.025,.975],axis=0)
    return {'candidate':candidate,'baseline':baseline,'observations':n,'block_sessions':block,
            'replications':replications,'seed':seed,'mean_log_loss_gain':float(gain.mean()),
            'gain_ci95_low':float(low[0]),'gain_ci95_high':float(high[0]),
            'gain_interval_contains_zero':bool(low[0]<=0<=high[0]),
            'accuracy_difference':float(accuracy.mean()),'accuracy_ci95_low':float(low[1]),'accuracy_ci95_high':float(high[1])}


def diagnostics(predictions,names):
    yearly=[];groups=[];calibration=[];sparse=[]
    for name in names:
        for year,g in predictions.groupby(predictions.decision_date.str[:4]):
            yearly.append({'year':year,'candidate':name,'rows':len(g),**probability_metrics(g.target.to_numpy(),g[name].to_numpy())})
        for inverted,g in predictions.groupby('curve_inverted'):
            groups.append({'asof_curve':'inverted' if inverted else 'non_inverted','candidate':name,'rows':len(g),**probability_metrics(g.target.to_numpy(),g[name].to_numpy())})
        bins=np.minimum((predictions[name].to_numpy()*5).astype(int),4)
        for bucket in range(5):
            g=predictions.loc[bins==bucket]
            calibration.append({'candidate':name,'bin_low':bucket/5,'bin_high':(bucket+1)/5,'rows':len(g),
                                'mean_probability':float(g[name].mean()) if len(g) else None,
                                'observed_positive_rate':float(g.target.mean()) if len(g) else None})
        g=predictions.iloc[::20]
        sparse.append({'candidate':name,'rows':len(g),**probability_metrics(g.target.to_numpy(),g[name].to_numpy())})
    return {'yearly':yearly,'groups':groups,'calibration':calibration,'every_twentieth':sparse}
