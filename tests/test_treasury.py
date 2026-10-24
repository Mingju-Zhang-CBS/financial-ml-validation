from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from ml_validation.treasury_data import (TENORS,VARIANTS,load_treasury,validate_yields,
    make_features,make_samples,feature_rows,purged_training,validate_samples)
from ml_validation.treasury_research import (BASELINES,development_subset,development_selection,
    freeze_selection,confirmation_predictions,fit_predict,baseline_probability,losses,
    paired_intervals,diagnostics,probability_metrics)


def small_curve(n=1800):
    rng=np.random.default_rng(901)
    moves=rng.normal(0,.025,(n,5))
    values=4+np.cumsum(moves,axis=0)
    return pd.DataFrame({'date':pd.bdate_range('2000-01-03',periods=n),**{f:values[:,j] for j,f in enumerate(TENORS)}})


@pytest.fixture(scope='module')
def experiment():
    data=small_curve();samples=make_samples(data,start='2000-01-01');features=make_features(data)
    _,_,protocol=load_treasury();protocol=deepcopy(protocol)
    protocol['development_folds']=[['2002-01-01','2002-12-31'],['2003-01-01','2003-12-31'],['2004-01-01','2004-12-31']]
    protocol['confirmation']=['2005-01-01','2006-12-31']
    dev=development_subset(samples,protocol)
    ranking,_,audit=development_selection(dev,features,protocol)
    frozen=freeze_selection(ranking,dev,protocol)
    return data,samples,features,protocol,ranking,audit,frozen


def test_public_snapshot_and_rights_record():
    data,source,protocol=load_treasury()
    assert len(data)==9005 and source['partial_tenor_dates']==0
    assert source['policy_url']=='https://www.federalreserve.gov/disclaimer.htm'
    assert 'Public domain' in source['license']
    assert data.date.max()==pd.Timestamp('2025-12-31')
    assert protocol['target_horizon_sessions']==20


@pytest.mark.parametrize('problem',['duplicate','target_column','infinite','missing_target_rate','wrong_units','reverse'])
def test_yield_schema_rejects_unsafe_inputs(problem):
    data=small_curve(80)
    if problem=='duplicate':data=pd.concat([data,data.iloc[:1]],ignore_index=True)
    if problem=='target_column':data['target']=1
    if problem=='infinite':data.loc[30,'y2']=np.inf
    if problem=='missing_target_rate':data.loc[30,'y10']=np.nan
    if problem=='wrong_units':data.loc[30,'y5']=400
    if problem=='reverse':data=data.iloc[::-1]
    with pytest.raises(ValueError):validate_yields(data)


def test_missing_other_tenor_is_not_forward_or_back_filled():
    data=small_curve(100);data.loc[30,'y2']=np.nan
    x=make_features(data)
    assert pd.isna(x.iloc[30].y2) and pd.isna(x.iloc[31].y2_change_1)


def test_label_begins_after_decision_not_at_old_feature_quote():
    data=small_curve(100)
    s=make_samples(data,start='2000-01-01')
    row=s.iloc[0];i=20
    assert row.ROW_ID==str(data.date.iloc[i].date())
    assert row.label_start_date==str(data.date.iloc[i+2].date())
    assert row.label_end_date==str(data.date.iloc[i+22].date())
    assert row.future_change_pp==pytest.approx(data.y10.iloc[i+22]-data.y10.iloc[i+2])
    assert row.feature_available_at<row.decision_at<row.label_available_at
    assert row.decision_at.tz_convert('America/New_York').hour==9
    assert pd.Timestamp(s.label_end_date.max())<data.date.iloc[-1]
    with pytest.raises(ValueError):make_samples(data,delay=1)


def test_future_yield_mutation_preserves_past_features():
    data=small_curve(160);other=data.copy();other.loc[100:,'y10']+=1
    a=make_features(data);b=make_features(other)
    pd.testing.assert_frame_equal(a.iloc[:100],b.iloc[:100])


def test_availability_stress_keeps_aligned_decision_outcomes_and_older_features():
    data=small_curve(300)
    a=make_samples(data,start='2000-01-01',delay=2)
    b=make_samples(data,start='2000-01-01',delay=4)
    joined=a.merge(b,on='decision_date',suffixes=('_base','_stress'),validate='one_to_one')
    assert (joined.target_base==joined.target_stress).all()
    assert (joined.label_end_date_base==joined.label_end_date_stress).all()
    assert (joined.observation_date_stress<joined.observation_date_base).all()


def test_purging_uses_label_availability_not_only_decision_date():
    samples=make_samples(small_curve(300),start='2000-01-01')
    boundary=samples.decision_at.iloc[180]
    train=purged_training(samples,boundary)
    assert len(train)<180 and (train.label_available_at<boundary).all()
    assert not train.ROW_ID.isin(samples.loc[samples.label_available_at>=boundary,'ROW_ID']).any()
    with pytest.raises(ValueError):purged_training(samples,pd.Timestamp('2000-10-01'))


def test_development_labels_finish_before_confirmation_and_folds_are_purged(experiment):
    _,samples,features,p,_,audit,_=experiment
    dev=development_subset(samples,p);cut=pd.Timestamp(p['confirmation'][0]).tz_localize('America/New_York')
    assert (dev.label_available_at<cut).all()
    assert len(dev)<int((samples.decision_at<cut).sum())
    for row in audit.itertuples():
        assert pd.Timestamp(row.last_training_label_available)<pd.Timestamp(row.first_validation_decision)
        assert row.purged_boundary_rows>0
    exposed=samples[samples.decision_date<p['confirmation'][0]].reset_index(drop=True)
    with pytest.raises(ValueError):development_selection(exposed,features,p)


def test_keyed_feature_alignment_and_target_whitelist(experiment):
    _,samples,x,_,_,_,_=experiment
    s=samples.iloc[:60]
    a=feature_rows(x,s,'combined')
    b=feature_rows(x.sample(frac=1,random_state=7),s,'combined')
    pd.testing.assert_frame_equal(a,b)
    with pytest.raises(ValueError):feature_rows(x.assign(target=1),s,'combined')
    with pytest.raises(ValueError):feature_rows(x,pd.concat([s,s.iloc[:1]]),'curve')
    with pytest.raises(ValueError):feature_rows(x.iloc[100:],s,'curve')


@pytest.mark.parametrize('model',['logistic','lightgbm','catboost'])
def test_models_ignore_test_labels_and_preprocessing_fits_training_only(experiment,model):
    _,samples,x,_,_,_,_=experiment
    test=samples.iloc[300:330].copy();train=purged_training(samples,test.decision_at.iloc[0])
    p,_,est=fit_predict(train,test,x,model,'combined')
    changed=test.copy();changed['target']=1-changed.target
    q,_,_=fit_predict(train,changed,x,model,'combined')
    np.testing.assert_array_equal(p,q)
    imputer=est.named_steps['preprocess'].named_transformers_['numeric'].named_steps['impute']
    expected=feature_rows(x,train,'combined').median().to_numpy()
    np.testing.assert_allclose(imputer.statistics_,expected)
    assert np.isfinite(p).all() and ((p>0)&(p<1)).all()


def test_confirmation_target_changes_cannot_change_frozen_choice_or_predictions(experiment):
    _,samples,x,p,ranking,_,frozen=experiment
    other=samples.copy();mask=other.decision_date>=p['confirmation'][0];other.loc[mask,'target']=1-other.loc[mask,'target']
    changed=freeze_selection(ranking,development_subset(other,p),p)
    assert frozen==changed
    a,_=confirmation_predictions(samples,x,frozen,p);b,_=confirmation_predictions(other,x,frozen,p)
    for name in [*BASELINES,*frozen['family_representatives'].values()]:np.testing.assert_array_equal(a[name],b[name])
    assert freeze_selection(ranking.sample(frac=1,random_state=8),development_subset(samples,p),p)==frozen


def test_baselines_fit_training_labels_only(experiment):
    _,s,_,_,_,_,_=experiment
    train=s.iloc[:100];test=s.iloc[200:230].copy()
    for name in BASELINES:
        a=baseline_probability(train,test,name);test['target']=1-test.target
        np.testing.assert_array_equal(a,baseline_probability(train,test,name))
    assert baseline_probability(train,test,'training_climatology')[0]==pytest.approx((train.target.sum()+1)/(len(train)+2))


def test_moving_blocks_preserve_paired_rows_and_match_manual_resampling():
    n=80;y=np.arange(n)%2;a=np.linspace(.2,.8,n);b=np.full(n,.55)
    frame=pd.DataFrame({'ROW_ID':np.arange(n).astype(str),'decision_date':pd.bdate_range('2020-01-01',periods=n).strftime('%Y-%m-%d'),'target':y,'a':a,'b':b})
    result=paired_intervals(frame,'a','b',block=9,replications=100,seed=10)
    delta=losses(y,b)-losses(y,a);rng=np.random.default_rng(10);draw=[]
    for _ in range(100):
        starts=rng.integers(0,n-9+1,int(np.ceil(n/9)))
        values=np.concatenate([delta[i:i+9] for i in starts])[:n]
        draw.append(values.mean())
    assert result['mean_log_loss_gain']==pytest.approx(delta.mean())
    assert result['gain_ci95_low']==pytest.approx(np.quantile(draw,.025))
    assert result['gain_ci95_high']==pytest.approx(np.quantile(draw,.975))
    same=paired_intervals(frame,'a','a',block=9,replications=100)
    assert same['gain_ci95_low']==same['gain_ci95_high']==0
    with pytest.raises(ValueError):paired_intervals(pd.concat([frame,frame.iloc[:1]]),'a','b',block=9,replications=100)


def test_reliability_bins_account_for_every_prediction_and_auc_can_be_undefined():
    y=np.array([0,1,0,1,0]);p=np.array([0.,.2,.4,.8,1.])
    s=pd.DataFrame({'target':y,'a':p,'decision_date':['2020-01-02']*5,'curve_inverted':[False]*5})
    d=diagnostics(s,['a'])
    assert sum(row['rows'] for row in d['calibration'])==5
    assert probability_metrics(np.ones(5),np.ones(5)*.6)['roc_auc'] is None


def test_invalid_probabilities_and_clock_order_fail_closed(experiment):
    _,samples,_,_,_,_,_=experiment
    with pytest.raises(ValueError):losses([0,1],[np.nan,.5])
    with pytest.raises(ValueError):losses([0,1],[.5])
    bad=samples.iloc[:10].copy();bad['feature_available_at']=bad.decision_at
    with pytest.raises(ValueError):validate_samples(bad)
