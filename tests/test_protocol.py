import numpy as np
import pandas as pd
import pytest
from ml_validation.data import align_training,feature_table,synthetic_panel,validate_features
from ml_validation.research import split_protocol,pipeline,date_bootstrap,scores

@pytest.fixture(scope="module")
def panel():
    x,y,order=synthetic_panel(dates=240,allocations=3)
    frame,target=align_training(x,y)
    return x,y,frame,target,order

def test_keyed_alignment_is_invariant_to_input_order(panel):
    x,y,frame,target,_=panel
    other,labels=align_training(x.sample(frac=1,random_state=77),y.sample(frac=1,random_state=88))
    pd.testing.assert_frame_equal(frame,other);np.testing.assert_array_equal(target,labels)
    _,lowercase=align_training(x,y.rename(columns={"TARGET":"target"}))
    np.testing.assert_array_equal(target,lowercase)
    with pytest.raises(ValueError):align_training(x,y.assign(target=1))
    with pytest.raises(ValueError):align_training(x,pd.concat([y,y.iloc[:1]]))
    with pytest.raises(ValueError):align_training(x,y.iloc[1:])
    with pytest.raises(ValueError):validate_features(x.assign(TARGET=1))
    with pytest.raises(ValueError):validate_features(x.assign(target=1))

def test_feature_whitelist_excludes_target_date_and_id(panel):
    x,_,frame,_,_=panel
    augmented=frame.assign(FUTURE_RETURN=1000)
    for variant in ["raw","summary","identity"]:
        f=feature_table(augmented,variant)
        assert not {"TS","ROW_ID","TARGET","FUTURE_RETURN"}&set(f)
        assert ("ALLOCATION" in f)==(variant=="identity")

def test_forward_purge_and_confirmation_date_isolation(panel):
    _,_,frame,_,order=panel
    splits,development,confirm,_=split_protocol(frame,date_order=order)
    position={d:i for i,d in enumerate(order)}
    for train,valid in splits:
        assert max(position[d] for d in frame.TS.iloc[train])+20<min(position[d] for d in frame.TS.iloc[valid])
        assert not set(frame.TS.iloc[valid])&set(frame.TS.iloc[confirm])
    assert max(position[d] for d in frame.TS.iloc[development])+20<min(position[d] for d in frame.TS.iloc[confirm])

def test_opaque_dates_use_whole_groups_without_calendar_assumptions(panel):
    _,_,frame,_,_=panel
    splits,development,confirm,label=split_protocol(frame)
    assert "opaque" in label
    assert not set(frame.TS.iloc[development])&set(frame.TS.iloc[confirm])
    for tr,va in splits:assert not set(frame.TS.iloc[tr])&set(frame.TS.iloc[va])

def test_preprocessing_fits_training_only_and_accepts_unknown_categories(panel):
    _,_,frame,y,_=panel
    features=feature_table(frame,"identity")
    train=features.iloc[:200].copy();valid=features.iloc[200:210].copy()
    train["RET_20"]=np.nan;valid["RET_1"]=1e6;valid["GROUP"]="UNSEEN";valid["ALLOCATION"]="NEW"
    model=pipeline(train,"logistic");model.fit(train,y[:200])
    imputer=model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    numeric=[c for c in train if c not in ["GROUP","ALLOCATION"]]
    assert imputer.statistics_[numeric.index("RET_1")]==pytest.approx(train.RET_1.median())
    p=model.predict_proba(valid)[:,1]
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()

@pytest.mark.parametrize("name",["lightgbm","catboost"])
def test_boosting_models_fit_and_produce_probabilities(panel,name):
    _,_,frame,y,_=panel;x=feature_table(frame,"summary")
    model=pipeline(x,name);model.fit(x.iloc[:300],y[:300]);p=model.predict_proba(x.iloc[300:320])[:,1]
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()

def test_cluster_bootstrap_preserves_pairing_and_weights_by_rows():
    y=np.array([1,0,1,0,1]);groups=np.array(["a","a","b","b","b"])
    p=np.array([.8,.8,.8,.2,.8]);base=np.full(5,.6)
    result=date_bootstrap(y,p,base,groups,repetitions=300)
    assert result["accuracy_difference"]==pytest.approx(.2)
    same=date_bootstrap(y,p,p,groups,repetitions=300)
    assert same["ci_low"]==same["ci_high"]==0
    with pytest.raises(ValueError):scores(y,np.full(5,1.1))

def test_bootstrap_rejects_broadcasting_and_nonfinite_probabilities():
    labels=np.array([0,1,0,1]);groups=np.array(["a","a","b","b"])
    with pytest.raises(ValueError):date_bootstrap(labels,[.8],np.full(4,.5),groups)
    with pytest.raises(ValueError):date_bootstrap(labels,np.full(4,np.nan),np.full(4,.5),groups)
    with pytest.raises(ValueError):date_bootstrap(labels,np.full(4,.5),np.full(4,.5),["a",None,"b","b"])
    with pytest.raises(ValueError):date_bootstrap(labels,np.full(4,.5),np.full(4,.5),groups,repetitions=100.5)

def test_score_contract_rejects_empty_or_nonbinary_labels():
    with pytest.raises(ValueError):scores([],[])
    with pytest.raises(ValueError):scores([0,2],[.2,.8])

def test_split_rejects_invalid_gap_or_fold_configuration(panel):
    frame=panel[2]
    with pytest.raises(ValueError):split_protocol(frame,gap=-1)
    with pytest.raises(ValueError):split_protocol(frame,folds=1)
