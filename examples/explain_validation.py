"""A small, runnable tour of alignment, split boundaries and fold-local fitting."""
import numpy as np
from ml_validation.data import synthetic_panel,align_training,feature_table
from ml_validation.research import split_protocol,pipeline,scores

def main():
    x,y,order=synthetic_panel(dates=180,allocations=3)
    frame,target=align_training(x.sample(frac=1,random_state=1),y.sample(frac=1,random_state=2))
    expected=y.assign(ROW_ID=y.ROW_ID.astype(str)).set_index('ROW_ID').TARGET.reindex(frame.ROW_ID).to_numpy()>0
    np.testing.assert_array_equal(target,expected)
    folds,development,confirmation,description=split_protocol(frame,date_order=order)
    position={name:i for i,name in enumerate(order)}
    train,valid=folds[0]
    latest_train=max(position[v] for v in frame.TS.iloc[train])
    earliest_valid=min(position[v] for v in frame.TS.iloc[valid])
    assert earliest_valid-latest_train>20
    print('Keyed alignment is invariant to X/y row permutation.')
    print('Protocol:',description)
    print('Rows: development=',len(development),'confirmation=',len(confirmation))
    print('First fold boundary positions:',latest_train,earliest_valid)
    features=feature_table(frame,'summary')
    estimator=pipeline(features,'logistic');estimator.fit(features.iloc[train],target[train])
    probability=estimator.predict_proba(features.iloc[valid])[:,1]
    print('Illustrative first-fold metrics, not the portfolio confirmation:',scores(target[valid],probability))

if __name__=='__main__':main()
