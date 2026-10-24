"""Real-data experiment artifacts and original English research figures."""
from __future__ import annotations
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .treasury_data import DATA,load_treasury,make_features,make_samples,purged_training
from .treasury_research import (BASELINES,development_subset,development_selection,freeze_selection,
    confirmation_predictions,probability_metrics,paired_intervals,diagnostics,fit_predict,baseline_probability,losses)
from .runtime import Measurement,dump,manifest,require_empty_output

PALETTE=['#167d8d','#b67535','#67608e','#788780','#a0a7ae']


def label(name):
    return {'training_climatology':'Baseline: climatology','training_fitted_direction_persistence':'Baseline: past-direction conditional'}.get(name,name.replace('/',' / '))


def figures(out,data,ranking,predictions,intervals,frozen,diag):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'#fbfaf7','axes.facecolor':'#fbfaf7','savefig.facecolor':'#fbfaf7'})
    fig,ax=plt.subplots(figsize=(11,5.4),layout='constrained')
    for tenor,name in [('y3m','3 months'),('y1','1 year'),('y2','2 years'),('y5','5 years'),('y10','10 years')]:
        ax.plot(data.date,data[tenor],label=name,lw=1.1,alpha=.85)
    ax.axvspan(pd.Timestamp('2020-01-01'),pd.Timestamp('2025-12-31'),color='#36b3a8',alpha=.09)
    ax.axvline(pd.Timestamp('2021-12-06'),color='#8c7051',ls=':',lw=1)
    ax.set(ylabel='Constant-maturity yield (% per annum)',xlabel='Measurement date · shaded: historical confirmation',
           title='01 / THE OBSERVATION IS A YIELD, NOT A TRADE\nOfficial public H.15 series · current historical vintage · 1990–2025')
    ax.legend(ncol=5,loc='upper right');ax.grid(alpha=.16)
    fig.savefig(out/'data_overview.png',dpi=170);plt.close(fig)

    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    display=ranking.sort_values('log_loss',ascending=False)
    colors=['#167d8d' if n==frozen['global_champion'] else '#b67535' if n in BASELINES else '#8b9bac' for n in display.candidate]
    ax.barh([label(n) for n in display.candidate],display.log_loss,color=colors)
    ax.set(xlabel='Pooled expanding-development log loss (lower is better)',title='02 / SELECT ON DEVELOPMENT ONLY\nNine fixed model/feature combinations + two training-fitted baselines')
    ax.grid(axis='x',alpha=.15)
    fig.savefig(out/'development_comparison.png',dpi=170);plt.close(fig)

    names=[*frozen['family_representatives'].values(),*BASELINES]
    table=pd.DataFrame([{'name':n,**probability_metrics(predictions.target.to_numpy(),predictions[n].to_numpy())} for n in names])
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained',sharey=True)
    for ax,metric,title in zip(axes,['log_loss','balanced_accuracy'],['Log loss ↓','Balanced accuracy ↑']):
        ax.barh(range(len(names)),table[metric],color=PALETTE)
        ax.set(yticks=range(len(names)),yticklabels=[label(n) for n in names],xlabel=title)
        if metric=='balanced_accuracy':ax.axvline(.5,color='#738091',ls='--',lw=1)
        ax.grid(axis='x',alpha=.15)
    fig.suptitle('03 / KEEP THE CHOICE FROZEN\n2020–2025 decision dates · one development-frozen feature set per family',fontsize=12,fontweight='bold')
    fig.savefig(out/'confirmation_comparison.png',dpi=170);plt.close(fig)

    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    family=list(frozen['family_representatives'].values())
    for i,name in enumerate(family):
        for j,block in enumerate((63,126)):
            r=intervals[name][str(block)];y=i+(j-.5)*.18
            ax.plot([r['gain_ci95_low'],r['gain_ci95_high']],[y,y],lw=3,color=PALETTE[j])
            ax.scatter(r['mean_log_loss_gain'],y,s=30,color='#101928',zorder=3)
    ax.axvline(0,color='#738091',ls='--')
    ax.set(yticks=range(len(family)),yticklabels=[label(n)+(' *' if n==frozen['global_champion'] else '') for n in family],
           xlabel='Paired mean log-loss gain: frozen baseline − classifier; positive favors classifier',
           title='04 / KEEP THE INTERVAL, INCLUDING ZERO\n95% moving-block intervals · teal: 63 sessions · amber: 126 · * frozen champion')
    ax.grid(axis='x',alpha=.15)
    fig.savefig(out/'uncertainty.png',dpi=170);plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
    calibration=pd.DataFrame(diag['calibration'])
    for name,color in zip(names,PALETTE):
        part=calibration[(calibration.candidate==name)&(calibration.rows>0)]
        ax.plot(part.mean_probability,part.observed_positive_rate,'o-',label=label(name),color=color,lw=1.2)
    ax.plot([0,1],[0,1],'--',color='#738091',lw=1)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability in fixed bin',ylabel='Observed fraction: future yield rises',
           title='05 / A PROBABILITY SHOULD EARN ITS CONFIDENCE\nFive fixed bins · descriptive reliability · no confirmation recalibration')
    ax.legend(fontsize=9,loc='upper left');ax.grid(alpha=.15)
    fig.savefig(out/'calibration.png',dpi=170);plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,4.7),layout='constrained')
    values=[];years=[]
    for year,g in predictions.groupby(predictions.decision_date.str[:4]):
        years.append(year)
        values.append(float((losses(g.target.to_numpy(),g[frozen['baseline']].to_numpy())-losses(g.target.to_numpy(),g[frozen['global_champion']].to_numpy())).mean()))
    ax.bar(years,values,color=[PALETTE[0] if v>=0 else PALETTE[1] for v in values]);ax.axhline(0,color='#738091',lw=1)
    ax.set(ylabel='Paired mean log-loss gain',xlabel='Confirmation decision year; final year has censored unfinished labels',
           title='06 / PERFORMANCE CHANGES ACROSS YEARS\nFrozen global champion vs frozen baseline · negative years remain visible')
    ax.grid(axis='y',alpha=.15);fig.savefig(out/'yearly.png',dpi=170);plt.close(fig)


def run_treasury(output):
    out=require_empty_output(output)
    data,source,protocol=load_treasury()
    features=make_features(data)
    samples=make_samples(data,delay=protocol['decision_delay_sessions'],horizon=protocol['target_horizon_sessions'],publication_lag=protocol['availability_complete_sessions'],start=protocol['observation_start'])
    dev=development_subset(samples,protocol)
    with Measurement() as measured:
        ranking,development_predictions,folds=development_selection(dev,features,protocol)
        frozen=freeze_selection(ranking,dev,protocol)
        # Persist selection before any confirmation metric is computed.
        dump(out/'frozen_selection.json',frozen)
        predictions,audit=confirmation_predictions(samples,features,frozen,protocol)
        names=[*frozen['family_representatives'].values(),*BASELINES]
        confirmation={n:probability_metrics(predictions.target.to_numpy(),predictions[n].to_numpy()) for n in names}
        intervals={n:{str(block):paired_intervals(predictions,n,frozen['baseline'],block=block,
                    replications=protocol['uncertainty']['replications'],seed=protocol['uncertainty']['seed']) for block in protocol['uncertainty']['blocks_sessions']} for n in frozen['family_representatives'].values()}
        diag=diagnostics(predictions,names)
        stress_samples=make_samples(data,delay=protocol['availability_stress_delay_sessions'],horizon=protocol['target_horizon_sessions'],publication_lag=protocol['availability_complete_sessions'],start=protocol['observation_start'])
        stress_test=stress_samples.loc[stress_samples.decision_date.between(*protocol['confirmation'])].reset_index(drop=True)
        stress_train=purged_training(stress_samples,stress_test.decision_at.iloc[0])
        champion,variant=frozen['global_champion'].split('/')
        stress_test[frozen['global_champion']],_,_=fit_predict(stress_train,stress_test,features,champion,variant)
        stress_scores={frozen['global_champion']:probability_metrics(stress_test.target.to_numpy(),stress_test[frozen['global_champion']].to_numpy())}
        for n in BASELINES:
            stress_test[n]=baseline_probability(stress_train,stress_test,n)
            stress_scores[n]=probability_metrics(stress_test.target.to_numpy(),stress_test[n].to_numpy())
        stress_gain=paired_intervals(stress_test,frozen['global_champion'],frozen['baseline'],block=63)
        aligned=predictions[['decision_date','target']].merge(stress_test[['decision_date','target']],on='decision_date',validate='one_to_one',suffixes=('_base','_stress'))
        if len(aligned)!=len(predictions) or len(aligned)!=len(stress_test) or not (aligned.target_base==aligned.target_stress).all():
            raise ValueError('Availability stress must align identical decision dates and outcomes')
    dump(out/'metrics.json',{'data_kind':'public-real','data_rows':len(data),'complete_samples':len(samples),
        'development_rows_with_known_labels':len(dev),'selection':frozen,'confirmation':confirmation,'paired_intervals':intervals,
        'availability_stress':{'decision_delay':protocol['availability_stress_delay_sessions'],'rows':len(stress_test),'scores':stress_scores,'paired_interval_63':stress_gain,'same_decision_dates_and_targets':True},
        'measurement':measured.result})
    audit.update({'raw_observations':len(data),'complete_timed_samples':len(samples),
        'development_known_label_rows':len(dev),'late_development_labels_excluded':int(((samples.decision_date<protocol['confirmation'][0])&~samples.ROW_ID.isin(dev.ROW_ID)).sum()),
        'confirmation_positive_fraction':float(predictions.target.mean()),'confirmation_ties':int((predictions.future_change_pp==0).sum()),
        'current_vintage_limitation':source['vintage_limitation'],
        'prediction_unit':'One decision date, with overlapping twenty-session labels; row count is not independent trials.'})
    dump(out/'audit.json',audit);dump(out/'protocol.json',protocol);dump(out/'source.json',source)
    ranking.to_csv(out/'development_ranking.csv',index=False,float_format='%.12g')
    folds.to_csv(out/'fold_audit.csv',index=False)
    samples.to_csv(out/'sample_clocks.csv',index=False,float_format='%.12g')
    predictions.to_csv(out/'confirmation_predictions.csv',index=False,float_format='%.12g')
    stress_test.to_csv(out/'availability_stress_predictions.csv',index=False,float_format='%.12g')
    pd.DataFrame([{'candidate':n,**m} for n,m in confirmation.items()]).to_csv(out/'confirmation_scores.csv',index=False,float_format='%.12g')
    fold_scores=[{'candidate':n,'fold':int(f),**probability_metrics(g.target.to_numpy(),g.probability.to_numpy())} for (n,f),g in development_predictions.groupby(['candidate','fold'])]
    pd.DataFrame(fold_scores).to_csv(out/'development_fold_scores.csv',index=False,float_format='%.12g')
    win=development_predictions[development_predictions.candidate==frozen['global_champion']]
    base=development_predictions[development_predictions.candidate==frozen['baseline']]
    paired=win.merge(base[['ROW_ID','fold','probability']],on=['ROW_ID','fold'],validate='one_to_one',suffixes=('_candidate','_baseline'))
    paired.to_csv(out/'development_champion_predictions.csv',index=False,float_format='%.12g')
    for name,rows in diag.items():pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False,float_format='%.12g')
    figures(out,data,ranking,predictions,intervals,frozen,diag)
    result=intervals[frozen['global_champion']]['63']
    summary='\n'.join(f"|{label(n)}|{m['accuracy']:.2%}|{m['balanced_accuracy']:.2%}|{m['log_loss']:.5f}|{m['brier']:.5f}|{m['roc_auc']:.3f}|" for n,m in confirmation.items())
    interval_rows='\n'.join(f"|{label(n)}|{block}|{r['mean_log_loss_gain']:.5f}|[{r['gain_ci95_low']:.5f}, {r['gain_ci95_high']:.5f}]|{'yes' if r['gain_interval_contains_zero'] else 'no'}|" for n,blocks in intervals.items() for block,r in blocks.items())
    conclusion=('The interval contains zero; this study does not establish a reliable incremental probability-forecasting advantage.' if result['gain_interval_contains_zero'] else
                'The interval lies below zero: the frozen classifier is worse than the baseline on log loss in this historical confirmation. Accuracy uncertainty and the other model comparisons remain visible; this does not prove universal model inferiority.' if result['gain_ci95_high']<0 else
                'This conditional historical interval is positive. The other intervals, baseline comparisons and vintage limitations still govern interpretation, and no tradable edge is inferred.')
    (out/'REPORT.md').write_text(f'''# Treasury yield-direction classification: real public data

**Data:** 9,005 official H.15 Treasury yield observations, 1990–2025; public-domain Board reuse policy, with attribution. Current historical vintage, retrieved in October 2026. **Task:** predict whether the 10-year constant-maturity yield rises over twenty observation sessions beginning on the future decision day. This is a yield-reference classification study, not bond-price or trading-return research.

## Main finding

The development-selected classifier is **{frozen['global_champion']}**; the development-selected baseline is **{frozen['baseline']}**. The overall development leader, including baselines, is **{frozen['overall_development_leader']}**: calling a configuration the best classifier does not mean it earned promotion over a simpler baseline. On {len(predictions):,} confirmation decision dates ({audit['first_confirmation_date']} to {audit['last_confirmation_date']}), the selected classifier's mean paired log-loss gain is **{result['mean_log_loss_gain']:.5f}**, with a 63-session block 95% interval **[{result['gain_ci95_low']:.5f}, {result['gain_ci95_high']:.5f}]**. Positive gain favors the classifier. {conclusion}

## Know what was available

Observation t supplies five yield levels and strictly trailing features. A full intervening observation session is allowed for H.15 publication; features are assumed available after that session, and the decision is at 09:00 New York time on t+2. The label starts with the later measurement on t+2 and ends twenty observed sessions afterward. The old feature-date yield is not the label's starting point. Both label endpoints are excluded from features. Label availability includes the same subsequent publication allowance.

Training labels must be complete and available strictly before a validation decision. All development-selection labels must also be available before 1 January 2020; late-2019 targets reaching confirmation are excluded. See `fold_audit.csv`, `audit.json` and all {len(samples):,} timed rows in `sample_clocks.csv`. No price/rate is forward-filled. A month is twenty supplied observations, not twenty calendar days.

Exact historical release clocks and revision vintages are unavailable. The current H.15 schedule supports a conservative timing assumption, not a reconstructed historical release archive. Historical familiarity and revisions mean these are chronological historical checks, not genuinely unseen prospective OOS evidence. Treasury changed its curve methodology on 6 December 2021; old and new official measurements remain in the realized series rather than a constant-methodology backfill.

![Yield data and evaluation period](data_overview.png)

## Development, selection and probability quality

Three expanding validation windows (2008–2011, 2012–2015 and 2016–2019) compare logistic regression, LightGBM and CatBoost across curve, dynamics and combined features. Hyperparameters, 0.50 threshold and feature definitions are fixed. Every training fold fits its own imputer and scaler. A climatology baseline and a past-direction-conditioned baseline use only eligible training labels, with Laplace smoothing. Selection minimizes pooled development log loss, with deterministic name tie-breaking; it does not maximize confirmation accuracy. No early stopping, threshold search or confirmation recalibration occurs.

One representative per model family, the global champion and the best of two baselines were persisted in `frozen_selection.json` before confirmation metrics. All three family representatives and both baselines are reported below; a different confirmation ranking does not replace the frozen champion. All nine development candidates and fold scores remain in the CSVs.

![All fixed development configurations](development_comparison.png)

|Frozen configuration|Accuracy|Balanced accuracy|Log loss ↓|Brier ↓|ROC AUC|
|---|---:|---:|---:|---:|---:|
{summary}

![Confirmation model comparison](confirmation_comparison.png)

## Dependence and uncertainty

Twenty-session labels overlap. Purging prevents unavailable training outcomes from entering fitting or selection; it does not make test rows independent. Paired non-circular moving blocks of 63 and 126 sessions, with 2,000 resamples and seed 20261024, preserve local serial dependence in losses. Intervals are descriptive, conditional on the development freeze, and rely on approximate stationarity. They do not account for the entire history of financial model discovery or create 1,000 independent monthly forecasts from daily rows. The every-twentieth-anchor diagnostic reduces label-window overlap without claiming independent observations.

|Frozen family representative|Block sessions|Mean paired log-loss gain|95% interval|Contains zero?|
|---|---:|---:|---:|---|
{interval_rows}

![All uncertainty intervals](uncertainty.png)

The following fixed-bin reliability diagram checks confidence rather than only sign accuracy. Bins are predefined at width 0.2; empty bins remain in the CSV. No calibration transformation is fitted to confirmation outcomes.

![Fixed-bin reliability](calibration.png)

![Annual gains, including negative years](yearly.png)

## Stress, artifacts and limitations

A four-session availability stress holds the base champion's model and feature variant fixed, rebuilds clocks, purges/refits only before confirmation and reports both baselines. Its paired log-loss gain is {stress_gain['mean_log_loss_gain']:.5f}, 63-session interval [{stress_gain['gain_ci95_low']:.5f}, {stress_gain['gain_ci95_high']:.5f}]. Base and stress predictions are aligned on identical decision dates and target endpoints; the longer delay makes the feature observation older. The observation-key ROW_ID therefore changes while the decision-date outcome remains the same. This is a robustness diagnostic, not a new model-selection stage. Annual, as-of curve-inversion and reduced-overlap diagnostics are retained without reselection.

The output contains ranking, fold audits/scores, frozen choice, keyed confirmation probabilities, sample clocks, six figures, fixed-bin reliability, year/group slices and availability-stress predictions. `manifest.json` binds data, source, protocol, environment and every generated file. Runtime and sampled RSS cover model comparisons, inference and stress, excluding plotting and file writing after the timed block. See [data rights](../../DATA_LICENSE.md), [methods](../../docs/METHODS.md), [walkthrough](../../docs/WALKTHROUGH.md) and [validation](../../docs/VALIDATION.md).

Reference yields are interpolated indicative measures. No bond maturity roll, duration/convexity, coupons, financing, execution or transaction costs are modeled; accuracy or AUC cannot be translated into trading return or Sharpe. Original course, employer and challenge data remain outside this independently implemented study. Negative and insignificant outcomes are retained. Development and documentation were AI-assisted.
''',encoding='utf-8')
    generated={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file()}
    record=manifest(protocol,source['dataset_sha256'],'public-real')
    record.update({'protocol_sha256':hashlib.sha256((DATA/'protocol.json').read_bytes()).hexdigest(),
        'source_record_sha256':hashlib.sha256((DATA/'source.json').read_bytes()).hexdigest(),'outputs_sha256':generated})
    dump(out/'manifest.json',record)
    print(json.dumps({'data_rows':len(data),'complete_samples':len(samples),'confirmation_rows':len(predictions),
                      'frozen_champion':frozen['global_champion'],'baseline':frozen['baseline'],'primary_interval':result,'measurement':measured.result},indent=2))
