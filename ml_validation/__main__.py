from __future__ import annotations
import argparse,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import align_training,synthetic_panel
from .research import evaluate
from .runtime import Measurement,manifest,dump,require_empty_output


def run(output,x_path=None,y_path=None):
    if bool(x_path)!=bool(y_path):raise ValueError("Supply both --x and --y")
    out=require_empty_output(output)
    if x_path:
        x,y=pd.read_csv(x_path,dtype={"ROW_ID":str}),pd.read_csv(y_path,dtype={"ROW_ID":str});order=None;kind="user_supplied_challenge"
    else:x,y,order=synthetic_panel();kind="synthetic"
    frame,targets=align_training(x,y)
    fingerprint=hashlib.sha256((frame.to_csv(index=False,float_format="%.17g")+pd.Series(targets).to_csv(index=False)).encode()).hexdigest()
    with Measurement() as measured:result,folds,ranking=evaluate(frame,targets,date_order=order)
    result.update(data_kind=kind,measurement=measured.result)
    dump(out/"metrics.json",result);dump(out/"frozen_selection.json",result["freeze"])
    config={"seed":42,"folds":3,"history_observations":20,"synthetic_dates":360 if kind=="synthetic" else None,
            "confirmation_fraction":.2,"threshold":.5,"models":["logistic","lightgbm","catboost"],
            "ablations":["raw","summary","identity"],"boosting_rounds":80}
    dump(out/"manifest.json",manifest(config,fingerprint,kind))
    folds.to_csv(out/"development_folds.csv",index=False);ranking.to_csv(out/"model_comparison.csv",index=False)
    pd.DataFrame(result["confirmation_groups"]).to_csv(out/"group_diagnostics.csv",index=False)
    fig,ax=plt.subplots(figsize=(10,5),layout="constrained")
    display=ranking.sort_values("accuracy")
    ax.barh(display.candidate,display.accuracy*100,color="#286a9a")
    ax.axvline(result["development_baseline"]["accuracy"]*100,color="#b34d25",linestyle="--",label="Fold-trained majority baseline")
    ax.set(xlabel="Development validation accuracy (%)",title=f"{kind.replace('_',' ').title()}: nine predeclared configurations")
    ax.legend(loc="lower right");ax.grid(axis="x",alpha=.2)
    fig.savefig(out/"model_comparison.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout="constrained")
    groups=pd.DataFrame(result["confirmation_groups"])
    ax.bar(groups.group,groups.accuracy*100,color="#286a9a")
    ax.axhline(result["baseline"]["accuracy"]*100,color="#b34d25",linestyle="--",label="Overall majority baseline")
    ax.set(xlabel="Allocation group",ylabel="Confirmation accuracy (%)",title="Frozen candidate: diagnostic slices, no reselection")
    ax.legend();ax.grid(axis="y",alpha=.2);fig.savefig(out/"group_diagnostics.png",dpi=160);plt.close(fig)
    b=result["paired_bootstrap"];m=result["confirmation"]
    table="\n".join(f"|{r.candidate}|{r.accuracy:.3%}|{r.balanced_accuracy:.3%}|{r.development_fit_seconds:.3f}|" for r in ranking.itertuples())
    (out/"REPORT.md").write_text(f"""# Financial classification and validation

Data: **{kind}**. {'This simulated panel is not the official QRT dataset and provides no evidence of market predictability or leaderboard performance.' if kind=='synthetic' else 'This user-supplied experiment is excluded from the initial public demo. Previously inspected challenge labels must not be described as pristine holdout data.'}

Protocol: {result['protocol']}. Development: {result['development_dates']} dates / {result['development_rows']} rows. Confirmation: {result['confirmation_dates']} dates / {result['confirmation_rows']} rows.

|Configuration|Development accuracy|Balanced accuracy|Fit seconds across folds|
|---|---:|---:|---:|
{table}

![Model comparison](model_comparison.png)

The frozen choice was **{result['selected']}**, selected solely on development accuracy with threshold 0.50. Confirmation accuracy: **{m['accuracy']:.3%}**; fold-independent confirmation majority baseline: **{result['baseline']['accuracy']:.3%}**. Paired accuracy difference: **{b['accuracy_difference']*100:.3f} percentage points**, with a date-cluster bootstrap 95% interval **[{b['ci_low']*100:.3f}, {b['ci_high']*100:.3f}]**. A negative or uncertain difference is retained, not tuned away.

![Group diagnostics](group_diagnostics.png)

## What is controlled

Labels join by ROW_ID; input order is irrelevant. Each preprocessing pipeline fits only its training rows. Dates, row IDs and targets are excluded from all features. The three fixed ablations use raw histories, summary statistics and optionally allocation identity. All nine configurations use the same development folds. No neural model, hyperparameter search, threshold search or public leaderboard feedback is used.

## Limits

Synthetic dates have a known generating order: forward development folds and confirmation are separated by 20 dates to isolate the 20-observation history. Official TS labels are anonymized and shuffled: the user-data protocol uses whole-date groups with a stable hash split, which does not establish true forward performance or remove unknown history overlap. Previously exposed official labels remain exposed even after a new split. Date bootstrap accounts for within-date dependence, not arbitrary dependence across dates. Confidence intervals are descriptive following development selection. Accuracy on a sign label is not a trading return, Sharpe or execution claim. This run does not upload predictions or measure public/private challenge scores.

`manifest.json` binds source, input fingerprint, configuration and environment. `metrics.json` includes timing and sampled process RSS; the freeze is recorded in `frozen_selection.json`.
""",encoding="utf-8")
    print(f"ML demo complete: data={kind}; selected={result['selected']}; confirmation={m['accuracy']:.3%}.")


def main():
    p=argparse.ArgumentParser(description="Keyed financial ML validation with an offline synthetic demo")
    p.add_argument("--x",help="Official feature CSV obtained independently from Challenge Data")
    p.add_argument("--y",help="Matching target CSV")
    p.add_argument("--output",default="local-runs/demo")
    a=p.parse_args();run(a.output,a.x,a.y)

if __name__=="__main__":main()
