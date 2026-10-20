"""Independent archive checks; checks do not train a model or alter reference data."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'export'/'csv'
REF=ROOT/'results'/'reference'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    checks=[]
    provenance=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text(encoding='utf8'))
    for item in provenance['files']:
        path=ROOT/item['path']
        assert path.is_file() and sha(path)==item['sha256'],item['path']
    checks.append({'check':'unchanged_source_copies','files':len(provenance['files']),'passed':True})
    manifest=ROOT/'checksums.sha256'
    if manifest.exists():
        for line in manifest.read_text(encoding='utf8').splitlines():
            digest,relative=line.split('  ',1)
            assert sha(ROOT/relative)==digest,relative
        checks.append({'check':'archive_checksums','passed':True})

    obs=pd.read_csv(DATA/'03_观测_10min.csv.gz').sort_values(['date','slot_index'])
    q2=pd.read_csv(DATA/'05_继承输入_Q2.csv.gz').sort_values(['date','slot_index'])
    rel=pd.read_csv(DATA/'06_发布时刻输入.csv.gz')
    ml=pd.read_csv(DATA/'10_ML训练表_10min.csv.gz').sort_values(['date','slot_index'])
    for label,frame,keys,n in [('observations',obs,['date','slot_index'],52560),
                              ('inherited',q2,['date','slot_index'],48096),
                              ('releases',rel,['date','release_hour','slot_index'],120240),
                              ('learning',ml,['date','slot_index'],48096)]:
        assert len(frame)==n and not frame.duplicated(keys).any(),label
        checks.append({'check':label+'_keys_and_rows','rows':n,'passed':True})
    assert obs.groupby('date').size().eq(144).all()
    dates=np.array(sorted(obs.date.unique()),dtype='datetime64[D]')
    assert len(dates)==365 and np.all(np.diff(dates)==np.timedelta64(1,'D'))
    assert dates[0]==np.datetime64('2025-01-01') and dates[-1]==np.datetime64('2025-12-31')
    assert ml.groupby('split').date.nunique().to_dict()=={'development':150,'validation':92,'test':92}
    for col in ('load_kw','pv_kw','price_yuan_per_kwh'):
        assert np.isfinite(obs[col]).all()
    assert (obs[['load_kw','pv_kw']]>=0).all().all() and (obs.price_yuan_per_kwh>0).all()
    assert (rel.latest_observation_slot<=rel.release_hour*6).all()
    assert (pd.to_datetime(rel.history_end_date)<pd.to_datetime(rel.date)).all()
    assert (rel.slot_index>rel.release_hour*6).all()
    for (date,hour),p in rel.groupby(['date','release_hour']):
        assert len(p)==144-int(hour)*6 and p.slot_index.min()==int(hour)*6+1 and p.slot_index.max()==144
    checks.append({'check':'calendar_physical_inputs_release_clock','passed':True})

    prices=obs.price_yuan_per_kwh.to_numpy().reshape(365,144)
    day_lookup={str(x):i for i,x in enumerate(sorted(obs.date.unique()))}
    di=ml.date.map(day_lookup).to_numpy(int)
    slots=ml.slot_index.to_numpy(int)-1
    errors={}
    for lag in (1,7,28):
        col=f'price_lag_{lag}d'
        errors[col]=float(np.max(np.abs(ml[col].to_numpy()-prices[di-lag,slots])))
        assert errors[col]<1e-8
    for window in (7,28):
        historic=prices[di[:,None]-np.arange(window,0,-1)[None,:],slots[:,None]]
        col=f'price_same_slot_mean_{window}d'
        # Upstream dataset_lib.py rounds historical means to 4 decimal places.
        # Half-unit tolerance handles floating-point ties in decimal rounding.
        errors[col]=float(np.max(np.abs(ml[col].to_numpy()-historic.mean(1))))
        assert errors[col]<=5.000001e-5
        if window==28:
            col='price_same_slot_std_28d'
            # The same source rounds the population standard deviation to 6 places.
            errors[col]=float(np.max(np.abs(ml[col].to_numpy()-historic.std(1))))
            assert errors[col]<=5.000001e-7
    checks.append({'check':'causal_price_lags_independently_recomputed','max_errors':errors,
                   'documented_rounding':{'historical_mean':4,'population_std':6},'passed':True})

    daily=pd.read_csv(REF/'expanded'/'policy_test_daily.csv')
    summ=pd.read_csv(REF/'expanded'/'policy_test_summary.csv').set_index('policy')
    assert len(daily)==4*92 and not daily.duplicated(['split','date']).any()
    for label,p in daily.groupby('split'):
        p=p.sort_values('date')
        assert len(p)==92
        assert np.allclose(p.total_cost_yuan,p.plan_cost_yuan+p.adjustment_cost_yuan+p.emergency_cost_yuan,rtol=0,atol=1e-6)
        assert np.allclose(p.soc_initial_kwh.iloc[1:],p.soc_final_kwh.iloc[:-1],rtol=0,atol=1e-5)
        assert np.allclose(p.emergency_day,(p.emergency_kwh>1e-6).astype(int))
        assert (p[['soc_initial_kwh','soc_final_kwh']]>=1200-1e-5).all().all()
        assert (p[['soc_initial_kwh','soc_final_kwh']]<=10800+1e-5).all().all()
        for col in ('total_cost_yuan','plan_cost_yuan','adjustment_cost_yuan','emergency_cost_yuan','emergency_kwh','adjusted_grid_kwh','solver_calls'):
            assert np.isclose(p[col].sum(),summ.loc[label,col],rtol=0,atol=1e-6),(label,col)
        assert summ.loc[label,'emergency_days']==p.emergency_day.sum()
    assert summ.plan_cost_yuan.nunique()==1 and summ.final_soc_kwh.eq(1200).all()
    assert daily.max_balance_residual_kwh.max()<1e-7
    assert daily.groupby('date').planned_grid_kwh.nunique().eq(1).all()
    v=pd.read_csv(REF/'expanded'/'policy_validation_daily.csv')
    assert len(v)==13*92 and not v.duplicated(['split','date']).any()
    checks.append({'check':'paired_policy_accounting_soc_and_workload','test_policy_days':len(daily),
                   'validation_policy_days':len(v),'passed':True})

    pivot=daily.pivot(index='date',columns='split',values='total_cost_yuan').sort_index()
    delta=(pivot.full_gain-pivot.half_gain).to_numpy()
    rng=np.random.default_rng(20261020)
    ids=(rng.integers(0,92,(4000,14))[:,:,None]+np.arange(7))%92
    ci=np.quantile(delta[ids.reshape(4000,-1)[:,:92]].mean(1),[.025,.975])
    stored=json.loads((REF/'expanded'/'paired_test_comparison.json').read_text(encoding='utf8'))['full_gain']
    assert np.allclose(ci,stored['paired_7day_block_bootstrap_95ci_mean_daily_yuan'],rtol=0,atol=1e-8)
    assert abs(delta.sum()-stored['candidate_minus_baseline_total_yuan'])<1e-6
    facts={'saving_cny':float(-delta.sum()),'mean_daily_saving_cny':float(-delta.mean()),
           'bootstrap_mean_daily_saving_95ci':[-float(ci[1]),-float(ci[0])],
           'exploratory_wilcoxon_p':float(wilcoxon(delta,zero_method='wilcox').pvalue),
           'test_cost_cny':{k:float(summ.loc[k,'total_cost_yuan']) for k in summ.index},
           'test_emergency_days':{k:int(summ.loc[k,'emergency_days']) for k in summ.index},
           'test_max_balance_residual_kwh':float(daily.max_balance_residual_kwh.max())}
    checks.append({'check':'paired_statistics_independently_recomputed','passed':True})

    gru=pd.read_csv(REF/'expanded'/'gru_predictions.csv')
    mask=gru.split.isin(['validation','test'])
    assert np.isfinite(gru.loc[mask,['gru_price','ensemble_std']]).all().all()
    assert gru.loc[~mask,'gru_price'].isna().all()
    gman=json.loads((REF/'expanded'/'gru_manifest.json').read_text(encoding='utf8'))
    assert gman['seeds']==[17,29,43]
    assert all(x['selected_epoch']==2 for x in gman['selected_models'])
    checks.append({'check':'saved_neural_predictions_and_training_metadata','scored_periods':int(mask.sum()),'passed':True})
    stress=pd.read_csv(REF/'initial'/'stress_daily.csv')
    assert len(stress)==5*2*92 and stress.max_balance_residual_kwh.max()<1e-7
    for _,p in stress.groupby(['stress_case','alpha','gain','reserve_scale']):
        p=p.sort_values('date')
        assert np.allclose(p.soc_initial_kwh.iloc[1:],p.soc_final_kwh.iloc[:-1],atol=1e-5,rtol=0)
    checks.append({'check':'stress_replay_records','policy_days':len(stress),'passed':True})
    report={'passed':True,'checks':checks,'verified_main_results':facts,
            'environment':{'python':sys.version,'platform':platform.platform(),
                'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','openpyxl']}},
            'scope':'Archive integrity, causal features and numerical reconciliation; independent data and model validity are distinct.'}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
