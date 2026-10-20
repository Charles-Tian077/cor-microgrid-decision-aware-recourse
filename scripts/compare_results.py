"""Compare numerical results, excluding inherently non-reproducible wall times."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT/'results'/'reference'


def compare_table(actual, expected, keys, fields, *, atol=1e-4, rtol=1e-10):
    a,b = pd.read_csv(actual),pd.read_csv(expected)
    if a.duplicated(keys).any() or b.duplicated(keys).any():
        raise AssertionError(f'Non-unique comparison keys: {actual.name}')
    a,b=a.set_index(keys).sort_index(),b.set_index(keys).sort_index()
    if not a.index.equals(b.index):
        raise AssertionError(f'Row keys differ: {actual.name}')
    deltas={}
    for name in fields:
        av,bv=a[name].to_numpy(float),b[name].to_numpy(float)
        if not np.allclose(av,bv,rtol=rtol,atol=atol,equal_nan=True):
            raise AssertionError(f'{actual.name}/{name}: max difference {np.nanmax(np.abs(av-bv))}')
        finite=np.isfinite(av)&np.isfinite(bv)
        deltas[name]=float(np.max(np.abs(av[finite]-bv[finite]))) if finite.any() else 0.
    return {'file':actual.name,'rows':len(a),'max_absolute_differences':deltas}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--kind',choices=['expanded','retrained','replay','initial','stress','training'],required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    checks=[]
    daily=['plan_cost_yuan','adjustment_cost_yuan','emergency_cost_yuan','total_cost_yuan',
           'adjusted_grid_kwh','emergency_kwh','emergency_day','soc_initial_kwh','soc_final_kwh']
    if args.kind in ('expanded','retrained','replay'):
        checks.append(compare_table(args.input/'policy_test_daily.csv',REF/'expanded'/'policy_test_daily.csv',['split','date'],daily))
        if args.kind in ('expanded','retrained'):
            checks.append(compare_table(args.input/'policy_validation_daily.csv',REF/'expanded'/'policy_validation_daily.csv',['split','date'],daily,
                                        atol=1e-4))
            checks.append(compare_table(args.input/'forecast_comparison.csv',REF/'expanded'/'forecast_comparison.csv',
                                        ['split','forecaster'],['slots','mae_yuan_per_kwh','rmse_yuan_per_kwh'],atol=1e-6 if args.kind=='retrained' else 1e-10))
            actual=json.loads((args.input/'policy_selection.json').read_text(encoding='utf8'))['selected_price_fusion_policies']
            expected=json.loads((REF/'expanded'/'policy_selection.json').read_text(encoding='utf8'))['selected_price_fusion_policies']
            assert actual==expected,(actual,expected)
    elif args.kind=='initial':
        keys=['date','alpha','gain','reserve_scale']
        checks.append(compare_table(args.input/'full_daily.csv',REF/'initial'/'full_daily.csv',keys,daily))
        checks.append(compare_table(args.input/'validation_daily.csv',REF/'initial'/'validation_daily.csv',keys,daily))
    elif args.kind=='stress':
        checks.append(compare_table(args.input/'stress_daily.csv',REF/'initial'/'stress_daily.csv',
                                    ['stress_case','date','alpha','gain','reserve_scale'],daily))
    else:
        # Floating-point neural training is not asserted to be bitwise identical across backends.
        a=pd.read_csv(args.input/'gru_predictions.csv').set_index(['date','slot_index']).sort_index()
        b=pd.read_csv(REF/'expanded'/'gru_predictions.csv').set_index(['date','slot_index']).sort_index()
        assert a.index.equals(b.index)
        mask=b['split'].isin(['validation','test'])
        assert np.isfinite(a.loc[mask,'gru_price']).all()
        diff=np.abs(a.loc[mask,'gru_price']-b.loc[mask,'gru_price'])
        checks.append({'file':'gru_predictions.csv','compared_periods':int(mask.sum()),
                       'mean_absolute_prediction_difference':float(diff.mean()),
                       'maximum_absolute_prediction_difference':float(diff.max()),
                       'bitwise_identity_required':False})
    args.out.mkdir(parents=True,exist_ok=True)
    report={'passed':True,'kind':args.kind,'timing_fields_excluded':True,
            'monetary_energy_tolerance':{'absolute':1e-4,'relative':1e-10},'checks':checks}
    if args.kind=='retrained':
        report['neural_backend_tolerances']={'validation_cost_energy_absolute':1e-4,'forecast_error_absolute':1e-6,
                                             'selected_test_policy_uses_standard_strict_tolerance':True}
    (args.out/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
