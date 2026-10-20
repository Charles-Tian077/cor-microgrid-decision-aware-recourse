"""Portable entry points; reference inputs/results are never overwritten."""
from __future__ import annotations
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
os.environ['PYTHONUTF8'] = '1'
DATA = ROOT / 'data' / 'export' / 'csv'
ORIGINAL = ROOT / 'code' / 'original' / 'expanded'
REFERENCE = ROOT / 'results' / 'reference'
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')


def output_dir(value):
    out = Path(value).expanduser().resolve()
    for protected in (ROOT/'data', REFERENCE, ROOT/'code'/'original'):
        if out == protected or protected in out.parents:
            raise ValueError(f'Output must not overwrite the archive: {protected}')
    out.mkdir(parents=True, exist_ok=True)
    return out


def engine():
    os.environ['COR_DATA_DIR'] = str(DATA)
    sys.path.insert(0, str(ORIGINAL))
    return importlib.import_module('cor_q4_experiment')


def replay(out, split):
    import pandas as pd
    model = engine()
    helper = importlib.import_module('evaluate_expansion')
    obs, q2, releases, ml, dates = model.read_data()
    selected = ml.loc[ml.split.eq(split), 'date'].unique()
    ds = sorted(str(x) for x in selected)
    # alpha=0 makes learned prices inactive; no fitted predictor is required.
    placeholder = ml[['date', 'slot_index']].assign(hgb_price=0.05)
    tables = model.prep_index(obs, q2, releases, placeholder)
    first = float(q2.loc[(q2.date == ds[0]) & q2.slot_index.eq(1), 'soc_start_kwh'].iloc[0])
    runs = []
    for label, gain, reserve in [('no_update', 0, 1), ('half_gain', .5, 1),
                                  ('full_gain', 1, 1), ('no_reserve', 1, 0)]:
        daily = helper.run_policy(ds, label, 0., gain, reserve, tables, first, model)
        daily['forecaster'] = 'causal'
        runs.append(daily)
    daily = pd.concat(runs, ignore_index=True)
    daily.to_csv(out/f'policy_{split}_daily.csv', index=False)
    summary = pd.DataFrame([{'policy':label, **helper.metrics(part)}
                           for label,part in daily.groupby('split', sort=False)])
    summary.to_csv(out/f'policy_{split}_summary.csv', index=False)
    print(summary.to_string(index=False))


def execute_original(phase, out):
    model = engine()
    model.OUT = out
    if phase == 'stress':
        shutil.copy2(REFERENCE/'initial'/'final_verdict.json', out/'final_verdict.json')
    saved = sys.argv
    try:
        sys.argv = ['cor_q4_experiment.py', '--phase', phase]
        model.main()
    finally:
        sys.argv = saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    verify = sub.add_parser('verify', help='Verify archived data, source copies and reference results')
    verify.add_argument('--out', default=str(ROOT/'runs'/'verification'))
    rep = sub.add_parser('replay', help='Re-solve four frozen policies, without retraining')
    rep.add_argument('--split', choices=['validation','test'], default='test')
    rep.add_argument('--out', default=str(ROOT/'runs'/'replay'))
    full = sub.add_parser('expanded', help='Refit tree forecasts; replay all 13 validation candidates and four test policies')
    full.add_argument('--predictions', type=Path, default=REFERENCE/'expanded')
    full.add_argument('--out', default=str(ROOT/'runs'/'expanded'))
    train = sub.add_parser('train', help='Retrain the three-seed GRU; GPU is optional with --allow-cpu')
    train.add_argument('--allow-cpu', action='store_true')
    train.add_argument('--epochs', type=int, default=30)
    train.add_argument('--out', default=str(ROOT/'runs'/'training'))
    for command in ('initial','stress'):
        p = sub.add_parser(command, help='Replay original control grid/334-day continuation' if command=='initial' else 'Re-solve the five stress cases')
        p.add_argument('--out', default=str(ROOT/'runs'/command))
    compare = sub.add_parser('compare', help='Numerically compare regenerated results to the archive; timing is excluded')
    compare.add_argument('--input', type=Path, required=True)
    compare.add_argument('--kind', choices=['expanded','retrained','replay','initial','stress','training'], default='expanded')
    compare.add_argument('--out', default=str(ROOT/'runs'/'comparison'))
    args = parser.parse_args()
    if args.command=='verify':
        subprocess.run([sys.executable,str(ROOT/'scripts'/'verify_archive.py'),'--out',str(output_dir(args.out))],check=True)
    elif args.command=='replay':
        replay(output_dir(args.out), args.split)
    elif args.command=='expanded':
        subprocess.run([sys.executable,str(ORIGINAL/'evaluate_expansion.py'), '--data',str(DATA),
                        '--gpu-results',str(args.predictions.resolve()),'--out',str(output_dir(args.out))],check=True)
    elif args.command=='train':
        cmd = [sys.executable,str(ORIGINAL/'train_gpu.py'),'--data',str(DATA),'--out',str(output_dir(args.out)),
               '--epochs',str(args.epochs),'--batch-size','512','--seeds','17','29','43']
        if args.allow_cpu: cmd.append('--allow-cpu')
        subprocess.run(cmd,check=True)
    elif args.command in ('initial','stress'):
        execute_original('full' if args.command=='initial' else 'stress', output_dir(args.out))
    else:
        subprocess.run([sys.executable,str(ROOT/'scripts'/'compare_results.py'),'--input',str(args.input.resolve()),
                        '--kind',args.kind,'--out',str(output_dir(args.out))],check=True)


if __name__=='__main__':
    main()
