"""Run validation/test inference from a reconstructed three-seed checkpoint set."""
from pathlib import Path
import argparse
import importlib
import json
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'/'original'/'expanded'))
train=importlib.import_module('train_gpu')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--stage',choices=['validation','test'],default='test')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(8)
    ml,hist,known,target,hashes=train.read_arrays(ROOT/'data'/'export'/'csv')
    mask=ml.split.eq(args.stage).to_numpy()
    predictions=[]
    for seed in (17,29,43):
        prefix='dev' if args.stage=='validation' else 'devval'
        saved=torch.load(args.models/f'gru_{prefix}_seed{seed}.pt',map_location='cpu',weights_only=True)
        assert saved['features']==train.FEATURES
        scale=saved['scale']
        hn=np.clip((hist[mask]-scale['target_mean'])/scale['target_std'],-10,10).astype(np.float32)
        kn=np.clip((known[mask]-np.asarray(scale['feature_mean'],dtype=np.float32))/np.asarray(scale['feature_std'],dtype=np.float32),-10,10).astype(np.float32)
        model=train.PriceGRU(known.shape[1])
        model.load_state_dict(saved['state']);model.eval()
        batches=[]
        with torch.no_grad():
            for start in range(0,len(hn),4096):
                h=torch.from_numpy(hn[start:start+4096,:,None])
                k=torch.from_numpy(kn[start:start+4096])
                batches.append(model(h,k))
        values=torch.cat(batches).numpy()
        predictions.append(np.maximum(values*scale['target_std']+scale['target_mean'],.05))
    pred=np.mean(predictions,axis=0)
    out=ml.loc[mask,['date','slot_index','split','price_yuan_per_kwh']].copy()
    out['gru_price']=pred;out['ensemble_std']=np.std(predictions,axis=0)
    args.out.mkdir(parents=True,exist_ok=True)
    out.to_csv(args.out/'checkpoint_predictions.csv',index=False)
    report={'stage':args.stage,'periods':len(out),'mae':float(np.abs(pred-target[mask]).mean()),
            'rmse':float(np.sqrt(np.mean((pred-target[mask])**2))),'input_sha256':hashes}
    original=args.models/'gru_predictions.csv'
    if original.exists():
        expected=pd.read_csv(original).query('split==@args.stage').set_index(['date','slot_index']).sort_index()
        actual=out.set_index(['date','slot_index']).sort_index()
        diff=np.abs(expected.gru_price.to_numpy()-actual.gru_price.to_numpy())
        report['max_difference_from_same_training_run']=float(diff.max())
        assert np.allclose(expected.gru_price,actual.gru_price,atol=2e-6,rtol=1e-6)
    (args.out/'checkpoint_inference.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
