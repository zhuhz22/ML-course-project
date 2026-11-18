"""Seed variability of Baseline 2, reported alongside the headline number."""
import sys, json, time; sys.path.insert(0, 'src')
import numpy as np, pandas as pd, xgboost as xgb
import nfl_eval as ne, nfl_xgb as nx

def run(seed):
    p = dict(max_depth=8, eta=0.08, subsample=0.8, colsample_bytree=0.8,
             min_child_weight=32, objective='reg:squarederror',
             tree_method='hist', nthread=40, max_bin=128, seed=seed)
    tr = pd.read_parquet('data/cache/targets_train.parquet')
    va = pd.read_parquet('data/cache/targets_val.parquet').reset_index(drop=True)
    Xtr, cats = nx.design_matrix(tr); Xva, _ = nx.design_matrix(va)
    Xva = nx.align_cats(Xva, cats)
    dtr = xgb.DMatrix(Xtr, feature_names=list(Xtr.columns))
    dva = xgb.DMatrix(Xva, feature_names=list(Xva.columns))
    out = va.copy(); iters = {}
    for axis in ('x', 'y'):
        dtr.set_label((tr[axis]-tr['cv_'+axis]).to_numpy(np.float32))
        dva.set_label((va[axis]-va['cv_'+axis]).to_numpy(np.float32))
        b = xgb.train(p, dtr, 1200, evals=[(dva,'val')], early_stopping_rounds=60, verbose_eval=False)
        iters[axis] = int(b.best_iteration)
        out['p'+axis] = out['cv_'+axis].to_numpy(np.float32) + b.predict(dva, iteration_range=(0,b.best_iteration+1))
    return {'seed': seed, 'rmse': ne.rmse(out.px, out.py, out.x, out.y), 'best_iter': iters}

res = []
for s in (0, 1, 2):
    r = run(s); res.append(r); print(json.dumps(r), flush=True)
    json.dump(res, open('out/seed_variation.json','w'), indent=1)
rs = [r['rmse'] for r in res]
print(f"\nmean {np.mean(rs):.4f}  std {np.std(rs, ddof=1):.4f}  min {min(rs):.4f}  max {max(rs):.4f}")
