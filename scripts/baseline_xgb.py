"""Baseline 2: XGBoost on constant-velocity residuals (x and y separately)."""
import sys, json, time, os; sys.path.insert(0, 'src')
import numpy as np, pandas as pd, xgboost as xgb
import nfl_eval as ne, nfl_xgb as nx

PARAMS = dict(max_depth=8, eta=0.08, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=32, objective='reg:squarederror',
              tree_method='hist', nthread=120, max_bin=128)
NROUND = 1200


def main():
    os.makedirs('out', exist_ok=True)
    tr = pd.read_parquet('data/cache/targets_train.parquet')
    va = pd.read_parquet('data/cache/targets_val.parquet')
    # subsample training plays to keep the run well under an hour if needed
    sub = float(os.environ.get('TRAIN_FRAC', '1.0'))
    if sub < 1.0:
        plays = tr[['game_id', 'play_id']].drop_duplicates().sample(frac=sub, random_state=0)
        tr = tr.merge(plays, on=['game_id', 'play_id'])
    print('train rows', len(tr), 'val rows', len(va), flush=True)

    Xtr, cats = nx.design_matrix(tr)
    Xva, _ = nx.design_matrix(va)
    Xva = nx.align_cats(Xva, cats)
    dtr = xgb.DMatrix(Xtr, label=(tr.x - tr.cv_x).to_numpy(np.float32), feature_names=list(Xtr.columns))
    dva = xgb.DMatrix(Xva, feature_names=list(Xtr.columns))

    preds = {}
    for axis, label in (('x', 'x - cv_x'), ('y', 'y - cv_y')):
        dtr.set_label((tr[axis] - tr['cv_' + axis]).to_numpy(np.float32))
        dva.set_label((va[axis] - va['cv_' + axis]).to_numpy(np.float32))
        t0 = time.time()
        bst = xgb.train(PARAMS, dtr, num_boost_round=NROUND,
                        evals=[(dva, 'val')], early_stopping_rounds=60, verbose_eval=100)
        preds[axis] = bst
        print(axis, 'best_iter', bst.best_iteration, f'{time.time()-t0:.0f}s', flush=True)
        bst.save_model(f'out/xgb_{axis}.json')

    out = va.copy()
    for axis in ('x', 'y'):
        r = preds[axis].predict(dva, iteration_range=(0, preds[axis].best_iteration + 1))
        out['xgb_' + axis] = out['cv_' + axis].to_numpy(np.float32) + r

    res = {'n_train': int(len(tr)), 'n_val': int(len(va)),
           'best_iter': {a: int(preds[a].best_iteration) for a in preds},
           'rmse_cv': ne.rmse(va.cv_x, va.cv_y, va.x, va.y),
           'rmse_xgb': ne.rmse(out.xgb_x, out.xgb_y, out.x, out.y)}
    print(json.dumps(res, indent=1))
    res['by_role'] = {r: [ne.rmse(g.xgb_x, g.xgb_y, g.x, g.y), ne.rmse(g.cv_x, g.cv_y, g.x, g.y), len(g)]
                      for r, g in out.groupby('player_role')}
    res['by_horizon'] = {str(k): [ne.rmse(g.xgb_x, g.xgb_y, g.x, g.y), ne.rmse(g.cv_x, g.cv_y, g.x, g.y), len(g)]
                         for k, g in out.groupby(pd.cut(out.frame_id, [0,2,4,6,8,10,15,20,200]), observed=True)}
    print('\nby role (xgb, cv, n):', {k:[round(v,3) for v in vv[:2]]+[vv[2]] for k,vv in res['by_role'].items()})
    print('by horizon (xgb, cv):', {k: [round(v[0],3), round(v[1],3)] for k,v in res['by_horizon'].items()})
    # feature importance
    imp = {}
    for axis in ('x','y'):
        d = preds[axis].get_score(importance_type='gain')
        imp[axis] = dict(sorted(d.items(), key=lambda kv:-kv[1])[:25])
    res['top_gain'] = imp
    print('\nTop gain (x):', list(imp['x'].items())[:12])
    json.dump(res, open('out/baseline_xgb.json','w'), indent=1)
    out[['game_id','play_id','nfl_id','frame_id','x','y','cv_x','cv_y','xgb_x','xgb_y']].to_parquet('out/xgb_val_preds.parquet')

main()
