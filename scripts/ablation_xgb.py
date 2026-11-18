"""Feature-group ablations for Baseline 2 (the mid-term version of Sec. 4 ablations)."""
import sys, json, time, os; sys.path.insert(0, 'src')
import numpy as np, pandas as pd, xgboost as xgb
import nfl_eval as ne, nfl_xgb as nx

PARAMS = dict(max_depth=8, eta=0.08, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=32, objective='reg:squarederror',
              tree_method='hist', nthread=60, max_bin=128)
NROUND, ESR = 1400, 60

VARIANTS = {
    'full':        dict(hist=True,  nbr=True,  land=True,  time=True),
    'no_history':  dict(hist=False, nbr=True,  land=True,  time=True),
    'no_neighbour':dict(hist=True,  nbr=False, land=True,  time=True),
    'history_last_frame': dict(hist=False, nbr=True, land=True, time=True, last_frame=True),
    'no_history_no_neighbour': dict(hist=False, nbr=False, land=True, time=True),
    'no_landing_geometry': dict(hist=True, nbr=True, land=False, time=True),
}


def columns_for(v, all_cols):
    drop = set()
    if not v['hist']:
        drop |= {c for c in all_cols if any(c.startswith(p) for p in
                 ('s_mean_', 's_max_', 'a_mean_', 'a_max_', 'path_', 'dir_std_', 's_acc_', 'straight_'))}
    if v.get('last_frame'):
        # keep the shortest (5-frame) summaries only; the longest window is the "full history" signal
        drop |= {c for c in all_cols if c.endswith('_10') or c.endswith('_20')}
    if not v['nbr']:
        drop |= {c for c in all_cols if c.startswith(('tm1_','tm2_','tm3_','op1_','op2_','op3_','n_tm','n_op',
                                                      'off_','def_','grp_','oth_','sep_'))}
    if not v['land']:
        drop |= {c for c in all_cols if c in ('b_dist','b_ang_sin','b_ang_cos','bx','by','closing','b_perp',
                                              't_to_land','reach_at_H','b_covers','x_from_land','y_from_land',
                                              'land_from_yardline')}
    return [c for c in all_cols if c not in drop]


def run(name, v):
    tr = pd.read_parquet('data/cache/targets_train.parquet')
    va = pd.read_parquet('data/cache/targets_val.parquet')
    Xtr, cats = nx.design_matrix(tr)
    Xva, _ = nx.design_matrix(va)
    Xva = nx.align_cats(Xva, cats)
    keep = columns_for(v, list(Xtr.columns))
    Xtr, Xva = Xtr[keep], Xva[keep]
    dtr = xgb.DMatrix(Xtr, feature_names=keep)
    dva = xgb.DMatrix(Xva, feature_names=keep)
    va_ = va.reset_index(drop=True)
    out = va_.copy()
    iters = {}
    for axis in ('x', 'y'):
        dtr.set_label((tr[axis] - tr['cv_' + axis]).to_numpy(np.float32))
        dva.set_label((va_[axis] - va_['cv_' + axis]).to_numpy(np.float32))
        bst = xgb.train(PARAMS, dtr, num_boost_round=NROUND, evals=[(dva, 'val')],
                        early_stopping_rounds=ESR, verbose_eval=False)
        iters[axis] = int(bst.best_iteration)
        r = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
        out['p' + axis] = out['cv_' + axis].to_numpy(np.float32) + r
    res = {'variant': name, 'n_features': len(keep), 'best_iter': iters,
           'rmse': ne.rmse(out.px, out.py, out.x, out.y),
           'rmse_cv': ne.rmse(out.cv_x, out.cv_y, out.x, out.y)}
    res['by_role'] = {r: ne.rmse(g.px, g.py, g.x, g.y) for r, g in out.groupby('player_role')}
    print(json.dumps(res), flush=True)
    return res


if __name__ == '__main__':
    names = sys.argv[1:] or list(VARIANTS)
    os.makedirs('out', exist_ok=True)
    res = []
    for n in names:
        t0 = time.time(); r = run(n, VARIANTS[n]); r['secs'] = round(time.time() - t0)
        res.append(r)
        json.dump(res, open('out/ablation_xgb.json', 'w'), indent=1)
    print('\n== summary ==')
    for r in sorted(res, key=lambda z: z['rmse']):
        print(f"{r['variant']:26s} feats={r['n_features']:4d}  RMSE={r['rmse']:.4f}")
