"""Error analysis for the mid-term: role, horizon, speed, direction change, play context."""
import sys, json; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
import nfl_eval as ne

def err(df, p):
    return np.sqrt(((df[p+'_x']-df.x)**2 + (df[p+'_y']-df.y)**2)/2)

def main():
    va = pd.read_parquet('data/cache/targets_val.parquet').reset_index(drop=True)
    pr = pd.read_parquet('out/xgb_val_preds.parquet')
    key = ['game_id', 'play_id', 'nfl_id', 'frame_id']
    out = va.drop(columns=['x','y','cv_x','cv_y']).merge(pr, on=key, validate='one_to_one')
    assert len(out) == len(va)
    out['e_cv'] = err(out, 'cv'); out['e_xgb'] = err(out, 'xgb')
    out['t'] = out.frame_id.astype(np.float32)
    res = {}
    def bucket(col, bins, label=None):
        k = pd.cut(out[col], bins, labels=label)
        g = out.groupby(k, observed=True)
        return {str(i): dict(n=int(len(v)),
                             rmse_cv=ne.rmse(v.cv_x, v.cv_y, v.x, v.y),
                             rmse_xgb=ne.rmse(v.xgb_x, v.xgb_y, v.x, v.y))
                for i, v in g}
    res['horizon'] = bucket('frame_id', [0,2,4,6,8,10,15,20,200])
    res['elapsed_s'] = bucket('t', [0.05,0.5,1,1.5,2,3,4,10])
    res['nfo'] = bucket('num_frames_output', [0,7,10,13,20,100])
    res['speed'] = bucket('s', [-.01,1,2,3,4,5,6,20])
    res['role'] = {r: dict(n=int(len(v)), rmse_cv=ne.rmse(v.cv_x,v.cv_y,v.x,v.y),
                           rmse_xgb=ne.rmse(v.xgb_x,v.xgb_y,v.x,v.y))
                   for r, v in out.groupby('player_role')}
    res['side'] = {r: dict(n=int(len(v)), rmse_cv=ne.rmse(v.cv_x,v.cv_y,v.x,v.y),
                           rmse_xgb=ne.rmse(v.xgb_x,v.xgb_y,v.x,v.y))
                   for r, v in out.groupby('player_side')}
    res['b_dist'] = bucket('b_dist', [0,2,5,10,15,20,30,120])
    res['dir_change'] = bucket('dir_o_diff', [0,10,30,60,90,180])
    res['straight'] = bucket('straight_10', [0,0.3,0.6,0.8,0.95,1.01])
    res['n_hist'] = bucket('n_obs', [0,10,15,20,25,30,40,100])

    # residual direction structure: distance error split into along/across landing direction
    dx = out.xgb_x - out.x; dy = out.xgb_y - out.y
    ux, uy = out.b_ang_sin.to_numpy(), out.b_ang_cos.to_numpy()
    out['e_along'] = dx*ux + dy*uy
    out['e_across'] = -dx*uy + dy*ux
    res['along_across'] = {
        'rmse_along': float(np.sqrt(np.mean(out.e_along**2))),
        'rmse_across': float(np.sqrt(np.mean(out.e_across**2))),
        'mean_along': float(out.e_along.mean()), 'mean_across': float(out.e_across.mean())}

    # worst plays
    pl = out.groupby(['game_id','play_id']).apply(
        lambda v: pd.Series({'n': len(v), 'rmse_xgb': ne.rmse(v.xgb_x,v.xgb_y,v.x,v.y),
                             'rmse_cv': ne.rmse(v.cv_x,v.cv_y,v.x,v.y),
                             'nfo': v.num_frames_output.iloc[0]}))
    res['worst_plays'] = pl.sort_values('rmse_xgb', ascending=False).head(10).reset_index().to_dict('records')
    res['best_plays'] = pl.sort_values('rmse_xgb').head(5).reset_index().to_dict('records')
    res['play_level'] = {'n_plays': int(len(pl)),
                         'median_play_rmse_xgb': float(pl.rmse_xgb.median()),
                         'median_play_rmse_cv': float(pl.rmse_cv.median()),
                         'p90_xgb': float(pl.rmse_xgb.quantile(.9)),
                         'frac_plays_xgb_better': float((pl.rmse_xgb < pl.rmse_cv).mean())}
    # share of total squared error from worst 5% of rows
    se = out.e_xgb**2
    res['error_concentration'] = {'top5pct_rows_share': float(se.nlargest(int(len(se)*.05)).sum()/se.sum()),
                                  'top1pct_rows_share': float(se.nlargest(int(len(se)*.01)).sum()/se.sum())}
    json.dump(res, open('out/error_analysis.json','w'), indent=1)
    r = lambda d: {k: round(v['rmse_xgb'], 4) for k, v in d.items()}
    print('role  ', {k: round(v['rmse_xgb'],4) for k,v in res['role'].items()})
    print('side  ', {k: round(v['rmse_xgb'],4) for k,v in res['side'].items()})
    print('horizon', r(res['horizon']))
    print('elapsed', r(res['elapsed_s']))
    print('nfo    ', r(res['nfo']))
    print('speed  ', r(res['speed']))
    print('b_dist ', r(res['b_dist']))
    print('dirchg ', r(res['dir_change']))
    print('straight', r(res['straight']))
    print('n_hist ', r(res['n_hist']))
    print('along/across', {k: round(v,4) for k,v in res['along_across'].items()})
    print('play level', res['play_level'])
    print('concentration', res['error_concentration'])
    print('worst plays', [(w['game_id'], w['play_id'], round(w['rmse_xgb'],2), w['n']) for w in res['worst_plays'][:5]])

main()
