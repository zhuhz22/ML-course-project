"""Baseline 1: constant-velocity extrapolation (Eq. 2), evaluated with Eq. 4."""
import sys; sys.path.insert(0, 'src')
import pandas as pd, numpy as np, json
import nfl_eval as ne

DT = 0.1

def run(name):
    t = pd.read_parquet(f'data/cache/targets_{name}.parquet')
    row = {'split': name, 'n_rows': int(len(t)),
           'n_plays': int(t.groupby(['game_id','play_id']).ngroups),
           'n_players': int(t.groupby(['game_id','play_id','nfl_id']).ngroups)}
    row['rmse_cv'] = ne.rmse(t.cv_x, t.cv_y, t.x, t.y)
    # reference points
    row['rmse_stationary'] = ne.rmse(t.x0, t.y0, t.x, t.y)
    row['rmse_landing'] = ne.rmse(t.ball_land_x, t.ball_land_y, t.x, t.y)
    # by role
    row['by_role'] = {r: ne.rmse(g.cv_x, g.cv_y, g.x, g.y)
                      for r, g in t.groupby('player_role')}
    # by forecast step bucket
    t = t.assign(bucket=pd.cut(t.frame_id, [0,2,4,6,8,10,15,20,200]))
    row['by_horizon'] = {str(k): [ne.rmse(g.cv_x, g.cv_y, g.x, g.y), int(len(g))]
                         for k, g in t.groupby('bucket', observed=True)}
    # by num_frames_output
    t = t.assign(nf_b=pd.cut(t.num_frames_output, [0,7,10,13,20,100]))
    row['by_nfo'] = {str(k): [ne.rmse(g.cv_x, g.cv_y, g.x, g.y), int(len(g))]
                     for k, g in t.groupby('nf_b', observed=True)}
    # by elapsed seconds
    row['by_time_s'] = {str(k): [ne.rmse(g.cv_x, g.cv_y, g.x, g.y), int(len(g))]
                        for k, g in t.groupby(pd.cut(t.frame_id*DT, np.arange(0,10.5,1.0)), observed=True)}
    # by speed at release
    row['by_speed'] = {str(k): [ne.rmse(g.cv_x, g.cv_y, g.x, g.y), int(len(g))]
                       for k, g in t.groupby(pd.cut(t.s, [-.01,1,2,3,4,5,6,20]), observed=True)}
    return row

if __name__ == '__main__':
    res = {n: run(n) for n in ('train', 'val')}
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)}
                      for k, v in res.items()}, indent=1))
    print('\nVAL by role:', {k: round(v,4) for k,v in res['val']['by_role'].items()})
    print('VAL by horizon:', {k: round(v[0],3) for k,v in res['val']['by_horizon'].items()})
    print('VAL by nfo:', {k: round(v[0],3) for k,v in res['val']['by_nfo'].items()})
    print('VAL by time:', {k: round(v[0],3) for k,v in res['val']['by_time_s'].items()})
    print('VAL by speed:', {k: round(v[0],3) for k,v in res['val']['by_speed'].items()})
    json.dump(res, open('out/baseline_cv.json','w'), indent=1)
