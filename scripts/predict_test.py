"""Run both baselines on the mock test set and validate submission format."""
import sys, os; sys.path.insert(0, 'src')
import numpy as np, pandas as pd, xgboost as xgb
import nfl_data as nd, nfl_xgb as nx, nfl_predict as npd

def main():
    test = pd.read_csv('data/raw/test_input.csv')
    test['week'] = -1
    for c in ['x','y','s','a','dir','o','ball_land_x','ball_land_y']:
        test[c] = test[c].astype(np.float32)
    test['height_in'] = test.player_height.str.split('-', n=1, expand=True).astype(float).pipe(lambda d: d[0]*12+d[1]).astype(np.float32)
    test['weight_lb'] = test.player_weight.astype(np.float32)
    test['age_days'] = (pd.to_datetime(test.game_id.astype(str).str[:8], format='%Y%m%d')
                        - pd.to_datetime(test.player_birth_date)).dt.days.astype(np.float32)
    feats = nd.build_split(test)
    req = pd.read_csv('data/raw/test.csv')
    print('test plays', test.groupby(['game_id','play_id']).ngroups,
          'requested rows', len(req))

    cv = npd.predict_cv(feats, req)
    assert len(cv) == len(req) and np.isfinite(cv[['x','y']].values).all()
    assert list(cv.drop(columns=['id']).columns) == ['x','y']
    cv.drop(columns=['id']).to_csv('out/submission_cv.csv', index=False)
    print('cv submission ok', cv.shape, '-> gate format (x,y) only')

    p = npd.expand_rows(feats, req)
    X, cats = nx.design_matrix(p)
    models = {a: xgb.Booster() for a in ('x','y')}
    for a in models: models[a].load_model(f'out/xgb_{a}.json')
    pm = p.copy()
    for a in ('x','y'):
        pm['xgb_'+a] = pm['cv_'+a].to_numpy(np.float32) + models[a].predict(xgb.DMatrix(X))
    sub = pd.DataFrame({'id': p.id.to_numpy(), 'x': pm.xgb_x.to_numpy(np.float32), 'y': pm.xgb_y.to_numpy(np.float32)})
    assert len(sub) == len(req) and np.isfinite(sub[['x','y']].values).all()
    sub.drop(columns=['id']).to_csv('out/submission_xgb.csv', index=False)
    print('xgb submission ok', sub.shape, '-> gate format (x,y) only')
    print(sub.describe().to_string())

main()
