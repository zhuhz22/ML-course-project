"""Convert raw Kaggle NFL BDB 2026 CSV weeks into a single compact parquet."""
import pandas as pd, numpy as np, glob, os, sys
COLS = ['game_id','play_id','player_to_predict','nfl_id','frame_id','play_direction',
        'absolute_yardline_number','player_position','player_side','player_role',
        'x','y','s','a','dir','o','num_frames_output','ball_land_x','ball_land_y',
        'player_height','player_weight','player_birth_date']
OUT='data/clean'
os.makedirs(OUT, exist_ok=True)
parts_i, parts_o = [], []
for f in sorted(glob.glob('data/raw/train/input_2023_w*.csv')):
    wk=int(f.split('_w')[1].split('.csv')[0])
    i=pd.read_csv(f, usecols=COLS)
    i['week']=np.int16(wk)
    parts_i.append(i)
    of=f.replace('input_','output_')
    o=pd.read_csv(of)
    o['week']=np.int16(wk)
    parts_o.append(o)
    print(wk, len(i), len(o), flush=True)
inp=pd.concat(parts_i, ignore_index=True)
out=pd.concat(parts_o, ignore_index=True)
# height -> inches
def h2in(s):
    p=s.str.split('-', n=1, expand=True).astype(float)
    return p[0]*12+p[1]
inp['height_in']=h2in(inp.player_height).astype(np.float32)
inp['weight_lb']=inp.player_weight.astype(np.float32)
inp['age_days']=(pd.to_datetime(inp.game_id.astype(str).str[:8], format='%Y%m%d')
                 - pd.to_datetime(inp.player_birth_date)).dt.days.astype(np.float32)
inp=inp.drop(columns=['player_height','player_weight','player_birth_date'])
for c in ['x','y','s','a','dir','o','ball_land_x','ball_land_y']:
    inp[c]=inp[c].astype(np.float32)
for c in ['game_id','play_id','nfl_id','frame_id','absolute_yardline_number','num_frames_output']:
    inp[c]=inp[c].astype(np.int64)
inp['player_to_predict']=inp.player_to_predict.astype(bool)
out['x']=out.x.astype(np.float32); out['y']=out.y.astype(np.float32)
for c in ['game_id','play_id','nfl_id','frame_id']: out[c]=out[c].astype(np.int64)
inp.to_parquet(f'{OUT}/input.parquet', index=False)
out.to_parquet(f'{OUT}/output.parquet', index=False)
print('input', inp.shape, 'output', out.shape)
print(inp.dtypes)
