"""Build cached feature + target tables for the temporal train/val split."""
import sys, os, time
sys.path.insert(0, 'src')
import pandas as pd, numpy as np
import nfl_data as nd

VAL_WEEKS = (17, 18)

def main():
    inp = pd.read_parquet('data/clean/input.parquet')
    out = pd.read_parquet('data/clean/output.parquet')
    os.makedirs('data/cache', exist_ok=True)
    for name, weeks in (('train', nd.window_split(inp.week.unique(), 'train', VAL_WEEKS)),
                        ('val', list(VAL_WEEKS))):
        t0 = time.time()
        sub = inp[inp.week.isin(weeks)]
        feats = nd.build_split(sub)
        tgt = nd.build_targets(feats, out)
        feats.to_parquet(f'data/cache/feats_{name}.parquet', index=False)
        tgt.to_parquet(f'data/cache/targets_{name}.parquet', index=False)
        print(name, 'weeks', min(weeks), '-', max(weeks),
              'plays', sub.groupby(['game_id','play_id']).ngroups,
              'feat rows', len(feats), 'target rows', len(tgt),
              f'{time.time()-t0:.0f}s', flush=True)

main()
