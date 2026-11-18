"""Numeric appendix of the data audit, written to out/audit.json."""
import sys, json; sys.path.insert(0, 'src')
import numpy as np, pandas as pd

inp = pd.read_parquet('data/clean/input.parquet')
out = pd.read_parquet('data/clean/output.parquet')
g = inp.groupby(['game_id', 'play_id'])
st = {}
st['n_rows_input'] = int(len(inp)); st['n_rows_output'] = int(len(out))
st['n_games'] = int(inp.game_id.nunique()); st['n_plays'] = int(g.ngroups)
st['n_players'] = int(inp.nfl_id.nunique())
st['n_players_pred'] = int(inp[inp.player_to_predict].nfl_id.nunique())
ppp = g.nfl_id.nunique(); st['players_per_play'] = ppp.describe().round(2).to_dict()
st['players_per_play_hist'] = {str(k): int(v) for k, v in ppp.value_counts().sort_index().items()}
npp = inp[inp.player_to_predict].groupby(['game_id','play_id']).nfl_id.nunique()
st['predict_per_play'] = npp.describe().round(2).to_dict()
st['predict_per_play_hist'] = {str(k): int(v) for k, v in npp.value_counts().sort_index().items()}
h = g.frame_id.max(); st['hist_frames'] = h.describe().round(2).to_dict()
st['hist_frames_pctl'] = h.quantile([.01,.05,.25,.5,.75,.95,.99]).round(0).astype(int).to_dict()
nfo = g.num_frames_output.first(); st['nfo'] = nfo.describe().round(2).to_dict()
st['nfo_pctl'] = nfo.quantile([.01,.05,.25,.5,.75,.95,.99]).round(0).astype(int).to_dict()
st['nfo_hist'] = {str(k): int(v) for k, v in nfo.value_counts().sort_index().items()}
st['nfo_gt_hist_frac'] = float((nfo > h).mean())
st['hist_lt_20_frac'] = float((h < 20).mean())
st['hist_lt_10_frac'] = float((h < 10).mean())
st['corr_hist_nfo'] = float(h.corr(nfo))
st['rows_per_play_out'] = out.groupby(['game_id','play_id']).size().describe().round(1).to_dict()
# NaN & duplicates
st['nan_cols_input'] = inp.isna().sum()[inp.isna().sum() > 0].to_dict()
st['nan_cols_output'] = out.isna().sum()[out.isna().sum() > 0].to_dict()
k = ['game_id','play_id','nfl_id','frame_id']
st['dup_input'] = int(inp.duplicated(k).sum()); st['dup_output'] = int(out.duplicated(k).sum())
# coverage: every prediction player-frame in output?
tgt = inp[inp.player_to_predict][k].drop_duplicates()
m = tgt.merge(out, on=k, how='left', indicator=True)
st['pred_frames_total'] = int(len(m)); st['pred_frames_unmatched'] = int((m._merge=='left_only').sum())
# output frames outside 1..nfo
nfo_map = inp[inp.player_to_predict].groupby(['game_id','play_id','nfl_id']).num_frames_output.first()
oo = out.set_index(['game_id','play_id','nfl_id']).join(nfo_map, rsuffix='_r')
st['out_rows_beyond_nfo'] = int((oo.frame_id > oo.num_frames_output).sum())
st['out_frame_min'] = int(out.frame_id.min())
# coordinate ranges
st['x_range'] = [float(inp.x.min()), float(inp.x.max())]
st['y_range'] = [float(inp.y.min()), float(inp.y.max())]
st['s_range'] = [float(inp.s.min()), float(inp.s.max())]
st['a_range'] = [float(inp.a.min()), float(inp.a.max())]
st['ox_range'] = [float(out.x.min()), float(out.x.max())]
st['oy_range'] = [float(out.y.min()), float(out.y.max())]
st['ball_x_range'] = [float(inp.ball_land_x.min()), float(inp.ball_land_x.max())]
st['ball_y_range'] = [float(inp.ball_land_y.min()), float(inp.ball_land_y.max())]
# out-of-bounds targets
st['out_x_oob'] = float(((out.x < 0) | (out.x > 120)).mean())
st['out_y_oob'] = float(((out.y < 0) | (out.y > 53.3)).mean())
# roles
st['role_counts'] = inp.player_role.value_counts().to_dict()
st['role_counts_pred'] = inp[inp.player_to_predict].player_role.value_counts().to_dict()
st['side_counts'] = inp.player_side.value_counts().to_dict()
st['position_counts'] = inp.player_position.value_counts().head(20).to_dict()
st['n_positions'] = int(inp.player_position.nunique())
# direction convention check
d = inp.sort_values(k).copy(); gg = d.groupby(['game_id','play_id','nfl_id'])
d['dx'] = gg.x.diff(); d['dy'] = gg.y.diff()
d = d.dropna(subset=['dx'])
th = np.deg2rad(d['dir'].to_numpy())
st['kin_corr_dx'] = float(np.corrcoef(d.dx, d.s*np.sin(th)*0.1)[0,1])
st['kin_corr_dy'] = float(np.corrcoef(d.dy, d.s*np.cos(th)*0.1)[0,1])
st['kin_resid_std'] = float(np.sqrt(np.mean((d.dx - d.s*np.sin(th)*0.1)**2 + (d.dy - d.s*np.cos(th)*0.1)**2)))
# ball landing vs direction of play
pl = inp.drop_duplicates(['game_id','play_id'])
st['land_rel_yardline'] = {k2: v.round(2).to_dict() for k2, v in
                           pl.assign(r=(pl.ball_land_x-pl.absolute_yardline_number)*np.where(pl.play_direction=='right',1,-1)).groupby('play_direction').r.describe().round(2).items()}
# test set
t = pd.read_csv('data/raw/test_input.csv'); tc = pd.read_csv('data/raw/test.csv')
st['test_plays'] = int(t.groupby(['game_id','play_id']).ngroups)
st['test_games'] = sorted(t.game_id.unique().tolist())
st['test_rows_req'] = int(len(tc))
st['test_players_req'] = int(tc.groupby(['game_id','play_id','nfl_id']).ngroups)
st['test_overlap_games'] = len(set(t.game_id) & set(inp.game_id))
st['test_nfo_hist'] = {str(k2): int(v) for k2, v in t.drop_duplicates(['game_id','play_id']).num_frames_output.value_counts().sort_index().items()}
# seasons
st['train_game_years'] = pd.Series(inp.game_id.astype(str).str[:4]).value_counts().to_dict()
json.dump(st, open('out/audit.json', 'w'), indent=1, default=str)
print(json.dumps({k2: v for k2, v in st.items() if not isinstance(v, dict)}, indent=1, default=str))
print('\nplayers_per_play_hist', st['players_per_play_hist'])
print('predict_per_play_hist', st['predict_per_play_hist'])
print('nfo_hist', st['nfo_hist'])
print('years', st['train_game_years'])
print('land_rel', st['land_rel_yardline'])
