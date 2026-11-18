"""Figures for the mid-term report."""
import sys, json; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .3,
                     'figure.dpi': 200, 'savefig.bbox': 'tight',
                     'axes.spines.top': False, 'axes.spines.right': False})
C_CV, C_GB = '#8c8c8c', '#1f6fb4'

ea = json.load(open('out/error_analysis.json'))
cv = json.load(open('out/baseline_cv.json'))
ab = json.load(open('out/ablation_xgb.json'))

def pretty(k):
    """'(0, 2]' -> '0-2' for axis labels."""
    lo, hi = k.strip('(]').split(',')
    return f'{float(lo):g}-{float(hi):g}'


def band(d, key, x):
    ks = list(d[key].keys())
    xs = [float(k.strip('(]').split(',')[0]) for k in ks]
    yc = [d[key][k]['rmse_cv'] if 'rmse_cv' in d[key][k] else d[key][k][0] for k in ks]
    yg = [d[key][k]['rmse_xgb'] if 'rmse_xgb' in d[key][k] else d[key][k][1] for k in ks]
    return ks, yc, yg

# --- Fig 1: horizon curve
ks, yc, yg = band(ea, 'horizon', 0)
fig, ax = plt.subplots(figsize=(4.6, 2.8))
x = np.arange(len(ks))
ax.plot(x, yc, 'o-', color=C_CV, label='constant velocity')
ax.plot(x, yg, 'o-', color=C_GB, label='XGBoost residual')
ax.set_xticks(x); ax.set_xticklabels([pretty(k) for k in ks], rotation=20, ha='right')
ax.set_xlabel('forecast frames since release  $t$'); ax.set_ylabel('RMSE (yards)')
ax.set_title('Error grows with the forecast horizon', fontsize=9.5)
ax.legend(frameon=False, fontsize=8)
fig.savefig('reports/fig/horizon.pdf'); plt.close(fig)

# --- Fig 2: ablation bar
fig, ax = plt.subplots(figsize=(5.4, 2.55))
rows = sorted(ab, key=lambda r: -r['rmse'])
PRETTY = {'full': 'full model', 'no_history': 'no history summaries',
          'no_neighbour': 'no interaction features',
          'history_last_frame': 'short (5-frame) history only',
          'no_history_no_neighbour': 'neither history nor interaction',
          'no_landing_geometry': 'no landing-point geometry'}
names = [PRETTY.get(r['variant'], r['variant']) for r in rows]
vals = [r['rmse'] for r in rows]
cols = [C_GB if r['variant'] == 'full' else '#b9c9d9' for r in rows]
b = ax.barh(names, vals, color=cols, height=.62)
ax.set_xlabel('validation RMSE (yards)')
ax.set_xlim(0.68, 1.045)
for rect, v, r in zip(b, vals, rows):
    ax.text(v + 0.006, rect.get_y() + rect.get_height() / 2,
            f'{v:.3f}  ({r["n_features"]} feats)', va='center', fontsize=7.2)
ax.set_title('Feature-group ablations (GBDT)', fontsize=9.5, pad=6)
ax.invert_yaxis()
fig.savefig('reports/fig/ablation.pdf'); plt.close(fig)

# --- Fig 3: role + speed panels
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
ax = axes[0]
roles = list(ea['role'].keys())
w = .36; xx = np.arange(len(roles))
ax.bar(xx-w/2, [ea['role'][r]['rmse_cv'] for r in roles], w, color=C_CV, label='CV')
ax.bar(xx+w/2, [ea['role'][r]['rmse_xgb'] for r in roles], w, color=C_GB, label='GBDT')
ax.set_xticks(xx); ax.set_xticklabels([r.replace(' ', '\n') for r in roles])
ax.set_ylabel('RMSE (yards)'); ax.set_title('By player role', fontsize=9.5)
ax.legend(frameon=False, fontsize=8)
ax = axes[1]
ks, yc, yg = band(ea, 'speed', 0)
x = np.arange(len(ks))
ax.plot(x, yc, 'o-', color=C_CV, label='CV')
ax.plot(x, yg, 'o-', color=C_GB, label='GBDT')
ax.set_xticks(x); ax.set_xticklabels([pretty(k) for k in ks], rotation=0, fontsize=7)
ax.set_xlabel('speed at release (yd/s)'); ax.set_title('By release speed', fontsize=9.5)
ax.legend(frameon=False, fontsize=8)
fig.savefig('reports/fig/role_speed.pdf'); plt.close(fig)

# --- Fig 4: distance to landing point
ks, yc, yg = band(ea, 'b_dist', 0)
fig, ax = plt.subplots(figsize=(4.4, 2.6))
x = np.arange(len(ks))
ax.plot(x, yc, 'o-', color=C_CV, label='constant velocity')
ax.plot(x, yg, 'o-', color=C_GB, label='XGBoost residual')
ax.set_xticks(x); ax.set_xticklabels([pretty(k) for k in ks], rotation=20, ha='right')
ax.set_xlabel('distance to landing point at release (yards)'); ax.set_ylabel('RMSE (yards)')
ax.set_title('Players far from the landing point are hardest', fontsize=9.5)
ax.legend(frameon=False, fontsize=8)
fig.savefig('reports/fig/bdist.pdf'); plt.close(fig)

# --- Fig 5: example trajectory (a real validation play, receiver + nearest defender)
va = pd.read_parquet('data/cache/targets_val.parquet')
inp = pd.read_parquet('data/clean/input.parquet')
pr = pd.read_parquet('out/xgb_val_preds.parquet')
key = ['game_id','play_id','nfl_id','frame_id']
m = va.merge(pr.drop(columns=['x','y','cv_x','cv_y']), on=key)
g = m.groupby(['game_id','play_id']).size().sort_values()
# pick a mid-horizon play with a targeted receiver
cand = m[(m.num_frames_output.between(10, 14))]
gp = cand.groupby(['game_id','play_id']).filter(lambda v: (v.player_role=='Targeted Receiver').any())
gj, pid = gp.iloc[0].game_id, gp.iloc[0].play_id
sel = m[(m.game_id==gj)&(m.play_id==pid)]
hist = inp[(inp.game_id==gj)&(inp.play_id==pid)]
fig, ax = plt.subplots(figsize=(6.2, 3.4))
for nid, v in sel.groupby('nfl_id'):
    role = v.player_role.iloc[0]
    col = '#d1495b' if role=='Targeted Receiver' else '#4a7fb5'
    h = hist[hist.nfl_id==nid].sort_values('frame_id')
    ax.plot(h['x'].to_numpy(), h['y'].to_numpy(), '-', color=col, lw=.8, alpha=.55)
    ax.plot(v['x'].to_numpy(), v['y'].to_numpy(), '-', color='k', lw=1.5)
    ax.plot(v['xgb_x'].to_numpy(), v['xgb_y'].to_numpy(), '--', color='#e0a300', lw=1.5)
    ax.scatter(v['x'].to_numpy()[0], v['y'].to_numpy()[0], s=22, color=col, zorder=5)
    ax.plot([v['x'].to_numpy()[-1]], [v['y'].to_numpy()[-1]], 'k^', ms=5, ls='')
    ax.plot([v['xgb_x'].to_numpy()[-1]], [v['xgb_y'].to_numpy()[-1]], color='#e0a300', marker='^', ms=5, ls='')
ax.scatter(sel.ball_land_x.to_numpy()[0], sel.ball_land_y.to_numpy()[0], marker='*', s=160, color='#2a9d8f', zorder=6)
ax.set_xlabel('x (yards, along field)'); ax.set_ylabel('y (yards, across field)')
ax.set_title('Example play: observed history (thin), truth (black), GBDT forecast (dashed)', fontsize=8.5)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],color='#d1495b',label='targeted receiver'),
                   Line2D([],[],color='#4a7fb5',label='defensive coverage'),
                   Line2D([],[],color='k',label='truth'),
                   Line2D([],[],color='#e0a300',ls='--',label='GBDT'),
                   Line2D([],[],marker='*',color='#2a9d8f',ls='',ms=9,label='ball landing point')],
          fontsize=7, frameon=False, loc='best')
fig.savefig('reports/fig/example_play.pdf'); plt.close(fig)
print('figs written')
