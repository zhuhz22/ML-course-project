"""Feature construction for NFL Big Data Bowl 2026 trajectory prediction.

Conventions
-----------
Coordinates are the competition's absolute field frame (x = yards along the
field, y = yards across). Play direction is already folded into `dir`/`o`
(verified: corr(dx, 0.1*s*sin(dir)) = 0.998), so no mirroring is applied.
The last observed frame of a play is the pass-release frame; all features are
built from frames at or before it, all targets are output frames >= 1.
"""
import numpy as np
import pandas as pd

DT = 0.1          # tracking sample period, seconds
HIST_FULL = 10    # frames used for full summaries (also compare 5 / 20)

KIN = ['x', 'y', 's', 'a', 'dir', 'o']


def _sin_cos(deg):
    r = np.deg2rad(deg.to_numpy(dtype=np.float32))
    return np.sin(r), np.cos(r)


def build_play_frame(inp: pd.DataFrame) -> pd.DataFrame:
    """One row per observed player in play, carrying features at pass release."""
    d = inp.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id']).copy()
    g = d.groupby(['game_id', 'play_id', 'nfl_id'], sort=False)

    d['dx'] = g.x.diff()
    d['dy'] = g.y.diff()
    d['dist_step'] = np.hypot(d['dx'], d['dy'])
    d['n_obs'] = g.x.transform('size')
    d['frame_rel'] = d.frame_id - g.frame_id.transform('max')   # 0 at release

    last = d[d.frame_rel == 0].copy()

    # ---- release-frame kinematics -------------------------------------
    th = np.deg2rad(last['dir'].to_numpy())
    last['vx'] = last['s'].to_numpy() * np.sin(th)
    last['vy'] = last['s'].to_numpy() * np.cos(th)
    orad = np.deg2rad(last['o'].to_numpy())
    last['o_sin'] = np.sin(orad)
    last['o_cos'] = np.cos(orad)
    last['dir_sin'] = np.sin(th)
    last['dir_cos'] = np.cos(th)
    # angle between motion and body orientation
    last['dir_o_diff'] = np.abs(((last['dir'] - last['o'] + 180) % 360) - 180)

    # ---- history window summaries -------------------------------------
    for W in (5, 10, 20):
        w = d[d.frame_rel >= -W]
        gg = w.groupby(['game_id', 'play_id', 'nfl_id'], sort=False)
        agg = gg.agg(**{
            f's_mean_{W}': ('s', 'mean'),
            f's_max_{W}': ('s', 'max'),
            f'a_mean_{W}': ('a', 'mean'),
            f'a_max_{W}': ('a', 'max'),
            f'path_{W}': ('dist_step', 'sum'),
            f'dir_std_{W}': ('dir', 'std'),
            f'n_{W}': ('s', 'size'),
            f'x_old_{W}': ('x', 'first'),
            f'y_old_{W}': ('y', 'first'),
        })
        # net displacement over the window and straightness ratio
        agg = agg.rename(columns={f'x_old_{W}': f'x_start_{W}', f'y_old_{W}': f'y_start_{W}'})
        last = last.merge(agg, left_on=['game_id', 'play_id', 'nfl_id'],
                          right_index=True, how='left')
        last[f'dx_net_{W}'] = last['x'] - last[f'x_start_{W}']
        last[f'dy_net_{W}'] = last['y'] - last[f'y_start_{W}']
        last[f's_acc_{W}'] = last['s'] - last[f's_mean_{W}']
        last[f'straight_{W}'] = (np.hypot(last[f'dx_net_{W}'], last[f'dy_net_{W}'])
                                 / last[f'path_{W}'].replace(0, np.nan))

    # ---- landing point geometry ---------------------------------------
    last['bx'] = last['ball_land_x'] - last['x']
    last['by'] = last['ball_land_y'] - last['y']
    last['b_dist'] = np.hypot(last['bx'], last['by'])
    last['b_ang_sin'] = last['bx'] / last['b_dist'].replace(0, np.nan)
    last['b_ang_cos'] = last['by'] / last['b_dist'].replace(0, np.nan)
    # closing speed toward the landing point / projected reach
    last['closing'] = (last['bx'] * last['vx'] + last['by'] * last['vy']) / last['b_dist'].replace(0, np.nan)
    last['b_perp'] = (last['bx'] * last['vy'] - last['by'] * last['vx']) / last['b_dist'].replace(0, np.nan)
    last['t_to_land'] = last['b_dist'] / last['s'].replace(0, np.nan)
    last['reach_at_H'] = last['s'] * DT * last['num_frames_output']
    last['b_covers'] = last['reach_at_H'] - last['b_dist']

    # ---- depth relative to the ball (progress along pass direction) ----
    pm = np.where(last['play_direction'] == 'right', 1.0, -1.0)
    last['dir_sign'] = pm
    last['x_from_land'] = (last['x'] - last['ball_land_x']) * pm
    last['y_from_land'] = last['y'] - last['ball_land_y']
    last['depth_from_yardline'] = (last['x'] - last['absolute_yardline_number']) * pm
    last['land_from_yardline'] = (last['ball_land_x'] - last['absolute_yardline_number']) * pm

    return last.reset_index(drop=True)


def add_neighbour_feats(last: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Relative geometry to the k nearest teammates / opponents at release."""
    key = ['game_id', 'play_id']
    gf = last[['game_id', 'play_id', 'nfl_id', 'player_side', 'x', 'y', 'vx', 'vy']].copy()
    gf['row_i'] = np.arange(len(gf))

    pairs = gf.merge(gf, on=key, suffixes=('', '_o'))
    pairs = pairs[pairs.nfl_id != pairs.nfl_id_o].copy()
    rx, ry = pairs.x_o - pairs.x, pairs.y_o - pairs.y
    pairs['dist'] = np.hypot(rx, ry)
    pairs['same'] = pairs.player_side == pairs.player_side_o
    pairs['rvx'] = pairs.vx_o - pairs.vx
    pairs['rvy'] = pairs.vy_o - pairs.vy
    pairs['closing'] = -(pairs.rvx * rx + pairs.rvy * ry) / pairs.dist.replace(0, np.nan)

    n = len(gf)
    out = {}
    for label, sub in (('tm', pairs[pairs.same]), ('op', pairs[~pairs.same])):
        sub = sub.sort_values(['row_i', 'dist'])
        # extract k-th nearest by position within group
        pos = sub.groupby('row_i').cumcount() + 1
        sub = sub.assign(pos=pos)
        for kk in range(1, k + 1):
            sel = sub[sub.pos == kk].set_index('row_i')
            out[f'{label}{kk}_dist'] = sel.dist.reindex(range(n))
            out[f'{label}{kk}_closing'] = sel.closing.reindex(range(n))
            out[f'{label}{kk}_relx'] = (sel.x_o - sel.x).reindex(range(n))
            out[f'{label}{kk}_rely'] = (sel.y_o - sel.y).reindex(range(n))
            out[f'{label}{kk}_rvx'] = sel.rvx.reindex(range(n))
            out[f'{label}{kk}_rvy'] = sel.rvy.reindex(range(n))
        out[f'n_{label}'] = sub.groupby('row_i').pos.max().reindex(range(n))

    fam = pd.DataFrame(out)
    # team shape: centroid and spread per side
    for side, tag in (('Offense', 'off'), ('Defense', 'def')):
        s_ = gf[gf.player_side == side].groupby(key)
        for c, v in (('cx', 'x'), ('cy', 'y'), ('sx', 'x'), ('sy', 'y'), ('n', 'x')):
            vv = s_[v].mean() if c in ('cx', 'cy') else (s_[v].std() if c in ('sx', 'sy') else s_[v].size())
            fam[f'{tag}_{c}'] = gf.set_index(key).index.map(vv)
    r = gf.join(fam)
    my = np.where(r.player_side == 'Offense', 'off', 'def')
    oth = np.where(r.player_side == 'Offense', 'def', 'off')
    r['grp_dx'] = r.x - np.where(my == 'off', r.off_cx, r.def_cx)
    r['grp_dy'] = r.y - np.where(my == 'off', r.off_cy, r.def_cy)
    r['oth_dx'] = r.x - np.where(oth == 'off', r.off_cx, r.def_cx)
    r['oth_dy'] = r.y - np.where(oth == 'off', r.off_cy, r.def_cy)
    r['sep_cx'] = r.def_cx - r.off_cx
    r['sep_cy'] = r.def_cy - r.off_cy
    return r.drop(columns=['row_i', 'player_side', 'x', 'y', 'vx', 'vy'])
