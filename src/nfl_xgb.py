"""Baseline 2: gradient-boosted trees on constant-velocity residuals.

Two XGBoost regressors predict (x - cv_x) and (y - cv_y) per player-frame row,
using release-frame kinematics, history summaries, landing-point geometry,
role encoding, forecast time, and neighbour/team-shape features.
"""
import numpy as np
import pandas as pd

CATS = ['player_side', 'player_role', 'player_position', 'play_direction']

# features constant within a play/player (release-time context)
BASE = ['s', 'a', 'dir_sin', 'dir_cos', 'o_sin', 'o_cos', 'vx', 'vy', 'dir_o_diff',
        'n_obs', 'height_in', 'weight_lb', 'age_days',
        'b_dist', 'b_ang_sin', 'b_ang_cos', 'bx', 'by', 'closing', 'b_perp',
        't_to_land', 'reach_at_H', 'b_covers', 'dir_sign',
        'x_from_land', 'y_from_land', 'depth_from_yardline', 'land_from_yardline',
        'num_frames_output', 'absolute_yardline_number']
HIST = [f'{n}_{W}' for W in (5, 10, 20)
        for n in ('s_mean', 's_max', 'a_mean', 'a_max', 'path', 'dir_std', 'n',
                  'dx_net', 'dy_net', 's_acc', 'straight')]
NBR = [f'{p}{k}_{q}' for p in ('tm', 'op') for k in (1, 2, 3)
       for q in ('dist', 'closing', 'relx', 'rely', 'rvx', 'rvy')] + \
      ['n_tm', 'n_op', 'off_cx', 'off_cy', 'off_sx', 'off_sy', 'off_n',
       'def_cx', 'def_cy', 'def_sx', 'def_sy', 'def_n',
       'grp_dx', 'grp_dy', 'oth_dx', 'oth_dy', 'sep_cx', 'sep_cy']
TIME = ['t', 't_norm']


def design_matrix(df: pd.DataFrame):
    """Numeric matrix + category codes; categories are taken from training data."""
    cats = {}
    for c in CATS:
        dt = df[c].astype('category')
        cats[c] = dt.cat.categories
    X = df[BASE + HIST + NBR].astype(np.float32).copy()
    for c in CATS:
        X[c] = pd.Categorical(df[c], categories=cats[c]).codes.astype(np.int8)
    X['t'] = df['t'].astype(np.float32)
    X['t_norm'] = df['t_norm'].astype(np.float32)
    X['t_sq'] = (df['t'] ** 2).astype(np.float32)
    X['cv_speed'] = np.hypot(df.vx, df.vy).astype(np.float32)
    # velocity of the constant-velocity path evaluated at t
    X['dt_x'] = (0.1 * df['t'] * df['vx']).astype(np.float32)
    X['dt_y'] = (0.1 * df['t'] * df['vy']).astype(np.float32)
    return X, cats


def align_cats(X, cats):
    for c in CATS:
        X[c] = pd.Categorical(X[c], categories=cats[c]).codes.astype(np.int8)
    return X
