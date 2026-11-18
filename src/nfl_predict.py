"""Inference: turn a release-frame feature table into Eq. 4-format predictions.

Serving mirrors the competition gateway: one batch per play, and no access to
post-release tracking. `feats` carries the release-frame row per player
(including its own `frame_id`), `request` lists the (player, frame) pairs to
predict, so the join renames the release frame away first.
"""
import numpy as np
import pandas as pd

DT = 0.1
JOIN = ['game_id', 'play_id', 'nfl_id']


def _joined(feats: pd.DataFrame, request: pd.DataFrame) -> pd.DataFrame:
    base = feats[feats.player_to_predict].rename(
        columns={'x': 'x0', 'y': 'y0', 'frame_id': 'frame_release'})
    p = request.merge(base, on=JOIN, how='inner', validate='many_to_one')
    p['t'] = p.frame_id.astype(np.float32)
    p['t_norm'] = p.t / p.num_frames_output
    p['cv_x'] = p.x0 + DT * p.t * p.vx
    p['cv_y'] = p.y0 + DT * p.t * p.vy
    return p


def predict_cv(feats: pd.DataFrame, request: pd.DataFrame) -> pd.DataFrame:
    """Constant velocity, evaluated only at the requested player-frames."""
    p = _joined(feats, request)
    return pd.DataFrame({
        'id': p.id.to_numpy(),
        'x': p.cv_x.to_numpy(np.float32),
        'y': p.cv_y.to_numpy(np.float32),
    })


def expand_rows(feats: pd.DataFrame, request: pd.DataFrame) -> pd.DataFrame:
    """Feature rows for every requested player-frame (what the GBDT consumes)."""
    return _joined(feats, request)
