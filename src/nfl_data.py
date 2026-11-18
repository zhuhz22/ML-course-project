"""Dataset assembly: expands release-frame features over the prediction horizon."""
import numpy as np
import pandas as pd
import nfl_feats as nf

DT = 0.1


def build_split(inp: pd.DataFrame) -> pd.DataFrame:
    """Feature table for all observed players of every play in `inp`."""
    last = nf.build_play_frame(inp)
    nbr = nf.add_neighbour_feats(last)
    return last.merge(nbr, on=['game_id', 'play_id', 'nfl_id'], how='left', validate='one_to_one')


def build_targets(feats: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """Join targets onto predicted players and attach the constant-velocity path."""
    tgt = feats[feats.player_to_predict].copy()
    tgt = tgt.merge(out, on=['game_id', 'play_id', 'nfl_id'], suffixes=('', '_true'))
    # `frame_id` from the input is the release frame; `frame_id_true` restarts at 1
    # for the prediction window and is the forecast step t in Eq. 2.
    tgt = tgt.rename(columns={'x': 'x0', 'y': 'y0', 'x_true': 'x', 'y_true': 'y',
                              'frame_id': 'frame_release', 'frame_id_true': 'frame_id'})
    t = tgt.frame_id.astype(np.float32)
    tgt['t'] = t
    tgt['t_norm'] = t / tgt.num_frames_output
    tgt['cv_x'] = tgt.x0 + DT * t * tgt.vx
    tgt['cv_y'] = tgt.y0 + DT * t * tgt.vy
    return tgt


def window_split(weeks, mode, val_weeks=(17, 18)):
    """Temporal split: train on earlier weeks, validate on the last two."""
    if mode == 'train':
        return [w for w in weeks if w not in val_weeks]
    if mode == 'val':
        return list(val_weeks)
    raise ValueError(mode)
