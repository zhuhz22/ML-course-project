"""Official competition metric (Eq. 4): pooled coordinate RMSE over player-frames."""
import numpy as np
import pandas as pd


def rmse(x_hat, y_hat, x, y):
    x_hat, y_hat, x, y = map(lambda v: np.asarray(v, dtype=np.float64), (x_hat, y_hat, x, y))
    n = len(x)
    return float(np.sqrt(np.sum((x_hat - x) ** 2 + (y_hat - y) ** 2) / (2.0 * n)))


def cv_rmse(df, hx='cv_x', hy='cv_y'):
    """Root-mean-square distance of the constant-velocity prediction to truth."""
    return rmse(df[hx], df[hy], df.x, df.y)


def strafe_equivalence(a, b, tol=1e-9):
    """Two metric implementations agree on a given frame set."""
    return abs(a - b) < tol
