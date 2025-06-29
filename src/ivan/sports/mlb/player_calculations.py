import os
import json
import pickle
from pathlib import Path
from scipy.stats import nbinom
import numpy as np
import pandas as pd

def predict_fpts_params_by_ops(ops_value: float,
                               model_dict: dict) -> tuple[pd.Interval, tuple[float,float]]:
    """
    Given an OPS value, find the closest OPS‐bin in model_dict and return
    (ops_bin, (r, p)).  model_dict maps pandas.Interval -> (r, p).
    """
    # extract midpoints for each interval
    intervals = list(model_dict.keys())
    mids = np.array([(iv.left + iv.right)/2 for iv in intervals])
    if len(mids) == 0:
        raise ValueError("No OPS bins in model")
    idx = int(np.argmin(np.abs(mids - ops_value)))
    chosen = intervals[idx]
    return chosen, model_dict[chosen]


def nb_pmf(x: np.ndarray, nb_params: tuple[float,float]) -> np.ndarray:
    """
    PMF of NB(r,p) at each x.
    """
    r, p = nb_params
    return nbinom.pmf(x, r, p)


def interpolate_nb_params(ops_value: float,
                          model_dict: dict) -> tuple[float,float]:
    """
    Linearly interpolate r and p between the two nearest OPS‐bin midpoints.
    """
    intervals = sorted(model_dict.keys(), key=lambda iv: iv.left)
    mids = np.array([(iv.left+iv.right)/2 for iv in intervals])
    r_vals = np.array([model_dict[iv][0] for iv in intervals])
    p_vals = np.array([model_dict[iv][1] for iv in intervals])

    # outside range?
    if ops_value <= mids[0]:
        return r_vals[0], p_vals[0]
    if ops_value >= mids[-1]:
        return r_vals[-1], p_vals[-1]

    # find lower bin
    i = np.searchsorted(mids, ops_value) - 1
    t = (ops_value - mids[i]) / (mids[i+1] - mids[i])
    r_interp = r_vals[i] + t*(r_vals[i+1] - r_vals[i])
    p_interp = p_vals[i] + t*(p_vals[i+1] - p_vals[i])
    return float(r_interp), float(p_interp)


def get_nb_params_for_ops(ops_value: float,
                          model_dict: dict,
                          interpolate: bool = True) -> tuple[float,float]:
    """
    Return (r, p) for a given OPS value, interpolating if desired.
    """
    if interpolate:
        return interpolate_nb_params(ops_value, model_dict)
    # else, pick nearest bin
    _, (r, p) = predict_fpts_params_by_ops(ops_value, model_dict)
    return r, p


def get_adjusted_nb_params(ops_value: float,
                           player_proj: float,
                           model_dict: dict,
                           interpolate: bool = True) -> tuple[float,float]:
    """
    Historical (r_hist, p_hist) via OPS → adjust p so that mean=r*(1−p)/p == player_proj.
    """
    r_hist, p_hist = get_nb_params_for_ops(ops_value, model_dict, interpolate=interpolate)
    # new p satisfying r_hist*(1−p_new)/p_new = player_proj
    p_new = float(r_hist / (r_hist + player_proj))
    return float(r_hist), p_new