from scipy import io
from scipy.interpolate import interp1d
import os
import pandas as pd
import numpy as np


def load(filename, file_format, **kwargs):
    if file_format == 'matlab':
        return io.loadmat(filename, squeeze_me=True)
    if file_format == 'excel':
        return pd.read_excel(filename, sheet_name=kwargs['sheet'])
    if file_format == 'csv':
        return pd.read_csv(filename,
                           **kwargs)


def listdir(directory):
    # a wrapper just to prepend DATA_ROOT
    return [f for f in os.listdir(directory)
            if not f.startswith('.')]


def pad(num_gaze_points, sample):
    sample = np.array(sample)
    num_zeros = num_gaze_points - len(sample)
    return np.pad(sample,
                  ((0, num_zeros), (0, 0)),
                  constant_values=0)


def interpolate_nans(trial):
    nans = np.isnan(trial)
    if not nans.any():
        return trial
    nan_idxs = np.where(nans)[0]
    not_nan_idxs = np.where(~nans)[0]
    not_nan_vals = trial[not_nan_idxs]
    trial[nans] = np.interp(nan_idxs, not_nan_idxs, not_nan_vals)
    return trial


def pull_coords_to_zero(coords):
    non_neg = coords.x >= 0
    coords.x[non_neg] -= coords.x[non_neg].min()
    non_neg = coords.y >= 0
    coords.y[non_neg] -= coords.y[non_neg].min()
    return coords


def downsample(trial, new_hz, old_hz):
    skip = int(old_hz / new_hz)
    trial.x = trial.x[::skip]
    trial.y = trial.y[::skip]
    return trial


def upsample(trial, new_hz, old_hz):
    factor = int(new_hz / old_hz)
    num_upsampled_points = len(trial.x) * factor
    points = np.arange(0, num_upsampled_points, factor)
    new_points = np.arange(0, num_upsampled_points - (factor - 1), 1)
    trial.x = interp1d(points, trial.x.reshape(1, -1), kind='cubic'
                       )(new_points).reshape(-1)
    trial.y = interp1d(points, trial.y.reshape(1, -1), kind='cubic'
                       )(new_points).reshape(-1)
    return trial
