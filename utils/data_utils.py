import numpy as np
from numba import njit


def correct_formatX(a):
    """
    X should be of format np.array([[..,..,..],[..,..,..],..])
    Thus:
    1.0                 -> np.array([[a]])
    if size 1, but:
        ndim = 0        -> np.array([a])    (make 1d, refer to next)
    1d X: [..,..,..]    -> np.array([a]).T  (make 2d, align amount of samples with first dimension, return)
    if 2d but list      -> np.array(a)
    """

    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if a.ndim == 0:
        a = np.array([a])
    if a.ndim == 1:
        return np.array([a]).T
    return a


def correct_format_hps(a):
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if a.ndim == 0:
        a = np.array([a])
    if a.ndim == 1:
        return np.array([a])
    return a


def return_unique(X):
    # # https://www.peterbe.com/plog/fastest-way-to-uniquify-a-list-in-python-3.6
    seq = [item for sublist in X for item in sublist]
    return np.array(list(dict.fromkeys(seq)))