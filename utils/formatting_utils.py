import numpy as np
from numba import njit


def correct_formatX(a,dim):
    """
    @param dim: dimension of a single sample. 
        Important for the case of 1 sample, to differentiate between array of 1d samples, and single item of multiple d? 
    
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
        a = np.array([a])
        if dim == 1:
            # then not shape (1,2) but (2,1)! so a.T
            a = a.T
    return a


def correct_fileformatX(X):
    """
    Assumes corrects input of X according to function correct_formatX.
    Makes sure X is stored as list of np arrays such that SF and MF use same format.
    """
    if not isinstance(X, list):
        return [X]
    return X


def correct_format_hps(a):
    if not isinstance(a, np.ndarray):
        a = np.array(a, dtype=np.float64)
    if a.ndim == 0:
        a = np.array([a], dtype=np.float64)
    if a.ndim == 1:
        return np.array([a], dtype=np.float64)
    return a
