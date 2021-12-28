import numpy as np


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