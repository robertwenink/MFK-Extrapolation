"""Infill criteria"""

# https://github.com/mid2SUPAERO/multi-fi-optimization/blob/master/MFK_tutorial.ipynb
import scipy as sci
import numpy as np

def EI(y, y_pred, sigma_pred):
    """
    Expected Improvement criterion, see e.g. (Jones 2001, Jones 1998)
    param y: the sampled value(s) belonging to X at level l, (this could be just one point, but it would have been sampled at the best location of previous level)
    param y_pred: (predicted) points at level l, locations X_pred
    param sigma_pred: (predicted) variance of points at level l, locations X_pred
    """
    f_min = np.min(y)
    u = (f_min - y_pred) / sigma_pred  # normalization
    EI = sigma_pred * (u * sci.stats.norm.cdf(u) + sci.stats.norm.pdf(u))
    return EI