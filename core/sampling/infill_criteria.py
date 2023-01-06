"""Infill criteria"""
# pyright: reportGeneralTypeIssues=false
# https://github.com/mid2SUPAERO/multi-fi-optimization/blob/master/MFK_tutorial.ipynb
import scipy as sci
from scipy import stats
import numpy as np

def EI(y_min, y_pred, sigma_pred):
    """
    Expected Improvement criterion, see e.g. (Jones 2001, Jones 1998)
    param y_min: the sampled value belonging to the best X at level l
    param y_pred: (predicted) points at level l, locations X_pred
    param sigma_pred: (predicted) variance of points at level l, locations X_pred
    """

    u = np.where(sigma_pred == 0, 0, (y_min - y_pred) / sigma_pred)  # normalization
    EI = sigma_pred * (u * sci.stats.norm.cdf(u) + sci.stats.norm.pdf(u))
    return EI