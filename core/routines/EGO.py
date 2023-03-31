import scipy.stats as scistats
import numpy as np


class EfficientGlobalOptimization():
    """This class performs the EGO algorithm on a given layer"""

    def __init__(self, setup, *args, **kwargs):
        # initial EI
        self.ei = 1
        self.ei_criterion = 0.0001 #2 * np.finfo(np.float32).eps


    @staticmethod
    def EI(y_min, y_pred, sigma_pred):
        """
        Expected Improvement criterion, see e.g. (Jones 2001, Jones 1998)
        param y_min: the sampled value belonging to the best X at level l
        param y_pred: (predicted) points at level l, locations X_pred
        param sigma_pred: (predicted) variance of points at level l, locations X_pred
        """
        # was:
        # u = np.where(sigma_pred == 0, 0, (y_min - y_pred) / sigma_pred)  # normalization
        # EI = sigma_pred * (u * scistats.norm.cdf(u) + scistats.norm.pdf(u))

        # maar, in dit geval (en igv reinterpolation) behoort een sample gebaseerd te worden op zijn waarde.
        # de expected improvement van een betere extrapolation, maar met sigma 0 is immers (y_min - y_pred)
        u = np.where(sigma_pred == 0, np.inf, (y_min - y_pred) / sigma_pred)  # normalization
        EI = sigma_pred * scistats.norm.pdf(u) + (y_min - y_pred) * scistats.norm.cdf(u)
        return float(EI)