# RESEARCH QUESTIONS:
# - Can we improve the effective sample efficiency/costs in multi fidelity Kriging by assuming correlation/dependence between fidelity levels?
#   - can we make accurate predictions about unseen higher fidelity levels based upon this assumption?
#   - can we avoid sampling the most expensive high fidelity levels based upon this assumption in a efficient global optimization setting?


"""
A 'solver' with the property of grid convergence; i.e. a converging function.
We could use it as well as modifier for transformed MF testcases.
"""
import matplotlib.pyplot as plt
import numpy as np

from sampling.initial_sampling import get_doe
from sampling.solvers.solver import get_solver
from sampling.infill.infill_criteria import EI
from models.kriging.method.OK import *


class Setup:
    """Dummy class"""
    pass


def Kriging(X_l, d_l, tuning=False, hps=None, train=True):
    """
    Setup a Kriging model, either for the discrepancy or a level
    """
    setup = Setup()
    setup.d = 1
    setup.kernel = "kriging"

    ok = OrdinaryKriging(setup)

    if tuning == False:
        if hps is None:
            raise NameError(
                "hps is not defined! When tuning=False, hps must be defined"
            )
        ok.hps = hps

    if train:
        ok.train(X_l, d_l, tuning)

    return ok


def Kriging_unknown(Z_k, d_k, d_pred, X, x_b, X_unique):
    """
    Scale the multiplicative d_pred using relative gains between points at x_b and x_other
    i.e. predict based on dependancy between levels!
    This approach is exact if the previous 2 levels (are sampled/exact and) are a linear transformation of the one we predict!

    @param Z_k: list containing all Kriging models of known levels
    @param d_k: list containing all Kriging models of known discrepancies
    @param d_pred: discrepancy between the first sampled point at the new level
     and the best point(s) of the current level, sampled at the same location.
     we use this discrepancy as a predictor for the rest of the unknown level.
     Can be an array.
    @param x_b: (best) location at current known top-level, i.e. location of d_pred. Can be an array.
    @param  X_unique: locations of points we want to predict on the new level
     (these locations will have a discrete sample at some level, i.e. all locations x that have been sampled at some time)
    @return discrete discrepancy funciton (np.array) between last known and new unknown level
    """

    Z0_k = Z_k[-2]
    Z1_k = Z_k[-1]

    Z0_b = Z0_k.predict(x_b)[0]  # this might be a Kriged value
    Z1_b = Z1_k.predict(x_b)[0]  # this is a known point only at the first step

    Z0 = Z0_k.predict(X_unique)[0]  # returns either sampled or Kriged values
    Z1 = Z1_k.predict(X_unique)[0]  # returns either sampled or Kriged values

    t = Z1_b * Z1
    d1 = (t - Z0 * Z1_b) / (t - Z0_b * Z1) * (d_pred - 1) + 1
    return d1


def Kriging_unknown_2(Z_k, z_pred, X, x_b, X_unique):

    Z0_k = Z_k[-2]
    Z1_k = Z_k[-1]

    # do calculations regarding the known point
    Z0_b, S0_b = Z0_k.predict(x_b)
    Z1_b, S1_b = Z1_k.predict(x_b)
    S0_b, S1_b = np.sqrt(S0_b), np.sqrt(S1_b)

    def func(lamb):
        """get expectation of the function according to
        integration of standard normal gaussian function lamb"""
        c1 = z_pred - Z1_b
        c2 = Z1_b - Z0_b
        c3 = S1_b - S0_b
        b1 = (c1 - lamb * S1_b) / (c2 + lamb * c3)

        # scipy.stats.norm.pdf(lamb) == np.exp(-x**2 / 2) / np.sqrt(2*np.pi)
        return b1 * np.exp(-(x ** 2) / 2) / np.sqrt(2 * np.pi)

    lambs = np.arange(-5, 5, 0.01)  # 1000 steps
    y = func(lambs)
    Exp_b1 = np.trapz(y, lambs, axis=1)  # TODO check of axis correct is

    # do the (corrected) prediction
    Z0_p, s0_p = Z0_k.predict(X_unique)
    Z1_p, s1_p = Z1_k.predict(X_unique)
    s0_p, s1_p = np.sqrt(s0_p), np.sqrt(s1_p)

    z2_p = Exp_b1 * (z1_p - z0_p) + z1_p
    s2_p = Exp_b1 * (s1_p - s0_p) + s1_p

    return z2_p, s2_p ** 2


def weighted_discrepancy_prediction(Z_k, d_k, D_pred, X, X_unique):
    """
    Function that weights the results of the function 'Kriging_unknown' for multiple samples at the (partly) unknown level
     based on spatial distance in a similar fashion to Kriging.
     In case convergence is linear over levels and location, the discrepancy prediction is exact. In case convergence is not linear,
     we could under or overestimate the discrepancy at other locations, something we could only solve by sampling and somehow weighing the solutions.

    @param Z_k, d_k: Kriging models of the levels and discrepancy.
    @param D_pred: The discrete discrepancy predictions, only at X_s. For instance, for 2 samples at level l+1 we have D.shape = (2, X_unique.shape[0], 1).
     The 1 is a bit weird, but originates from X having multiple dimensions, and d should be compatible.
    @param X: list of all level`s sample locations, includes:
           X_s = X[-1]: Locations at which we have sampled at the new level, and at which the discrepancy predictions have been formulated.
    @param X_unique: all unique and previously sampled locations at which we have defined a predictive discrepancy, based on the points in X_s
    """
    # The problem of converting this to Kriging is that we do not have a single point with a single value, but
    # a single point that instigates multiple values across the domain. This means we need beforehand, for each point in X_unique, weigh each contribution
    # in D / the points in X_s, i.e. construct a correlation matrix. Ofcourse, we already have a correlation matrix with (approximately tuned) hyperparameters,
    # known from our Kriging model!

    # So, we can
    # 1) use the same Kriging correlation matrix method,
    # 2) and then scale its rows to summarize to 1; retrieving a coefficient c_{i,j} for each combination.
    # 3) Then for each point in X_unique, we can retrieve the contribution of each X_s and summate.
    # 4) We now have per-point values, which we can Krige again!
    # Both times we 'Krige' over X_unique, so we can use the same hyperparameters.

    # TODO what is interesting in the weighing, is that the first discrepancy d_x0 is exact,
    # but each later sample might be relative to a Kriged result, thus including uncertainty.

    hps = d_k[-1].hps
    X_s = correct_formatX(X[-1])
    X_unique = correct_formatX(X_unique)

    # recalculate all predictive discrepency arrays
    # TODO vectorize, but no priority, non-vectorized is better for development
    D = []
    for i in range(X_s.shape[0]):
        D.append(Kriging_unknown(Z_k, d_k, D_pred[i], X, X_s[i], X_unique))
    D = np.array(D)

    # choice of weighting function
    km = Kriging(X_s, None, hps=hps, train=False)
    c = km.corr(X_s, X_unique, *hps)

    # NOTE by scaling, values of D_pred are not necessarily seen back exact in the returned array
    #      and thus the result is not exact anymore at sampled locations!! Therefore this fix.
    # if we dont do this, EI does not work!!
    # if we do this, we might get very spurious results.... -> we require regression!
    mask = c == 1.0
    idx = mask.any(axis=0)
    c[:, idx] = 0
    c += mask
    c_scaled = np.divide(c, np.sum(c, axis=0))  # sums to 1 over axis=1

    return np.sum(np.multiply(D, c_scaled), axis=0)


def return_unique(X):
    # # https://www.peterbe.com/plog/fastest-way-to-uniquify-a-list-in-python-3.6
    seq = [item for sublist in X for item in sublist]
    return np.array(list(dict.fromkeys(seq)))
    # return np.concatenate(X).ravel()


