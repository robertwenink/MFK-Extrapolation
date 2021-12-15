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


def Kriging_unknown_z(x_b, X_unique, z_pred, Z_k):

    Z0_k = Z_k[-2]
    Z1_k = Z_k[-1]

    # get values corresponding to x_b of previous 2 levels
    z0_b, s0_b = Z0_k.predict(x_b)
    z1_b, s1_b = Z1_k.predict(x_b)
    s0_b, s1_b = np.sqrt(s0_b), np.sqrt(s1_b)

    # get all other values of previous 2 levels
    Z0, S0 = Z0_k.predict(X_unique)
    Z1, S1 = Z1_k.predict(X_unique)
    S0, S1 = np.sqrt(S0), np.sqrt(S1)

    def func(lamb):
        """get expectation of the function according to
        integration of standard normal gaussian function lamb"""
        c1 = z_pred - z1_b
        c2 = z1_b - z0_b
        c3 = s1_b - s0_b
        b1 = (c1 - lamb * s1_b) / (c2 + lamb * c3)

        # scipy.stats.norm.pdf(lamb) == np.exp(-x**2 / 2) / np.sqrt(2*np.pi)
        return b1 * np.exp(-(lamb ** 2) / 2) / np.sqrt(2 * np.pi)

    # 1000 steps; evaluates norm to 0.9999994113250346
    lambs = np.arange(-5, 5, 0.01)

    # evaluate expectation
    y = func(lambs)
    Exp_b1 = np.trapz(y, lambs, axis=0)

    # retrieve the (corrected) prediction + std
    Z2_p = Exp_b1 * (Z1 - Z0) + Z1
    S2_p = Exp_b1 * (S1 - S0) + S1

    return Z2_p, S2_p ** 2


def weighted_prediction(X, X_unique, Z, Z_k):
    """
    Function that weights the results of the function 'Kriging_unknown' for multiple samples
    at the (partly) unknown level based on spatial distance in a similar fashion to Kriging.
    In case convergence is linear over levels and location, the discrepancy prediction is exact.
    In case convergence is not linear, we get discrepancies,
     something we try to solve by sampling and weighing the solutions.

    @param X: list of all level`s sample locations, includes:
           X_s = X[-1]: Locations at which we have sampled at the new level.
    @param X_unique: all unique previously sampled locations in X.
    @param Z: list of all level`s sample locations, including the new level.
    @param Z_k: Kriging models of the levels, should only contain known levels.
    """

    X_s = correct_formatX(X[-1])
    Z_s = Z[-1]
    X_unique = correct_formatX(X_unique)

    D, D_mse = [], []
    for i in range(X_s.shape[0]):
        Z_p, mse_p = Kriging_unknown_z(X_s[i], X_unique, Z_s[i], Z_k)
        D.append(Z_p)
        D_mse.append(mse_p)
    D, D_mse = np.array(D), np.array(D_mse)

    # choice of weighting function
    hps = Z_k[-1].hps
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

    # TODO extend c scaled
    return np.sum(np.multiply(D, c_scaled), axis=0), np.sum(np.multiply(D_mse, c_scaled), axis=0)


def return_unique(X):
    # # https://www.peterbe.com/plog/fastest-way-to-uniquify-a-list-in-python-3.6
    seq = [item for sublist in X for item in sublist]
    return np.array(list(dict.fromkeys(seq)))
