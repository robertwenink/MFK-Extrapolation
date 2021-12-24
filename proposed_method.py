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


def Kriging(X_l, d_l, tuning=False, R_diagonal=None, hps=None, train=True):
    """
    Setup a Kriging model, either for the discrepancy or a level
    """
    setup = Setup()
    setup.d = 1
    setup.kernel = "kriging"
    setup.regression = False

    ok = OrdinaryKriging(setup)

    if tuning == False:
        if hps is None:
            raise NameError(
                "hps is not defined! When tuning=False, hps must be defined"
            )
        ok.hps = hps

    if train:
        ok.train(X_l, d_l, tuning, R_diagonal)

    return ok


def Kriging_unknown_z(x_b, X_unique, z_pred, Z_k):
    """
    Assume the relative differences at one location are representative for the
     differences elsewhere in the domain, if we scale according to the same convergence.
     Use this to predict a location, provide an correction/expectation based on chained variances.

    @param x_b: a (presumably best) location at which we have sampled the new level.
    @param X_unique: all unique previously sampled locations in X.
    @param z_pred: new level`s sampled locations.
    @param Z_k: Kriging models of the levels, should only contain known levels.
    """

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

    def exp_b1(lamb):
        """get expectation of the function according to
        integration of standard normal gaussian function lamb"""
        c1 = z_pred - z1_b
        c2 = z1_b - z0_b
        c3 = (s1_b + s0_b) # NOTE should be s1_b - s0_b if they are assumed to move in the same direction
        b1 = (c1 - lamb * s1_b) / (c2 + lamb * c3 + np.finfo(np.float32).eps)

        # scipy.stats.norm.pdf(lamb) == np.exp(-x**2 / 2) / np.sqrt(2*np.pi)
        return b1 * np.exp(-(lamb ** 2) / 2) / np.sqrt(2 * np.pi), b1

    # 1000 steps; evaluates norm to 0.9999994113250346
    lambs = np.arange(-5, 5, 0.01)

    # evaluate expectation
    y, b1 = exp_b1(lambs)
    Exp_b1 = np.trapz(y, lambs, axis=0)

    def var_b1(lamb):
        # E[(X-E[X])]
        return (b1 - Exp_b1) ** 2 * np.exp(-(lamb ** 2) / 2) / np.sqrt(2 * np.pi)

    y2 = var_b1(lambs)
    Var_b1 = np.sqrt(np.trapz(y2, lambs, axis=0))

    # retrieve the (corrected) prediction + std
    # NOTE this does not retrieve z_pred at x_b if sampled at kriged locations.
    Z2_p = Exp_b1 * (Z1 - Z0) + Z1

    # TODO add extra term based on distance.
    # NOTE (Eb1+s_b1)*(E[Z1-Z0]+s_[Z1-Z0]), E[Z1-Z0] is just the result of the Kriging 
    # with s_[Z1-Z0] approximated as S1 + S0 for a pessimistic always oppositely moving case
    Var_b2 = (S1 + S0)
    S2_p = abs(Exp_b1) * Var_b2 + abs(Z1 - Z0) * Var_b1 + Var_b2 * Var_b1 + S1

    # get index in X_unique of x_b
    ind = X_unique == x_b

    # set Z2_p to z_pred at that index
    Z2_p[ind] = z_pred

    # set S2_p to 0 at that index
    S2_p[ind] = 0

    return Z2_p, S2_p ** 2


def weighted_prediction(X, X_unique, Z, Z_k):
    """
    Function that weights the results of the function 'Kriging_unknown' for multiple samples
    at the (partly) unknown level.

    In case convergence is linear over levels and location,
     the "Kriging_unknown" prediction is exact if we have exact data.
    In case convergence is not linear, we get discrepancies;
     something we try to solve by sampling and weighing the solutions.

    @param X: list of all level`s sample locations, includes:
           X_s = X[-1]: Locations at which we have sampled at the new level.
    @param X_unique: all unique previously sampled locations in X.
    @param Z: list of all level`s sample locations, including the new level.
    @param Z_k: Kriging models of the levels, should only contain known levels.
    """

    X_s = correct_formatX(X[-1])
    Z_s = Z[-1]

    " Collect new results "
    # NOTE if not in X_unique, we could just add a 0 to all previous,
    # might be faster but more edge-cases
    D, D_mse = [], []
    for i in range(X_s.shape[0]):
        Z_p, mse_p = Kriging_unknown_z(X_s[i], X_unique, Z_s[i], Z_k)
        D.append(Z_p), D_mse.append(mse_p)
    D, D_mse = np.array(D), np.array(D_mse)

    " Weighing "
    # 1) distance based: take the (tuned) Kriging correlation function

    hps = Z_k[-1].hps
    km = Kriging(X_s, None, hps=hps, train=False)
    c = km.corr(X_s, correct_formatX(X_unique), hps)

    # 2) variance based: predictions without variances involved are presumably more reliable
    #    we want to keep the correlation/weighing the same if there is no variance,
    #    and otherwise reduce it.
    #    We could simply do: sigma += 1 and divide.
    #    However, sigma is dependend on scale of Z, so we should better use e^-sigma.
    #    This decreases distance based influence if sigma > 0.
    #    NOTE TODO this is not (yet) a fully math-informed decision in terms of efffectiveness.

    mult = np.exp(-D_mse)
    c = c * mult

    " Scale to sum to 1; correct for sampled locations "
    # Contributions coefficients should sum up to 1.
    # Furthermore, sampled locations should be the only contribution for themselves.
    #  if we dont do this, EI does not work!
    #  however, we might get very spurious results -> we require regression!

    mask = c == 1.0  # then sampled point
    idx = mask.any(axis=0)  # get columns with sampled points
    c[:, idx] = 0  # set to zero
    c = c + mask  # retrieve samples exactly

    # Scale for Z
    c_z = np.divide(c, np.sum(c, axis=0))
    Z_pred = np.sum(np.multiply(D, c_z), axis=0)

    # Scale for mse
    c_mse = c_z - mask  # always 0 variance for samples
    mse = np.sum(np.multiply(D_mse, c_mse), axis=0)

    return Z_pred, mse


def return_unique(X):
    # # https://www.peterbe.com/plog/fastest-way-to-uniquify-a-list-in-python-3.6
    seq = [item for sublist in X for item in sublist]
    return np.array(list(dict.fromkeys(seq)))
