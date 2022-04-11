"""
A 'solver' with the property of grid convergence; i.e. a converging function.
We could use it as well as modifier for transformed MF testcases.
"""
import matplotlib.pyplot as plt
import numpy as np
from numba import vectorize, float64

from sampling.DoE import get_doe
from sampling.infill.infill_criteria import EI
from models.kriging.method.OK import Kriging
from utils.data_utils import return_unique, correct_formatX


def Kriging_unknown_z(x_b, X_unique, z_pred, Z_k):
    """
    Assume the relative differences at one location are representative for the
     differences elsewhere in the domain, if we scale according to the same convergence.
     Use this to predict a location, provide an correction/expectation based on chained variances.

    @param x_b: a (presumably best) location at which we have/will sample(d) the new level.
    @param X_unique: all unique previously sampled locations in X.
    @param z_pred: new level`s sampled locations.
    @param Z_k: Kriging models of the levels, should only contain known levels.
    @return extrapolated predictions Z2_p, corresponding mse S2_p (squared!)
    """

    Z0_k = Z_k[-2]
    Z1_k = Z_k[-1]

    # get values corresponding to x_b of previous 2 levels
    z0_b, s0_b = Z0_k.predict(x_b)
    z1_b, s1_b = Z1_k.predict(x_b)
    s0_b, s1_b = np.sqrt(s0_b), np.sqrt(s1_b)

    # this is only the base noise,actual noise can deviate due to correlations
    # TODO unused yet
    S0_noise = Z0_k.sigma_hat * Z0_k.hps[-1]
    S1_noise = Z1_k.sigma_hat * Z1_k.hps[-1]

    # get all other values of previous 2 levels
    Z0, S0 = Z0_k.predict(X_unique)
    Z1, S1 = Z1_k.predict(X_unique)
    S0, S1 = np.sqrt(S0), np.sqrt(S1)

    def exp_f(lamb1, lamb2):
        """get expectation of the function according to
        integration of standard normal gaussian function lamb"""
        c1 = z_pred - z1_b
        c2 = z1_b - z0_b
        f = (c1 - lamb1 * s1_b) / (
            c2 + lamb1 * s1_b + lamb2 * s0_b + np.finfo(np.float64).eps
        )

        # scipy.stats.norm.pdf(lamb) == np.exp(-x**2 / 2) / np.sqrt(2*np.pi)
        # bivariate : https://mathworld.wolfram.com/BivariateNormalDistribution.html
        return f * np.exp(-(lamb1 ** 2 + lamb2 ** 2) / 2) / (2 * np.pi), f

    # 1000 steps; evaluates norm to 0.9999994113250346
    lambs = np.arange(-5, 5, 0.01)
    lamb1, lamb2 = np.meshgrid(lambs, lambs)

    # evaluate expectation
    Y, f = exp_f(lamb1, lamb2)
    Ef = np.trapz(
        np.trapz(Y, lambs, axis=0), lambs, axis=0
    )  # axis=0 because the last axis (default) are the dimensions.

    def var_f(lamb1, lamb2):
        # Var X = E[(X-E[X])**2]
        # down below: random variable - exp random variable, squared, times the pdf of rho (N(0,1));
        # we integrate this function and retrieve the expectation of this expression, i.e. the variance.
        return (f - Ef) ** 2 * np.exp(-(lamb1 ** 2 + lamb2 ** 2) / 2) / (2 * np.pi)

    Y = var_f(lamb1, lamb2)
    Sf = np.sqrt(
        np.trapz(np.trapz(Y, lambs, axis=0), lambs, axis=0)
    )  # NOTE not a normal gaussian variance, but we use it as such

    # retrieve the (corrected) prediction + std
    # NOTE this does not retrieve z_pred at x_b if sampled at kriged locations.
    Z2_p = Ef * (Z1 - Z0) + Z1

    # NOTE (Eb1+s_b1)*(E[Z1-Z0]+s_[Z1-Z0]), E[Z1-Z0] is just the result of the Kriging
    # with s_[Z1-Z0] approximated as S1 + S0 for a pessimistic always oppositely moving case
    t0 = np.min([(S0 - S1) * Ef, np.zeros_like(S0)], axis=0)
    t1 = S1 * Ef + S1
    t2 = abs(Z1 - Z0) * Sf
    t = t0 + t1 + t2

    S2_p = S1 + abs((S1 - S0) * Ef) + abs(Z1 - Z0) * Sf

    # TODO get the max uncertainty contribution at an hifi unsampled location`s point!
    # S1 + abs((S1 - S0) * Ef) should be compared to the total KRIGED variance.
    # i.e. above might not be equal to that.
    # NOTE new idea for checking assumption? i.e. if estimated variance deviates too much from kriged variance?
    #
    # we might include as well the expected improvement, i.e. if the variance is reduced,
    # do we expect to still see an ei?
    # Further, even if S1 contributes most, if this is the case, we might only want to sample S0.

    # get index in X_unique of x_b
    ind = np.all(X_unique == x_b, axis=1)

    # set Z2_p to z_pred at that index
    Z2_p[ind] = z_pred

    # set S2_p to 0 at that index (we will later include variance from noise.)
    S2_p[ind] = 0

    return Z2_p, S2_p ** 2, Sf ** 2


def weighted_prediction(setup, X_s, X_unique, Z_s, Z_k):
    """
    Function that weights the results of the function 'Kriging_unknown' for multiple samples
    at the (partly) unknown level.

    In case convergence is linear over levels and location,
     the "Kriging_unknown" prediction is exact if we have exact data.
    In case convergence is not linear, we get discrepancies;
     something we try to solve by sampling and weighing the solutions.

    @param X_s = X[-1]: Locations at which we have sampled at the new level.
    @param X_unique: all unique previously sampled locations in X.
    @param Z_s: list hifi level sample locations
    @param Z_k: Kriging models of the levels, should only contain *known* levels.
    """

    X_s = correct_formatX(X_s, setup.d)

    if len(Z_s) == 1:
        Z_pred, mse_pred, _ = Kriging_unknown_z(X_s, X_unique, Z_s, Z_k)
        return Z_pred, mse_pred

    " Collect new results "
    # NOTE if not in X_unique, we could just add a 0 to all previous,
    # might be faster but more edge-cases
    D, D_mse, D_Sf = [], [], []
    for i in range(X_s.shape[0]):
        Z_p, mse_p, Sf = Kriging_unknown_z(X_s[i], X_unique, Z_s[i], Z_k)
        D.append(Z_p), D_mse.append(mse_p), D_Sf.append(Sf)
    D, D_mse, D_Sf = np.array(D), np.array(D_mse), np.array(D_Sf)

    " Weighing "
    # 1) distance based: take the (tuned) Kriging correlation function
    # NOTE one would say retrain/ retune on only the sampled Z_s (as little as 2 samples), to get the best/most stable weighing.
    # However, we cannot tune on such little data, the actual scalings theta are best represented by the (densely sampled) previous level.
    km = Kriging(setup, X_s, Z_s, hps_init=Z_k[-1].hps, train=False)
    c = km.corr(X_s, correct_formatX(X_unique, setup.d), km.hps)
 
    mask = c == 1.0  # then sampled point
    idx = mask.any(axis=0)  # get columns with sampled points

    # 2) variance based: predictions based on fractions without large variances Sf involved are more reliable
    #    we want to keep the correlation/weighing the same if there is no variance,
    #    and otherwise reduce it.
    #    We could simply do: sigma += 1 and divide.
    #    However, sigma is dependend on scale of Z, so we should better use e^-sigma.
    #    This decreases distance based influence if sigma > 0.
    #    We take the variance of the fraction, S_f

    mult = np.exp(-D_Sf)
    c = (c.T * mult).T

    " Scale to sum to 1; correct for sampled locations "
    # Contributions coefficients should sum up to 1.
    # Furthermore, sampled locations should be the only contribution for themselves.
    #  if we dont do this, EI does not work!
    #  however, we might get very spurious results -> we require regression!
    c[:, idx] = 0  # set to zero
    c = c + mask  # retrieve samples exactly

    # Scale for Z
    c_z = np.divide(
        c, np.sum(c, axis=0) + np.finfo(np.float64).eps
    )  # to avoid division by (near) zero for higher d
    Z_pred = np.sum(np.multiply(D, c_z), axis=0)

    # Scale for mse
    # NOTE noise is added to the samples regardless of what this function returns
    c_mse = c_z - mask  # always 0 variance for samples
    mse_pred = np.sum(np.multiply(D_mse, c_mse), axis=0)

    return Z_pred, mse_pred
