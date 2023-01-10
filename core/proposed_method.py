"""
A 'solver' with the property of grid convergence; i.e. a converging function.
We could use it as well as modifier for transformed MF testcases.
"""
import numpy as np

from core.kriging.mf_kriging import ProposedMultiFidelityKriging
from utils.formatting_utils import correct_formatX

def Kriging_unknown_z(x_b, X_unique, z_pred, K_mf):
    """
    Assume the relative differences at one location are representative for the
     differences elsewhere in the domain, if we scale according to the same convergence.
     Use this to predict a location, provide an correction/expectation based on chained variances.

    @param x_b: a (presumably best) location at which we have/will sample(d) the new level.
    @param X_unique: all unique previously sampled locations in X.
    @param z_pred: new level`s sampled location value(s).
    @param K_mf: Kriging models of the levels, should only contain known levels.
    @return extrapolated predictions Z2_p, corresponding mse S2_p (squared!)
    """

    K0 = K_mf[0]
    K1 = K_mf[1]

    # get values corresponding to x_b of previous 2 levels
    z0_b, s0_b = K0.predict(x_b)
    z1_b, s1_b = K1.predict(x_b)
    s0_b, s1_b = np.sqrt(s0_b), np.sqrt(s1_b)

    # get all other values of previous 2 levels
    Z0, S0 = K0.predict(X_unique)
    Z1, S1 = K1.predict(X_unique)
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

    # to find precision, use Y = np.exp(-(lamb1 ** 2 + lamb2 ** 2) / 2) / (2 * np.pi) and use the double trapz part
    # 1000 steps; evaluates norm to 0.9999994113250346 ( met 1000x1000 evaluations!!)
    # 200 steps to 0.9999986775391365
    # 100 to       0.9999984361381121
    # 20 to        0.9999895499778308
    lambs = np.arange(-5, 5, 0.1)
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
    
    # scaling according to reliability of Ef based on ratio Sf/Ef, and correlation with surrounding samples
    # important for: suppose the scenario where a sample at L1 lies between samples of L2. 
    # Then it would not be reasonable to discard all weighted information and just take the L1 value even if Sf/Ef is high.

    # so:   if high Sf/Ef -> use correlation (sc = correlation)
    #       if low Sf/Ef -> use prediction (sc = 1)
    
    # TODO this is the point at which we can integrate other MF methods!!!!
    # e.g. this is exactly what other Mf methods do: combine multiple fidelities and weigh their contributions automatically.
    # then, we can use Sf not to switch back to L1, but to switch back to the other model if our prediction turns out to be unreliable.
    # bcs there is no hi-fi info available for the other MF method I presume it will use L1 values further away from our L2 values, just like we want!
    # so TODO now it is time to get to the MF-EGO approach of Meliani.

    corr = K1.corr(correct_formatX(x_b,X_unique.shape[1]), X_unique, K_mf[-1].hps).flatten()
    lin = np.exp(-1/2 * abs(Sf/Ef)) # = 1 for Sf = 0, e.g. when the prediction is perfectly reliable (does not say anything about non-linearity); 

    w = 1 * lin + corr * (1 - lin) 
    w = lin
    # * np.exp(-abs(Sf/Ef))

    # retrieve the (corrected) prediction + std
    # NOTE this does not retrieve z_pred at x_b if sampled at kriged locations.
    Z1_alt = Z1 # TODO should become equal to external method
    Z2_p = w * (Ef * (Z1 - Z0) + Z1) + (1-w) * Z1_alt # TODO diminish the first term if Sf is high! 

    # NOTE (Eb1+s_b1)*(E[Z1-Z0]+s_[Z1-Z0]), E[Z1-Z0] is just the result of the Kriging
    # with s_[Z1-Z0] approximated as S1 + S0 for a pessimistic always oppositely moving case
    S1_alt = S1 # TODO should become equal to external method
    S2_p = w * (S1 + abs((S1 - S0) * Ef) + abs(Z1 - Z0) * Sf ) + (1 - w) * S1_alt

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

    # not Sf/Ef bcs for large Ef, Sf will already automatically be smaller by calculation!!
    return Z2_p, S2_p ** 2, Sf ** 2, Ef

# TODO 0.46 seconden per call! zware functie! gemiddeld 4 Kriging+unkown_z calls per call, gemiddelde 0.119s
def weighted_prediction(mf_model : ProposedMultiFidelityKriging, X_s = [], Z_s = [], assign : bool = True, X_test = np.array([])):
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
    @param K_mf: Kriging models of the levels, should only contain *known* levels.
    @param assign: Assign the results Z_pred and mse_pred as property to the mf_model.
    """

    # data pre-setting
    X_unique, K_mf =  mf_model.X_unique, mf_model.K_mf
    if X_test.size != 0:
        X_unique = X_test

    if not (np.any(X_s) and np.any(Z_s)):
        X_s, Z_s = mf_model.X_mf[-1],  mf_model.Z_mf[-1]
    
    # In case there is only one sample, nothing to weigh
    if len(Z_s) == 1:
        Z_pred, mse_pred, _, Ef_weighed = Kriging_unknown_z(X_s, X_unique, Z_s, K_mf)
    else:
        " Collect new results "
        # NOTE if not in X_unique, we could just add a 0 to all previous,
        # might be faster but more edge-cases
        n = X_s.shape[0]
        D, D_mse, D_Sf, D_Ef = [], [], [], []
        for i in range(X_s.shape[0]):
            Z_p, mse_p, Sf, Ef = Kriging_unknown_z(X_s[i], X_unique, Z_s[i], K_mf)
            D.append(Z_p), D_mse.append(mse_p), D_Sf.append(Sf), D_Ef.append(Ef) # type: ignore
        D, D_mse, D_Sf, D_Ef = np.array(D), np.array(D_mse), np.array(D_Sf), np.array(D_Ef)

        " Weighing "
        # 1) distance based: take the (tuned) Kriging correlation function
        # NOTE one would say retrain/ retune on only the sampled Z_s (as little as 2 samples), to get the best/most stable weighing.
        # However, we cannot tune on such little data, the actual scalings theta are best represented by the (densely sampled) previous level.
        c = K_mf[-1].corr(X_s, X_unique, K_mf[-1].hps)

        mask = np.isclose(c,1.0)  # then sampled point
        idx = mask.any(axis=0)  # get columns with sampled points

        # 2) variance based: predictions based on fractions without large variances Sf involved are more reliable
        #    we want to keep the correlation/weighing the same if there is no variance,
        #    and otherwise reduce it.
        #    We could simply do: sigma += 1 and divide.
        #    However, sigma is dependend on scale of Z, so we should better use e^-sigma.
        #    This decreases distance based influence if sigma > 0.
        #    We take the variance of the fraction, S_f
    
        mult = np.exp(-D_Sf/np.mean(D_Sf))
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

        Ef_weighed = np.dot(D_Ef, c_z)


    # assign the found values to the mf_model, in order to be easily retrieved.
    if assign:
        mf_model.Z_pred = Z_pred
        mf_model.mse_pred = mse_pred

    return Z_pred, mse_pred, Ef_weighed
