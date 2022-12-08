import numpy as np
from core.proposed_method import *

def overlap_check(s1, e1, s2, e2):
    """Does the range (s1, e1) overlap with (s2, e2)?"""
    return np.logical_and(e1 >= s2, e2 >= s1)


def overlap_amount(s1, e1, s2, e2):
    interval1 = e1 - s1
    interval2 = e2 - s2

    def helper(s1, e1, s2, e2):
        # vectorized if elif statement
        t1 = (s2 >= s1) * np.min([e1 - s2, e2 - s2], axis=0)
        t2 = (e2 <= e1) * (s2 < s1) * np.min([e2 - s1, e2 - s2], axis=0)
        return t1 + t2

    check = overlap_check(s1, e1, s2, e2)
    overlap = helper(s1, e1, s2, e2)

    # if <=0 not necessary no overlap, but one of the intervals might be 0
    overlap[overlap <= 0] = check[overlap <= 0]

    # the minimum interval size determines the score
    intmin = np.min([interval1, interval2], axis=0)
    intmin[intmin == 0] = 1  # unless one of them is zero!

    overlap_fraction = overlap / intmin
    overlap_fraction[overlap_fraction > 1] += check[overlap_fraction > 1]

    return overlap_fraction


def check_linearity(setup, X, X_unique, Z, K_mf, pp, L):
    # TODO deze manier werkt alleen als er noise aanwezig is;
    # als er geen noise aanwezig is moet de aanname PERFECT kloppen, wat nooit zo zal zijn
    print("### Checking linearity")
    assert Z[-1].shape[0] >= 2, "\nMinimum of 2 samples at highest level required"
    

    # need the list without X[-1] too

    Z_pred_full, mse_pred_full = weighted_prediction(setup, X[-1], X_unique, Z[-1], K_mf)
    K_mf_full = Kriging(setup, X_unique, Z_pred_full, hps_init=K_mf[-1].hps, hps_noise_ub = True, tune = True, R_diagonal=mse_pred_full / K_mf[-1].sigma_hat)
    K_mf_full.reinterpolate()

    Z_interval = np.max(Z_pred_full) - np.min(Z_pred_full)
    deviation_allowed = .1 * Z_interval # 5 percent noise/deviation allowed in absolute value

    start_full = Z_pred_full - 5 * mse_pred_full
    end_full = Z_pred_full + 5 * mse_pred_full

    # TODO extremely ugly
    inds = [
        i
        for sublist in [np.where(X[-1] == item)[0].tolist() for item in X_unique]
        for i in sublist
    ]

    linear = True
    nr_samples = Z[-1].shape[0]
    pp.draw_current_levels(X, Z, [*K_mf, K_mf_full], X_unique, L)
    for i_exclude in range(nr_samples):
        i_include = np.delete(inds, i_exclude)

        Z_pred_partial, mse_pred_partial = weighted_prediction(
            setup,
            np.delete(X[-1], i_exclude, 0),
            X_unique,
            np.delete(Z[-1], i_exclude, 0),
            K_mf,
        )

        # TODO fault: mse_pred is not the prediction mse!!! maakt dit uit?
        start_partial = Z_pred_partial - 4 * mse_pred_partial
        end_partial = Z_pred_partial + 4 * mse_pred_partial

        K_mf_partial = Kriging(
            setup, X_unique, Z_pred_partial, hps_init=K_mf_full.hps, hps_noise_ub = True, tune = False, R_diagonal=mse_pred_partial / K_mf_full.sigma_hat
        )
        K_mf_partial.reinterpolate()

        # 0.5 corresponds with a fraction representing one side of the 100% confidence interval
        # i_include are sampeled points, so we exclude them ...
        overlap = np.delete(
            overlap_amount(start_full, end_full, start_partial, end_partial), i_include, 0
        )
        difference = np.abs(Z_pred_full-Z_pred_partial)
        ov = np.any(overlap < 0.25)
        if ov:
            print("Too little overlap!")
        diff = np.any(difference > deviation_allowed)
        if diff:
            print("Too large absolute difference!")

        if ov and diff:
            linear = False
            print("NOT LINEAR ENOUGH")
        
        # draw the result
        pp.draw_current_levels(X, Z, [*Z_k, Z_k_partial, Z_k_full], X_unique, L)

    return linear
