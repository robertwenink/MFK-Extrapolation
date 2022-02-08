import numpy as np
from proposed_method import *

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


def check_linearity(setup, X, X_unique, Z, Z_k, pp):
    # TODO deze manier werkt alleen als er noise aanwezig is;
    # als er geen noise aanwezig is moet de aanname PERFECT kloppen, wat nooit zo zal zijn
    print("### Checking linearity")
    assert Z[-1].shape[0] >= 2, "\nMinimum of 2 samples at highest level required"

    # need the list without X[-1] too

    Z_pred_full, mse_pred_full = weighted_prediction(setup, X[-1], X_unique, Z[-1], Z_k)
    Z_k_full = Kriging(setup, X_unique, Z_pred_full, hps_init=Z_k[-1].hps, tune=True)

    Z_interval = np.max(Z_pred_full) - np.min(Z_pred_full)
    deviation_allowed = .1 * Z_interval # 5 percent noise/deviation allowed in absolute value

    start_full = Z_pred_full - 4 * mse_pred_full
    end_full = Z_pred_full + 4 * mse_pred_full

    # TODO extremely ugly
    inds = [
        i
        for sublist in [np.where(X[-1] == item)[0].tolist() for item in X_unique]
        for i in sublist
    ]

    linear = True
    nr_samples = Z[-1].shape[0]
    pp.draw_current_levels(X, [*Z_k, Z_k_full], X_unique)
    for i_exclude in range(nr_samples):
        i_include = np.delete(inds, i_exclude)

        Z_pred_partial, mse_pred_partial = weighted_prediction(
            setup,
            np.delete(X[-1], i_exclude),
            X_unique,
            np.delete(Z[-1], i_exclude),
            Z_k,
        )

        start_partial = Z_pred_partial - 4 * mse_pred_partial
        end_partial = Z_pred_partial + 4 * mse_pred_partial

        # draw the result
        Z_k_partial = Kriging(
            setup, X_unique, Z_pred_partial, hps_init=Z_k[-1].hps, tune=True
        )

        # 0.5 corresponds with a fraction representing one side of the 100% confidence interval
        # i_include are sampeled points, so we exclude them ...
        overlap = np.delete(
            overlap_amount(start_full, end_full, start_partial, end_partial), i_include
        )
        difference = np.abs(Z_pred_full-Z_pred_partial)
        if np.any(overlap < 0.5) and any(difference > deviation_allowed):
            linear = False
            print("NOT LINEAR ENOUGH")
        pp.draw_current_levels(X, [*Z_k, Z_k_partial, Z_k_full], X_unique)
    
    return linear
