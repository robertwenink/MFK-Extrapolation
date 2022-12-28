# pyright: reportGeneralTypeIssues=false
import numpy as np
import matplotlib.pyplot as plt

from core.proposed_method import *
from core.kriging.mf_kriging import ProposedMultiFidelityKriging
from postprocessing.plotting import Plotting

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


def check_linearity(mf_model : ProposedMultiFidelityKriging, pp : Plotting):
    """
    There should already be a high-fidelity / truth Kriging model as part of mf_model.
    # TODO zonder noise is de (geplotte) oplossing toch anders, hoe kan dat? Zou nml perfect moeten zij.
    """
    # TODO deze manier werkt alleen als er noise aanwezig is;
    # als er geen noise aanwezig is moet de aanname PERFECT kloppen, wat nooit zo zal zijn
    print("Checking linearity ...",end='\r')
    assert mf_model.Z_mf[-1].shape[0] >= 2, "\nMinimum of 2 samples at highest level required"
    

    # need the list without X[-1] too

    Z_interval = np.max(mf_model.Z_pred) - np.min(mf_model.Z_pred)
    deviation_allowed = .1 * Z_interval # 5 percent noise/deviation allowed in absolute value

    start_full = mf_model.Z_pred - 5 * mf_model.mse_pred
    end_full = mf_model.Z_pred + 5 * mf_model.mse_pred

    # TODO extremely ugly ( but correct )
    inds = [
        i
        for sublist in [np.where(mf_model.X_mf[-1] == item)[0].tolist() for item in mf_model.X_unique]
        for i in sublist
    ]

    linear = True
    nr_samples = mf_model.Z_mf[-1].shape[0]
    
    for i_exclude in range(nr_samples):
        i_include = np.delete(inds, i_exclude)
        X_s = np.delete(mf_model.X_mf[-1], i_exclude, 0)
        Z_s = np.delete(mf_model.Z_mf[-1], i_exclude, 0)
        Z_pred_partial, mse_pred_partial = weighted_prediction(
            mf_model,
            X_s = X_s,
            Z_s = Z_s,
            assign=False
        )

        # TODO fault: mse_pred is not the prediction mse!!! maakt dit uit?
        start_partial = Z_pred_partial - 4 * mse_pred_partial
        end_partial = Z_pred_partial + 4 * mse_pred_partial

        K_mf_partial = mf_model.create_level(mf_model.X_unique, Z_pred_partial, tune = True, name = "Linearity Check {}".format(i_exclude), append = False, hps_noise_ub = True, R_diagonal=mf_model.mse_pred / mf_model.K_mf[-1].sigma_hat)
        K_mf_partial.reinterpolate()
        K_mf_partial.X_s = X_s
        K_mf_partial.Z_s = Z_s

        # 0.5 corresponds with a fraction representing one side of the 100% confidence interval
        # i_include are sampeled points, so we exclude them ...
        overlap = np.delete(
            overlap_amount(start_full, end_full, start_partial, end_partial), i_include, 0
        )
        difference = np.abs(mf_model.Z_pred-Z_pred_partial)
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
        # TODO een optie opnemen om een extra level weer te geven zoals de truth.
        # K_mf_alt = [*Z_k, Z_k_partial, Z_k_full] # zonder *Z_k wss
        pp.draw_current_levels(mf_model, K_mf_extra=K_mf_partial)
        plt.pause(1)

    return linear
