import numpy as np
from utils.selection_utils import isin_indices
from core.sampling.solvers.internal import TestFunction

def RMSE(Z, Z_predict):
    """Test the root mean squared error of Z compared to the prediction"""
    assert len(Z) == len(Z_predict), "length of prediction and sample sets is not equal!"
    RMSE = np.sqrt(1 / Z.shape[0] * np.sum((Z - Z_predict) ** 2)) * 100
    return RMSE


def RMSE_norm(Z, Z_predict):
    return RMSE(Z, Z_predict) / np.mean(abs(Z))


def MAE_norm(Z, Z_predict):
    """Calculates the normalized mean absolute error"""
    assert len(Z) == len(Z_predict), "length of prediction and sample sets is not equal!"
    MAE = np.sum(np.abs(Z - Z_predict)) / np.sum(Z)

    return MAE


def RMSE_norm_MF(mf_model, no_samples = False):
    """
    Calculate multifidelity RMSE.
    @param no_samples (bool): only calculate RMSE based on predicted points i.e. points not sampled at the highest level.
    """
    # RMSE zou eigenlijk alleen moeten worden bepaald op plekken waar er GEEN samples op level 2 zijn, alleen predicted points!
    # anders gaat de RMSE automatisch omlaag, wat misleidend is.
    RMSE = []
    for i, K_l in enumerate(mf_model.K_mf):
        if no_samples:
            inds = isin_indices(mf_model.X_truth,mf_model.X_mf[-1],inversed=True)
            RMSE.append(RMSE_norm(mf_model.Z_truth[inds], K_l.predict(mf_model.X_truth[inds])[0]))
        else:
            RMSE.append(RMSE_norm(mf_model.Z_truth, K_l.predict(mf_model.X_truth)[0]))
    
    return RMSE

def RMSE_focussed(mf_model, focus_perc):
    """
    Version of 'RMSE_norm_MF' where the RMSE is calculated based on 
    specific areas that are within focus_perc % of the optimum value.
    Because there could be barely any to 0 samples within these areas of interest, 
    instead the prediction of the truth surface is used.
    
    @param focus_perc (float): percentage of the datarange that we asses, above the (known) optimum value.
    """
    z_infill_truth = mf_model.K_truth.predict(mf_model.X_infill)[0]
    data_range = np.max(z_infill_truth) - np.min(z_infill_truth)

    # determining the 'optimum' baseline value
    if isinstance(mf_model.solver,TestFunction):
        # if available, use known optimum
        X_opt, Z_opt = mf_model.solver.get_optima()
        z_opt = np.min(Z_opt)
    else:
        # use the current values of the truth
        z_opt = np.min(z_infill_truth)

    mask = np.where(z_infill_truth >= z_opt + data_range * focus_perc / 100)
    
    RMSE = []
    for K_l in mf_model.K_mf:
        RMSE.append(RMSE_norm(z_infill_truth[mask], K_l.predict(mf_model.X_infill[mask])[0]))

    return RMSE

