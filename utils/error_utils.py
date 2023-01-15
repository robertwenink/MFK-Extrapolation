import numpy as np
from utils.selection_utils import isin_indices

def RMSE(Z, Z_predict):
    """Test the root mean squared error of Z compared to the prediction"""
    assert len(Z) == len(Z_predict), "length of prediction and sample sets is not equal!"
    RMSE = np.sqrt(1 / Z.shape[0] * np.sum((Z - Z_predict) ** 2))
    return RMSE


def RMSE_norm(Z, Z_predict):
    return RMSE(Z, Z_predict) / np.mean(abs(Z))


def MAE_norm(Z, Z_predict):
    """Calculates the normalized mean absolute error"""
    assert Z == Z_predict, "length of prediction and sample sets is not equal!"
    MAE = np.sum(Z - Z_predict) / np.sum(Z)

    return MAE


def RMSE_norm_MF(X, Z_truth, mf_model, no_samples = False):
    """
    Calculate multifidelity RMSE.
    @param no_samples (bool): only calculate RMSE based on predicted points i.e. points not sampled at the highest level.
    """
    # RMSE zou eigenlijk alleen moeten worden bepaald op plekken waar er GEEN samples op level 2 zijn, alleen predicted points!
    # anders gaat de RMSE automatisch omlaag, wat misleidend is.
    RMSE = []
    for i, K_l in enumerate(mf_model.K_mf):
        if no_samples:
            inds = isin_indices(X,mf_model.X_mf[-1],inversed=True)
            RMSE.append("{:.2f} %".format(RMSE_norm(Z_truth[inds], K_l.predict(X[inds])[0])*100))
        else:
            RMSE.append("{:.2f} %".format(RMSE_norm(Z_truth, K_l.predict(X)[0])*100))
    
    return RMSE

def RMSE_focussed(X, Z_truth, mf_model, focus_perc, no_samples = False):
    """
    Version of 'RMSE_norm_MF' where the RMSE is calculated based on 
    specific areas that are within focus_perc % of the optimum value.
    """
    pass

