import numpy as np
from beautifultable import BeautifulTable

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


def RMSE_norm_MF(X, Z_truth, K_mf):
    #TODO RMSE zou eigenlijk alleen moeten worden bepaald op plekken waar er GEEN samples op level 2 zijn, alleen predicted points!
    # anders gaat de RMSE automatisch omlaag, wat misleidend is.
    levels = []
    RMSE = []
    for i, K_l in enumerate(K_mf):
        levels.append("Level " + str(i))
        RMSE.append("{:.2f} %".format(RMSE_norm(Z_truth, K_l.predict(X)[0])*100))
    
    return RMSE