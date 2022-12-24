import numpy as np


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
    for i, K_l in enumerate(K_mf):
        RMSE = RMSE_norm(Z_truth, K_l.predict(X)[0])
        print("RMSE of level {} = {:.4f} %".format(i,RMSE*100))

