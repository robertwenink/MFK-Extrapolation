import numpy as np


def RMSE(Z, Z_predict):
    """Test the root mean squared error of Z compared to the prediction"""
    assert len(Z) == len(Z_predict), "length of prediction and sample sets is not equal!"
    RMSE = np.sqrt(1 / Z.shape[0] * np.sum((Z - Z_predict) ** 2))
    return RMSE


def RMSE_norm(Z, Z_predict):
    return RMSE(Z, Z_predict) / np.mean(Z)


def MAE_norm(Z, Z_predict):
    """Calculates the normalized mean absolute error"""
    assert Z == Z_predict, "length of prediction and sample sets is not equal!"
    MAE = np.sum(Z - Z_predict) / np.sum(Z)

    return MAE


def check_all_RMSE(X, Z_hifi, Z_k_list):
    for i, Z_k in enumerate(Z_k_list):
        RMSE = RMSE_norm(Z_hifi, Z_k.predict(X)[0])
        print("RMSE of level {} = {:.4f} %".format(i,RMSE*100))

