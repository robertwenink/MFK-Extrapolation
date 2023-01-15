import numpy as np
import core.kriging.OK as ok
import core.kriging.mf_kriging as mf # like this because circular import!
from utils.formatting_utils import correct_formatX

def isin(x,X):
    """Test if any x is in X"""    
    return isin_indices(x,X).any()

def isin_indices(X_test, X, inversed = False):
    """
    return an array of True for indices of X_test where X_test is not in X.
    X_test can be 1 or multiple locations.
    """
    assert X_test.shape[1:] == X.shape[1:], "x and X not of same format"
    mask = (X_test == X[:, None]).all(axis=-1).any(axis=0)
    if inversed:
        return ~mask
    return mask

def get_best_sample(model, arg = False):
    if isinstance(model,mf.MultiFidelityKriging):
        return np.argmin(model.Z_mf[-1]) if arg else np.min(model.Z_mf[-1])
    elif isinstance(model,ok.OrdinaryKriging):
        return np.argmin(model.y) if arg else np.min(model.y)

def get_best_prediction(model):
    """
    returns X_best, y_best """
    if isinstance(model,mf.MultiFidelityKriging):
        y_pred, _ = model.K_mf[-1].predict(model.X_infill) # type: ignore
    else:
        y_pred, _ = model.predict(model.X_infill)
    ind = np.argmin(y_pred)
    return correct_formatX(model.X_infill[ind,:],model.d), y_pred[ind] # type: ignore