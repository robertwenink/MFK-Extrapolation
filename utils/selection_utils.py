import numpy as np
from core.ordinary_kriging.OK import OrdinaryKriging
import core.mfk.mfk_base as mf # like this because circular import!
from utils.formatting_utils import correct_formatX

def create_X_infill(d, lb, ub, n_infill_per_d):
    """
    Build a (1d or 2d) grid used for finding the maximum EI. Specify n points in each dimension d.
    !! DICTATES THE PRECISION AND THEREBY AMOUNT OF SAMPLING BEFORE STOPPING !!
    Dimensions d are specified by the dimensions to plot in setup.d_plot.
    Coordinates of the other dimensions are fixed in the centre of the range.
    """
    # n_infill_per_d = 601 # NOTE quite high! is for branin resolution of 0.025
    lin = np.linspace(lb, ub, n_infill_per_d, endpoint=True)
    
    # otherwise weird meshgrid of np.arrays
    lis = [lin[:, i] for i in range(d)]

    # create plotting meshgrid
    X_infill = np.stack(np.meshgrid(*lis), axis=-1).reshape(-1, d)
    return X_infill

def isin(x,X):
    """Test if any x is in X"""    
    return isin_indices(x,X).any()

def isin_indices(X_test, X, inversed = False):
    """
    return an array of True for indices of X_test where X_test is in X.
    if inversed = True, return True for indices where X_test is not in X
    returned array length is length of X_test
    If you want to test for the indices where X is in X_test, change input order!
    """
    assert X_test.shape[1:] == X.shape[1:], "x and X not of same format"
    mask = (X_test == X[:, None]).all(axis=-1).any(axis=0)
    if inversed:
        return ~mask
    return mask

def get_best_sample(model, arg = False):
    """
    returns the best sample as X, y
    if arg = True, return only the sample index 
    """
    if isinstance(model,mf.MultiFidelityKrigingBase):
        best_ind = np.argmin(model.Z_mf[-1])
        if not arg:
            return correct_formatX(model.X_mf[-1][best_ind], model.d), model.Z_mf[-1][best_ind] 

    elif isinstance(model,OrdinaryKriging):
        best_ind = np.argmin(model.y)
        if not arg:
            return correct_formatX(model.X[best_ind], model.d), model.y[best_ind] 

    return best_ind # type: ignore

def get_best_prediction(model, x_best = None):
    """
    @param x_best: location of interest (i.e. best prediction or highest EI)
    returns X_best, y_best 
    """
    if isinstance(model,mf.MultiFidelityKrigingBase):
        predictor = model.K_mf[-1]
    else:
        predictor = model
    
    if np.all(x_best == None) and hasattr(model,'X_infill'):
        # expensive due to large X_infill !
        y_pred, _ = predictor.predict(model.X_infill) # type: ignore
        ind = np.argmin(y_pred)
        return correct_formatX(model.X_infill[ind,:], model.X_infill.shape[1]), y_pred[ind] # type: ignore
    elif np.all(x_best != None):
        y_pred, _ = predictor.predict(x_best) # type: ignore
        return x_best, y_pred
    return 0, 0
    
        
