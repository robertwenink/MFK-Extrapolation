"""
Kernel, also known as basis function. The covariance function then is sigma**2 times this basis.  
The kernel determines the relation between points based on distance or radius, the correlation matrix.
The used kernel determines which hyperparameters need to be tuned.
For Kriging we have to use a kernel that is able to scale the dimensions according to the corresponding hyperparameters. (Jones 2001)
"""

import numpy as np

from numba import njit, prange


def get_available_kernel_names():
    """Helper function for the GUI"""
    return ["kriging"]


def get_kernel(setup):
    """
    Function to return the correct kernel function according to the setup object,
    including the hyperparameters, its names and constraints.
    """
    if setup.kernel == "kriging":
        # define function to use
        func = corr_matrix_kriging

        # define the format/type of the hyperparameters, place in a list that is easy to unpack
        hps = np.array([[1] * setup.d, [2] * setup.d]).reshape(-1,1)        
        
        # we should always include this regression hyperparameter, otherwise functions break down
        hps = np.append(hps, [0])

        # uses the same structure as setup.search_space.
        # par > x is encoded as x + np.finfo(float).eps for lowerbounds; likewise,
        # par < x is encoded as x - np.finfo(float).eps for upperbounds.
        # We can now externally always use >= and <= respectively.
        # hps_constraints = [["theta", "p"], np.array([0, 0 + np.finfo(np.float32).eps]), np.array([np.inf, 2])]
        hps_constraints = np.array(
            [[[0, 1000]] * setup.d, [[2, 2]] * setup.d]
        ).reshape(-1,2)

        if setup.noise_regression:
            hps_constraints = np.append(hps_constraints,[[0, 0.001]], axis=0)
        else:
            hps_constraints = np.append(hps_constraints,[[0, 0]], axis=0)        

        return func, hps, hps_constraints


# NOTE currently unused, but of importance for the linear, cubic, thin plate spline, multiquadric, etc kernels
def _dist_matrix(X):
    """
    Function that calculates the matrix of distances between all points according to the L2 / Euclidian norm
    NOTE this is actually a symmetric matrix, can we exploit this in terms of speed?
    """
    return np.power(np.sum(np.power(X[:, np.newaxis, :] - X, 2), axis=2), 1 / 2)


@njit(cache=True, fastmath=True)
def diff_matrix(X, X_other):
    """
    Calculate matrix of distances per dimension"""
    diff = np.zeros((X.shape[0], X_other.shape[0], X.shape[1]))
    for i in range(X_other.shape[0]):
        diff[:, i, :] = np.abs(X - X_other[i, :])
    return diff


@njit(cache=True)
def corr_matrix_kriging_tune(hps, diff_matrix,R_diagonal=0):
    """
    Function used in tuning. Creates the correlation matrix for the whole population.
    Includes terms for noise and R_diagonal regression.
    """
    
    # amount of different sets of hyperparameters we are trying
    n = hps.shape[0]

    # dimension
    d = diff_matrix.shape[-1]

    # size of single R
    l = diff_matrix.shape[0]

    # initialise stack of different correlation matrices
    R = np.zeros((n, l, l))

    for i in range(n):
        R[i] = corr_matrix_kriging_tune_inner(
            diff_matrix, hps[i, :d], hps[i, d : 2 * d]
        )
        # add correlation with self
        np.fill_diagonal(R[i], np.diag(R[i]) + hps[i, 2 * d] + R_diagonal)
    return R


@njit(cache=True, fastmath=True)
def corr_matrix_kriging_tune_inner(diff_matrix, theta, p):
    arr = np.zeros((diff_matrix.shape[:-1]))
    i = np.arange(diff_matrix.shape[0])
    for d in range(diff_matrix.shape[-1]):
        arr[i, :] += theta[d] * diff_matrix[i, :, d] ** p[d]
    return np.exp(-arr)


@njit(cache=True, fastmath=True)
def corr_matrix_kriging(X, X_other, hps):
    """
    Kriging basis function according to (Jones 2001) and (Sacks 1989).
    This function should not include regression.
    :param X, X_other:  datapoint locations in format [[x11,x12,..],[x21,..],..]
                        for matrix R we use X_other = X,
                        for r we use X_other = X_predict
    :param theta: hyperparameters scaling the (relative) relevance of each dimension
    :param p: hyperparameters for scaling the relevance of distance in each dimension
    """
    d = X_other.shape[1]
    theta, p = hps[:d], hps[d : 2*d]
    
    # ugly but fast due to prange and memory efficiency
    arr = np.zeros((X.shape[0], X_other.shape[0]))
    for i in prange(X_other.shape[0]):
        diff = np.abs(X - X_other[i, :])
        for di in range(d):
            arr[:, i] += theta[di] * diff[:, di] ** p[di]
    return np.exp(-arr)
