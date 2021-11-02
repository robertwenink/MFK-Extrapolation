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
        hps = [[1] * setup.d, [2] * setup.d]

        # uses the same structure as setup.search_space.
        # par > x is encoded as x + np.finfo(float).eps for lowerbounds; likewise,
        # par < x is encoded as x - np.finfo(float).eps for upperbounds.
        # We can now externally always use >= and <= respectively.
        hp_constraints = [["theta", "p"], [0, 0 + np.finfo(np.float32).eps], [np.inf, 2]]
        
        return func, hps, hp_constraints

# NOTE currently unused, but of importance for the linear, cubic, thin plate spline, multiquadric, etc kernels
def _dist_matrix(X):
    """
    Function that calculates the matrix of distances between all points according to the L2 / Euclidian norm
    NOTE this is actually a symmetric matrix, can we exploit this in terms of speed?
    """
    return np.power(np.sum(np.power(X[:, np.newaxis, :] - X, 2), axis=2), 1 / 2)

@njit(cache=True, fastmath=True, parallel=True)
def corr_matrix_kriging(X, X_other, theta, p):
    """
    Kriging basis function according to (Jones 2001) and (Sacks 1989).
    This function does not include noise.
    :param X X_other:   datapoint locations in format [[x11,x12,..],[x21,..],..]
                        for matrix R we use X_other = X,
                        for r we use X_other = X_predict
    :param theta: hyperparameters scaling the (relative) relevance of each dimension
    :param p: hyperparameters for scaling the relevance of distance in each dimension
    """

    # ugly but fast due to prange and memory efficiency
    arr = np.zeros((X.shape[0], X_other.shape[0]))
    for i in prange(X_other.shape[0]):
        diff = np.abs(X - X_other[i,:])
        for d in range(X_other.shape[1]):
            arr[:,i]+=theta[d] * diff[:,d]**p[d]
    return np.exp(-arr) 
    
    # ugly but faster due to memory efficiency
    # arr = np.zeros((X.shape[0], X_other.shape[0]))
    # for i in prange(X_other.shape[0]):
    #     diff = np.abs(X - X_other[i,:])
    #     arr[:,i] = np.exp(
    #         -np.sum(
    #             np.multiply(theta, np.power(diff, p)),
    #             axis=-1
    #         )
    #     )
    # return arr 
    
    # nice/pythonic but slow
    # return np.exp(
    #     -np.sum(
    #         np.multiply(theta, np.power(np.abs(X[:, np.newaxis, :] - X_other[np.newaxis, :, :]), p)),
    #         axis=2
    #     )
    # )
