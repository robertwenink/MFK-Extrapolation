"""
Ordinary Kriging
"""
import numpy as np
from scipy import linalg
from models.kriging.kernel import get_kernel, diff_matrix, corr_matrix_kriging_tune
from numba import njit, prange
import time

from models.kriging.hyperparameter_tuning.GA import geneticalgorithm as ga


def correct_formatX(a):
    """
    X should be of format np.array([[..,..,..],[..,..,..],..])
    Thus:
    1.0                 -> np.array([[a]])
    if size 1, but:
        ndim = 0        -> np.array([a])    (make 1d, refer to next)
    1d X: [..,..,..]    -> np.array([a]).T  (make 2d, align amount of samples with first dimension, return)
    if 2d but list      -> np.array(a)
    """

    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if a.ndim == 0:
        a = np.array([a])
    if a.ndim == 1:
        return np.array([a]).T
    return a


@njit(cache=True)
def mu_hat(R_in, y):
    """Estimated constant mean of the Gaussian process, according to (Jones 2001)"""
    return np.sum(np.dot(R_in, y)) / np.sum(R_in)


@njit(cache=True)
def sigma_hat(R_in, y, mu_hat, n):
    """Estimated variance of the Gaussian process, according to (Jones 2001)"""
    t = y - mu_hat
    return np.dot(t.T, np.dot(R_in, t)) / n


# @njit is slower!
def mse(R_in, r, rtR_in, sigma_hat):
    """Mean squared error of the predictor, according to (Jones 2001)"""
    # hiervan willen we de mse allen op de diagonaal, i.e. de mse van een punt naar een ander onsampled punt is niet onze interesse.
    # t0 = 1 - np.diag(np.dot(rtR_in , r)) # to make this faster we only need to calculates the columns/rows corresponding to the diagonals.
    t = 1 - np.sum(np.multiply(rtR_in, r.T), axis=-1)

    # np.abs because numbers close to zero / e-13 can get negative due to rounding errors (in R_in?)
    mse = np.abs(sigma_hat * (t + t ** 2 / np.sum(R_in)))
    return mse


@njit(cache=True, fastmath=True)
def predictor(R_in, r, y, mu_hat):
    """Kriging predictor, according to (Jones 2001)"""
    rtR_in = np.dot(r.T, R_in)
    y_hat = mu_hat + np.dot(rtR_in, y - mu_hat)
    return y_hat, rtR_in


@njit(cache=True, fastmath=True)
def sigma_mu_hat(R_in, y, n):
    t = y - (np.sum(np.dot(R_in, y)) / np.sum(R_in))
    sigma_hat = np.dot(t.T, np.dot(R_in, t)) / n
    return sigma_hat


@njit(cache=True, fastmath=True)
def fitness_func_loop(R_in_list, y, R):
    """This function joins the left + right hand side of the concentrated log likelihood"""
    n_pop = R_in_list.shape[0]
    fit = np.zeros(n_pop)
    n = y.shape[0]
    for i in range(n_pop):
        fit[i] = -n / 2 * np.log(sigma_mu_hat(R_in_list[i], y, n)) - (1 / 2) * np.log(
            np.linalg.det(R[i])
        )
    return fit


# @njit(cache=True)
def fitness_func(hps, diff_matrix, y, corr):
    R = corr_matrix_kriging_tune(hps, diff_matrix)

    # computes many inverses simultaneously = much faster; no njit
    R_in = np.linalg.inv(R)
    
    return fitness_func_loop(R_in, y, R)


def fitness_func_mf_regression(hps, diff_matrix, y, corr, R_diagonal):
    """
    fitness function if regression is included. Add values to the diagonal of the correlation matrix.
    R_diagonal: factors determining the correlation with a point with itself. 
                0 if sampled, 0> if originating from lower level/ not sampled on current.
    hps_reg: regression hyperparameter, scaling R_diagonal contributions. 
             This hyperparameter is only dependent on levels, so always 1d.
    """
    R = corr_matrix_kriging_tune(hps, diff_matrix,R_diagonal)

    # computes many inverses simultaneously = much faster; no njit
    R_in = np.linalg.inv(R)

    return fitness_func_loop(R_in, y, R)


class OrdinaryKriging:
    def __init__(self, setup):
        self.corr, self.hps, self.hps_constraints = get_kernel(setup)
        self.hps_reg = 0.01

    def predict(self, X_new):
        """Predicts and returns the prediction and associated mean square error"""
        X_new = correct_formatX(X_new)

        # NOTE this does not involve regression terms. Those come back through R(_in) in both sigma and the prediction.
        self.r = self.corr(self.X, X_new, *self.hps)

        y_hat, rtR_in = predictor(self.R_in, self.r, self.y, self.mu_hat)
        mse_var = mse(self.R_in, self.r, rtR_in, self.sigma_hat)
        return y_hat, mse_var

    def train(self, X, y, tune=False, R_diagonal = None):
        """Train the class on matrix X and the corresponding sampled values of array y"""
        self.X = correct_formatX(X)
        self.y = y

        if tune:
            self.tune(R_diagonal)

        R = self.corr(self.X, self.X, *self.hps)
        
        # add correlation with self
        if R_diagonal is not None:
            np.fill_diagonal(R, np.diag(R) + R_diagonal * self.hps_reg)
    
        n = self.X.shape[0]

        # we only want to calculate the inverse once, so it has to be object oriented or passed around
        self.R_in = linalg.inv(R)
        self.mu_hat = mu_hat(self.R_in, y)
        self.sigma_hat = sigma_hat(self.R_in, y, self.mu_hat, n)

    def tune(self, R_diagonal=None):
        """Tune the Kernel hyperparameters according to the concentrated log-likelihood function (Jones 2001)"""
        diff_m = diff_matrix(self.X, self.X)

        # select fitness function and arguments based on inclusion of regression
        if R_diagonal is None:
            ff = fitness_func
            other_function_arguments=[diff_m, self.y, self.corr]
            hps_constraints = self.hps_constraints.reshape(-1,2)
            dim = self.hps.size
        else:
            ff = fitness_func_mf_regression
            other_function_arguments=[diff_m, self.y, self.corr, R_diagonal]

            # add the regression hyperparameter constraint
            hps_constraints = self.hps_constraints.reshape(-1,2)
            hps_constraints = np.append(hps_constraints,[[0, 0.1]], axis=0)
            dim = self.hps.size + 1

        # run model and time it
        start = time.time()
        model = ga(
            function=ff,
            dimension=dim,
            other_function_arguments=other_function_arguments,
            hps_constraints=hps_constraints,
            progress_bar=True,
            convergence_curve=True,
        )
        model.run()
        t = time.time() - start

        # assign results to Kriging object
        hps = model.output_dict["variable"]
        self.hps_reg = hps[-1]
        self.hps = hps[:-1].reshape(self.hps_constraints.shape[:-1])
        # print results of tuning
        print(
            "Tuning completed with fitness {} and time {} s".format(
                model.output_dict["function"], t
            )
        )


# NOTE TODO
# But I'd be amiss if I didn't point out that this is very rarely really necessary:
# anytime you need to compute a product A−1b, you should instead solve the linear system Ax=b
# (e.g., using numpy.linalg.solve) and use x instead -- this is much more stable, and can be done
# (depending on the structure of the matrix A) much faster. If you need to use A−1 multiple times,
#  you can precompute a factorization of A (which is usually the most expensive part of the solve) and reuse that later.
