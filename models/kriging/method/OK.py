"""
Ordinary Kriging
"""
import numpy as np
from scipy import linalg
from models.kriging.kernel import get_kernel, diff_matrix, corr_matrix_kriging_tune
from numba import njit,prange
import time

from models.kriging.hyperparameter_tuning.GA import geneticalgorithm as ga

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
    mse = np.abs(sigma_hat * (t + np.power(t, 2) / np.sum(R_in)))
    return mse


@njit(cache=True, fastmath=True)
def predictor(R_in, r, y, mu_hat):
    """Kriging predictor, according to (Jones 2001)"""
    rtR_in = np.dot(r.T, R_in)
    y_hat = mu_hat + np.dot(rtR_in, y - mu_hat)
    return y_hat, rtR_in


@njit(cache=True, fastmath=True,parallel=False) 
def sigma_mu_hat_log_det(R_in_list, y, R):
    """This function contains the left + right hand side of the concentrated log likelihood"""
    n_pop = R_in_list.shape[0]
    fit = np.zeros(n_pop)
    for i in prange(n_pop):
        R_in = R_in_list[i]
        n = y.shape[0]
        t = y - (np.sum(np.dot(R_in, y)) / np.sum(R_in))
        det = np.linalg.det(R[i])
        fit[i] = -n / 2 * np.log(np.dot(t.T, np.dot(R_in, t)) / n) - (1 / 2) * np.log(det)
    return fit

# @njit(cache=True)
def fitness_func(hps, diff_matrix, y, corr):
    R = corr_matrix_kriging_tune(hps,diff_matrix)
    R_in = np.linalg.inv(R)
    return sigma_mu_hat_log_det(R_in, y, R)

class OrdinaryKriging:
    def __init__(self, setup):
        self.corr, self.hps, self.hp_constraints = get_kernel(setup)

    def predict(self, X_new):
        """Predicts and returns the prediction and associated mean square error"""
        r = self.corr(self.X, X_new, *self.hps)
        print("Predicting")
        y_hat, rtR_in = predictor(self.R_in, r, self.y, self.mu_hat)
        print("Calculating mse")
        mse_var = mse(self.R_in, r, rtR_in, self.sigma_hat)
        print("Done with predict")
        return y_hat, mse_var

    def train(self, X, y):
        """Train the class on matrix X and the corresponding sampled values of array y"""
        R = self.corr(X, X, *self.hps)
        self.X = X
        self.y = y
        n = X.shape[0]

        # we only want to calculate the inverse once, so it has to be object oriented or passed around
        start = time.time()
        self.R_in = linalg.inv(R)
        end = time.time()
        print("time for inv: {}".format(end - start))
        self.mu_hat = mu_hat(self.R_in, y)
        print("mu_hat = {}".format(self.mu_hat))
        self.sigma_hat = sigma_hat(self.R_in, y, self.mu_hat, n)

    def tune(self):
        """Tune the Kernel hyperparameters according to the concentrated log-likelihood function (Jones 2001)"""
        diff_m = diff_matrix(self.X, self.X)
        # print("Now tuning, starting with fitness {}".format(fitness_func(np.array([self.hps]),diff_m, self.y, self.corr)))
        start = time.time()
        model=ga(function=fitness_func,dimension=self.hps.size,other_function_arguments = [diff_m, self.y, self.corr], \
            variable_boundaries=self.hp_constraints,progress_bar=True,convergence_curve=False)
        model.run()
        self.hps =model.output_dict['variable']

        t = time.time() - start
        print("Tuning completed with fitness {} and time {} s".format(model.output_dict['function'], t))


# NOTE TODO
# But I'd be amiss if I didn't point out that this is very rarely really necessary:
# anytime you need to compute a product A−1b, you should instead solve the linear system Ax=b
# (e.g., using numpy.linalg.solve) and use x instead -- this is much more stable, and can be done
# (depending on the structure of the matrix A) much faster. If you need to use A−1 multiple times,
#  you can precompute a factorization of A (which is usually the most expensive part of the solve) and reuse that later.
