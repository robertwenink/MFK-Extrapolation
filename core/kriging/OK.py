"""
Ordinary Kriging
"""
# pyright: reportGeneralTypeIssues=false

import numpy as np
from scipy import linalg
from core.kriging.kernel import get_kernel, diff_matrix, corr_matrix_kriging_tune
from numba import njit
import time
from beautifultable import BeautifulTable

# from core.kriging.hp_tuning import GeneticAlgorithm as ga
from core.kriging.hp_tuning import MultistartHillclimb as ga
from utils.formatting_utils import correct_formatX


class OrdinaryKriging:
    def __init__(self, kernel, d, hps_init=None, name = "", *args, **kwarg):
        super().__init__(*args, **kwarg)

        # TODO y naar z veranderen? -> in dat geval ook bij plotting voor linearity check veranderen.

        self.corr, self.hps, self.hps_constraints = kernel
        self.d = d
        self.name = name

        if hps_init is not None:
            assert self.hps.shape == hps_init.shape, "Provide hps_init of correct size!"
            self.hps = hps_init

    def predict(self, X_new):
        """Predicts and returns the prediction and associated mean square error"""
        X_new = correct_formatX(X_new, self.d)

        # NOTE correlation function for r should not involve regression terms (forrester2006).
        # Those come back through R(_in) in both sigma and the prediction.
        self.r = self.corr(self.X, X_new, self.hps)

        y_hat, rtR_in = _predictor(self.R_in, self.r, self.y, self.mu_hat)
        mse_var = _mse(self.R_in, self.r, rtR_in, self.sigma_hat)
        return y_hat, mse_var


    def train(self, X, y, tune : bool = False, retuning : bool = True, R_diagonal=0):
        """Train the class on matrix X and the corresponding sampled values of array y"""
        self.X = correct_formatX(X, self.d)
        self.y = y

        self.diff_matrix = diff_matrix(self.X, self.X)
        self.R_diagonal = R_diagonal

        if tune:
            self.tune(R_diagonal, retuning)

        R = self.corr(self.X, self.X, self.hps)

        # add regularization constant!! 
        # I prefer this over pseudoinverse pinv because pinv is slower
        self.regularization = np.finfo(np.float32).eps

        # add prediction regression terms
        if R_diagonal is not None:
            np.fill_diagonal(R, np.diag(R) + R_diagonal + self.regularization)
        else:
            np.fill_diagonal(R, np.diag(R) + self.hps[-1] + self.regularization)
        
        n = self.X.shape[0]

        # we only want to calculate the inverse once, so it has to be object oriented or passed around
        self.R_in = linalg.pinv(R)
        self.mu_hat = _mu_hat(self.R_in, y)
        self.sigma_hat = _sigma_hat(self.R_in, y, self.mu_hat, n)
        

    def reinterpolate(self):
        # retrieve the noisy predictions
        y_noise = self.predict(self.X)[0]

        # set noise to 0
        self.hps[-1] = 0
        
        # retrain
        self.train(
            self.X,
            y_noise,
            tune=False,
            R_diagonal=self.R_diagonal,
        )
        
    def fitness_func(self, hps):
        """
        Fitness function for the tuning process.

        Regression is optionally included by adding values to the diagonal of the correlation matrix.
        R_diagonal: factors determining the correlation with a point with itself.
                    0 if sampled, 0> if originating from lower level/ not sampled on current.
        """

        R = corr_matrix_kriging_tune(hps, self.diff_matrix, self.R_diagonal)

        if hps.shape[0] == 1:
            # in case of single evaluation.
            if numba_cond(R[0]) < 1/np.finfo(np.float64).eps: 
                #NOTE cond is 1/6 totale sim tijd; svd als subroutine wordt door zowel cond als pinv gebruikt, is deze overlap bruikbaar?
                # cond ~~ min(LA.svd(a, compute_uv=False))*min(LA.svd(LA.inv(a), compute_uv=False)) https://numpy.org/doc/stable/reference/generated/numpy.linalg.cond.html
                # inv is 4x zo snel als pinv!! 6 vs 26 seconden voor ongeveer zelfde hoeveelheid calls.

                # https://stackoverflow.com/questions/13249108/efficient-pythonic-check-for-singular-matrix
                # computes many inverses simultaneously = somewhat faster; no njit
                R_in = np.linalg.inv(R[0])
                # R_in = numba_inv(R[0])
                # return _fitness_func_loop(R_in, self.y, R).item()
                return fitness_direct(R_in,self.y,R[0])
            else:
                return -1e8
        else:
            print("WARNING: NO COND CHECK YET")
            # in case of batch evaluation.
            R_in = np.linalg.inv(R)
            return _fitness_func_loop(R_in, self.y, R)

    def tune(self, R_diagonal = 0, retuning : bool = True):
        """
        Tune the Kernel hyperparameters according to the concentrated log-likelihood function (Jones 2001)
        """
        # run model and time it
        start = time.time()

        if not hasattr(self, "model"):
            self.model = ga(
                function=self.fitness_func,
                hps_init=self.hps,
                hps_constraints=self.hps_constraints,
                progress_bar=True,
                convergence_curve=False,
                retuning = retuning #! set explicitly to False if creating a new OK!
            )
        else:
            self.model.set_retuning(retuning)
            self.model.run()

        t = time.time() - start

        # assign results to Kriging object, this sets hps_init for the tuning as well.
        self.hps = self.model.output_dict["hps"].reshape(self.hps.shape)

        # print results of tuning
        table = BeautifulTable()
        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        tuning_time = "{}{}{}".format("Re-t" if hasattr(self, "model") and retuning else "T","uning time"," of {}".format(self.name) if self.name != "" else "")
        table.columns.header = [tuning_time, "Fitness"] + ["\u03F4" + str(i).translate(SUB) for i in range(self.d)] +  ["p" + str(i).translate(SUB) for i in range(self.d)] + ["noise"]
        row = ["{:.4f} s".format(t), self.model.output_dict["function"]] + list(self.model.best_hps)
        table.rows.append(row)
        print(end='\r')
        print(table)


    def get_state(self):
        """
        Get the state of the model. 
        Excludes correlation function 'corr', which should be set at class initiation.
        """
        # OK.__dict__:  dict_keys(['corr', 'hps', 'hps_constraints', 'd', 'X', 'y', 'diff_matrix', 'R_diagonal', 'R_in', 'mu_hat', 'sigma_hat', 'r'])
        # corr is a (compiled) function
        return {k: self.__dict__[k] for k in set(list(self.__dict__.keys())) - set({'corr', 'model', 'r', 'diff_matrix', 'R_in', 'mu_hat', 'sigma_hat'})}

    def set_state(self, data_dict):
        """
        Set the state of the already initialised model.
        """
        # best to do this explicitly!
        # then if something misses we will know instead of being blind
        keylist = ['hps', 'hps_constraints', 'd', 'X', 'y', 'R_diagonal']
        for key in keylist:
            setattr(self, key, data_dict[key])

        # such that we do not have to save:
        # r, diff_matrix, R_in, mu_hat, sigma_hat
        self.train(self.X, self.y, tune = False, retuning = True, R_diagonal = self.R_diagonal)

@njit(cache=True)
def numba_cond(R):
    """Factor 2 faster with numba!"""
    return np.linalg.cond(R)

@njit(cache=True)
def numba_inv(R):
    # TODO, mogelijk sneller, maar slechtere resultaten?
    return np.linalg.inv(R)

@njit(cache=True)
def _mu_hat(R_in, y):
    """Estimated constant mean of the Gaussian process, according to (Jones 2001)"""
    return np.sum(np.dot(R_in, y)) / np.sum(R_in)


@njit(cache=True)
def _sigma_hat(R_in, y, mu_hat, n):
    """Estimated variance sigma squared of the Gaussian process, according to (Jones 2001)"""
    t = y - mu_hat
    return np.dot(t.T, np.dot(R_in, t)) / n


# @njit is slower!
def _mse(R_in, r, rtR_in, sigma_hat):
    """Mean squared error of the predictor, according to (Jones 2001)"""
    # hiervan willen we de mse allen op de diagonaal, i.e. de mse van een punt naar een ander onsampled punt is niet onze interesse.
    # t0 = 1 - np.diag(np.dot(rtR_in , r)) # to make this faster we only need to calculates the columns/rows corresponding to the diagonals.
    t = 1 - np.sum(np.multiply(rtR_in, r.T), axis=-1)
    t2 = t ** 2 / np.sum(R_in)
    # t2 = 1 - np.sum(rtR_in,axis=-1)
    # np.abs because numbers close to zero / e-13 can get negative due to rounding errors (in R_in?)
    mse = np.abs(sigma_hat * (t + t2))
    return mse


@njit(cache=True, fastmath=True)
def _predictor(R_in, r, y, mu_hat):
    """Kriging predictor, according to (Jones 2001)"""
    rtR_in = np.dot(r.T, R_in)
    y_hat = mu_hat + np.dot(rtR_in, y - mu_hat)
    return y_hat, rtR_in


@njit(cache=True)
def _sigma_mu_hat(R_in, y, n):
    """
    Convenience function with sigma mu hat calculations in one.
    log for more numerically stable tuning optimization.
    """
    t = y - np.sum(R_in * y) / np.sum(R_in)
    return -n * np.log(np.dot(t.T, np.dot(R_in, t)) / n)

@njit(cache=True)
def fitness_direct(R_in,y,R):
    """
    Slightly less overhead than fitness_func_loop for the single data case.
    """
    return _sigma_mu_hat(R_in, y, y.shape[0]) - np.linalg.slogdet(R)[1]

@njit(cache=True)
def _fitness_func_loop(R_in_list, y, R):
    """This function joins the left + right hand side of the concentrated log likelihood"""
    n_pop = R_in_list.shape[0]
    fit = np.zeros(n_pop)
    n = y.shape[0]
    for i in range(n_pop):
        # MLE, omitting factor 1/2
        fit[i] = _sigma_mu_hat(R_in_list[i], y, n) - np.linalg.slogdet(R[i])[1]
    return fit
