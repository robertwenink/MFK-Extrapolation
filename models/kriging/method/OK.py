"""
Ordinary Kriging
"""
import numpy as np

from models.kriging.kernel import get_kernel


def mu_hat(R_in, y):
    """Estimated constant mean of the Gaussian process, according to (Jones 2001)"""
    I = np.ones((R_in.shape[0]))
    return np.dot(I.T, np.dot(R_in, y)) / np.dot(I.T, np.dot(R_in, I))


def sigma_hat(R_in, y, mu_hat):
    """Estimated variance of the Gaussian process, according to (Jones 2001)"""
    I = np.ones((R_in.shape[0]))
    i = I * mu_hat
    t = y - i
    return np.dot(t.T, np.dot(R_in, t)) / R_in.shape[0]


def mse(R_in, r, sigma_hat):
    """Mean squared error of the predictor, according to (Jones 2001)"""
    I = np.ones((R_in.shape[0]))

    # t = (1 - np.dot(r.T , np.dot( R_in , r)))
    # hiervan willen we de mse allen op de diagonaal, i.e. de mse van een punt naar een ander onsampled punt is niet onze interesse.
    # onderstaande: https://stackoverflow.com/questions/14758283/is-there-a-numpy-scipy-dot-product-calculating-only-the-diagonal-entries-of-the
    t = 1 - np.einsum("ji,jk,ki->i", r, R_in, r)
    mse = sigma_hat * (t + t ** 2 / np.dot(I.T, np.dot(R_in, I)))
    return mse


def predictor(R_in, r, y, mu_hat):
    """Kriging predictor, according to (Jones 2001)"""
    I = np.ones((R_in.shape[0]))
    y_hat = mu_hat + np.dot(r.T, np.dot(R_in, (y - I * mu_hat)))
    return y_hat


class OrdinaryKriging:
    def __init__(self, setup):
        self.corr, self.hps, self.hp_constraints = get_kernel(setup)

    def predict(self, X_new):
        """Predicts and returns the prediction and associated mean square error"""
        r = self.corr(self.X, X_new, *self.hps)
        y_hat = predictor(self.R_in, r, self.y, self.mu_hat)
        mse_var = mse(self.R_in, r, self.sigma_hat)
        return y_hat, mse_var

    def train(self, X, y):
        """Train the class on matrix X and the corresponding sampled values of array y"""
        R = self.corr(X, X, *self.hps) 
        self.X = X
        self.y = y

        # we only want to calculate the inverse once, so it has to be object oriented or passed around
        self.R_in = np.linalg.inv(R)
        self.mu_hat = mu_hat(self.R_in, y)
        self.sigma_hat = sigma_hat(self.R_in, y, self.mu_hat)
