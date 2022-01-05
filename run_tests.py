import time
import numpy as np
import matplotlib.pyplot as plt

from sampling.initial_sampling import get_doe
from sampling.solvers.solver import get_solver
from models.kriging.method.OK import *
from postprocessing.plotting import plot_kriging

class Setup():
    pass

setup = Setup()
setup.kernel = "kriging"
setup.doe = "LHS"
# setup.doe = "grid"
setup.solver_str = "Branin"
setup.d = 2
setup.search_space = [["x0", "x1"], np.array([-5, 0]), np.array([10, 15])]

doe = get_doe(setup)
X = doe(setup, 9)

solver = get_solver(setup)
y, _ = solver.solve(X)

ok = OrdinaryKriging(setup)
ok.train(X, y, True)

# test if sigma_hat from tuning is equal to that of prediction regimes
assert (
    ok.sigma_hat == _sigma_mu_hat(ok.R_in, y, y.shape[0])
), "\n sigma_hat not equal between tuning and training"

# test if r.T R_in r == 1 if r build using subset of R_in -> then np.all(mse == 0)
ok.predict(np.array([X[0]]))
res = np.diag(np.linalg.multi_dot((ok.r.T, ok.R_in, ok.r)))
rtR_in = np.dot(ok.r.T, ok.R_in)

res0 = np.sum(np.multiply(rtR_in, ok.r.T), axis=-1)

assert(np.all(res == res0)), '\n mse diagonalisation method not correct.'

assert (
    np.allclose(res,1)
), "\n r.T R_in r != 1, with result: {}".format(res)

assert (
    np.allclose(res,1)
), "\n r.T R_in r != 1, with result: {}".format(res)

assert (np.all(_mse(ok.R_in, ok.r, rtR_in, ok.sigma_hat) == 0)), '\n mse not 0 when it should'


# test if r R_in r == 0 at index i if r_i build using subset of R_in -> then mse[i] == 0
X_test = doe(setup, 4)



setup.live_plot = True
setup.d_plot = [0, 1]
setup.type_of_plot = 'Surface'
plot_kriging(setup,X, y, ok)
plt.show()

print('Testing done')
import sys
sys.exit()