from preprocessing.input import Input
from sampling.initial_sampling import get_doe
import numpy as np
import matplotlib.pyplot as plt
from sampling.solvers.solver import get_solver
from models.kriging.kernel import _dist_matrix, corr_matrix_kriging
from models.kriging.method.OK import OrdinaryKriging
from postprocessing.plotting import plot_kriging

# Testing ala beun
# import run_tests

setup = Input(2)
setup.regression = False
doe = get_doe(setup)
if hasattr(setup,'X'):
    X = setup.X
else:
    X = doe(setup, 20)

solver = get_solver(setup)()
y = solver.solve(X)

ok = OrdinaryKriging(setup)
ok.train(X, y,True)


plot_kriging(setup,X, y, ok)
plt.show()

if setup.SAVE_DATA:
    setup.X = X
    setup.create_input_file()
