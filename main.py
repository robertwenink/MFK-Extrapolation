from preprocessing.input import Input
from sampling.initial_sampling import get_doe, grid
import numpy as np
import matplotlib.pyplot as plt
from sampling.solvers.solver import get_solver
from models.kriging.kernel import _dist_matrix, corr_matrix_kriging
from models.kriging.method.OK import OrdinaryKriging
from postprocessing.plotting import plot_kriging

setup = Input(0)
doe = get_doe(setup)
if hasattr(setup,'X'):
    X = setup.X
else:
    X = doe(setup, 20)

X_new = grid(setup, 100)
print('Generated grids')

solver = get_solver(setup)()
y = solver.solve(X)

print(np.mean(y))

ok = OrdinaryKriging(setup)
ok.train(X, y)
y_hat, mse = ok.predict(X_new)

ok.tune()

# print(np.sqrt(mse))

plot_kriging(setup,X, y, ok)

plt.show()

if setup.SAVE_DATA:
    setup.X = X
    setup.create_input_file()
