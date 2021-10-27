from preprocessing.input import Input
from sampling.initial import grid
import numpy as np
import matplotlib.pyplot as plt
from sampling.solvers.solver import get_solver
from models.kriging.kernel import _dist_matrix, corr_matrix_kriging
from models.kriging.method.OK import OrdinaryKriging
from postprocessing.plotting import plot2d

setup = Input(0)
X = grid.generate(setup, 6)
X_new = grid.generate(setup, 3)/1.5

R = corr_matrix_kriging(X,X,[4,1],[1,2])
solver = get_solver(setup)()
# solver.plot2d(X)
y = solver.solve(X)
ok = OrdinaryKriging(setup)
ok.train(X, y)
y_hat, mse = ok.predict(X_new)
print(y_hat in y)
print(np.sqrt(mse))

plot2d(X,y,X_new,y_hat,mse)

plt.show()

if setup.SAVE_DATA:
    "update the input file explicitly by setting the corresponding value of data_dict."
    setup.data_dict['X'] = X.tolist()
    setup.create_input_file()
