from sampling.DoE import get_doe, plot_doe
import numpy as np
import matplotlib.pyplot as plt
from sampling.solvers.solver import get_solver
from models.kriging.method.OK import OrdinaryKriging
from postprocessing.plotting import Plotting

from preprocessing.input import Input

# plot_doe()

setup = Input(0)
pp = Plotting(setup)

if __name__ == "__main__":
    doe = get_doe(setup)
    if hasattr(setup,'X'):
        X = setup.X
    else:
        X = doe(setup, 2*setup.d)
    # X = np.append(X,[[2,2,2]],axis=0)
    solver = get_solver(setup)
    y, _ = solver.solve(X)
    
    ok = OrdinaryKriging(setup)
    ok.train(X, y,True)
    pp.plot(X,ok)

    # ok2 = OrdinaryKriging(setup)
    # ok2.train(X, y+1,True)
    # pp.plot(X,ok2)
    
    if setup.SAVE_DATA:
        setup.X = X

        setup.create_input_file()

    plt.show()


    