from sampling.initial_sampling import get_doe
import numpy as np
import matplotlib.pyplot as plt
from sampling.solvers.solver import get_solver
from models.kriging.method.OK import OrdinaryKriging
from postprocessing.plotting import plot_kriging

from preprocessing.input import Input
from scipy.optimize import minimize

setup = Input(2)

if __name__ == "__main__":
    doe = get_doe(setup)
    if hasattr(setup,'X'):
        X = setup.X
    else:
        X = doe(setup, 10*setup.d)
    
    solver = get_solver(setup)
    y, _ = solver.solve(X,2)
    
    ok = OrdinaryKriging(setup)
    ok.train(X, y,True)
    
    
    plot_kriging(setup,X, y, ok)
    plt.show()
    
    if setup.SAVE_DATA:
        setup.X = X
        setup.create_input_file()
    