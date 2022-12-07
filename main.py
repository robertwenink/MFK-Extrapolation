from core.sampling.DoE import get_doe, plot_doe
import numpy as np
import matplotlib.pyplot as plt
from core.sampling.solvers.solver import get_solver
from core.kriging.OK import OrdinaryKriging
from postprocessing.plotting import Plotting

from preprocessing.input import Input

# plot_doe()

L = [2,3,3.5]

setup = Input(0)
pp = Plotting(setup)

doe = get_doe(setup)
if hasattr(setup,'X') and isinstance(setup.X, list):
    X = setup.X
else:
    if isinstance(setup.X, np.ndarray):
        X0 = setup.X 
    else:
        X0 = doe(setup, 10*setup.d)
    
    # take all the same X    
    for l in L:
        X.append(X0)

Z = []
for l in range(len(L)):
    solver = get_solver(setup)
    z, _ = solver.solve(X[l],L[l])
    Z.append(z)
        
    ok = OrdinaryKriging(setup)
    ok.train(X[l], z,True)
    pp.plot(X[l],z,ok)

if setup.SAVE_DATA:
    setup.X = X
    setup.Z = Z

    setup.create_input_file()

plt.show()


    