from sampling.DoE import get_doe, plot_doe
import numpy as np
import matplotlib.pyplot as plt
from sampling.solvers.solver import get_solver
from kriging.OK import OrdinaryKriging
from postprocessing.plotting import Plotting

from preprocessing.input import Input

from utils.correlation_utils import check_correlations


setup = Input(0)
pp = Plotting(setup)

L = [1,1.5,2,2.5,3,3.5,4]
# L = [1] 
Z = []
X = []

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

for l in range(len(L)):
    solver = get_solver(setup)
    z, _ = solver.solve(X[0],L[l])
    Z.append(z)

for i in range(2,len(L)):
    for j in range(i):
        for k in range(j):
            print("###### Checking the refinements: {}, {}, {} ######".format(L[k],L[j],L[i]))
            check_correlations(Z[k], Z[j], Z[i])
        



