from core.sampling.DoE import get_doe, plot_doe
import numpy as np
import matplotlib.pyplot as plt
from core.sampling.solvers.solver import get_solver
from core.ordinary_kriging.OK import OrdinaryKriging
from postprocessing.plotting import Plotting

from preprocessing.input import Input

from utils.correlation_utils import check_correlations
from utils.convergence_utils import plot_grid_convergence, plot_grid_convergence_tt

setup = Input(0)

L = [1,2,3,4]
# L = [1] 
Z = []
TT = []
X = []

doe = get_doe(setup)
if hasattr(setup,'X'):
    if isinstance(setup.X, list) and len(setup.X) == len(L):
        #NOTE redundant in this setup
        X = setup.X
    elif isinstance(setup.X, np.ndarray):
        X0 = setup.X 
    else: 
        X0 = setup.X[0]
else:
    X0 = doe(setup, 10*setup.d)
    
X = []
# take all the same X    
for l in L:
    X.append(X0)

for l in range(len(L)):
    solver = get_solver(setup)
    z, _, tt = solver.solve(X[0],L[l],get_time_trace = True)
    Z.append(z)
    TT.append(tt)

for i in range(2,len(L)):
    for j in range(i):
        for k in range(j):
            print("###### Checking the refinements: {}, {}, {} ######".format(L[k],L[j],L[i]))
            check_correlations(Z[k], Z[j], Z[i])
        

plot_grid_convergence(X, Z, L, solver)
plot_grid_convergence_tt(X, TT, L, solver)
plt.show()