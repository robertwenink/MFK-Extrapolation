import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import sys


from core.sampling.solvers.solver import get_solver
from core.mfk.proposed_mfk import ProposedMultiFidelityKriging
from preprocessing.input import Input
from utils.correlation_utils import check_correlations
from postprocessing.convergence_utils import plot_grid_convergence, plot_grid_convergence_tt

setup = Input(0)
mf_model = ProposedMultiFidelityKriging(setup, MFK_kwargs = {})
if not 'EVA' in setup.solver_str: # type:ignore
    print("This plotting solution is EVA / containing a timetrace specific.")
    sys.exit()

if hasattr(setup,'model_end'):
    mf_model.set_state(deepcopy(setup.model_end)) #type:ignore
elif hasattr(setup,'model'):
    mf_model.set_state(deepcopy(setup.model)) #type:ignore
else:
    print("No model with data saved to this .json file")
    sys.exit()

mf_model.set_L([0.5, 1, 1.5])

Z, TT = [], []
nlvl = len(mf_model.L)
for l in range(nlvl):
    solver = get_solver(setup)
    z, _, tt = solver.solve(mf_model.X_truth,mf_model.L[l], get_time_trace = True)
    Z.append(z)
    TT.append(tt)

for i in range(2,nlvl):
    for j in range(i):
        for k in range(j):
            print("###### Checking the refinements: {}, {}, {} ######".format(mf_model.L[k],mf_model.L[j],mf_model.L[i]))
            # check_correlations(Z[k], Z[j], Z[i])
        

plot_grid_convergence(mf_model, Z)
plot_grid_convergence_tt(mf_model, TT)
plt.show()