import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import sys


from core.sampling.solvers.solver import get_solver
from core.mfk.proposed_mfk import ProposedMultiFidelityKriging
from preprocessing.input import Input
from utils.correlation_utils import check_correlations
from postprocessing.plot_grid_convergences import plot_grid_convergence
from utils.selection_utils import isin_indices

setup = Input(0)
MFK_kwargs = {'print_global' : False,
                'print_training' : True,
                'print_prediction' : False,
                'eval_noise' : True,# always true
                # 'eval_noise' : setup.noise_regression, 
                'propagate_uncertainty' : False,  
                'optim_var' : False, # true: HF samples is forced to zero; = reinterpolation
                'hyper_opt' : 'Cobyla', # [‘Cobyla’, ‘TNC’] Cobyla standard
                'n_start': 30, # 10 = default, but I want it a bit more robust ( does not always tune to the same -> major influence to own result!)
                'corr' : 'squar_exp',
                }
mf_model = ProposedMultiFidelityKriging(setup, MFK_kwargs = MFK_kwargs)

mf_model.optim_var = True
mf_model.set_L([0.5, 1, 2])
mf_model.prepare_initial_surrogate(setup)  

if hasattr(setup,'model_end') and False:
    print("Set end model")
    mf_model.set_state(deepcopy(setup.model_end)) #type:ignore
elif hasattr(setup,'model'):
    print("Set start model")
    mf_model.set_state(deepcopy(setup.model)) #type:ignore
else:
    print("No model with data saved to this .json file")
    sys.exit()

mf_model.set_L([0.5, 1, 2])

# Z, TT = [], []
nlvl = len(mf_model.L)
# for l in range(nlvl):
#     solver = get_solver(setup)
#     z, _, tt = solver.solve(mf_model.X_truth,mf_model.L[l], get_time_trace = True)
#     Z.append(z)
#     TT.append(tt)

# for i in range(2,nlvl):
#     for j in range(i):
#         for k in range(j):
#             print("###### Checking the refinements: {}, {}, {} ######".format(mf_model.L[k],mf_model.L[j],mf_model.L[i]))
#             if i == nlvl - 1:
#                 Z_top = mf_model.Z_truth
#             else:
#                 Z_top = mf_model.Z_mf[i]
#             inds = isin_indices(mf_model.X_mf[j],mf_model.X_truth,inversed=False)
#             _, inds_sort = np.unique(mf_model.Z_mf[k][inds], axis = 0, return_index = True)
#             check_correlations(mf_model.Z_mf[k][inds][inds_sort], mf_model.Z_mf[j][inds][inds_sort], Z_top)
        
 
# plot_grid_convergence_Z(mf_model, Z)
# plot_grid_convergence_tt(mf_model, TT)
plot_grid_convergence(mf_model)
plt.show()