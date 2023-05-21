import os
import numpy as np
from core.routines.mfk_EGO import MultiFidelityEGO
from core.sampling.DoE import LHS
from core.sampling.solvers.solver import get_solver
from preprocessing.input import Input
from utils.correlation_utils import print_pearson_correlations


solver_str = "Branin"
# solver_str = "Rosenbrock"
d = 2
# d = 5

MFK_kwargs = {'print_global' : False,
                'print_training' : False,
                'print_prediction' : False,
                # 'eval_noise' : False,
                'eval_noise' : True, # Always true, but not always add noise!
                'propagate_uncertainty' : False,  
                'optim_var' : False, # true: HF samples is forced to zero; = reinterpolation
                'hyper_opt' : 'Cobyla', # [‘Cobyla’, ‘TNC’] Cobyla standard
                'n_start': 30, # 10 = default, but I want it a bit more robust ( does not always tune to the same -> major influence to own result!)
                'corr' : 'squar_exp',
                }

# Define the DoE based on a fake input
ss = get_solver(name = solver_str).get_preferred_search_space(d)
ss[1], ss[2] = np.array(ss[1]), np.array(ss[2])
setup = lambda x: 0
setup.search_space = ss
setup.d = d
setup.kernel_name = "kriging"
setup.noise_regression = False # Always true, but not always add noise!
setup.mf = True
setup.solver_str = solver_str

X_l = LHS(setup, n_per_d = 10)

def run(conv_type, conv_mod):
    # set the variable settings
    setup.conv_type = conv_type
    setup.conv_mod = conv_mod
    setup.solver_noise = 0.0
    
    " Creating new model "
    # NOTE optim_var was true voor 2d!! False voor 5d
    mf_model = MultiFidelityEGO(setup, proposed = False, optim_var = False, initial_nr_samples = 3, max_cost = np.inf, MFK_kwargs = MFK_kwargs, printing = False)
    mf_model.set_L([2, 3, None])
    mf_model.set_L_costs([1,10,1000])   
    mf_model.prepare_initial_surrogate(setup, X_l)   

    print_pearson_correlations(mf_model, print_raw=True)

run("Stable up",0)
run("Stable up",1)
run("Stable up",2)
run("Alternating",0)
run("Alternating",1)
run("Alternating",2)

" EVA "
setup = Input(0)
if setup.solver_str != "EVA":
    print("SELECTED WRONG SOLVER!!")

mf_model = MultiFidelityEGO(setup, proposed = False, optim_var = False, initial_nr_samples = 3, max_cost = np.inf, MFK_kwargs = MFK_kwargs, printing = False)
mf_model.set_L([0.5, 1, 2])
mf_model.solver.base_path = os.path.normpath("./EVA_Results/Wedge_optimization_constant_mass_low")
mf_model.prepare_initial_surrogate(setup)   
print_pearson_correlations(mf_model, print_raw=True)

mf_model = MultiFidelityEGO(setup, proposed = False, optim_var = False, initial_nr_samples = 3, max_cost = np.inf, MFK_kwargs = MFK_kwargs, printing = False)
mf_model.set_L([0.5, 1, 2])
mf_model.solver.base_path = os.path.normpath("./EVA_Results/Wedge_optimization_constant_mass_high")
mf_model.prepare_initial_surrogate(setup)   
print_pearson_correlations(mf_model, print_raw=True)
