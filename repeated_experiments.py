# pyright: reportGeneralTypeIssues=false, reportOptionalCall=false
import numpy as np
import time

from utils.error_utils import RMSE_norm_MF, RMSE_focussed
np.set_printoptions(precision=3,linewidth = 150,sign=' ')

import warnings
warnings.simplefilter("ignore", RuntimeWarning) # Scipy optimize can throw x out of bounds, but this is really consequential

import pandas as pd
import subprocess
from multiprocessing.pool import ThreadPool, Pool
from multiprocessing import cpu_count

from core.sampling.DoE import LHS
from utils.selection_utils import create_X_infill

from core.sampling.solvers.internal import TestFunction
from core.sampling.solvers.solver import get_solver
from postprocessing.plot_live_metrics import ConvergencePlotting
from postprocessing.plotting import Plotting
from utils.linearity_utils import check_linearity

from preprocessing.input import Input

# options to pick starting class from
from core.mfk.mfk_base import MultiFidelityKrigingBase
from core.mfk.mfk_smt import MFK_smt
from core.mfk.mfk_ok import MFK_org
from core.mfk.proposed_mfk import ProposedMultiFidelityKriging
from core.routines.mfk_EGO import MultiFidelityEGO

def optimize_with_plotting(setup,mf_model):
    # incl plotting etc
    setup.d_plot = np.arange(d)[:2]
    pp = Plotting(setup, plotting_pause = 0.001, plot_once_every=1, fast_plot=True, make_video=False)
    pp.set_zoom_inset([0,3], x_rel_range = [0.05,0.2])
    cp = ConvergencePlotting(setup)
    mf_model.optimize(pp,cp)

# variables 
conv_mods = [0, 1, 2, 3]
solver_noises = [0.0, 0.02, 0.1] # to be passed as solver.noise_level = .. 
conv_types = ["Stable up", "Stable down", "Alternating"] # average these results
# conv_types = ["Stable up"] # average these results

# Solver / scenario dependent variables
initial_nr_samples = 1 # 1 en 3 voor Own, alleen 3 voor reference.
solver_str = 'Rosenbrock' # 'Rosenbrock'
d = 5 # 2 if branin, 5 or 10 if Rosenbrock

# Define the DoE based on a fake input
solver = get_solver(name = solver_str)
ss = solver.get_preferred_search_space(d)
ss[1], ss[2] = np.array(ss[1]), np.array(ss[2])
setup = lambda x: 0
setup.search_space = ss
setup.d = d
X_l = LHS(setup, n_per_d = 10)

# RMSE requirements
X_RMSE = create_X_infill(setup.d, setup.search_space[1], setup.search_space[2], int(1000**(1/d)))
RMSE_focuss_percentage = 10

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


def worker(conv_mod, conv_type, solver_noise):
    print(f"Started run: conv_mod {conv_mod}, {f'{conv_type},':<12} solver_noise {solver_noise}\n", end="\r")

    " Creating setup object"
    # fake setup as a function object
    setup = lambda x: 0

    # standard setup settings
    setup.kernel_name = "kriging"
    setup.noise_regression = True # Always true, but not always add noise!
    setup.solver_noise = True
    setup.mf = True
    
    setup.solver_str = solver_str
    setup.d = d
    setup.search_space = ss

    # set the variable settings
    setup.conv_mod = conv_mod
    setup.conv_type = conv_type
    setup.solver_noise = solver_noise
    
    " Creating new model "
    mf_model = MultiFidelityEGO(setup, initial_nr_samples = initial_nr_samples, max_cost = np.inf, MFK_kwargs = MFK_kwargs, printing = False)
    mf_model.set_L([2, 3, None])
    mf_model.set_L_costs([1,10,1000])   
    mf_model.prepare_proposed(setup, X_l)   

    # if mf_model.X_mf[-1].shape[0] >= 3:
    #     if not check_linearity(mf_model, None):
    #         print("WARNING Linearity check: NOT LINEAR enough!")
    #     else:
    #         print("Linearity check: LINEAR enough!")
    # else:
    #     print("Too little hifi samples for reliable linearity check!")

    " run "
    mf_model.optimize()
    # optimize_with_plotting(setup, mf_model)

    " get and provide outputs "
    # prepare
    range = mf_model.solver.value_metrics[1] - mf_model.solver.value_metrics[0]
    x_opt, z_opt = mf_model.solver.get_optima()
    x_opt, z_opt = x_opt[0], z_opt[0]

    x_best = mf_model.get_best_sample()[0][0]
    z_best = round(mf_model.get_best_sample()[1],4)

    # Sampling costs, given as (total sampling cost | high-fidelity sampling cost)
    cost_total = mf_model.costs_total
    cost_high  = mf_model.costs_per_level[2]

    # The number of high-fidelity and low-fidelity samples, reported as (high | low).
    nr_samples_high = mf_model.X_mf[-1].shape[0]
    nr_samples_low = mf_model.X_mf[0].shape[0]

    # Difference of optimized objective value to the truth; reported as (absolute | relative to the true objective value range)
    z_diff_absolute = abs(z_best - z_opt) 
    z_diff_relative = abs(z_best - z_opt) / range * 100

    # Euclidian (parameter-)distance of the optimization‘s end-solution to the exact known location of the optimum
    x_dist_euclidian = np.sqrt((x_best[0] - x_opt[1])**2 + (x_best[0] - x_opt[1])**2)

    # Normalized RMSE of the optimization‘s end-solution with respect to the known exact response surface:
    # – For the full surrogate
    RMSE = RMSE_norm_MF(mf_model, no_samples=True)

    # – For the ’focussed’ surrogate: only taking into account the responses with objective values 10% above the known optimum
    RMSE_focus = RMSE_focussed(mf_model, X_RMSE, RMSE_focuss_percentage)

    return cost_total, cost_high, nr_samples_high, nr_samples_low, z_diff_absolute, z_diff_relative, x_dist_euclidian, RMSE, RMSE_focus


amnt = len(conv_mods) * len(conv_types) * len(solver_noises)

def progress(unused_result):
    global amnt
    amnt = amnt - 1
    if amnt == 0:
        print(f" DONE {'':<110}")
    else:
        print(f"{amnt} runs of {len(conv_mods) * len(conv_types) * len(solver_noises)} left in queue", end='\r') # \033[A voor ook omhoog

if __name__ == '__main__':
    # setup dataframe    
    colum_names = ["cost_total", "cost_high", "nr_samples_high", "nr_samples_low", "z_diff_absolute", "z_diff_relative", "x_dist_euclidian", "RMSE", "RMSE_focus"]
    index_arrays = [
        np.repeat(conv_mods, len(solver_noises) * len(conv_types)),
        np.array(list(np.repeat(solver_noises, len(conv_types))) * len(conv_mods)),
        np.array(conv_types * len(solver_noises) * len(conv_mods))
    ] 

    result = pd.DataFrame(index = index_arrays, columns= colum_names)

    # setup process pool    
    tp = Pool(int(cpu_count()))
    workers = []
    for conv_mod in conv_mods:    
        for solver_noise in solver_noises:
            for conv_type in conv_types:
                # workers los toevoegen, .get() is uiteraard blocking, dus we moeten eerst close en join aanroepen.
                workers.append(tp.apply_async(worker, (conv_mod, conv_type, solver_noise), callback = progress))

    tp.close()
    tp.join()

    i = 0
    for conv_mod in conv_mods:    
        for solver_noise in solver_noises:
            for conv_type in conv_types:
                result.loc[conv_mod, solver_noise, conv_type] = workers[i].get()
                i += 1

    print(result)
