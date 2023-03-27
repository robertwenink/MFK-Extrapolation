#%%
# pyright: reportGeneralTypeIssues=false, reportOptionalCall=false
import logging
logging.basicConfig(filename='toy_experiments.log', encoding='utf-8', filemode='w')


import numpy as np
import time
from copy import deepcopy

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

# NOTE testing only
# conv_types = ["Stable up"] # average these results
# solver_noises = [0.0] # to be passed as
# conv_mods = [0]


# Solver / scenario dependent variables
proposed_and_samples = [(True, 1), (True, 3), (False, 3)] # 1 en 3 voor Own, alleen 3 voor reference.
solver_str = 'Branin' # 'Rosenbrock'
d = 2 # 2 if branin, 5 or 10 if Rosenbrock
scenarios = [['Branin',2],
             ['Rosenbrock',2],
             ['Rosenbrock',5]]

# Define the DoE based on a fake input
solver = get_solver(name = solver_str)
ss = solver.get_preferred_search_space(d)
ss[1], ss[2] = np.array(ss[1]), np.array(ss[2])
setup = lambda x: 0
setup.search_space = ss
setup.d = d
X_l = LHS(setup, n_per_d = 10)

# RMSE requirements
# X_RMSE = create_X_infill(setup.d, setup.search_space[1], setup.search_space[2], int(1000**(1/d)))
X_RMSE = LHS(setup, n_per_d=40)
RMSE_focuss_percentage = 10

MFK_kwargs = {'print_global' : False,
                'print_training' : False,
                'print_prediction' : False,
                # 'eval_noise' : False,
                'eval_noise' : True, # Always true, but not always add noise!
                'propagate_uncertainty' : False,  
                'optim_var' : True, # true: HF samples is forced to zero; = reinterpolation
                'hyper_opt' : 'Cobyla', # [‘Cobyla’, ‘TNC’] Cobyla standard
                'n_start': 30, # 10 = default, but I want it a bit more robust ( does not always tune to the same -> major influence to own result!)
                'corr' : 'squar_exp',
                }


def worker(proposed, nr_samples, conv_mod, solver_noise, conv_type):
    try:
        if proposed:
            print(f"Started run: Proposed,  {nr_samples} start samples, conv_mod {conv_mod}, {f'{conv_type},':<12} solver_noise {solver_noise}\n", end="\r")
        else:
            print(f"Started run: Reference, {nr_samples} start samples, conv_mod {conv_mod}, {f'{conv_type},':<12} solver_noise {solver_noise}\n", end="\r")

        # return 1,1,1,1,1,1,1,1,1,1,1

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
        mf_model = MultiFidelityEGO(setup, proposed = proposed, initial_nr_samples = nr_samples, max_cost = np.inf, MFK_kwargs = MFK_kwargs, printing = False)
        mf_model.set_L([2, 3, None])
        mf_model.set_L_costs([1,10,1000])   
        mf_model.prepare_initial_surrogate(setup, X_l)   

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
        x_dist_euclidian = np.sqrt((x_best[0] - x_opt[0])**2 + (x_best[1] - x_opt[1])**2)

        # Normalized RMSE of the optimization‘s end-solution with respect to the known exact response surface:
        # – For the full surrogate
        RMSE = RMSE_norm_MF(mf_model, no_samples=True)

        # – For the ’focussed’ surrogate: only taking into account the responses with objective values 10% above the known optimum
        RMSE_focus = RMSE_focussed(mf_model, X_RMSE, RMSE_focuss_percentage)

        return cost_total, cost_high, nr_samples_high, nr_samples_low, z_diff_absolute, z_diff_relative, x_dist_euclidian, RMSE[1], RMSE[2], RMSE_focus[1], RMSE_focus[2]
    except:
        print(f"WARNING: Run {proposed, nr_samples, conv_mod, solver_noise, conv_type} failed!")
        logging.exception(f"Run {proposed, nr_samples, conv_mod, solver_noise, conv_type} failed with following message:")
        logging.info('\n')
        # return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


amnt = len(conv_mods) * len(conv_types) * len(solver_noises) * len(proposed_and_samples)
amnt_tot = deepcopy(amnt)
def progress(unused_result):
    global amnt
    amnt = amnt - 1

    print(f"{amnt} runs of {amnt_tot} left in queue  ", end='\r') # \033[A voor ook omhoog
    if amnt == 0:
        print("")


filename = f"{solver_str}_{d}d.csv"

#%%
if __name__ == '__main__':
    # setup dataframe    
    
    colum_names = ["cost_total", "cost_high", "nr_samples_high", "nr_samples_low", "z_diff_absolute", \
                   "z_diff_relative", "x_dist_euclidian", "RMSE_low", "RMSE", "RMSE_focus_low", "RMSE_focus"]
    new_indexes = pd.MultiIndex.from_product([colum_names, proposed_and_samples])
    tuple_indexlist = [(method, samples, proposed) for method, (samples, proposed) in new_indexes.values]
    index_names = ["Metric", "Proposed_method", "Nr_initial_samples"]
    new_indexes = pd.MultiIndex.from_tuples(tuple_indexlist, names = index_names)
    new_columns = pd.MultiIndex.from_product([conv_mods, solver_noises, conv_types], 
                                             names = ["Convergence_modifier", "Solver_noise", "Convergence_types"])
    df = pd.DataFrame(index = new_indexes, columns = new_columns)

    # setup process pool    
    tp = Pool(int(cpu_count()))
    workers = []
    for proposed, nr_samples in proposed_and_samples:
        for conv_mod in conv_mods:    
            for solver_noise in solver_noises:
                for conv_type in conv_types:
                    # workers los toevoegen, .get() is uiteraard blocking, dus we moeten eerst close en join aanroepen.
                    workers.append(tp.apply_async(worker, (proposed, nr_samples, conv_mod, solver_noise, conv_type), callback = progress))

    # close and join pool
    tp.close()
    tp.join()
    
    # write results to dataframe
    i = 0
    idx = pd.IndexSlice # needed bcs we cant do .loc[bla,bla2,bla3]["c","c2"] etc omdat dat een slice/ projectie is
    for proposed, nr_sample in proposed_and_samples:
        for conv_mod in conv_mods:    
            for solver_noise in solver_noises:
                for conv_type in conv_types:
                    df.loc[idx[:, proposed, nr_sample], idx[conv_mod, solver_noise, conv_type]] = workers[i].get() 
                    i += 1
    
    print(df)

    # save the complete dataframe
    df.to_csv(filename)

    #%%
    # reading into multi-index
    filename = f"{solver_str}_{d}d.csv"
    df = pd.read_csv(filename, index_col=[0,1,2], header = [0,1,2])

    # take the mean over axis 1, grouped by the first two levels (so except convergence_types)
    df = df.groupby(level=[0,1], axis = 1).mean().apply(lambda x: round(x, 3))

    idx = pd.IndexSlice 
    # print(df.loc[idx[["RMSE_low", "RMSE"],:,:]])
    proj = df.loc[idx[["RMSE_low", "RMSE"],:,:]].groupby(level = [1,2]).agg(tuple)
    proj = df.loc[idx[["RMSE_focus_low", "RMSE_focus"],:,:]].groupby(level = [1,2]).agg(tuple)

    # prepend the correct indices again (groupby loses these)
    df.loc[idx["RMSE",:,:]] = pd.concat({'RMSE': proj}, names=['Metric'])
    df.loc[idx["RMSE_focus",:,:]] = pd.concat({'RMSE_focus': proj}, names=['Metric'])

    # remove non-needed parts now
    df.drop(('RMSE_low'), axis=0, inplace=True)
    df.drop(('RMSE_focus_low'), axis=0, inplace=True)

    filename = f"{solver_str}_{d}d_processed.csv"

    df.to_csv(filename, float_format='%.3f')

    # %%
