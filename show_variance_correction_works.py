#%%
# pyright: reportGeneralTypeIssues=false, reportOptionalCall=false
import logging
logging.basicConfig(filename='toy_experiments.log', encoding='utf-8', filemode='w')


import numpy as np
import time
from copy import deepcopy

from utils.error_utils import RMSE_norm_MF, RMSE_focussed
np.set_printoptions(precision=4,linewidth = 150,sign=' ')

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

def create_Xs(solver_str, d, nr_repetitions):
    # Define the DoE based on a fake input
    ss = get_solver(name = solver_str).get_preferred_search_space(d)
    ss[1], ss[2] = np.array(ss[1]), np.array(ss[2])
    setup = lambda x: 0
    setup.search_space = ss
    setup.d = d

    XX_l = []
    for repetition in range(nr_repetitions):
        X_l = LHS(setup, n_per_d = 10, random_state = repetition)
        XX_l.append(X_l)

    return XX_l

def worker(conv_mod, solver_noise, conv_type, X_l, solver_str, d):
    try:
        print(f"\rStarted run: conv_mod {conv_mod}, {f'{conv_type},':<12} solver_noise {solver_noise}\n", end="\r")
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
        ss = get_solver(name = solver_str).get_preferred_search_space(d)
        ss[1], ss[2] = np.array(ss[1]), np.array(ss[2])
        setup.search_space = ss

        # set the variable settings
        setup.conv_mod = conv_mod
        setup.conv_type = conv_type
        setup.solver_noise = solver_noise

        mf_model = MultiFidelityEGO(setup, proposed = True, optim_var = True, initial_nr_samples = 1, max_cost = np.inf, MFK_kwargs = MFK_kwargs, printing = False)
        mf_model.method_weighing = False
        mf_model.set_L([2, 3, None])
        mf_model.set_L_costs([1,10,1000])       
        mf_model.prepare_initial_surrogate(setup, X_l)   
        RMSE_corrected = RMSE_norm_MF(mf_model, no_samples=True)
        Ef_corrected = mf_model.last_Ef

        mf_model = MultiFidelityEGO(setup, proposed = True, optim_var = True, initial_nr_samples = 1, max_cost = np.inf, MFK_kwargs = MFK_kwargs, printing = False)
        mf_model.use_uncorrected_Ef = True
        mf_model.method_weighing = False
        mf_model.set_L([2, 3, None])
        mf_model.set_L_costs([1,10,1000])       
        mf_model.prepare_initial_surrogate(setup, X_l)   
        RMSE_naive = RMSE_norm_MF(mf_model, no_samples=True)
        Ef_naive = mf_model.last_Ef

        return RMSE_naive[-1], RMSE_corrected[-1], Ef_naive, Ef_corrected
    except:
        print(f"WARNING: Run {conv_mod, solver_noise, conv_type} failed!")
        logging.exception(f"Run {conv_mod, solver_noise, conv_type} failed with following message:")
        logging.info('\n')


def progress(unused_result):
    global amnt
    amnt = amnt - 1

    # works untill 1 day, afterwards gives +1 day answer
    amnt_completed = amnt_tot - amnt
    t_end_exp = time.strftime('%a %H:%M:%S', time.localtime(time.time() + (time.time() - t_start_scenario)/amnt_completed * amnt))

    print(f"{amnt} runs of {amnt_tot} left in queue. Estimated end scenario at {t_end_exp}", end='\r') # \033[A voor ook omhoog
    if amnt == 0:
        print("")

def run_scenario(solver_str, d, XX_l):
    string = f"~~~~ Running {solver_str} {d}d scenario with {amnt_tot} runs ~~~~"
    print("~"*len(string))
    print(string)
    print("~"*len(string))
    filename = f"show_Ef_correction_{solver_str}_{d}d.csv"
    nr_repetitions = len(XX_l)

    # setting up the dataframe with multi_indexes
    column_names = ["RMSE","Ef"]
    indexes = pd.MultiIndex.from_product([[f"{solver_str}_{d}d"],column_names, ["Naive","Corrected"]])
    new_columns = pd.MultiIndex.from_product([conv_mods, solver_noises, conv_types, np.arange(nr_repetitions)], 
                                             names = ["Convergence_modifier", "Solver_noise", "Convergence_type", "Repetition"])
    df = pd.DataFrame(index = indexes, columns = new_columns)

    # setup process pool    
    tp = Pool(int(cpu_count()))
    workers = []
    for conv_mod in conv_mods:    
        for solver_noise in solver_noises:
            for conv_type in conv_types:
                for run_nr in range(nr_repetitions):
                    # workers los toevoegen, .get() is uiteraard blocking, dus we moeten eerst close en join aanroepen.
                    workers.append(tp.apply_async(worker, (conv_mod, solver_noise, conv_type, XX_l[run_nr], solver_str, d), callback = progress))

    # close and join pool
    tp.close()
    tp.join()
    
    # write results to dataframe
    i = 0
    idx = pd.IndexSlice # needed bcs we cant do .loc[bla,bla2,bla3]["c","c2"] etc omdat dat een slice/ projectie is
    for conv_mod in conv_mods:    
        for solver_noise in solver_noises:
            for conv_type in conv_types:
                for run_nr in range(nr_repetitions):
                    df.loc[idx[:,:,:], idx[conv_mod, solver_noise, conv_type, run_nr]] = workers[i].get() 
                    i += 1
    
    print(df)
    
    # save the complete dataframe
    df.to_csv(filename)
    t = (time.time() - t_start_scenario)
    print(f"Batch took {np.floor(t/3600)}:{round(t%3600/60)} hours")

if __name__ == '__main__':
    global amnt_batch, amnt, amnt_tot, solver_str, d, conv_types, solver_noises, conv_mods, t_start_scenario

    # variables 
    conv_mods = [0, 1, 2, 3]
    solver_noises = [0.0, 0.02, 0.1] # to be passed as solver.noise_level = .. 
    conv_types = ["Stable up", "Stable down", "Alternating"] # average these results

    nr_repetitions = 10

    scenarios = [('Branin',2),
            ('Rosenbrock',2),
            ('Rosenbrock',5)]
 
    # run scenarios
    for solver_str, d in scenarios:    
        amnt_tot = nr_repetitions * len(conv_mods) * len(conv_types) * len(solver_noises)
        amnt = deepcopy(amnt_tot)

        XX_l = create_Xs(solver_str, d, nr_repetitions)

        t_start_scenario = time.time()
        
        run_scenario(solver_str, d, XX_l)

    # process scenario output files
    df_all = []
    for solver_str, d in scenarios:    
        filename = f"show_Ef_correction_{solver_str}_{d}d"
        df_all.append(pd.read_csv(filename+".csv", index_col=[0,1,2], header = [0,1,2,3]))

    df_all = pd.concat(df_all)
    df_all = df_all.groupby(level=[0,1], axis = 1).mean()#.apply(lambda x: f"{round(x, 2):.2f}")
    df_all.to_csv("show_Ef_processed.csv", float_format='%.6f')

    

