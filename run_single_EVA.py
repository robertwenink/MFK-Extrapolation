# inits based on input settings
import numpy as np
from core.routines.mfk_EGO import MultiFidelityEGO
from preprocessing.input import Input


setup = Input(0)  
MFK_kwargs = {'print_global' : False,
                'print_training' : True,
                'print_prediction' : False,
                'eval_noise' : True,# always true
                # 'eval_noise' : setup.noise_regression, 
                'propagate_uncertainty' : False,  
                'optim_var' : True, # true: HF samples is forced to zero; = reinterpolation
                'hyper_opt' : 'Cobyla', # [‘Cobyla’, ‘TNC’] Cobyla standard
                'n_start': 30, # 10 = default, but I want it a bit more robust ( does not always tune to the same -> major influence to own result!)
                'corr' : 'squar_exp',
                }
mf_model = MultiFidelityEGO(setup, proposed = True, optim_var = True, initial_nr_samples = 2, max_cost = np.inf, MFK_kwargs = MFK_kwargs)

x = np.array([[0.5676, 0.0044]]) # high
# x = np.array([[0.8809, 0.5214]]) # low
refinement = 2.0
mf_model.solver.solve(x, refinement)
print("Done")

# to view in paraview, do following:
# click the eye next to the file
# click +y (in de balk)
# make two clips:
# set to scalar
# first one  fraction (partial) , set cutoff at 0.3
# second one body fraction (partial), set cutoff at 0.1
# set colors by pressing enter after clicking ball
# set in left pane the "camera parallel projection"
# onder file -> save animation -> set background color white (als nog niet gedaan)
