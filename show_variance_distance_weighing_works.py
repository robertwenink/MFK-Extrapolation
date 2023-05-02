import numpy as np

from core.sampling.solvers.internal import TestFunction
from core.sampling.solvers.solver import get_solver
from utils.error_utils import RMSE_norm
from utils.selection_utils import isin_indices
np.set_printoptions(precision=4,linewidth = 150,sign=' ',suppress = True)
import sys
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11})

import warnings
warnings.simplefilter("ignore", RuntimeWarning) # Scipy optimize can throw x out of bounds, but this is really consequential

# np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)  # type: ignore
from copy import deepcopy, copy
# pyright: reportGeneralTypeIssues=false, reportOptionalCall=false

from utils.linearity_utils import check_linearity

from preprocessing.input import Input

# options to pick starting class from
from core.mfk.mfk_base import MultiFidelityKrigingBase
from core.mfk.mfk_smt import MFK_smt
from core.mfk.mfk_ok import MFK_org
from core.mfk.proposed_mfk import ProposedMultiFidelityKriging
from core.routines.mfk_EGO import MultiFidelityEGO

from postprocessing.plotting import Plotting
from postprocessing.plot_live_metrics import ConvergencePlotting


# init plotting etc
setup = Input(0)
pp = Plotting(setup, plotting_pause = 0.001, plot_once_every=1, fast_plot=False, make_video=True)
pp.set_zoom_inset([0,3], x_rel_range = [0.05,0.2])
pp.plot_NRMSE_text = True

conv_types = ["Stable up", "Stable down", "Alternating"]

setup.conv_mod = 1
setup.conv_type = conv_types[2]
setup.solver_noise = 0.1

reuse_values = False
reload_endstate = False
# NOTE deze waardes aanpassen werkt alleen als reuse_values en reload_endstate uitstaat!
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
   
# NOTE zonder noise werkt zonder optim var beter voor reference
# mf_model = MFK_smt(setup, max_cost = 150000, initial_nr_samples = 1, **MFK_kwargs)# NOTE cant use one (1) because of GLS in smt!
mf_model = MultiFidelityEGO(setup, proposed = True, optim_var= False, initial_nr_samples = 1, max_cost = np.inf, MFK_kwargs = MFK_kwargs)

mf_model.method_weighing = False
mf_model.variance_weighing = False
mf_model.distance_weighing = False

# startpunt van voorbeeld eerste hifi punt moet zijn: x = 0.4161
mf_model.set_L([1, 2, None])
mf_model.number_of_levels = 3

X = [[0.4161], [0.1942], [0.7413]]
Z0 = mf_model.solver.solve(np.atleast_2d(X), mf_model.L[0])[0]
Z1 = mf_model.solver.solve(np.atleast_2d(X), mf_model.L[1])[0]
Z2 = mf_model.solver.solve(np.atleast_2d(X), mf_model.L[2])[0]

if True:
    X_mf = [np.array([[ 0.3198],
        [ 0.8034],
        [ 0.5142],
        [ 0.9624],
        [ 0.6783],
        [ 0.2412],
        [ 0.1672],
            X[0],
        [ 0.0593],
        [ 0.1673],
        [ 0.1671],
            X[1],
            X[2]]), np.array([[ 0.3198],
        [ 0.8034],
        [ 0.5142],
        [ 0.9624],
        [ 0.6783],
        [ 0.2412],
        [ 0.1672],
            X[0],
        [ 0.0593],
        [ 0.1673],
        [ 0.1671],
            X[1],
            X[2]]), np.array([X[0]])]
    X_truth = np.array([[ 0.0593],
        [ 0.1672],
        [ 0.2412],
        [ 0.3198],
            X[0] ,
        [ 0.5142],
        [ 0.6783],
        [ 0.8034],
        [ 0.9624]])
    Z_mf = [np.array([  4.238 , -15.4531,  -3.0895,  20.7512, -14.3335,   5.6141,   4.3662,  Z0[0],  -1.5417,   4.384 ,   4.253, Z0[1], Z0[2]]), np.array([-2.1245, -0.1465,  2.9541,  9.803 ,  1.6303, -3.1132, -3.3135,  Z1[0],  1.426 , -3.3298, -3.2937, Z1[1], Z1[2]]), np.array([Z2[0]])]
    Z_truth = np.array([ 0.3952, -0.9063, -0.2733, -0.0011,  Z2[0],  0.9723, -3.6018, -4.7647, 13.5881])
    mf_model.X_mf, mf_model.X_truth, mf_model.Z_mf, mf_model.Z_truth, mf_model.X_unique = X_mf, X_truth, Z_mf, Z_truth, X_mf[0]
    mf_model.set_L_costs([1,10,1000])   

    # mf_model.create_update_K_truth()
    mf_model.train()
    mf_model.weighted_prediction()
    mf_model.create_update_K_pred()
    pp.fig.suptitle(pp.figure_title + ": No weighing routines", x = 0.625, y = 0.95)
    pp.draw_current_levels(mf_model)

    X_mf = [np.array([[ 0.3198],
        [ 0.8034],
        [ 0.5142],
        [ 0.9624],
        [ 0.6783],
        [ 0.2412],
        [ 0.1672],
            X[0],
        [ 0.0593],
        [ 0.1673],
        [ 0.1671],
            X[1],
            X[2]]), np.array([[ 0.3198],
        [ 0.8034],
        [ 0.5142],
        [ 0.9624],
        [ 0.6783],
        [ 0.2412],
        [ 0.1672],
            X[0],
        [ 0.0593],
        [ 0.1673],
        [ 0.1671],
            X[1],
            X[2]]), np.array([X[0],X[1]])]
    X_truth = np.array([[ 0.0593],
        [ 0.1672],
        [ 0.2412],
        [ 0.3198],
            X[0] ,
        [ 0.5142],
        [ 0.6783],
        [ 0.8034],
        [ 0.9624],
            X[1]])
    Z_mf = [np.array([  4.238 , -15.4531,  -3.0895,  20.7512, -14.3335,   5.6141,   4.3662,  Z0[0],  -1.5417,   4.384 ,   4.253, Z0[1], Z0[2]]), np.array([-2.1245, -0.1465,  2.9541,  9.803 ,  1.6303, -3.1132, -3.3135,  Z1[0],  1.426 , -3.3298, -3.2937, Z1[1], Z1[2]]), np.array([Z2[0], Z2[1]])]
    Z_truth = np.array([ 0.3952, -0.9063, -0.2733, -0.0011,  Z2[0],  0.9723, -3.6018, -4.7647, 13.5881, Z2[1]])
    mf_model.X_mf, mf_model.X_truth, mf_model.Z_mf, mf_model.Z_truth, mf_model.X_unique = X_mf, X_truth, Z_mf, Z_truth, X_mf[0]
    mf_model.set_L_costs([1,10,1000])   

    # mf_model.create_update_K_truth()
    mf_model.train()
    mf_model.weighted_prediction()
    mf_model.create_update_K_pred()

    # remove the setin axin ax from hereon out
    pp.axes[0].axin.remove()
    del pp.axes[0].axin

    pp.draw_current_levels(mf_model)

X_mf = [np.array([[ 0.3198],
       [ 0.8034],
       [ 0.5142],
       [ 0.9624],
       [ 0.6783],
       [ 0.2412],
       [ 0.1672],
        X[0],
       [ 0.0593],
       [ 0.1673],
       [ 0.1671],
        X[1],
        X[2]]), np.array([[ 0.3198],
       [ 0.8034],
       [ 0.5142],
       [ 0.9624],
       [ 0.6783],
       [ 0.2412],
       [ 0.1672],
        X[0],
       [ 0.0593],
       [ 0.1673],
       [ 0.1671],
        X[1],
        X[2]]), np.array([X[0],X[1],X[2]])]
X_truth = np.array([[ 0.0593],
       [ 0.1672],
       [ 0.2412],
       [ 0.3198],
        X[0] ,
       [ 0.5142],
       [ 0.6783],
       [ 0.8034],
       [ 0.9624],
        X[1],
        X[2]])
Z_mf = [np.array([  4.238 , -15.4531,  -3.0895,  20.7512, -14.3335,   5.6141,   4.3662,  Z0[0],  -1.5417,   4.384 ,   4.253, Z0[1], Z0[2]]), np.array([-2.1245, -0.1465,  2.9541,  9.803 ,  1.6303, -3.1132, -3.3135,  Z1[0],  1.426 , -3.3298, -3.2937, Z1[1], Z1[2]]), np.array([Z2[0], Z2[1], Z2[2]])]
Z_truth = np.array([ 0.3952, -0.9063, -0.2733, -0.0011,  Z2[0],  0.9723, -3.6018, -4.7647, 13.5881, Z2[1], Z2[2]])
mf_model.X_mf, mf_model.X_truth, mf_model.Z_mf, mf_model.Z_truth, mf_model.X_unique = X_mf, X_truth, Z_mf, Z_truth, X_mf[0]
mf_model.set_L_costs([1,10,1000])   

# mf_model.create_update_K_truth()
mf_model.train()
mf_model.weighted_prediction()
mf_model.create_update_K_pred()
pp.draw_current_levels(mf_model)

# " sample from the predicted distribution in EGO fashion"
# if isinstance(mf_model, MultiFidelityEGO):
#     # do just one step, get next EI
#     mf_model.optimize(pp, cp, max_iter = 1)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~ Only distance weighing ~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Enable distance weighing
mf_model.method_weighing = False
mf_model.variance_weighing = False
mf_model.distance_weighing = True
mf_model.train()
mf_model.weighted_prediction()
mf_model.create_update_K_pred()
pp.fig.suptitle(pp.figure_title + ": Distance weighing", x = 0.625, y = 0.95)
pp.draw_current_levels(mf_model)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~ Only Variance weighing ~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Also enable variance weighing
mf_model.method_weighing = False
mf_model.variance_weighing = True
mf_model.distance_weighing = False
mf_model.train()
mf_model.weighted_prediction()
mf_model.create_update_K_pred()
pp.fig.suptitle(pp.figure_title + ": Variance weighing", x = 0.625, y = 0.95)
pp.draw_current_levels(mf_model)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~ Distance + Variance weighing ~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Also enable variance weighing
mf_model.method_weighing = False
mf_model.variance_weighing = True
mf_model.distance_weighing = True
mf_model.train()
mf_model.weighted_prediction()
mf_model.create_update_K_pred()
pp.fig.suptitle(pp.figure_title + ": Distance + Variance weighing", x = 0.625, y = 0.95)
pp.draw_current_levels(mf_model)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~ Distance + Method weighing ~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Also enable method_weighing weighing
mf_model.method_weighing = True
mf_model.variance_weighing = False
mf_model.distance_weighing = True
mf_model.train()
mf_model.weighted_prediction()
mf_model.create_update_K_pred()
pp.fig.suptitle(pp.figure_title + ": Distance + Method (MFK) weighing", x = 0.625, y = 0.95)
pp.draw_current_levels(mf_model)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~ Distance + Variance + Method weighing MFK ~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Also enable method_weighing weighing
mf_model.method_weighing = True
mf_model.variance_weighing = True
mf_model.distance_weighing = True
mf_model.train()
mf_model.weighted_prediction()
mf_model.create_update_K_pred()
pp.fig.suptitle(pp.figure_title + ": Distance + Variance + Method (MFK) weighing", x = 0.625, y = 0.95)
pp.draw_current_levels(mf_model)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~ Distance + Variance + Method weighing Z1 ~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Also enable method_weighing weighing
mf_model.method_weighing = True
mf_model.variance_weighing = True
mf_model.distance_weighing = True
mf_model.try_use_MFK = False
mf_model.train()
mf_model.weighted_prediction()
mf_model.create_update_K_pred()
pp.fig.suptitle(pp.figure_title + ": Distance + Variance + Method (Z1) weighing", x = 0.625, y = 0.95)
pp.draw_current_levels(mf_model)



pp.render_video()
print(" Simulation finished ")
plt.show()