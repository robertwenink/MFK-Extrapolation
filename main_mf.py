import numpy as np
np.set_printoptions(precision=3,linewidth = 150,sign=' ')
import sys
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11})

np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)  # type: ignore
# pyright: reportGeneralTypeIssues=false, reportOptionalCall=false

from utils.linearity_utils import check_linearity
from utils.formatting_utils import correct_formatX

from preprocessing.input import Input

from core.proposed_method import weighted_prediction
from core.sampling.DoE import get_doe
from core.kriging.mf_kriging import MultiFidelityKriging, ProposedMultiFidelityKriging, MultiFidelityEGO

from postprocessing.plotting import Plotting
from postprocessing.plot_convergence import ConvergencePlotting

from smt.applications import MFK


# inits based on input settings
setup = Input(0)

mf_model = MultiFidelityEGO(setup, initial_nr_samples = 1, max_cost = 150000)
# mf_model = ProposedMultiFidelityKriging(setup, max_cost = 150000)

doe = get_doe(setup)

pp = Plotting(setup, plotting_pause = 0.001, plot_once_every=1, fast_plot=True)
pp.set_zoom_inset([0,3], x_rel_range = [0.1,0.2]) # not set: inset_rel_limits = [[]], 
cp = ConvergencePlotting(setup)

###############################
# main
###############################

" level 0 and 1 : setting 'DoE' and 'solve' "
reuse_values = True
reload_endstate = True
tune = True

used_endstate = hasattr(setup,'model_end') and reload_endstate
if used_endstate:
    mf_model.set_state(setup.model_end)
    cp.set_state(setup)
elif hasattr(setup,'model') and reuse_values:
    mf_model.set_state(setup.model)
else:
    X_l = doe(setup, n_per_d = 10)
    
    # list of the convergence levels we pass to solvers; different to the Kriging level we are assessing.
    mf_model.set_L([2, 3, None])
    
    # create Krigings of levels, same initial hps
    if setup.d == 2:
        hps = np.array([-1.42558281e+00, -2.63967644e+00, 2.00000000e+00, 2.00000000e+00, 1.54854970e-04])
        mf_model.create_level(X_l, tune=tune, hps_init=hps)
    elif setup.d == 1:
        hps = np.array([1.26756467e+00, 2.00000000e+00, 9.65660595e-04])
        mf_model.create_level(X_l, tune=tune, hps_init=hps)
        # forrester moet ongeveer gefocust rond x = 0.62, y = -1.3
    else:
        mf_model.create_level(X_l, tune=tune)
        
    mf_model.create_level(X_l, tune=tune)

    " level 2 / hifi initialisation "
    mf_model.sample_initial_hifi(setup)

    mf_model.sample_truth()

    " initial prediction "
    Z_pred, mse_pred, _ = weighted_prediction(mf_model)
    K_mf_new = mf_model.create_level(mf_model.X_unique, Z_pred, append = True, tune = tune, hps_noise_ub = True, R_diagonal= mse_pred / mf_model.K_mf[-1].sigma_hat)
    K_mf_new.reinterpolate()
    
    setup.create_input_file(mf_model, cp, endstate = False)

mf_model.set_L_costs([1,9,10000])   

sm = MFK(theta_bounds = [1e-1, 20],
         use_het_noise = False,
         propagate_uncertainty=True, 
         n_start=10)

for i, K in enumerate(mf_model.K_mf):
    X = K.X
    y = K.y.reshape(-1,1)
    print(f"{X.shape:} {y.shape:}")
    if i == mf_model.number_of_levels - 1:
        sm.set_training_values(X, y)
    else:
        sm.set_training_values(X, y, name = i)

# sm.set_training_values(mf_model.X_mf,mf_model.Z_mf)
sm.train()

# HF
# x = correct_formatX(X[0],setup.d)
x = X

# for the HF  
y = sm.predict_values(x)
var = sm.predict_variances(x)

# values, including intermediate levels
y0 = sm._predict_intermediate_values(x, 1)
y1 = sm._predict_intermediate_values(x, 2)
y2 = sm.predict_values(x)

# mse`s including, including intermediate levels
MSE, sigma2_rhos = sm.predict_variances_all_levels(x)
l = 0 # level
std = np.sqrt(MSE[:,l].reshape(-1,1))

do_check = False
if do_check and not check_linearity(mf_model, pp):
    print("Linearity check: NOT LINEAR enough, but continueing for now.")
else:
    print("Linearity check: LINEAR enough!!")



" sample from the predicted distribution in EGO fashion"
if isinstance(mf_model, MultiFidelityEGO):
    mf_model.optimize(pp,cp)

setup.create_input_file(mf_model, cp, endstate = True)
cp.plot_convergence(mf_model)

print("Simulation finished")

plt.show()

