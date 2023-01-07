import numpy as np
import sys 

np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)  # type: ignore
# pyright: reportGeneralTypeIssues=false, reportOptionalCall=false

import matplotlib.pyplot as plt

from preprocessing.input import Input

from core.proposed_method import weighted_prediction
from core.sampling.DoE import get_doe
from core.sampling.solvers.solver import get_solver
from core.sampling.infill_criteria import EI
from core.kriging.mf_kriging import MultiFidelityKriging, ProposedMultiFidelityKriging, MultiFidelityEGO
from core.kriging.kernel import get_kernel

from postprocessing.plotting import Plotting

from utils.formatting_utils import correct_formatX
from utils.linearity_utils import check_linearity
from utils.selection_utils import isin


# inits based on input settings
setup = Input(0)

solver = get_solver(setup)
kernel = get_kernel(setup.kernel_name, setup.d, setup.noise_regression) 
mf_model = MultiFidelityEGO(kernel, setup.d, solver, max_cost = 150000)

doe = get_doe(setup)
pp = Plotting(setup, plotting_pause = 0.01, plot_once_every=5)

###############################
# main
###############################

" level 0 and 1 : setting 'DoE' and 'solve' "
reuse_values = True
tune = True
if hasattr(setup,'model') and reuse_values:
    mf_model.set_state(setup.model)
else:
    X_l = doe(setup, n_per_d = 10)
    
    # list of the convergence levels we pass to solvers; different to the Kriging level we are assessing.
    mf_model.set_L([2, 3, 6])
    
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

setup.create_input_file(mf_model)

# draw the result
pp.set_zoom_inset([0], xlim  = [[0.7,0.82]], y_zoom_centres = [-8]) # not set: inset_rel_limits = [[]], 
pp.draw_current_levels(mf_model)

do_check = False
if do_check and not check_linearity(mf_model, pp):
    print("Linearity check: NOT LINEAR enough, but continueing for now.")
else:
    print("Linearity check: LINEAR enough!!")

# TODO MF EGO gedeelte naar de bijbehorend klasse porten
# oftewel alles hieronder moet naar ten eerste een EGO class; die samen met MFKriging of proposedMFKriging parent is voor MF-EGO
ei_criterion = 2 * np.finfo(np.float32).eps * 1e3

" sample from the predicted distribution in EGO fashion"
while np.sum(mf_model.costs_total) < mf_model.max_cost and isinstance(mf_model, MultiFidelityEGO):
    # select points to asses expected improvement
    X_infill = pp.X_pred  # TODO does not work for d>2 omdat X-pred de median neemt voor alle waardes in d not in dplot

    # predict and calculate Expected Improvement
    y_pred, sigma_pred = mf_model.K_mf[-1].predict(X_infill) 
    y_min = np.min(mf_model.Z_mf[-1])

    ei = EI(y_min, y_pred, np.sqrt(sigma_pred))

    # select best point to sample
    print("Maximum EI: {:4f}".format(np.max(ei)))
    x_new = correct_formatX(X_infill[np.argmax(ei)], setup.d)

    # terminate if criterion met, or when we revisit a point
    # NOTE level 2 is reinterpolated, so normally 0 EI at sampled points per definition!
    # however, we use regulation (both through R_diagonal of the proposed method as regulation constant for steady inversion)
    # this means the effective maximum EI can be higher than the criterion!
    if np.all(ei < ei_criterion) or isin(x_new,mf_model.X_mf[-1]):
        break

    # TODO implement level selection
    _, _, Ef_weighed = weighted_prediction(mf_model, X_test=x_new)
    l = mf_model.level_selection(x_new, Ef_weighed)
    print("Sampling on level {} at {}".format(l,x_new))
    
    mf_model.sample_nested(l, x_new)

    # weigh each prediction contribution according to distance to point.
    Z_pred, mse_pred, _ = weighted_prediction(mf_model)

    # build new kriging based on prediction
    # NOTE, we normalise mse_pred here with the previous known process variance, since otherwise we would arrive at a iterative formulation.
    # NOTE for top-level we require re-interpolation if we apply noise

    mf_model.K_mf[-1].train(
        mf_model.X_unique, Z_pred, tune=tune, R_diagonal=mse_pred / mf_model.K_mf[-1].sigma_hat
    )
    mf_model.K_mf[-1].reinterpolate()

    # " output "
    pp.draw_current_levels(mf_model)


print("Simulation finished")
pp.plot_once_every = 1
pp.draw_current_levels(mf_model)
plt.show()

