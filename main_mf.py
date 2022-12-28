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
from core.kriging.mf_kriging import MultiFidelityKriging, ProposedMultiFidelityKriging
from core.kriging.kernel import get_kernel

from postprocessing.plotting import Plotting

from utils.formatting_utils import correct_formatX
from utils.linearity_utils import check_linearity


# inits based on input settings
setup = Input(0)

solver = get_solver(setup)
kernel = get_kernel(setup.kernel_name, setup.d, setup.noise_regression) 
mf_model = ProposedMultiFidelityKriging(kernel, setup.d, solver, max_cost = 1500e5)

doe = get_doe(setup)
pp = Plotting(setup)

# list of the convergence levels we pass to solvers; different to the Kriging level we are assessing.
# mf_model.set_L([1, 3, 4])

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

    " initial prediction "
    Z_pred, mse_pred = weighted_prediction(mf_model)
    K_mf_new = mf_model.create_level(mf_model.X_unique, Z_pred, append = True, tune = tune, hps_noise_ub = True, R_diagonal= mse_pred / mf_model.K_mf[-1].sigma_hat)
    K_mf_new.reinterpolate()

# # if not check_linearity(mf_model, pp):
# #     print("Not linear enough, but continueing for now.")

mf_model.sample_truth()
setup.create_input_file(mf_model)

# draw the result
pp.set_zoom_inset([0], xlim  = [[0.7,0.82]], y_zoom_centres = [-8]) # not set: inset_rel_limits = [[]], 
pp.draw_current_levels(mf_model)

plt.draw()
plt.pause(1)
plt.show()
sys.exit()

# TODO MF EGO gedeelte naar de bijbehorend klasse porten
# oftewel alles hieronder moet naar ten eerste een EGO class; die samen met MFKriging of proposedMFKriging parent is voor MF-EGO
ei_criterion = 2 * np.finfo(np.float32).eps

" sample from the predicted distribution in EGO fashion"
while np.sum(mf_model.costs_total) < mf_model.max_cost:
    # select points to asses expected improvement
    X_infill = pp.X_pred  # TODO does not work for d>2

    # predict and calculate Expected Improvement
    y_pred, sigma_pred = K_mf_new.predict(X_infill)
    y_min = np.min(mf_model.Z_mf[-1])
    ei = np.zeros(X_infill.shape[0])
    for i in range(len(y_pred)):
        ei[i] = EI(y_min, y_pred[i], np.sqrt(sigma_pred[i]))

    # select best point to sample
    x_new = correct_formatX(X_infill[np.argmax(ei)], setup.d)

    # terminate if criterion met, or for some buggy reason we revisit a point
    if np.all(ei < ei_criterion) or x_new in mf_model.X_mf[2]:
        break

    # TODO implement level selection
    # select level l on which we will sample, based on location of ei and largest source of uncertainty there (corrected for expected cost).
    # source of uncertainty is from each component in proposed method which already includes approximated noise levels.
    # in practice this means: only for small S1 will we sample S0, otherwise S1. Nested DoE not required.
    l = 2

    # 1) check of de kriging variances gelijk zijn, aka of de predicted variances van de kriging toplevel 
    # ongeveer overeenkomen met de prediction gebaseerd op lagere levels. Dus punt x_new virtueel aan X_unique voor weighted_prediction toevoegen en vergelijken.
    # 1) variances van Kriging unknown scheiden
    # 2) meenemen in de weighing
    # 3) scalen mbv cost_exp naar Meliani, i.e. argmax(sigma_red[l]/cost_exp[l])
    
    mf_model.sample_nested(l, x_new)

    # weigh each prediction contribution according to distance to point.
    Z_pred, mse_pred = weighted_prediction(mf_model)

    # build new kriging based on prediction
    # NOTE, we normalise mse_pred here with the previous known process variance, since otherwise we would arrive at a iterative formulation.
    # NOTE for top-level we require re-interpolation if we apply noise

    K_mf_new.train(
        mf_model.X_unique, Z_pred, tune=tune, R_diagonal=mse_pred / K_mf_new.sigma_hat
    )
    K_mf_new.reinterpolate()

    # " output "
    X_unique_exc = mf_model.return_unique_exc(mf_model.X_mf[mf_model.l_hifi])
    pp.draw_current_levels(mf_model)
    plt.pause(1)



#####################################
# Postprocessing
#####################################



# if setup.SAVE_DATA:
#     setup.X = X
#     setup.Z = Z
#     setup.create_input_file()
# mf_model.get_state()


# # check_correlations(Z[0], Z[1], Z[2])
# RMSE_norm(Z_hifi_full, K_mf[-1].predict(X0))

print("Simulation finished")
show = True
if show: 
    plt.show()
else:
    plt.draw()
    plt.pause(10)
