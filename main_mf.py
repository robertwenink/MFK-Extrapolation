import numpy as np
np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)  # type: ignore
# pyright: reportGeneralTypeIssues=false, reportOptionalCall=false

import matplotlib.pyplot as plt
import sys

from preprocessing.input import Input

from core.proposed_method import weighted_prediction
from core.sampling.DoE import get_doe
from core.sampling.solvers.solver import get_solver
from core.kriging.mf_kriging import MultiFidelityKriging
from core.kriging.kernel import get_kernel

# from postprocessing.plotting import Plotting

from utils.formatting_utils import correct_formatX
from utils.linearity_utils import check_linearity
from utils.correlation_utils import check_correlations
from utils.error_utils import RMSE_norm_MF

# inits based on input settings
setup = Input(0)

solver = get_solver(setup)
doe = get_doe(setup)
kernel = get_kernel(setup.kernel_name, setup.d, setup.noise_regression) 
# pp = Plotting(setup)

mf_model = MultiFidelityKriging(kernel, setup.d, solver, max_cost = 1500e5)

# list of the convergence levels we pass to solvers; different to the Kriging level we are assessing.
mf_model.set_L([1, 3, 4])

################
# Settings
################

" inits and settings"
n_samples_l0 = 10 


###############################
# main
###############################

" level 0 and 1 : setting 'DoE' and 'solve' "
reuse_values = True
if reuse_values:
    mf_model.set_state(setup)

    # sequentially retrain all
else:
    X_l = doe(setup)
    
    # create Krigings of levels, same initial hps
    mf_model.add_level(X_l, tune=True)
    mf_model.add_level(X_l, tune=True)


" level 2 / hifi initialisation "
mf_model.sample_initial_hifi(setup)

# if not check_linearity(setup, X, X_unique, Z, K_mf, pp, L):
#     print("Not linear enough, but continueing for now.")


" initial prediction "
Z_pred, mse_pred = weighted_prediction(mf_model)
K_mf_new = mf_model.add_level(setup, mf_model.X_unique, Z_pred, hps_noise_ub = True, tune = True, R_diagonal=mse_pred / mf_model.K_mf[-1].sigma_hat)
K_mf_new.reinterpolate()


# draw the result
pp.draw_current_levels(X, Z, [*K_mf, K_mf_new], X_unique_exc, L)
pp.plot_kriged_truth(setup,X0, Z_hifi_full, hps_init = K_mf[-1].hps)
plt.show()

sys.exit(0)

" sample from the predicted distribution in EGO fashion"
while np.sum(costs) < max_cost:
    # select points to asses expected improvement
    X_infill = pp.X_pred  # TODO does not work for d>2

    # predict and calculate Expected Improvement
    y_pred, sigma_pred = K_mf_new.predict(X_infill)
    y_min = np.min(Z[-1])
    ei = np.zeros(X_infill.shape[0])
    for i in range(len(y_pred)):
        ei[i] = EI(y_min, y_pred[i], np.sqrt(sigma_pred[i]))

    # select best point to sample
    x_new = correct_formatX(X_infill[np.argmax(ei)], setup.d)

    # terminate if criterion met, or for some buggy reason we revisit a point
    if np.all(ei < ei_criterion) or x_new in X[2]:
        break

    # TODO implement level selection
    # select level l on which we will sample, based on location of ei and largest source of uncertainty there (corrected for expected cost).
    # source of uncertainty is from each component in proposed method which already includes approximated noise levels.
    # in practice this means: only for small S1 will we sample S0, otherwise S1. Nested DoE not required.

    # 1) check of de kriging variances gelijk zijn, aka of de predicted variances van de kriging toplevel 
    # ongeveer overeenkomen met de prediction gebaseerd op lagere levels. Dus punt x_new virtueel aan X_unique voor weighted_prediction toevoegen en vergelijken.
    # 1) variances van Kriging unknown scheiden
    # 2) meenemen in de weighing
    # 3) scalen mbv cost_exp naar Meliani, i.e. argmax(sigma_red[l]/cost_exp[l])
        
    sample_nested(l, x_new, X, Z, K_mf)

    # recalculate X_unique
    X_unique, X_unique_exc = return_unique(X, setup.d, X[-1])

    # weigh each prediction contribution according to distance to point.
    Z_pred, mse_pred = weighted_prediction(mf_model)

    # build new kriging based on prediction
    # NOTE, we normalise mse_pred here with the previous known process variance, since otherwise we would arrive at a iterative formulation.
    # NOTE for top-level we require re-interpolation if we apply noise

    K_mf_new.train(
        X_unique, Z_pred, tune=tune, R_diagonal=mse_pred / K_mf_new.sigma_hat
    )
    K_mf_new.reinterpolate()

    # " output "
    pp.draw_current_levels(X, Z, [*K_mf, K_mf_new], X_unique_exc, L)


#####################################
# Postprocessing
#####################################



if setup.SAVE_DATA:
    setup.X = X
    setup.Z = Z
    setup.create_input_file()


# check_correlations(Z[0], Z[1], Z[2])
RMSE_norm(Z_hifi_full, K_mf[-1].predict(X0))

print("Simulation finished")
show = True
if show: 
    plt.show()
else:
    plt.draw()
    plt.pause(10)