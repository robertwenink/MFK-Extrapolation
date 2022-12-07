import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
import os

from preprocessing.input import Input

from core.proposed_method import *
from core.sampling.DoE import get_doe, LHS_subset

from postprocessing.plotting import *

from utils.formatting_utils import return_unique, correct_formatX
from utils.linearity_utils import check_linearity
from utils.correlation_utils import check_correlations
from utils.error_utils import check_all_RMSE


setup = Input(0)
solver = get_solver(setup)
doe = get_doe(setup)
pp = Plotting(setup)


################
# Settings
################

" inits and settings"
# list of the convergence levels we pass to solvers; different to the Kriging level we are assessing.
L = [1, 3, 4] 



n_samples_l0 = 10 
max_cost = 1500e5
max_nr_levels = 3
reuse_values = True
pc = 3 # parallel capability, used for sampling the hifi in assumption verifications

ei = 1
ei_criterion = 2 * np.finfo(np.float32).eps

Z_k, Z_pred = [], []
costs, cost_exp = [0 for _ in range(len(L))], [0 for _ in range(len(L))]

############################### 
# helper functions
###############################

def update_costs(cost,X,l):
    # list of costs per level
    costs[l] += cost

    # expected cost per sample at a level
    cost_exp[l] = costs[l] / X[l].shape[0]

    # print current total costs (all levels)
    if l == 2:
        print("Current cost: {}    ; {} hifi samples".format(np.sum(costs),X[-1].shape[0]))
    else:
        print("Current cost: {}".format(np.sum(costs)))

def sample(l, x_new):
    """
    Function that does all required actions for core.sampling.
    lists X, Z, and costs are changed inplace by reference
    x_new is the value / an np.ndarray of to sample location(s)
    """

    # check if not already sampled at this level
    # TODO for loop, parallel.
    if np.all(np.isin(x_new,X[l],invert=True)):
        z_new, cost = solver.solve(x_new, L[l])
        X[l] = np.append(X[l], x_new, axis=0)
        Z[l] = np.append(Z[l], z_new)
        update_costs(cost, X, l)

def sample_nested(l, x_new, X, Z, Z_k):
    """
    Function that does all required actions for nested core.sampling.
    """

    # Sampled HIFI location should be sampled at all lower levels as well, i.e. nested DoE; this because of Sf and optimally making use of the hifi point.
    # TODO do this for all levels? i.e. only the for loop?
    # NOTE leave for loop out to only sample the highest level
    sampled_levels = [l]
    sample(l, x_new)
    if l == len(L) - 1:
        for i in range(l):
            sample(i, x_new)
            sampled_levels.append(i)
    
    # retrain the sampled levels with the newly added sample.
    for i in sampled_levels:
        if i != 2:
            Z_k[i].train(X[i], Z[i])


def sample_initial_hifi(setup, X,Z,X_unique):
    # select best sampled (!) point of previous level and sample again there
    x_b = X[-1][np.argmin(Z[-1])]

    # select other points for cross validation, in LHS style
    X_hifi = correct_formatX(LHS_subset(setup, X_unique, x_b, pc),setup.d)

    Z_pred, cost = solver.solve(X_hifi, L[l])
    X.append(X_hifi)
    Z.append(Z_pred)
    update_costs(cost, X, l)

    sample_nested(l, X_hifi, X, Z, Z_k)

###############################
# main
###############################
np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)

" level 0 and 1 : setting 'DoE' and 'solve' "
if hasattr(setup, "X") and reuse_values:
    X = setup.X
    
    # for now only keep first 2 levels
    del X[2]
else:
    X = []
    X.append(doe(setup, n_samples_l0 * setup.d))
    X.append(doe(setup, n_samples_l0 * setup.d))
    # NOTE use same X at lower levels / nested
    X[1] = X[0]
    

# fill Z, either by reading old results or new (test) solves.
Z = []
for l in range(2):
    z_new, cost = solver.solve(X[l], L[l])
    Z.append(z_new)
    update_costs(cost, X, l)


# create Krigings of levels, same initial hps
Z_k.append(Kriging(setup, X[0], Z[0], tune=True))
Z_k.append(Kriging(setup, X[1], Z[1], hps_init=Z_k[0].hps, tune=True))


" level 2 initialisation "
# we are now on level 2!    
l = 2

#  X_unique are the locations at which we will evaluate our prediction
# NOTE return_unique does not scale well, should instead append the list each new level
X_unique, X_unique_exc = return_unique(X, setup.d)

if len(Z) == 2: # then we do not have a sample on level 2 yet.
    sample_initial_hifi(setup, X, Z, X_unique)

# if not check_linearity(setup, X, X_unique, Z, Z_k, pp, L):
#     print("Not linear enough, but continueing for now.")


" initial prediction "
Z_pred, mse_pred = weighted_prediction(setup, X[-1], X_unique, Z[-1], Z_k)
Z_k_new = Kriging(setup, X_unique, Z_pred, hps_init=Z_k[-1].hps, hps_noise_ub = True, tune = True, R_diagonal=mse_pred / Z_k[-1].sigma_hat)
Z_k_new.reinterpolate()


X0 = X[0]
# RMSE = RMSE_norm(solver.solve(X0,L[-1])[0], Z_k_new.predict(X0)[0])
# print("RMSE: {}".format(RMSE))
Z_hifi_full = solver.solve(X0,L[-1])[0]
check_all_RMSE(X0, Z_hifi_full, [*Z_k, Z_k_new])
check_correlations(Z[0], Z[1], Z_hifi_full)

# draw the result
pp.draw_current_levels(X, Z, [*Z_k, Z_k_new], X_unique_exc, L)
pp.plot_kriged_truth(setup,X0, Z_hifi_full, hps_init = Z_k[-1].hps)
plt.show()

sys.exit(0)

" sample from the predicted distribution in EGO fashion"
while np.sum(costs) < max_cost:
    # select points to asses expected improvement
    X_infill = pp.X_pred  # TODO does not work for d>2

    # predict and calculate Expected Improvement
    y_pred, sigma_pred = Z_k_new.predict(X_infill)
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
        
    sample_nested(l, x_new, X, Z, Z_k)

    # recalculate X_unique
    X_unique, X_unique_exc = return_unique(X, setup.d, X[-1])

    # weigh each prediction contribution according to distance to point.
    Z_pred, mse_pred = weighted_prediction(setup, X[-1], X_unique, Z[-1], Z_k)

    # build new kriging based on prediction
    # NOTE, we normalise mse_pred here with the previous known process variance, since otherwise we would arrive at a iterative formulation.
    # NOTE for top-level we require re-interpolation if we apply noise

    Z_k_new.train(
        X_unique, Z_pred, tune=tune, R_diagonal=mse_pred / Z_k_new.sigma_hat
    )
    Z_k_new.reinterpolate()

    # " output "
    pp.draw_current_levels(X, Z, [*Z_k, Z_k_new], X_unique_exc, L)


#####################################
# Postprocessing
#####################################



if setup.SAVE_DATA:
    setup.X = X
    setup.Z = Z
    setup.create_input_file()


# check_correlations(Z[0], Z[1], Z[2])
RMSE_norm(Z_hifi_full, Z_k[-1].predict(X0))

print("Simulation finished")
show = True
if show: 
    plt.show()
else:
    plt.draw()
    plt.pause(10)
