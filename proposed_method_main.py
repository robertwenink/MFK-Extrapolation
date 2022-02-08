import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
import os

from proposed_method import *
from linearity_check import *
from preprocessing.input import Input
from sampling.initial_sampling import get_doe
from postprocessing.plotting import *

from preprocessing.input import Input
from utils.data_utils import return_unique

setup = Input(0)
solver = get_solver(setup)
doe = get_doe(setup)
pp = Plotting(setup)


################
# Settings
################

" inits and settings"
# list of the convergence levels we pass to solvers; different to the Kriging level we are assessing.
L = [0, 1, 3] 

n_samples_l0 = 10
max_cost = 1500
max_nr_levels = 3
reuse_values = False
pc = 3 # parallel capability, used for sampling the hifi in assumption verifications

ei = 1
ei_criterion = 2 * np.finfo(np.float32).eps

Z_k, Z_pred = [], []
costs, cost_exp = [0 for _ in range(len(L))], [0 for _ in range(len(L))]

############################### 
# helper functions
###############################

def update_costs(cost,X,l):
    costs[l] += cost
    cost_exp[l] = costs[l] / X[l].shape[0]
    if l == 2:
        print("Current cost: {}    ; {} hifi samples".format(np.sum(costs),X[-1].shape[0]))
    else:
        print("Current cost: {}".format(np.sum(costs)))

def sample(l, x_new):
    """
    Function that does all required actions for sampling.
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

def sample_nested(l, x_new):
    """
    Function that does all required actions for nested sampling.
    """

    # Sampled HIFI location should be sampled at all lower levels as well, i.e. nested DoE; this because of Sf and optimally making use of the hifi point.
    # TODO do this for all levels? i.e. only the for loop?
    # NOTE leave for loop out to only sample the highest level
    sample(l, x_new)
    if l == len(L) - 1:
        for i in range(l):
            sample(i, x_new)

def LHS_subset(X_unique,x_init,amount):
    """
    TODO the result should be sampled via some maximin criterion again including boundaries. For now just random.
    """ 
    X_list = [x_init]
    while len(X_list) < amount:
        item = X_unique[np.random.randint(0,X_unique.shape[0])]
        if item not in X_list:
            X_list.append(item)
    return np.array(X_list)


def sample_initial_hifi(X,Z,X_unique):
    # select best sampled (!) point of previous level and sample again there
    x_b = X[-1][np.argmin(Z[-1])]

    # select other points for cross validation, in LHS style
    amount = os.cpu_count()
    amount = 3
    X_hifi = correct_formatX(LHS_subset(X_unique, x_b, amount),setup.d)

    Z_pred, cost = solver.solve(X_hifi, L[l])
    X.append(X_hifi)
    Z.append(Z_pred)
    update_costs(cost, X, l)

    sample_nested(l, X_hifi)

###############################
# main
###############################

" level 0 and 1 : setting 'DoE' and 'solve' "
if hasattr(setup, "X") and reuse_values:
    X = setup.X
    Z = setup.Z
    
    # for now only keep first 2 levels
    del X[2]
    del Z[2]
else:
    X, Z = [], []
    X.append(doe(setup, n_samples_l0 * setup.d))
    X.append(doe(setup, n_samples_l0 * setup.d))
    X[1] = X[0]
    
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
X_unique, X_unique_exc = return_unique(X)

if len(Z) == 2: # then we do not have a sample on level 2 yet.
    sample_initial_hifi(X, Z, X_unique)

if not check_linearity(setup, X, X_unique, Z, Z_k, pp):
    plt.pause(10)
    sys.exit()

" initial prediction "
Z_pred, mse_pred = weighted_prediction(setup, X[-1], X_unique, Z[-1], Z_k)
Z_k_new = Kriging(setup, X_unique, Z_pred, hps_init=Z_k[-1].hps, tune = True)

# draw the result
pp.draw_current_levels(X, [*Z_k, Z_k_new], X_unique_exc)


" sample from the predicted distribution in EGO fashion"
while np.sum(costs) < max_cost:
    # select points to asses expected improvement
    X_infill = pp.X_pred  # TODO does not work for d>2

    # predict and calculate Expected Improvement
    y_pred, sigma_pred = Z_k_new.predict(X_infill)
    y_min = np.min(Z[-1])
    ei = np.zeros(X_infill.shape[0])
    for i in range(len(y_pred)):
        ei[i] = EI(y_min, y_pred[i], sigma_pred[i])

    # select best point to sample
    x_new = correct_formatX(X_infill[np.argmax(ei)], setup.d)

    # terminate if criterion met, or for some buggy reason we revisit a point
    if np.all(ei < ei_criterion) or x_new in X[2]:
        break

    # TODO implement level selection
    # select level l on which we will sample, based on location of ei and largest source of uncertainty there (corrected for expected cost).
    # source of uncertainty is from each component in proposed method which already includes approximated noise levels.
    # in practice this means: only for small S1 will we sample S0, otherwise S1. Nested DoE not required.

    sample_nested(l, x_new)

    # TODO retrain all sampled levels.

    # recalculate X_unique
    X_unique, X_unique_exc = return_unique(X, X[-1])

    # weigh each prediction contribution according to distance to point.
    Z_pred, mse_pred = weighted_prediction(setup, X[-1], X_unique, Z[-1], Z_k)

    # build new kriging based on prediction
    # NOTE, we normalise mse_pred here with the previous known process variance, since otherwise we would arrive at a iterative formulation.
    # NOTE for top-level we require re-interpolation if we apply noise
    if "Z_k_new_noise" not in locals():
        Z_k_new_noise = Kriging(
            setup,
            X_unique,
            Z_pred,
            R_diagonal=mse_pred / Z_k[-1].sigma_hat,
            hps_init=Z_k[-1].hps,
            tune=True,
        )
    else:
        Z_k_new_noise.train(
            X_unique, Z_pred, tune=True, R_diagonal=mse_pred / Z_k_new_noise.sigma_hat
        )

    # NOTE noise should decrease when we increase the fidelity, therefore, extrapolate the noise of the previous levels.
    # in a real case this should be something like a log over the resolution, here we just take the previous result.
    # furthermore, points closeby will always have benefit from a larger noise + the predictions we make build upon the noise -> more noise.
    # at the same time, for points closeby, if there is little noise, we will require little noise.
    # we should thus not tune for noise at the prediction level, but maybe re-use a previous estimate in some way.
    # TODO maybe only tune the noise bases on the sampled points?? i.e. exclude the prediction points as Kriging points.
    Z_k_new_noise.hps[-1] = Z_k[-1].hps[-1]

    " reinterpolate upper level the non-math way "
    Z_k_new = Z_k_new_noise
    Z_k_new.hps[-1] = 0  # set noise to 0
    Z_k_new.train(
        X_unique,
        Z_k_new_noise.predict(X_unique)[0],
        tune=False,
        R_diagonal=mse_pred / Z_k_new_noise.sigma_hat,
    )
    print("Z_k_new.hps : {}".format(Z_k_new.hps))

    # " output "
    pp.draw_current_levels(X, [*Z_k, Z_k_new], X_unique_exc)


#####################################
# Postprocessing
#####################################

if setup.SAVE_DATA:
    setup.X = X
    setup.Z = Z
    setup.create_input_file()

print("Simulation finished")
show = True
if show:
    plt.show()
else:
    plt.draw()
    plt.pause(10)
