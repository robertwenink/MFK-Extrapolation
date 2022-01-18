import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from proposed_method import *
from preprocessing.input import Input
from sampling.initial_sampling import get_doe
from postprocessing.plotting import *

from preprocessing.input import Input
from utils.data_utils import return_unique

setup = Input(2)
solver = get_solver(setup)
doe = get_doe(setup)

pp = Plotting(setup)

" inits and settings"
X, Z, Z_k, costs, Z_pred = [], [], [], [], []
n_samples_l0 = 10
max_cost = 300
l = 0
max_level = l + 2

" level 0 and 1 : setting 'DoE' and 'solve' "
# level 0 DoE
# X0 = np.arange(0, 1 + np.finfo(np.float32).eps, 1 / n_samples_l0)
X0 = doe(setup, n_samples_l0*setup.d)
X.append(X0)

# sample level 0
Z_l, cost_l = solver.solve(X0, l)
Z.append(Z_l)
costs.append(cost_l)
l += 1

# level 1 DoE + sampling
X.append(X0)  # X1 = X0
Z_l, cost_l = solver.solve(X0, l)
Z.append(Z_l)
costs.append(cost_l)
l += 1

# create Krigings of levels, same hps
Z_k.append(Kriging(setup, X[0], Z[0], tuning=True))
Z_k.append(Kriging(setup, X[1], Z[1], hps_init=Z_k[0].hps))

# draw result
pp.draw_current_levels(X, Z_k, X0)

print("Initial cost: {}".format(np.sum(costs)))
while np.sum(costs) < max_cost and l <= max_level:
    "sample first point at new level"
    # select best sampled (!) point of previous level and sample again there
    x_b_ind = np.argmin(Z[-1])
    x_b = correct_formatX(X[-1][x_b_ind])

    # sample the best location, increase costs
    z_pred, cost = solver.solve(x_b, l)
    costs.append(cost)
    print("Current cost: {}".format(np.sum(costs)))

    # add initial sample on the level to known list of samples
    X.append(x_b)
    Z.append(z_pred)

    #  X_unique are the locations at which we will evaluate our prediction
    # NOTE return_unique does not scale well, should instead append the list each new level
    X_unique = return_unique(X)

    " predict based on single point"
    Z_new_p, mse_new_p = Kriging_unknown_z(x_b, X_unique, z_pred, Z_k)
    Z_k_new = Kriging(setup, X_unique, Z_new_p, hps_init=Z_k[-1].hps)

    " output "
    pp.draw_current_levels(X, [*Z_k, Z_k_new], X_unique)

    " sample from the predicted distribution in EGO fashion"
    ei = 1
    ei_criterion = 2 * np.finfo(np.float32).eps
    while np.sum(costs) < max_cost:
        # select points to asses expected
        X_infill = pp.X_pred # TODO does not work for d>2

        # predict and calculate Expected Improvement
        y_pred, sigma_pred = Z_k_new.predict(X_infill)
        y_min = np.min(Z[-1]) 
        ei = np.zeros(X_infill.shape[0])
        for i in range(len(y_pred)):
            ei[i] = EI(y_min, y_pred[i], sigma_pred[i])

        # terminate if criterion met
        if np.all(ei < ei_criterion):
            break

        # select best point to sample
        x_new =correct_formatX( X_infill[np.argmax(ei)])

        # sample, append lists
        z_new, cost = solver.solve(x_new, l)
        X[-1] = np.append(X[-1],x_new,axis=0)
        Z[-1] = np.append(Z[-1],z_new)
        costs.append(cost)
        print("Current cost: {}".format(np.sum(costs)))

        # recalculate X_unique
        X_unique = return_unique(X)

        # weigh each prediction contribution according to distance to point.
        Z_pred, mse_pred = weighted_prediction(setup,X, X_unique, Z, Z_k)

        # build new kriging based on prediction
        # NOTE, we normalise mse_pred here with the previous known process variance, since otherwise we would arrive at a iterative formulation.
        # NOTE for top-level we require re-interpolation if we apply noise
        if "Z_k_new_noise" not in locals():
            Z_k_new_noise = Kriging(setup,
                X_unique,
                Z_pred,
                R_diagonal=mse_pred / Z_k[-1].sigma_hat,
                hps_init=Z_k[-1].hps,
                tuning=True,
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
        # Z_k_new = deepcopy(Z_k_new_noise)
        Z_k_new = Z_k_new_noise
        Z_k_new.hps[-1] = 0 # set noise to 0
        Z_k_new.train(
                X_unique, Z_k_new_noise.predict(X_unique)[0], tune=False, R_diagonal=mse_pred / Z_k_new_noise.sigma_hat
            )
        
        # " output "
        pp.draw_current_levels(X, [*Z_k, Z_k_new], X_unique)


    # update kriging models of levels and discrepancies before continueing to next level
    Z_k.append(Z_k_new)
    if "Z_k_new_noise"  in locals():
        del Z_k_new_noise

    # moving to the next level we will be sampling at
    l += 1

print("Simulation finished")
show = True
if show:
    plt.show()
else:
    plt.draw()
    plt.pause(10)
