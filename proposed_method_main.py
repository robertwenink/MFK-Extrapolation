from proposed_method import *

from postprocessing.plotting import *
from dummy_mf_solvers import *

import matplotlib.pyplot as plt
from copy import deepcopy

if __name__ == "__main__":
    solver = solve_sq
    # solver = solve_sq_inverse
    # solver = solve_ah

    # plotting
    if solver.__name__ == "solve_ah":
        text = "Harmonic"
    elif solver.__name__ == "solve_sq":
        text = "Stable, converging from "
    else:
        text = "Stable, converging from below"

    draw_convergence(solver, text)

    " inits and settings"
    X, Z, Z_k, costs, Z_pred = [], [], [], [], []
    n_samples_l0 = 20
    max_cost = 100
    l = 0
    max_level = l + 2
    runEGO = True

    " level 0 and 1 : setting 'DoE' and 'solve' "
    # level 0 DoE
    X0 = np.arange(0, 1 + np.finfo(np.float32).eps, 1 / n_samples_l0)
    X.append(X0)

    # sample level 0
    Z_l, cost_l = mf_forrester2008(X0, l, solver)
    Z.append(Z_l)
    costs.append(cost_l)
    l += 1

    # level 1 DoE + sampling
    X.append(X0)  # X1 = X0
    Z_l, cost_l = mf_forrester2008(X0, l, solver)
    Z.append(Z_l)
    costs.append(cost_l)
    l += 1

    # create Krigings of levels
    # Z_k.append(Kriging(X[0], Z[0], hps=np.array([[140 / (10 / n_samples_l0)], [2]])))
    Z_k.append(Kriging(X[0], Z[0], tuning=True))
    Z_k.append(Kriging(X[1], Z[1], hps=Z_k[0].hps))

    # draw result
    X_plot = np.arange(0, 1 + np.finfo(np.float32).eps, 1 / 200)
    ax = draw_current_levels(X, Z, Z_k, X0, X_plot, solver)

    print("Initial cost: {}".format(np.sum(costs)))
    while np.sum(costs) < max_cost and l <= max_level:
        "sample first point at new level"
        # select best sampled (!) point of previous level and sample again there
        x_b_ind = np.argmin(Z[-1])
        x_b = X[-1][x_b_ind]

        # sample the best location, increase costs
        z_pred, cost = mf_forrester2008(x_b, l, solver)
        costs.append(cost)

        # add initial sample on the level to known list of samples
        X.append([x_b])
        Z.append([z_pred])

        #  X_unique are the locations at which we will evaluate our prediction
        # NOTE return_unique does not scale well, should instead append the list each new level
        X_unique = return_unique(X)

        " predict based on single point"
        Z_new_p, mse_new_p = Kriging_unknown_z(x_b, X_unique, z_pred, Z_k)
        Z_k_new = Kriging(X_unique, Z_new_p, hps=Z_k[-1].hps)

        " output "
        print("Current cost: {}".format(np.sum(costs)))
        ax = draw_current_levels(X, Z, [*Z_k, Z_k_new], X_unique, X_plot, solver, ax)

        " sample from the predicted distribution in EGO fashion"
        ei = 1
        ei_criterion = 2 * np.finfo(np.float32).eps
        while np.sum(costs) < max_cost and runEGO:
            # select points to asses expected
            X_infill = X_plot

            # predict and calculate Expected Improvement
            y_pred, sigma_pred = Z_k_new.predict(X_infill)
            y_min = np.min(Z_k_new.predict(X_unique)[0])
            ei = np.zeros(X_infill.shape[0])
            for i in range(len(y_pred)):
                ei[i] = EI(y_min, y_pred[i], sigma_pred[i])

            if np.all(ei < ei_criterion):
                break

            # select best point to sample
            x_new = X_infill[np.argmax(ei)]

            # sample, append lists
            z_new, cost = mf_forrester2008(x_new, l, solver)
            X[-1].append(x_new)
            Z[-1].append(z_new)
            costs.append(cost)

            # recalculate X_unique
            X_unique = return_unique(X)

            # weigh each prediction contribution according to distance to point.
            Z_pred, mse_pred = weighted_prediction(X, X_unique, Z, Z_k)

            # build new kriging based on prediction
            # NOTE, we normalise mse_pred here with the previous known process variance, since otherwise we would arrive at a iterative formulation.
            # TODO for top-level we require re-interpolation if we apply noise
            if "Z_k_new_noise" not in locals():
                Z_k_new_noise = Kriging(
                    X_unique,
                    Z_pred,
                    R_diagonal=mse_pred / Z_k[-1].sigma_hat,
                    hps=Z_k[-1].hps,
                    tuning=True,
                )
            else:
                Z_k_new_noise.train(
                    X_unique, Z_pred, tune=True, R_diagonal=mse_pred / Z_k_new_noise.sigma_hat
                )

            " reinterpolate upper level the non-math way "
            Z_k_new = deepcopy(Z_k_new_noise)
            Z_k_new.hps[-1] = 0
            Z_k_new.train(
                    X_unique, Z_k_new_noise.predict(X_unique)[0], tune=False, R_diagonal=mse_pred / Z_k_new_noise.sigma_hat
                )

            # " output "
            print("Current cost: {}".format(np.sum(costs)))
            ax = draw_current_levels(
                X, Z, [*Z_k, Z_k_new], X_unique, X_plot, solver, ax
            )

        # update kriging models of levels and discrepancies before continueing to next level
        Z_k.append(Z_k_new)
        X[-1] = np.array(X[-1]).ravel()
        Z[-1] = np.array(Z[-1]).ravel()
        del Z_k_new_noise

        # moving to the next level we will be sampling at
        l += 1

    print("Simulation finished")
    show = False
    if show:
        plt.show()
    else:
        plt.draw()
        plt.pause(10)
