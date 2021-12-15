from proposed_method import *

from postprocessing.plotting import *
from dummy_mf_solvers import *

import matplotlib.pyplot as plt

if __name__ == "__main__":
    solver = solve_sq
    # solver = solve_sq_inverse
    # solver = solve_ah

    # plotting
    if solver.__name__ == "solve_ah":
        text = "Harmonic"
    elif solver.__name__ == "solve_sq":
        text = "Stable, converging from above"
    else:
        text = "Stable, converging from below"

    draw_convergence(solver, text)

    " inits "
    X = []
    Z = []
    d = []
    Z_k = []
    d_k = []
    costs = []
    n_samples_l0 = 20
    max_cost = 1000
    max_level = 2
    runEGO = True

    " level 0 and 1 : setting 'DoE' and 'solve' "
    test_multfactor = 10 / n_samples_l0
    sample_interval = 0.1 * test_multfactor
    X0 = np.arange(0, 1 + np.finfo(np.float32).eps, sample_interval)
    X.append(X0)

    Z_l, cost_l = mf_forrester2008(X0, 0, solver)
    Z.append(Z_l)
    costs.append(cost_l)

    X.append(X0)  # X1 = X0
    Z_l, cost_l = mf_forrester2008(X0, 1, solver)
    Z.append(Z_l)
    costs.append(cost_l)

    # first level multiplicative discrepancy d0
    d0 = Z[1] / Z[0]
    d.append(d0)

    # d_k.append(Kriging(X0, d0, tuning=True))
    d_k.append(
        Kriging(X0, d0, tuning=False, hps=np.array([[140 / test_multfactor], [2]]))
    )
    Z_k.append(Kriging(X0, Z[0], hps=d_k[0].hps))
    Z_k.append(Kriging(X0, Z[1], hps=d_k[0].hps))

    X_plot = np.arange(0, 1 + np.finfo(np.float32).eps, 1 / 200)
    ax = draw_current_levels(X, Z, Z_k, d_k, X0, X_plot, solver)

    # define level we are on
    l = 1

    print("Initial cost: {}".format(np.sum(costs)))
    while np.sum(costs) < max_cost and l < max_level:
        # moving to the next level we will be sampling at
        l += 1

        "prediction"
        # select best sampled (!) point of previous level and sample again there
        x_b_ind = np.argmin(Z[-1])
        x_b = X[-1][x_b_ind]

        # sample the best location, increase costs
        Z_l_0, cost = mf_forrester2008(x_b, l, solver)
        costs.append(cost)

        # add initial sample on the level to known list of samples
        X.append([x_b])
        Z.append([Z_l_0])

        # predictive multiplicative gain, only mu, no Kriging possible yet
        d_pred = Z_l_0 / Z[-2][x_b_ind]

        # Due to non-linearity we might experience extremities, i.e. when Z0 and Z1 are almost equal but Z2/Z_l_0 is not
        # We might thus want to clip the multiplications, since this is only representative for the local non-linearity.
        # TODO
        # prev_b = d_k[-1].predict(x_b)[0]
        # d_pred = np.clip(d_pred, 0, max(prev_b, 1 / prev_b))

        " discrepancy extrapolation "
        #  X_unique are the locations at which we will evaluate our prediction
        #  X_unique should be the ordered set of unique locations X, used over all levels
        # NOTE return_unique does not scale well, should instead append the list each new level
        X_unique = return_unique(X)
        d_new = Kriging_unknown(Z_k, d_k, d_pred, X, x_b, X_unique)
        Z_new_p = Z_k[-1].predict(X_unique)[0] * d_new

        d_k_new = Kriging(X_unique, d_new, hps=d_k[-1].hps)
        Z_k_new = Kriging(X_unique, Z_new_p, hps=d_k_new.hps)

        # init a placeholder for all the predictive discrepancies corresponding with a sample at the new level
        D_pred = [d_pred]

        " output "
        print("Current cost: {}".format(np.sum(costs)))
        ax = draw_current_levels(
            X,
            Z,
            [*Z_k, Z_k_new],
            [*d_k, d_k_new],
            X_unique,
            X_plot,
            solver,
            ax,
        )

        " sample from the predicted distribution in EGO fashion"
        # TODO better stopping criterion?
        ei = 1
        criterion = 2 * np.finfo(np.float32).eps
        while np.any(ei > criterion) and np.sum(costs) < max_cost and runEGO:
            # select points to asses expected
            X_infill = X_plot

            # predict and calculate Expected Improvement
            y_pred, sigma_pred = Z_k_new.predict(X_infill)
            y_min = np.min(Z_new_p)
            ei = np.zeros(X_infill.shape[0])
            for i in range(len(y_pred)):
                ei[i] = EI(y_min, y_pred[i], sigma_pred[i])

            # select best point to sample
            x_new = X_infill[np.argmax(ei)]

            # sample, append lists
            z_new, cost = mf_forrester2008(x_new, l, solver)
            X[-1].append(x_new)
            Z[-1].append(z_new)
            costs.append(cost)

            # recalculate X_unique
            X_unique = return_unique(X)

            # calculate predictive discrepancy belonging to sample x_new
            D_pred.append(z_new / Z_k[-1].predict(x_new)[0])

            # weigh each prediction contribution according to distance to point.
            d_l = weighted_discrepancy_prediction(Z_k, d_k, D_pred, X, X_unique)

            Z_new_p = (
                Z_k[-1].predict(X_unique)[0] * d_l
            )  # is exact at sampled locations!!
            d_k_new = Kriging(X_unique, d_l, hps=d_k[-1].hps)
            Z_k_new = Kriging(X_unique, Z_new_p, hps=d_k_new.hps)

            " output "
            print("Current cost: {}".format(np.sum(costs)))
            ax = draw_current_levels(
                X,
                Z,
                [*Z_k, Z_k_new],
                [*d_k, d_k_new],
                X_unique,
                X_plot,
                solver,
                ax,
            )

        # update kriging models of levels and discrepancies before continueing to next level
        Z_k.append(Z_k_new), d_k.append(d_k_new)
        X[-1] = np.array(X[-1]).ravel()
        Z[-1] = np.array(Z[-1]).ravel()

    show = True
    if show:
        plt.show()
    else:
        plt.draw()
        plt.pause(2)