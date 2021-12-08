# RESEARCH QUESTIONS:
# - Can we improve the effective sample efficiency/costs in multi fidelity Kriging by assuming correlation/dependence between fidelity levels?
#   - can we make accurate predictions about unseen higher fidelity levels based upon this assumption?
#   - can we avoid sampling the most expensive high fidelity levels based upon this assumption in a efficient global optimization setting?


"""
A 'solver' with the property of grid convergence; i.e. a converging function.
We could use it as well as modifier for transformed MF testcases.
"""
import itertools
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
from mpl_toolkits.mplot3d import axes3d

from sampling.initial_sampling import get_doe
from sampling.solvers.solver import get_solver
from sampling.infill.infill_criteria import EI
from models.kriging.method.OK import *
from postprocessing.plotting import *
from dummy_mf_solvers import *


class Setup:
    """Dummy class"""
    pass


def Kriging(X_l, d_l, tuning=False, hps=None, train=True):
    """
    Setup a Kriging model, either for the discrepancy or a level
    """
    setup = Setup()
    setup.d = 1
    setup.kernel = "kriging"

    ok = OrdinaryKriging(setup)

    if tuning == False:
        if hps is None:
            raise NameError(
                "hps is not defined! When tuning=False, hps must be defined"
            )
        ok.hps = hps

    if train:
        ok.train(X_l, d_l, tuning)

    return ok


def Kriging_unknown(Z_k, d_k, d_pred, X, x_b, X_unique):
    """
    Scale the multiplicative d_pred using relative gains between points at x_b and x_other
    i.e. predict based on dependancy between levels!
    This approach is exact if the previous 2 levels (are sampled/exact and) are a linear transformation of the one we predict!

    @param Z_k: list containing all Kriging models of known levels
    @param d_k: list containing all Kriging models of known discrepancies
    @param d_pred: discrepancy between the first sampled point at the new level
     and the best point(s) of the current level, sampled at the same location.
     we use this discrepancy as a predictor for the rest of the unknown level.
     Can be an array.
    @param x_b: (best) location at current known top-level, i.e. location of d_pred. Can be an array.
    @param  X_unique: locations of points we want to predict on the new level
     (these locations will have a discrete sample at some level, i.e. all locations x that have been sampled at some time)
    @return discrete discrepancy funciton (np.array) between last known and new unknown level
    """

    Z0_k = Z_k[-2]
    Z1_k = Z_k[-1]

    Z0_b = Z0_k.predict(x_b)[0]  # this might be a Kriged value
    Z1_b = Z1_k.predict(x_b)[0]  # this is a known point only at the first step

    Z0 = Z0_k.predict(X_unique)[0]  # returns either sampled or Kriged values
    Z1 = Z1_k.predict(X_unique)[0]  # returns either sampled or Kriged values

    t = Z1_b * Z1
    d1 = (t - Z0 * Z1_b) / (t - Z0_b * Z1) * (d_pred - 1) + 1
    return d1


def weighted_discrepancy_prediction(Z_k, d_k, D_pred, X, X_unique):
    """
    Function that weights the results of the function 'Kriging_unknown' for multiple samples at the (partly) unknown level
     based on spatial distance in a similar fashion to Kriging.
     In case convergence is linear over levels and location, the discrepancy prediction is exact. In case convergence is not linear,
     we could under or overestimate the discrepancy at other locations, something we could only solve by sampling and somehow weighing the solutions.
    
    @param Z_k, d_k: Kriging models of the levels and discrepancy.
    @param D_pred: The discrete discrepancy predictions, only at X_s. For instance, for 2 samples at level l+1 we have D.shape = (2, X_unique.shape[0], 1).
     The 1 is a bit weird, but originates from X having multiple dimensions, and d should be compatible.
    @param X: list of all level`s sample locations, includes: 
           X_s = X[-1]: Locations at which we have sampled at the new level, and at which the discrepancy predictions have been formulated.
    @param X_unique: all unique and previously sampled locations at which we have defined a predictive discrepancy, based on the points in X_s
    """
    # The problem of converting this to Kriging is that we do not have a single point with a single value, but
    # a single point that instigates multiple values across the domain. This means we need beforehand, for each point in X_unique, weigh each contribution
    # in D / the points in X_s, i.e. construct a correlation matrix. Ofcourse, we already have a correlation matrix with (approximately tuned) hyperparameters,
    # known from our Kriging model!

    # So, we can
    # 1) use the same Kriging correlation matrix method,
    # 2) and then scale its rows to summarize to 1; retrieving a coefficient c_{i,j} for each combination.
    # 3) Then for each point in X_unique, we can retrieve the contribution of each X_s and summate.
    # 4) We now have per-point values, which we can Krige again!
    # Both times we 'Krige' over X_unique, so we can use the same hyperparameters.

    # TODO what is interesting in the weighing, is that the first discrepancy d_x0 is exact,
    # but each later sample might be relative to a Kriged result, thus including uncertainty.

    hps = d_k[-1].hps
    X_s = correct_formatX(X[-1])
    X_unique = correct_formatX(X_unique)

    # recalculate all predictive discrepency arrays
    # TODO vectorize, but no priority, non-vectorized is better for development
    D = []
    for i in range(X_s.shape[0]):
        D.append(Kriging_unknown(Z_k, d_k, D_pred[i], X, X_s[i], X_unique))
    D = np.array(D)

    # choice of weighting function
    km = Kriging(X_s, None, hps=hps, train=False)
    c = km.corr(X_s, X_unique, *hps)

    # NOTE by scaling, values of D_pred are not necessarily seen back exact in the returned array
    #      and thus the result is not exact anymore at sampled locations!! Therefore this fix.
    # if we dont do this, EI does not work!!
    # if we do this, we might get very spurious results....
    mask = (c == 1.0)
    idx = mask.any(axis=0)
    c[:, idx] = 0
    c += mask

    c_scaled = np.divide(c, np.sum(c, axis=0))  # sums to 1 over axis=1
    
    return np.sum(np.multiply(D, c_scaled), axis=0)


def return_unique(X):
    # # https://www.peterbe.com/plog/fastest-way-to-uniquify-a-list-in-python-3.6
    seq = [item for sublist in X for item in sublist]
    return np.array(list(dict.fromkeys(seq)))
    # return np.concatenate(X).ravel()


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

    X_plot = np.arange(0, 1 + np.finfo(np.float32).eps, 1/200)
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
        criterion = 2*np.finfo(np.float32).eps
        while np.any(ei>criterion) and np.sum(costs) < max_cost and runEGO:
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