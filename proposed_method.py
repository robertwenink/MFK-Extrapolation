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
from models.kriging.method.OK import *
from postprocessing.plotting import plot_kriging

marker = itertools.cycle(("o", "v", "^", "<", ">", "s", "8", "p"))

# Power describing the speed of convergence. Higher is faster
CONVERGENCE_SPEED = 2


class Setup:
    pass


def solve_sq(l, p=2):
    """return sum of series reciprocals up to number l.
    p=2; Basel problem converges to pi**2 / 6"""
    n = np.arange(
        1, l ** CONVERGENCE_SPEED + 2
    )  # +2, because then indexing from l=0 and less crazy first convergence step
    mod = 0
    return (np.pi ** 2 / 6 + mod) / (np.sum(np.divide(1, n ** 2)) + mod)


def solve_sq_inverse(l, p=2):
    return 1 / solve_sq(l, p)


def solve_ah(l):
    """alternating harmonic"""
    n = np.arange(2, l ** CONVERGENCE_SPEED + 2)
    n[::2] *= -1
    harmonic = 1 + np.sum(np.divide(1, n))
    return 1 + (np.log(2) - harmonic) * 2


def forrester2008(x):
    # last term is not from forrester2008
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4) + np.sin(8 * 2 * np.pi * x)


def solver_wrapper(solver_v, x):
    """this function provides a wrapper for the solver, introducing irregular convergence"""
    hard = True
    if hard:
        return solver_v + (1 - solver_v) * -(0.5 + np.sin(solver_v * x * 2 * np.pi))
    else:
        return solver_v + 2 * (1 - solver_v) * -(np.sin(x * 2 * np.pi))


def mf_forrester2008(x, l, solver):
    x = correct_formatX(x)
    conv = solver_wrapper(solver(l), x)
    A = conv
    linear_gain_mod = 2
    B = 10 * (conv - 1) * linear_gain_mod
    C = 5 * (conv - 1)
    ret = A * forrester2008(x) + B * (x - 0.5) + C
    return ret.ravel(), sampling_costs(l) * x.shape[0]


def sampling_costs(l):
    return l ** 4


def draw_convergence(solver, solver_type_text):
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig = plt.figure()
    ax1 = fig.add_subplot(221, projection="3d")
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(212)
    fig.suptitle("Convergence type: {}".format(solver_type_text))
    x = np.arange(0, 1 + np.finfo(np.float32).eps, 0.05)

    "show convergence profile"
    L = np.arange(10)
    y = []
    for l in L:
        y.append(solver_wrapper(solver(l), x))
    X, L_mesh = np.meshgrid(x, L)
    y = np.array(y)
    ax1.plot_surface(X, L_mesh, y, alpha=0.9)
    # ax1.plot(x, L, y)
    ax1.set_ylabel("Fidelity level")
    ax1.set_xlabel("x location")
    ax1.set_title("Convergence profile")

    # "show costs"
    ax2.plot(L, sampling_costs(L))
    ax2.set_xlabel("Fidelity level")
    ax2.set_title("Sampling costs")

    "show mf function"
    for l in range(0, 6, 1):
        y, _ = mf_forrester2008(x, l, solver)
        ax3.plot(x, y, label="Level = {}".format(l))
    ax3.plot(x, forrester2008(x), ".-", label="Fully converged")
    ax3.set_xlabel("X: 1D search space")
    ax3.set_title("Evaluation function responses per fidelity")

    # # ax1.set_aspect(1.0 / ax1.get_data_ratio(), adjustable="box")
    # ax2.set_aspect(1.0 / ax2.get_data_ratio(), adjustable="box")
    # ax3.set_aspect(1.0 / ax3.get_data_ratio(), adjustable="box")

    plt.legend()


def draw_current_levels(X, Z, Z_k, d_k, x_b, X_plot, solver, ax=None):
    if ax == None:
        fig, ax = plt.subplots(1, 1)
        fig.suptitle("{}".format(solver.__name__))
    else:
        ax.clear()

    for i in range(len(X) - 1):
        color = next(ax._get_lines.prop_cycler)["color"]
        ax.plot(
            X[i],
            Z[i],
            linestyle="",
            markeredgecolor="none",
            marker=next(marker),
            color=color,
        )
        ax.plot(
            X_plot,
            Z_k[i].predict(X_plot)[0],
            linestyle="-",
            color=color,
            label="Kriging level {}".format(i),
        )

    i = i + 1
    color = next(ax._get_lines.prop_cycler)["color"]
    ax.plot(
        X[i],
        Z[i],
        linestyle="",
        markeredgecolor="none",
        marker=next(marker),
        color=color,
    )
    ax.plot(
        X_plot, Z_k[i].predict(X_plot)[0], label="pred level {}".format(i), color=color
    )
    Y_plot_true, _ = mf_forrester2008(X_plot, i, solver)
    ax.plot(
        X_plot,
        Y_plot_true,
        linestyle="--",
        label="true level {}".format(i),
        color=color,
    )

    ax.plot(X_plot, forrester2008(X_plot), "--", label="truth Kriging")

    # ax.scatter(X[0], forrester2008(X[0]), label="truth sampled")
    ax.scatter(x_b, Z_k[-2].predict(x_b)[0], label="current best")

    plt.legend()

    plt.draw()
    plt.pause(2)
    return ax


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


def Kriging_unknown(Z_k, d_k, d_pred, x_b, X_unique):
    """
    @param Z_k: list containing all Kriging models of known levels
    @param d_k: list containing all Kriging models of known discrepancies
    @param d_pred: discrepancy between the first sampled point at the new level
     and the best point of the current level, sampled at the same location.
     we use this discrepancy as a predictor for the rest of the unknown level.
    @param x_b: best location at current known top-level, i.e. location of d_pred
    @param  X_unique: locations of points we want to predict on the new level
     (these locations will have a discrete sample at some level, i.e. all locations x that have been sampled at some time)
    returns discrete discrepancy funciton (np.array) between last known and new unknown level
    """

    Z0_k = Z_k[-2]
    Z1_k = Z_k[-1]

    # Scale the multiplicative d_pred using relative gains between points at x_b and x_other
    # i.e. predict based on dependancy between levels!
    # This approach is exact if the previous 2 levels (are sampled/exact and) are a linear transformation of the one we predict!
    # NOTE the above is thus true under the assumption of equal convergence behaviour over the full search range.
    # TODO implement differently scaled convergence types, dependent on x.
    Z0_b = Z0_k.predict(x_b)[0]  # this might be a Kriged value
    Z1_b = Z1_k.predict(x_b)[0]  # this is always a known point

    Z0 = Z0_k.predict(X_unique)[0]  # returns either sampled or Kriged values
    Z1 = Z1_k.predict(X_unique)[0]  # returns either sampled or Kriged values

    # TODO make d_pred have multiple sources/sample points; weigh each term according to the distance towards the sample point.
    # This sounds like a new Kriging, however, the sum of the coefficients should always be 1 at each point.
    # It is however based upon a correlation.
    t = Z1_b * Z1
    d1 = (t - Z0 * Z1_b) / (t - Z0_b * Z1) * (d_pred - 1) + 1
    return d1


def weigh_discrepancy_predictions(D, X_s, X_unique, hps):
    """
    Function that weights the results of the function 'Kriging_unknown' for multiple samples at the (partly) unknown level
     based on spatial distance in a similar fashion to Kriging.
     In case convergence is linear over levels and location, the discrepancy prediction is exact. In case convergence is not linear,
     we could under or overestimate the discrepancy at other locations, something we could only solve by sampling and somehow weighing the solutions.
    @param D: The discrete discrepancy predictions. For instance, for 2 samples at level l+1 we have D.shape = (2, X_unique.shape[0], 1).
     The 1 is a bit weird, but originates from X having multiple dimensions, and d should be compatible.
    @param X_s: Locations at which we have sampled at the new level, and with which the discrepancy predictions have been formulated.
    @param X_unique: all unique and previously sampled locations at which we have defined a predictive discrepancy, based on the points in X_s
    """
    # The problem of converting this to Kriging is that we do not have a single point with a single value, but
    # a single point that instigates multiple values across the domain. This means we need beforehand, for each point in X_unique, weigh each contribution
    # in D / the points in X_s, i.e. construct a correlation matrix. Ofcourse, we already have a correlation matrix with (approximately tuned) hyperparameters,
    # known from our Kriging model!
    #
    # So, we can
    # 1) use the same Kriging correlation matrix method,
    # 2) and then scale its rows to summarize to 1; retrieving a coefficient c_{i,j} for each combination.
    # 3) Then for each point in X_unique, we can retrieve the contribution of each X_s and summate.
    # 4) We now have per-point values, which we can Krige again!
    # Both times we 'Krige' over X_unique, so we can use the same hyperparameters.

    # TODO what is interesting in the weighing, is that the first discrepancy d_x0 is exact,
    # but each later sample might be relative to a Kriged result, thus including uncertainty.
    D = correct_formatX(D)
    X_s = correct_formatX(X_s)
    X_unique = correct_formatX(X_unique)

    km = Kriging(X_s, None, hps=hps, train=False)
    c = km.corr(X_s, X_unique, *hps)
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

    X_plot = np.arange(0, 1 + np.finfo(np.float32).eps, 0.01)
    ax = draw_current_levels(X, Z, Z_k, d_k, X[-1][np.argmin(Z[-1])], X_plot, solver)

    # define level we are on
    l = 1
    
    print("Initial cost: {}".format(np.sum(costs)))
    while np.sum(costs) < max_cost:
        # moving to the next level we will be sampling at
        l += 1

        "prediction"
        # select best point of previous level and sample again there
        x_b_ind = np.argmin(Z[-1])
        x_b = X[-1][x_b_ind]

        Z_l_0, cost = mf_forrester2008(x_b, l, solver)
        costs.append(cost)
        print("Current cost: {}".format(np.sum(costs)))

        # predictive multiplicative gain, only mu, no Kriging possible yet
        d_pred = Z_l_0 / Z[-1][x_b_ind]

        # Larger than previous would mean a diverging solution, while we strictly converge
        # however, due to introduced non-linearity it might seem at some points we are diverging
        # therefore we alter d_pred
        # TODO this prevents the most extreme deviations, but only a quick fix
        # prev_b = d_k[-1].predict(x_b)[0]
        # d_pred = np.clip(d_pred, 0, max(prev_b, 1 / prev_b))  

        " discrepancy extrapolation "
        #  X_unique are the locations at which we will evaluate our prediction
        #  X_unique should be the ordered set of unique locations X, used over all levels
        # TODO return_unique does not scale well, should instead append the list each new level
        X_unique = return_unique(X)
        d_new = Kriging_unknown(Z_k, d_k, d_pred, x_b, X_unique)
        Z_new = Z[-1] * d_new

        d_k_new = Kriging(X_unique, d_new, hps=d_k[-1].hps)
        Z_k_new = Kriging(X_unique, Z_new, hps=d_k_new.hps)

        " plot all "
        ax = draw_current_levels(
            [*X, X_unique],
            [*Z, Z_new],
            [*Z_k, Z_k_new],
            [*d_k, d_k_new],
            x_b,
            X_plot,
            solver,
            ax,
        )

        " sample from the predicted distribution"
        X_s = [x_b]
        D = [d_new]

        # TODO proper stopping criterion
        for i in range(0,round(n_samples_l0/(l-1))):

            # TODO select based on expected improvement
            x_new_ind = round(n_samples_l0 / 2)
            x_new = X_unique[x_new_ind]
            X_s.append(x_new)

            z_new, cost = mf_forrester2008(X_s[-1], l, solver)
            costs.append(cost)
            print("Current cost: {}".format(np.sum(costs)))

            # NOTE TODO still part of X_unique, if we sample really a new point, add to X_unique
            d_pred = z_new / Z_k[-1].predict(x_new)[0]
            D.append(Kriging_unknown(Z_k, d_k, d_pred, x_new, X_unique))

            # weigh each prediction contribution according to distance to point.
            d_l = weigh_discrepancy_predictions(D, X_s, X_unique, d_k[-1].hps)

            Z_new = Z[-1] * d_l
            d_k_new = Kriging(X_unique, d_l, hps=d_k[-1].hps)
            Z_k_new = Kriging(X_unique, Z_new, hps=d_k_new.hps)

            " plot all "
            ax = draw_current_levels(
                [*X, X_unique],
                [*Z, Z_new],
                [*Z_k, Z_k_new],
                [*d_k, d_k_new],
                x_b,
                X_plot,
                solver,
                ax,
            )
            break

        X.append(X_unique), Z.append(Z_new), Z_k.append(Z_k_new), d_k.append(d_k_new)

    show = True
    if show:
        plt.show()
    else:
        plt.draw()
        plt.pause(2)
