import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sampling.solvers.solver import get_solver


def plot_kriging(setup, X, y, predictor):
    setup.n_per_d = 50

    if setup.live_plot:
        if setup.d == 1 or len(setup.d_plot) == 1:
            plot_1d(setup, X, y, predictor)
        else:
            plot_2d(setup, X, y, predictor)


def grid_plot(setup, n_per_d=None):
    """
    Build a (1d or 2d) grid used for plotting and sample n points in each dimension d.
    Dimensions d are specified by the dimensions to plot in setup.d_plot.
    Coordinates of the other dimensions are fixed in the centre of the range.
    """
    d = setup.d
    d_plot = setup.d_plot
    if n_per_d is None:
        n_per_d = setup.n_per_d

    lb = setup.search_space[1]
    ub = setup.search_space[2]

    lin = np.linspace(lb[d_plot], ub[d_plot], n_per_d)
    lis = [lin[:, i] for i in range(len(d_plot))]
    res = np.meshgrid(*lis)
    Xx = np.stack(res, axis=-1).reshape(-1, len(d_plot))
    X = np.ones((len(Xx), d)) * (lb + (ub - lb) / 2)
    X[:, d_plot] = Xx
    return X


def plot_2d(setup, X, y, predictor):
    d_plot = setup.d_plot
    d = setup.d
    n_per_d = setup.n_per_d

    X_new = grid_plot(setup)
    y_hat, mse = predictor.predict(X_new)

    X_predict = X_new[:, d_plot].reshape(n_per_d, n_per_d, -1)
    y_hat = y_hat.reshape(n_per_d, n_per_d)
    std = np.sqrt(mse.reshape(n_per_d, n_per_d))

    fig = plt.figure()
    fig.suptitle("{}".format(setup.solver_str))  # set_title if ax

    nrows = 1
    ncols = 2
    ax = [[0] * ncols] * nrows

    ax[0][0] = fig.add_subplot(1, 2, 1, projection="3d")
    ax[0][0].plot_surface(X_predict[:, :, 0], X_predict[:, :, 1], y_hat, alpha=0.9)
    ax[0][0].plot_surface(
        X_predict[:, :, 0], X_predict[:, :, 1], y_hat - std, alpha=0.4
    )
    ax[0][0].plot_surface(
        X_predict[:, :, 0], X_predict[:, :, 1], y_hat + std, alpha=0.4
    )
    ax[0][0].set_zlabel("Z")
    ax[0][0].scatter(X[:, d_plot[0]], X[:, d_plot[1]], y)
    ax[0][0].scatter(X[:, d_plot[0]], X[:, d_plot[1]], predictor.predict(X)[0])

    ax[0][1] = fig.add_subplot(1, 2, 2)
    ax[0][1].set_aspect("equal")
    ax[0][1].contour(X_predict[:, :, 0], X_predict[:, :, 1], y_hat)

    for row in ax:
        for col in row:
            col.set_xlabel(setup.search_space[0][d_plot[0]])
            col.set_ylabel(setup.search_space[0][d_plot[1]])


def plot_1d(setup, X, y, predictor):
    pass


##############################################################
from proposed_method import *
import itertools
from dummy_mf_solvers import *
import matplotlib.pyplot as plt

def draw_convergence(solver, solver_type_text):

    marker = itertools.cycle(("^", "o", ">", "s", "<", "8", "v", "p"))
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


def draw_current_levels(X, Z, Z_k, X_unique, X_plot, solver, ax=None):
    marker = itertools.cycle(("^", "o", "s", "<", "8", "v", "p"))
    if ax == None:
        fig, ax = plt.subplots(1, 1)
        # if we want to set the figure size.
        fig.set_figheight(4)
        fig.set_figwidth(10)

        # title
        fig.suptitle("{}".format(solver.__name__))
    else:
        ax.clear()

    for i in range(max(len(X) - 1, 2)):
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

    if len(X) > 2:  # first two levels always known
        i += 1
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
            X_unique,
            Z_k[i].predict(X_unique)[0],
            linestyle="",
            markeredgecolor="none",
            marker=">",
            color="black",
        )
        ax.plot(
            X_plot,
            Z_k[i].predict(X_plot)[0],
            label="prediction level {}".format(i),
            color=color,
        )
        Y_plot_true, _ = mf_forrester2008(X_plot, i, solver)
        ax.plot(
            X_plot,
            Y_plot_true,
            linestyle="--",
            label="true level {}".format(i),
            color=color,
        )
        i -= 1

    ax.plot(X_plot, forrester2008(X_plot), "--", label="truth", color="black")

    # ax.scatter(X[0], forrester2008(X[0]), label="truth sampled")
    best = np.argmin(Z[-1])
    ax.plot(X[-1][best], Z[-1][best], "*", label="current best")

    plt.legend()

    plt.draw()
    plt.pause(1)
    return ax
