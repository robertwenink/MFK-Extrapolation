import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sampling.solvers.solver import get_solver
from sampling.solvers.internal import TestFunction

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
    contour = False

    # retrieve the solver and check type
    solver = get_solver(setup)
    testfunc = False
    if isinstance(solver,TestFunction):
        testfunc = True
        
    # retrieve some plotting parameters    
    d_plot = setup.d_plot
    d = setup.d
    n_per_d = setup.n_per_d
    X_new = grid_plot(setup)
    X_plot = X_new[:, d_plot].reshape(n_per_d, n_per_d, -1)

    " retrieve predictions "
    y_hat, mse = predictor.predict(X_new)

    # reshape predictions to be a grid
    y_hat = y_hat.reshape(n_per_d, n_per_d)
    std = np.sqrt(mse.reshape(n_per_d, n_per_d))

    " retrieve exact solution "
    y_exact, _ = solver.solve(X_new)

    # reshape predictions to be a grid
    y_exact = y_exact.reshape(n_per_d, n_per_d)


    " initialise figure "
    fig = plt.figure()
    fig.suptitle("{}".format(setup.solver_str))  # set_title if ax
    axes = []

    nrows = 1 + contour 
    ncols = 1 + testfunc # then we can depict the exact solution

    " plot prediction surface "
    ax = fig.add_subplot(nrows, ncols, 1, projection="3d")
    ax.set_title("prediction surface")
    axes.append(ax)

    ax.plot_surface(X_plot[:, :, 0], X_plot[:, :, 1], y_hat, alpha=0.9)
    ax.plot_surface(
        X_plot[:, :, 0], X_plot[:, :, 1], y_hat - 2*std, alpha=0.4
    )
    ax.plot_surface(
        X_plot[:, :, 0], X_plot[:, :, 1], y_hat + 2*std, alpha=0.4
    )
    ax.set_zlabel("Z")

    # add samplepoints
    ax.scatter(X[:, d_plot[0]], X[:, d_plot[1]], y)
    ax.scatter(X[:, d_plot[0]], X[:, d_plot[1]], predictor.predict(X)[0])

    " plot exact surface "
    if testfunc:
        ax = fig.add_subplot(nrows, ncols, 2, projection="3d")
        ax.set_title("exact surface")
        axes.append(ax)

        ax.plot_surface(X_plot[:, :, 0], X_plot[:, :, 1], y_exact, alpha=0.9)
        ax.set_zlabel("Z")

        # add samplepoints
        ax.scatter(X[:, d_plot[0]], X[:, d_plot[1]], y)
        ax.scatter(X[:, d_plot[0]], X[:, d_plot[1]], predictor.predict(X)[0])

    if contour:
        " Plot prediction contour "
        ax = fig.add_subplot(nrows, ncols, 3)
        ax.set_title("prediction contour")
        axes.append(ax)

        ax.set_aspect("equal")
        ax.contour(X_plot[:, :, 0], X_plot[:, :, 1], y_hat)

        " Plot exact contour "
        if testfunc:
            ax = fig.add_subplot(nrows, ncols, 4)
            ax.set_title("exact contour")
            axes.append(ax)

            ax.set_aspect("equal")
            ax.contour(X_plot[:, :, 0], X_plot[:, :, 1], y_exact)

    " set axes properties "
    for ax in axes:
            ax.set_xlabel(setup.search_space[0][d_plot[0]])
            ax.set_ylabel(setup.search_space[0][d_plot[1]])


def plot_1d(setup, X, y, predictor):
    pass


##############################################################
from proposed_method import *
import itertools
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


def draw_current_levels(X, Z, Z_k, X_unique, X_plot, solver, ax=None, Z_k_new_noise = None):
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

    " plot known kriging levels"
    for i in range(max(len(X) - 1, 2)):
        color = next(ax._get_lines.prop_cycler)["color"]
        plot_single_1d_kriging(X[i],Z[i],next(marker),Z_k[i], X_plot, ax, color,"Kriging level {}".format(i),alpha=0.2)


    " plot prediction kriging "
    if len(X) > 2:  # first two levels always known
        i += 1
        color = next(ax._get_lines.prop_cycler)["color"]

        ax.plot(
            X_unique,
            Z_k[i].predict(X_unique)[0],
            linestyle="",
            markeredgecolor="none",
            marker=">",
            color="black",
        )

        # prediction
        mark = next(marker)
        plot_single_1d_kriging(X[i],Z[i],mark,Z_k[i], X_plot, ax, color,"prediction level {}".format(i))

        if Z_k_new_noise is not None: 
            color = next(ax._get_lines.prop_cycler)["color"]
            plot_single_1d_kriging(X[i],Z[i],mark,Z_k_new_noise, X_plot, ax, color,"prediction level {} with noise".format(i))
            

        " plot truth "
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


def plot_single_1d_kriging(Xi,Zi,marker,Z_k, X_plot, ax, color,label,alpha=0.4):
    z_pred, mse_pred = Z_k.predict(X_plot)

    # plot sample points
    ax.plot(
        Xi,
        Zi,
        linestyle="",
        markeredgecolor="none",
        marker=marker,
        color=color,
    )

    # plot prediction and mse
    ax.plot(
        X_plot,
        z_pred,
        label=label,
        color=color,
    )
    ax.plot(
        X_plot,
        z_pred + 2*np.sqrt(mse_pred),
        color=color,
        alpha=alpha,
    )
    ax.plot(
        X_plot,
        z_pred - 2*np.sqrt(mse_pred),
        color=color,
        alpha=alpha,
    )