import numpy as np
import math
import itertools

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from sampling.solvers.solver import get_solver
from sampling.solvers.internal import TestFunction
from proposed_method import *


class Plotting():
    def __init__(self,setup):
        self.n_per_d = 50
        self.d_plot = setup.d_plot
        self.d = setup.d
        self.plot_contour = setup.plot_contour

        # create self.X_plot, and X_pred
        self.axis_labels = [setup.search_space[0][i] for i in self.d_plot]
        self.lb = setup.search_space[1]
        self.ub = setup.search_space[2]
        self.create_grid()

        # we can plot the exact iff our solver is a self.plot_exacttion
        self.solver = get_solver(setup)
        self.plot_exact = isinstance(self.solver,TestFunction)
        self.figure_title = setup.solver_str + " {}d".format(setup.d)

        if setup.live_plot:
            if setup.d == 1 or len(setup.d_plot) == 1:
                self.plot = self.plot_1d
            else:                                                                 
                self.plot = self.plot_2d
                self.init_2d_fig()
        else:
            self.plot = lambda X,y,predictor: None

    def create_grid(self):
        """
        Build a (1d or 2d) grid used for plotting and sample n points in each dimension d.
        Dimensions d are specified by the dimensions to plot in setup.d_plot.
        Coordinates of the other dimensions are fixed in the centre of the range.
        """
        lin = np.linspace(self.lb[self.d_plot], self.ub[self.d_plot], self.n_per_d)

        # otherwise weird meshgrid of np.arrays
        lis = [lin[:, i] for i in range(len(self.d_plot))] 

        # create plotting meshgrid
        self.X_plot = np.array(np.meshgrid(*lis))
        xs=np.stack(self.X_plot, axis=-1)

        # create X for prediction, including centre domain values for other dimensions.
        self.X_pred = np.ones((self.X_plot[0].size, self.d)) * (self.lb + (self.ub - self.lb) / 2)
        self.X_pred[:,self.d_plot] = xs.reshape(-1, len(self.d_plot))

    def transform_X(self,X):
        """Transform X to have constant values in dimensions unused for plotting."""
        assert (X.ndim ==2), "\nProvide X in 2d shape."
        X_t = np.ones(X.shape) * (self.lb + (self.ub - self.lb) / 2)
        X_t[:,self.d_plot] = X[:,self.d_plot]
        return X_t

    def init_2d_fig(self):
        " initialise figure "
        fig = plt.figure()
        fig.suptitle("{}".format(self.figure_title))
        
        axes = []
        nrows = 1 + self.plot_contour    # if contour, we need another column
        ncols = 1 + self.plot_exact # if plot_exact, we need another column

        # predictions surface
        ax = fig.add_subplot(nrows, ncols, 1, projection="3d")
        ax.set_title("prediction surface")
        ax.set_zlabel("Z")
        axes.append(ax)

        # exact surface
        if self.plot_exact:    
            ax = fig.add_subplot(nrows, ncols, 2, projection="3d")
            ax.set_title("exact surface")
            ax.set_zlabel("Z")
            axes.append(ax)

        if self.plot_contour:
            ax = fig.add_subplot(nrows, ncols, 3)
            ax.set_title("prediction contour")
            ax.set_aspect("equal")
            axes.append(ax)

            if self.plot_exact:
                ax = fig.add_subplot(nrows, ncols, 4)
                ax.set_title("exact contour")
                ax.set_aspect("equal")
                axes.append(ax)

        " set axes properties "
        for ax in axes:
            ax.set_xlabel(self.axis_labels[0])
            ax.set_ylabel(self.axis_labels[1])

        self.fig = fig
        self.axes = axes



    def plot_2d(self, X, predictor):
        """
        Plotting for surfaces and possibly contours of the predicted function in 2d.
        Exact function values plotted if available.
        """                        

        " retrieve predictions "
        y_hat, mse = predictor.predict(self.X_pred)
        y,_ = self.solver.solve(self.transform_X(X))

        # reshape to X_plot shape
        y_hat = y_hat.reshape(self.X_plot[0].shape)
        std = np.sqrt(mse.reshape(self.X_plot[0].shape))

        " plot prediction surface "
        self.axes[0].plot_surface(self.X_plot[0], self.X_plot[1], y_hat, alpha=0.9)
        self.axes[0].plot_surface(self.X_plot[0], self.X_plot[1], y_hat - 2*std, alpha=0.4)
        self.axes[0].plot_surface(self.X_plot[0], self.X_plot[1], y_hat + 2*std, alpha=0.4)

        # add exact sampled
        self.axes[0].scatter(X[:, self.d_plot[0]], X[:, self.d_plot[1]], y)

        " plot exact surface "
        if self.plot_exact:                
            # retrieve exact solution
            y_exact, _ = self.solver.solve(self.X_pred)
            y_exact = y_exact.reshape(self.X_plot[0].shape)

            # plot
            self.axes[1].plot_surface(self.X_plot[0], self.X_plot[1], y_exact, alpha=0.9)

            # add samplepoints
            self.axes[1].scatter(X[:, self.d_plot[0]], X[:, self.d_plot[1]], y)

        if self.plot_contour:
            " Plot prediction contour "
            self.axes[2].contour(self.X_plot[0], self.X_plot[1], y_hat)

            " Plot exact contour "
            if self.plot_exact:
                self.axes[3].contour(self.X_plot[0], self.X_plot[1], y_exact)


    def plot_1d(setup, X, y, predictor):
        pass



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