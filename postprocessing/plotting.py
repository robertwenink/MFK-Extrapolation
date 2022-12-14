# pyright: reportGeneralTypeIssues=false

import numpy as np
import itertools

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from core.sampling.solvers.solver import get_solver
from core.sampling.solvers.internal import TestFunction
from core.proposed_method import *

from core.kriging.mf_kriging import MultiFidelityKriging, ProposedMultiFidelityKriging
  
def fix_colors(surf):
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

class Plotting:
    def __init__(self, setup):
        self.n_per_d = 100
        self.d_plot = setup.d_plot
        self.d = setup.d
        self.plot_contour = setup.plot_contour

        # create self.X_plot, and X_pred
        self.axes = []
        self.markers = itertools.cycle(("^", "o", "p"))
        self.axis_labels = [setup.search_space[0][i] for i in self.d_plot]
        self.lb = setup.search_space[1]
        self.ub = setup.search_space[2]
        self.create_grid()

        # we can plot the exact iff our solver is a self.plot_exacttion
        self.solver = get_solver(setup)
        self.plot_exact = isinstance(self.solver, TestFunction) # Can we cheaply retrieve the exact surface of that level?
        self.figure_title = setup.solver_str + " {}d".format(setup.d)

        if setup.live_plot:
            if setup.d == 1 or len(setup.d_plot) == 1:
                self.plot = self.plot_1d
                self.init_1d_fig()
            else:
                self.plot = self.plot_2d
                self.init_2d_fig()
        else:
            self.plot = lambda X, y, predictor: None


    def iterate_color_marker(func):
        def wrapper(self, *args, **kwargs):
            self.marker = next(self.markers)
            self.color = next(self.axes[0]._get_lines.prop_cycler)["color"]
            func(self, *args, **kwargs)
            self.reset_color_cycle()
        return wrapper

    def reset_color_cycle(self):
        plt.gca().set_prop_cycle(None)

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
        xs = np.stack(self.X_plot, axis=-1)

        # create X for prediction, including centre domain values for other dimensions.
        self.X_pred = np.ones((self.X_plot[0].size, self.d)) * (
            self.lb + (self.ub - self.lb) / 2
        )
        self.X_pred[:, self.d_plot] = xs.reshape(-1, len(self.d_plot))

    def transform_X(self, X):
        """Transform X to have constant values in dimensions unused for plotting."""
        assert X.ndim == 2, "\nProvide X in 2d shape."
        X_t = np.ones(X.shape) * (self.lb + (self.ub - self.lb) / 2)
        X_t[:, self.d_plot] = X[:, self.d_plot]
        return X_t

    def init_1d_fig(self):
        fig, ax = plt.subplots(1, 1, figsize = (4,10))
        fig.suptitle("{}".format(self.figure_title))
        self.axes.append(ax)
    
    
    @iterate_color_marker
    def plot_1d(self, X, y, predictor, l, label=""):
        """
        X : X where we have actually sampled
        predictor : K_mf
        """
        ax = self.axes[0]

        # plot sample points
        ax.plot(
            X,
            y,
            linestyle="",
            markeredgecolor="none",
            marker=self.marker,
            color=self.color,
            label="" if label =="" else "Samples level {}".format(l) if l is not None else "", 
        )

        # retrieve predictions
        y_hat, mse = predictor.predict(self.X_pred)
        std = np.sqrt(mse)

        # plot prediction and mse
        ax.plot(*self.X_plot, y_hat, label=label, color=self.color, alpha=0.7)
        ax.plot(*self.X_plot, y_hat + 2 * std, color=self.color, alpha=0.2)
        ax.plot(*self.X_plot, y_hat - 2 * std, color=self.color, alpha=0.2)


    def init_2d_fig(self):
        "initialise figure"
        fig = plt.figure(figsize=(10,10))
        fig.suptitle("{}".format(self.figure_title))

        axes = []
        # first expand horizontally, then vertically; depending on plotting options
        ncols = 1 + (self.plot_exact or self.plot_contour) 
        nrows = 1 + (self.plot_exact and self.plot_contour)
        ind = 1

        # predictions surface
        ax = fig.add_subplot(nrows, ncols, ind, projection="3d")
        ax.set_title("prediction surface")
        ax.set_zlabel("Z")
        axes.append(ax)
        ind += 1

        # exact surface
        if self.plot_exact:
            ax = fig.add_subplot(nrows, ncols, ind, projection="3d")
            ax.set_title("exact surface")
            ax.set_zlabel("Z")
            axes.append(ax)
            ind += 1

        if self.plot_contour:
            ax = fig.add_subplot(nrows, ncols, ind)
            ax.set_title("prediction contour")
            ax.set_aspect("equal")
            axes.append(ax)
            ind += 1

            if self.plot_exact:
                ax = fig.add_subplot(nrows, ncols, ind)
                ax.set_title("exact contour")
                ax.set_aspect("equal")
                axes.append(ax)

        " set axes properties "
        for ax in axes:
            ax.set_xlabel(self.axis_labels[0])
            ax.set_ylabel(self.axis_labels[1])

        self.fig = fig
        self.axes = axes

    @iterate_color_marker
    def plot_2d(self, X, y, predictor, l = None, label=""):
        """
        Plotting for surfaces and possibly contours of the predicted function in 2d.
        Exact function values plotted if available.
        @param label: define a custom label for the prediction surface. If given, all other labels will not be used. 
        """      

        if label == "":
            plt.legend()

        # marker used for handling multiple axes
        ind = 0

        " retrieve predictions "
        y_hat, mse = predictor.predict(self.X_pred)

        # reshape to X_plot shape
        y_hat = y_hat.reshape(self.X_plot[0].shape)
        std = np.sqrt(mse.reshape(self.X_plot[0].shape))

        " plot prediction surface "
        ax, ind = self.axes[ind], ind + 1
        fix_colors(ax.plot_surface(*self.X_plot, y_hat, alpha=0.6, color=self.color, label=label if label != "" else "Prediction surface"))
        ax.plot_surface(*self.X_plot, y_hat - 2 * std, alpha=0.1, color=self.color)
        ax.plot_surface(*self.X_plot, y_hat + 2 * std, alpha=0.1 , color=self.color)

        # add sample locations
        ax.scatter(*X[:, self.d_plot].T, y, c=self.color, marker=self.marker, label="" if label == "" else "Samples level {}".format(l) if l is not None else "")

        " plot exact surface "
        if self.plot_exact:

            # retrieve exact solution
            y, _ = self.solver.solve(self.transform_X(X))
            y_exact, _ = self.solver.solve(self.X_pred)
            y_exact = y_exact.reshape(self.X_plot[0].shape)

            # plot
            ax, ind = self.axes[ind], ind + 1
            fix_colors(ax.plot_surface(*self.X_plot, y_exact, alpha=0.9))

            # add samplepoints
            ax.scatter(*X[:, self.d_plot].T, y, c=self.color, marker=self.marker)

        if self.plot_contour:
            "Plot prediction contour"
            ax, ind = self.axes[ind], ind + 1
            CS = ax.contour(*self.X_plot, y_hat, colors=self.color)
            CS.collections[-1].set_label("Kriging level {}".format(l) if l is not None else "")
            ax.scatter(*X[:, self.d_plot].T, c=self.color, marker=self.marker, label = "Samples level {}".format(l) if l is not None else "")

            " Plot exact contour "
            if self.plot_exact:
                # TODO deze wordt gewoon zovaak het level bestaat geplot!!
                # TODO per ax een eigen plotfunction maken, veel overzichtelijker!
                #       wel even nadenken hoe je dat dan met plot_1d plot_2d doet
                # TODO colorcycle / iterator kunnen resetten -> kan zolang colormarker per functie zelf een decorator is??
                ax, ind = self.axes[ind], ind + 1
                CS = ax.contour(*self.X_plot, y_exact, colors=self.color)
                ax.clabel(CS, inline=1, fontsize=10)
                CS.collections[-1].set_label("Exact level {}".format(l) if l is not None else "Exact") # zou ook met ax ipv CS kunnen

        plt.tight_layout()

    def plot_kriged_truth(self, mf_model : MultiFidelityKriging):
        """
        Use to plot the fully sampled hifi truth Kriging with the prediction core.kriging.
        """
        print("Creating Kriging model of truth")
        K = mf_model.create_level(mf_model.X_truth, mf_model.Z_truth, tune = True, append = False)

        self.plot(mf_model.X_truth, mf_model.Z_truth, K, label="Kriging truth")
        # self.plot(X_hifi_full, Z_hifi_full, predictor, label="Kriging truth")

    def draw_current_levels(self, mf_model : MultiFidelityKriging, K_mf_alt = None):
        """

        TODO for plotting: 
        1) if there is a sampled truth present, i.e. mf_model.X_truth, then plot 'exact' too.
        This has exactly the same plot as the regular one, but only contains the truth and prediction -> segment/refactor code!!
        2) add K_mf_alt, this is an option that can/will be used by the linearity check.

        @param X: !! 3D array, different in format, it is now an array of multiple X in the standard format
                    for the last level, this contains only the sampled locations.
        @param Z: list of the sampled z (or y) values; these can be different than the predictions when noise is involved.
        @param K_mf: list of the Kriging predictors of each level
        @param X_unique: unique X present in X_l. At these locations we estimate our prediction points.
        """
        X = mf_model.X_mf # contains the hifi sampled level too
        Z = mf_model.Z_mf # contains the hifi sampled level too
        K_mf = mf_model.K_mf
        X_unique = mf_model.X_unique
        L = mf_model.L
        if K_mf_alt != None:
            K_mf = K_mf_alt

        # reset axes and colorcycle
        axes = self.axes
        self.markers = itertools.cycle(("^", "o", "p"))
        for ax in axes:
            ax.clear()
            ax._get_lines.set_prop_cycle(None)

        " plot known kriging levels"
        has_prediction = isinstance(mf_model,ProposedMultiFidelityKriging) and mf_model.l_hifi + 1 == mf_model.number_of_levels
        for l in range(mf_model.number_of_levels - 1 if has_prediction else mf_model.number_of_levels):
            self.plot(X[l], Z[l], K_mf[l], l, label="Kriging level {}".format(l))

        " plot prediction kriging "
        if has_prediction:
            l = mf_model.l_hifi

            # plot our prediction, estimated points in black
            inds = np.array([np.all(X_unique == x, axis=1) for x in X[-1]]).any(axis=0)
            X_plot_est = np.delete(X_unique,inds,axis = 0)

            self.axes[0].scatter(
                *X_plot_est[:, self.d_plot].T,
                K_mf[l].predict(self.transform_X(X_plot_est))[0],
                c="black",
                marker="+",
                label = "Extrapolated points", 
                s = 70
            )

            # prediction line, this is re-interpolated if we used noise and might not cross the sample points exactly
            self.plot(X[l], Z[l], K_mf[l], l, label="Kriging level {}".format(l))
            

            # best out of all the *sampled* locations
            best = np.argmin(Z[-1])
            t =  X[l][best, self.d_plot].T
            self.axes[0].scatter(*X[l][best, self.d_plot].T, Z[-1][best], s = 70, marker = "*", color = 'red', zorder = 10, label="Current best")
            if self.plot_contour:
                self.axes[1].scatter(*X[l][best, self.d_plot].T, s = 70, marker = "*", color = 'red', zorder = 10, label="Current best")


            if self.plot_exact:
                # NOTE this only works for test functions.
                # exact result of the level we try to predict
                y_pred_truth = self.solver.solve(self.X_pred, l = L[-1])[0].reshape(
                    self.X_plot[0].shape
                )
                kwargs = {
                    "label":"true level {}".format(l),
                    "color":self.color,
                    "alpha":0.5
                }
                if self.d == 1:
                    ax=self.axes[0]
                    ax.plot(*self.X_plot, y_pred_truth, '--', **kwargs ) 
                else:
                    ax=self.axes[1]
                    fix_colors(ax.plot_surface(*self.X_plot, y_pred_truth, **kwargs))

            
        " plot truth "
        if self.plot_exact:
            # exact hifi truth
            y_exact = self.solver.solve(self.X_pred)[0].reshape(self.X_plot[0].shape)

            kwargs2 = {
                "label":"truth", 
                "color":"black", 
                "alpha": 0.3
            }
            
            # use correct plotting function
            if self.d == 1:
                ax=self.axes[0]
                ax.plot(*self.X_plot, y_exact, '--', **kwargs2)
            else:
                ax=self.axes[1]
                fix_colors(ax.plot_surface(*self.X_plot, y_exact, **kwargs2))
        
        " plot 'full' Kriging level in case of linearity check"
        if len(K_mf)>3:
            self.plot(X[-1], Z[-1], K_mf[-1], label="Full Kriging (linearity check)")

        # legend and plot settings
        # self.axes[0].legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
        for ax in axes:
            leg = ax.legend()
            for t in leg.texts:
                t.set_alpha(1)
            for lh in leg.legendHandles:
                lh.set_alpha(1)
                # lh.set_color(self.color) # TODO reset cycler!!
        
        plt.tight_layout()
        plt.draw()
        # plt.pause(2)

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
    
