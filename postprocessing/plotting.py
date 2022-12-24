# pyright: reportGeneralTypeIssues=false

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import axes3d

from core.sampling.solvers.solver import get_solver
from core.sampling.solvers.internal import TestFunction
from core.proposed_method import *

from core.kriging.mf_kriging import MultiFidelityKriging, ProposedMultiFidelityKriging

# print(plt.style.available)
# ['Solarize_Light2', '_classic_test_patch', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']
# plt.style.use("seaborn")


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
        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.markers = ["^", "o", "p", "8"] # "^", "o", ">", "s", "<", "8", "v", "p"
        self.s_standard = plt.rcParams['lines.markersize'] ** 2
        self.std_mult = 5 # 5 is 100% confidence interval

        self.axis_labels = [setup.search_space[0][i] for i in self.d_plot]
        
        self.lb = setup.search_space[1]
        self.ub = setup.search_space[2]
        self.create_plotting_grid()

        # we can plot the exact iff our solver is a self.plot_exact_possibletion
        self.solver = get_solver(setup)
        self.plot_exact_possible = isinstance(self.solver, TestFunction) # Can we cheaply retrieve the exact surface of that level?
        self.try_show_exact = False
        self.figure_title = setup.solver_str + " {}d".format(setup.d)

        if setup.live_plot:
            if setup.d == 1 or len(setup.d_plot) == 1:
                self.plot = self.plot_1d
                self.init_1d_fig()
            else:
                self.plot = self.plot_2d
                self.init_2d_fig()
        else:
            self.plot = lambda X, y, predictor, l: None


    def create_plotting_grid(self):
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
        fig, self.axes = plt.subplots(2, 1, figsize = (11,7.5))
        fig.suptitle("{}".format(self.figure_title))
        for ax in self.axes:
            ax.colors = []

    def plot_1d_ax(self, ax, predictor, l, X_sample = None, y_sample = None, show_exact: bool = True, is_truth : bool = False, label="", color = "", marker = ""):
        """
        plot 1d function
        """

        l, label, label_samples, label_exact, color, marker, show_exact, color_exact = self.get_standards(l, label, color, marker, is_truth, show_exact)

        ax.set_ylabel('y')
 
        # retrieve predictions
        y_hat, mse = predictor.predict(self.X_pred)
        std = np.sqrt(mse)

        # plot prediction and mse
        ax.plot(*self.X_plot, y_hat, linestyle= '--' if is_truth else "-", label=label, color=color, alpha=0.7)
        ax.plot(*self.X_plot, y_hat + self.std_mult * std, color=color, alpha=0.2)
        ax.plot(*self.X_plot, y_hat - self.std_mult * std, color=color, alpha=0.2)
        ax.colors.append(color)

        if X_sample is not None and y_sample is not None:
            # plot sample points
            ax.scatter(X_sample, y_sample, marker=marker, s = self.s_standard + 10 if is_truth else self.s_standard, color = color, linewidth=2, facecolor = "none" if is_truth else color, label = label_samples)
            ax.colors.append(color)

        if show_exact and self.plot_exact_possible:
            # NOTE this only works for test functions.
            # exact result of the level we try to predict
            y_exact, _ = self.solver.solve(self.X_pred) #l argument possible
            y_exact = y_exact.reshape(self.X_plot[0].shape)
            ax.plot(*self.X_plot, y_exact, '--', label = label_exact, color = color_exact, alpha = 0.5 ) 
            ax.colors.append(color_exact)

    def plot_1d(self, predictor, l, X_sample = None, y_sample = None, show_exact: bool = False, is_truth : bool = False, label="", color = "", marker = ""):
        self.plot_1d_ax(self.axes[0],predictor,l,X_sample,y_sample,show_exact,is_truth,color,marker)
        if l >= self.l_hifi:
            show_exact = self.try_show_exact
            self.plot_1d_ax(self.axes[1],predictor,l,X_sample,y_sample,show_exact,is_truth,label,color,marker)



    def init_2d_fig(self):
        "initialise figure"
        fig = plt.figure(figsize=(15,10))
        fig.suptitle("{}".format(self.figure_title))

        axes = []
        # first expand horizontally, then vertically; depending on plotting options
        ncols = 1 + (self.plot_exact_possible or self.plot_contour) 
        nrows = 1 + (self.plot_exact_possible and self.plot_contour)
        ind = 1

        # predictions surface
        ax = fig.add_subplot(nrows, ncols, ind, projection="3d")
        ax.set_title("prediction surface")
        ax.set_zlabel("Z")
        axes.append(ax)
        ind += 1

        # exact surface
        if self.plot_exact_possible:
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

            if self.plot_exact_possible:
                ax = fig.add_subplot(nrows, ncols, ind)
                ax.set_title("exact contour")
                ax.set_aspect("equal")
                axes.append(ax)

        # keep track of a list of plotted colors.
        for ax in axes:
            ax.colors = []

        self.fig = fig
        self.axes = axes

    def get_labels(self, l, label = "", is_truth=False):
        """
        Get the used labels.
        """
        
        if is_truth:
            label = label if label != "" else "Kriging level {} (truth)".format(l)
            label_samples = "Samples Kriged truth"
        else:
            label = label if label != "" else "Kriging level {}".format(l)
            label_samples = "Samples level {}".format(l)
        label_exact = "Exact level {}".format(l)

        return label, label_samples, label_exact

    def get_standards(self, l, label, color, marker, is_truth, show_exact):
        # Setting the labels used
        label, label_samples, label_exact = self.get_labels(l,label,is_truth)

        if is_truth:
            l += 1

        if show_exact:
            if not is_truth:
                show_exact = False
        
        # color and marker of this level
        if color == "":
            color = self.colors[l]
        if marker == "":
            marker = self.markers[3] # we only display one set of samples, so better be consistent!
        
        color_exact = self.colors[l+1]

        return l, label, label_samples, label_exact, color, marker, show_exact, color_exact

    def plot_2d_ax(self,ax,predictor, l, X_sample = None, y_sample = None, show_exact: bool = False, is_truth : bool = False, label="", color = "", marker = ""):
        """
        There are 4 desired axes / modes of plotting:
        1) Normal surfaces (+ exact if possible), (with or without samples)
        2) Normal contours (+ exact if possible), (with or without samples)
        3) Predicted + truth surfaces (+ exact if possible), (with or without samples)
        4) Predicted + truth contours (+ exact if possible), (with or without samples)
        """

        l, label, label_samples, label_exact, color, marker, show_exact, color_exact = self.get_standards(l, label, color, marker, is_truth, show_exact)

        " STEP 1: data preparation "
        # retrieve predictions from the provided predictor
        y_hat, mse = predictor.predict(self.X_pred)

        # reshape to X_plot shape
        y_hat = y_hat.reshape(self.X_plot[0].shape)
        std = np.sqrt(mse.reshape(self.X_plot[0].shape))

        # ADDITIONALLY get (and later plot) the exact level solution simultaneously
        y_exact = 0
        if show_exact and self.plot_exact_possible:
            y_exact, _ = self.solver.solve(self.X_pred)
            y_exact = y_exact.reshape(self.X_plot[0].shape)


        " STEP 2: plotting "
        if ax.name == "3d":
            # if the axis is 3d, we will plot a surface
    
            fix_colors(ax.plot_surface(*self.X_plot, y_hat, alpha=0.5, color=color, label=label))
            ax.plot_surface(*self.X_plot, y_hat - self.std_mult * std, alpha=0.1, color=color)
            ax.plot_surface(*self.X_plot, y_hat + self.std_mult * std, alpha=0.1 , color=color)
            ax.colors.append(color)

            if X_sample is not None and y_sample is not None:
                # add sample locations
                ax.scatter(*X_sample[:, self.d_plot].T, y_sample, marker = marker, s = self.s_standard + 10 if is_truth else self.s_standard, color = color, linewidth=2, facecolor = "none" if is_truth else color, label = label_samples) # needs s argument, otherwise smaller in 3d!
                ax.colors.append(color)


            if show_exact and self.plot_exact_possible:
                fix_colors(ax.plot_surface(*self.X_plot, y_exact, alpha=0.5, label = label_exact, color = color_exact))
                ax.colors.append(color_exact)
        else:
            # then we are plotting a contour, 2D data (only X, no y)
            # there is no difference between plotting the normal and truth here
            CS = ax.contour(*self.X_plot, y_hat, colors=color)
            CS.collections[-1].set_label(label)
            ax.colors.append(color)
            ax.CS = CS # Used for giving only the last contour inline labelling

            if X_sample is not None:
                ax.scatter(*X_sample[:, self.d_plot].T, marker = marker, s = self.s_standard + 10 if is_truth else self.s_standard, color = color, linewidth=2, facecolor = "none" if is_truth else color, label= label_samples)
                ax.colors.append(color)

            if show_exact and self.plot_exact_possible:
                CS = ax.contour(*self.X_plot, y_exact, colors=color_exact)
                CS.collections[-1].set_label(label_exact)
                ax.colors.append(color_exact)
                ax.CS = CS # Used for giving only the last contour inline labelling
                


    def plot_2d(self, predictor, l, X_sample = None, y_sample = None, show_exact: bool = False, is_truth : bool = False, label="", color = "", marker = ""):
        """
        Plotting for surfaces and possibly contours of the predicted function in 2d.
        Exact function values plotted if available.
        @param label: define a custom label for the prediction surface. If given, all other labels will not be used. 
        """      
        for i, ax in enumerate(self.axes):
            if i == 1 or i == 3: # right two plots, here we only want the truth / +exact
                show_exact = self.try_show_exact # NOTE not showing exact for now!!
                if l >= self.l_hifi:
                    self.plot_2d_ax(ax,predictor,l,X_sample,y_sample,show_exact,is_truth,label,color,marker)
            else:
                if not is_truth:
                    self.plot_2d_ax(ax,predictor,l,X_sample,y_sample,show_exact,is_truth,label,color,marker)


    def plot_kriged_truth(self, mf_model : MultiFidelityKriging, tune : bool = True):
        """
        Use to plot the fully sampled hifi truth Kriging with the prediction core.kriging.
        """
        print("Creating Kriging model of truth")
        # TODO
        tune = False

        K = mf_model.create_level(mf_model.X_truth, mf_model.Z_truth, tune = tune, append = False)
        self.plot(K, mf_model.l_hifi, mf_model.X_truth, mf_model.Z_truth, is_truth=True) #  color = 'black',

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
        L = mf_model.L
        self.l_hifi = mf_model.l_hifi

        if K_mf_alt != None:
            K_mf = K_mf_alt

        # reset axes for live plotting purposes
        axes = self.axes
        for ax in axes:
            ax.clear()

        " plot known kriging levels"
        has_prediction = isinstance(mf_model,ProposedMultiFidelityKriging) and mf_model.l_hifi + 1 == mf_model.number_of_levels
        for l in range(mf_model.number_of_levels - 1 if has_prediction else mf_model.number_of_levels):
            if mf_model.d == 1:
                self.plot(K_mf[l], l, X[l], Z[l]) # for 1d the added points is not too confusing
            else:
                self.plot(K_mf[l], l)
            # self.plot(X[l], Z[l], K_mf[l], l, label="Kriging level {}".format(l)) # with samples

        " plot prediction kriging "
        if has_prediction:
            l = self.l_hifi

            # prediction line, this is re-interpolated if we used noise and might not cross the sample points exactly
            # includes sample points, but not the predicted points!
            self.plot(K_mf[l], l, X[l], Z[l], label="Predicted level {}".format(l))
            
            # set up to plot our prediction`s` estimated part of points seperately in black
            X_plot_est = mf_model.return_unique_exc(X_exclude=X[-1])
            Z_plot_est = K_mf[l].predict(self.transform_X(X_plot_est))[0]

            # set up to plot best out of all the *sampled* hifi locations
            # NOTE not per definition the best location of all previous levels too
            best = np.argmin(Z[l])

            # plot the methods prediction points
            for ax in self.axes:
                color_predicted = 'black'
                color_best = 'black'
                ax.colors.append(color_predicted)
                ax.colors.append(color_best)
                label_predicted = "Predicted points level {}".format(l)
                marker_predicted = "+"

                # 3D scatter plot
                if ax.name == "3d" or  mf_model.d == 1:
                    # plot predicted points
                    ax.scatter(
                        *X_plot_est[:, self.d_plot].T,
                        Z_plot_est,
                        c=color_predicted,
                        marker=marker_predicted,
                        label = label_predicted, 
                        s = 70,
                        zorder = 5,
                    )

                    # plot best point
                    ax.scatter(*X[l][best, self.d_plot].T, Z[-1][best], s = 300, marker = "*", color = color_best, zorder = 6, facecolor="none", label="Current best sample")

                else: # the 2D scatter plot or 1d
                        
                    # plot predicted points
                    ax.scatter(
                        *X_plot_est[:, self.d_plot].T,
                        c=color_predicted,
                        marker=marker_predicted,
                        label = label_predicted, 
                        s = 70,
                        zorder = 5,
                    )

                    # plot best point
                    ax.scatter(*X[l][best, self.d_plot].T, s = 300, marker = "*", color = color_best, zorder = 6, facecolor="none", label="Current best sample")

        " plot 'full' Kriging level in case of linearity check"
        if len(K_mf)>3:
            self.plot(X[-1], Z[-1], K_mf[-1], label="Full Kriging (linearity check)")

        if hasattr(mf_model, "X_truth"):
            self.plot_kriged_truth(mf_model, tune = True)

        self.set_axis_props(mf_model)
        # plt.pause(2)


    def set_axis_props(self, mf_model):
        """ set axes properties like color and legend """
        # legend and plot settings
        # self.axes[0].legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
        
        for ax_nr, ax in enumerate(self.axes):
            is_line_or_surface = []
            is_line_or_surface_colors = []
            is_other = []
            counter = 0

            leg = ax.legend()
            if mf_model.d == 1:
                for i, lh in enumerate(leg.legendHandles):
                    if isinstance(lh,matplotlib.patches.Rectangle) or isinstance(lh,matplotlib.lines.Line2D):
                        is_line_or_surface.append(i)
                        is_line_or_surface_colors.append(lh._color)
                    else:
                        # still if the label color is exactly the same as of line or surface, we want to add it after!
                        loc = -1
                        for jj, j in enumerate(is_line_or_surface_colors):
                            if lh._original_edgecolor == j:
                                print("{} and {} for colors {} and {}".format(i,j,lh._original_edgecolor,ax.colors[jj]))
                                loc = jj + 1 + counter
                                break
                        if loc >= 0:
                            is_line_or_surface.insert(loc,i)
                            counter += 1
                            print("new line_surface: {}".format(is_line_or_surface))
                        else:
                            is_other.append(i)
            else: 
                for i, lh in enumerate(leg.legendHandles):
                    if isinstance(lh,matplotlib.patches.Rectangle) or isinstance(lh,matplotlib.lines.Line2D):
                        is_line_or_surface.append(i)
                    else:
                        if len(is_line_or_surface) > 0 and ax.colors[i]==ax.colors[is_line_or_surface[-1]]:
                            is_line_or_surface.append(i)
                        else:
                            is_other.append(i)            

            # reorder, surfaces / lines flushed forwards! 
            order = is_line_or_surface + is_other
            if (mf_model.d != 1) and ax_nr == 1 or ax_nr == 3:
                loc = 'left'
                bb = (1.08, 1.0)
            else:
                loc = 'right'
                bb = (-0.08, 1.0)
            
            extra = []
            extra_text = []
            if ax_nr == 0 or ax_nr == 1: # works for both 1d as 2d plotting
                extra_text = [u"\nVariance displayed with\n\u00B1 {} standard deviations".format(self.std_mult)]
                extra = [Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)]

            handles, labels = ax.get_legend_handles_labels()
            leg = ax.legend([handles[idx] for idx in order] + extra,[labels[idx] for idx in order] + extra_text,loc='upper '+loc, bbox_to_anchor=bb)

            # now reset the color and alpha! (calling ax.legend resets this)
            # NOTE this is only fucked for 3d surface plots
            if ax.name == '3d':
                for i in range(len(is_line_or_surface)):
                    lh = leg.legendHandles[i]
                    without_facecolor = False
                    try:
                        if not any(lh.get_facecolor()):
                            without_facecolor = True
                    except:
                        pass

                    if not without_facecolor:
                        lh.set_alpha(1)
                        lh.set_color(ax.colors[is_line_or_surface[i]])
                for t in leg.texts:
                    t.set_alpha(1)
            else:
                try:
                    ax.clabel(ax.CS, inline=1, fontsize=10)
                except:
                    pass
            
            # # make the start of 'current best' smaller
            # for lh in leg.legendHandles:
            #     if 'best' in lh._label:
            #         lh._legmarker.set_markersize(15) # FAKKING BULLSHIT

        for ax in self.axes:
            ax.set_xlabel(self.axis_labels[0])
            if mf_model.d > 1:
                ax.set_ylabel(self.axis_labels[1])

        
        plt.tight_layout()
        plt.draw()

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
    
