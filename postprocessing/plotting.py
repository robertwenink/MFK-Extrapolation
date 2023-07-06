# pyright: reportGeneralTypeIssues=false

from matplotlib.transforms import Bbox
import numpy as np
import time 
import os 
import shutil
import glob
import imageio
from PIL import Image
import tkinter as tk
import ctypes

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import axes3d

from preprocessing.input import Input

from core.sampling.solvers.solver import get_solver
from core.sampling.solvers.internal import TestFunction
from core.mfk.mfk_base import MultiFidelityKrigingBase
from core.mfk.proposed_mfk import ProposedMultiFidelityKriging

from utils.error_utils import RMSE_norm_MF


def fix_colors(surf):
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d
    return surf


class Plotting:
    def __init__(self, setup : Input, inset_kwargs = None, plotting_pause : float = 0, plot_once_every = 1, fast_plot = True, make_video = False):
        self.d_plot = setup.d_plot
        self.d = setup.d

        # plotting resolution
        if fast_plot and self.d > 1:
            self.n_per_d = 75
            self.sub_skip = 4
            plt.rcParams["figure.dpi"] = 80
        else:
            self.sub_skip = 1
            self.n_per_d = 150
            
        if len(self.d_plot) == 1: 
            # then no need for reducing 
            self.sub_skip = 1
            self.n_per_d = 1000

        # plotting options and settings
        self.axes = []
        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.markers = ["^", "o", "p", "8"] # "^", "o", ">", "s", "<", "8", "v", "p"
        self.s_standard = plt.rcParams['lines.markersize'] ** 2 # 6 ^ 2 = 36
        self.truth_s_increase = 40
        self.std_mult = 5 # 5 is 100% confidence interval
        self.axis_labels = [setup.search_space[0][i] for i in self.d_plot]
        
        # create self.X_plot, and X_pred
        self.lb = setup.search_space[1]
        self.ub = setup.search_space[2]
        self.create_plotting_grid()

        # we can plot the exact iff our solver is a self.plot_exact_possibletion
        self.solver = get_solver(setup)
        self.plot_exact_possible = isinstance(self.solver, TestFunction) # Can we cheaply retrieve the exact surface of that level?
        self.try_show_exact = True
        self.plot_truth = not self.plot_exact_possible
        self.figure_title = setup.solver_str + " {}d".format(setup.d)

        # plot seperate ax with only response?
        self.plot_double_axes = False
        self.plot_NRMSE_text = False
        self.show_legend = True

        if setup.d == 1 or len(setup.d_plot) == 1:
            self.plot = self.plot_1d
            self.init_1d_fig()
        else:
            self.plot = self.plot_2d
            self.init_2d_fig()
        self.plotting_pause = plotting_pause

        # option to directly set the inset at init
        if inset_kwargs != None:
            self.set_zoom_inset(**inset_kwargs)

        self.counter = 0
        self.tight = False
        self.plot_once_every = plot_once_every if self.d > 1 else 1

        self.make_video = make_video
        self.save_svg = False
        create_folder_per_run = True
        if make_video:
            self.frames = []     
            self.video_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output_files', os.path.split(setup.filename)[-1].split('.')[0])
            if not os.path.exists(self.video_path):
                os.makedirs(self.video_path)
            
            if create_folder_per_run:
                dirs = glob.glob(self.video_path+os.path.sep+"*"+os.path.sep)
                i = len(dirs)
                self.video_path = os.path.join(self.video_path,f"run_{i}")  
                os.makedirs(self.video_path)

                # copy the input file
                shutil.copy(setup.file_path, self.video_path)
                self.setup_file_path = setup.file_path

                print(f"Writing results to {os.path.join(*self.video_path.split(os.sep)[-3:])}")
            else:
                img_paths_list = sorted(glob.glob(self.video_path + os.path.sep + "image_*.png"), key=os.path.getmtime)
                for img in img_paths_list:
                    if os.path.isfile(img):
                        os.remove(img)
                print("Deleted previous png`s!!")

    ###########################################################################
    ## Helper functions                                                     ###
    ###########################################################################

    def create_plotting_grid(self):
        """
        Build a (1d or 2d) grid used for plotting and sample n points in each dimension d.
        Dimensions d are specified by the dimensions to plot in setup.d_plot.
        Coordinates of the other dimensions are fixed in the centre of the range.
        """
        lin = np.linspace(self.lb[self.d_plot], self.ub[self.d_plot], self.n_per_d)

        # otherwise weird meshgrid of np.arrays
        lis = [lin[:, i] for i in range(len(self.d_plot))]
        lis_sub = [lin[::self.sub_skip, i] for i in range(len(self.d_plot))]

        # create plotting meshgrid
        self.X_plot = np.array(np.meshgrid(*lis))
        self.X_plot_sub = np.array(np.meshgrid(*lis_sub))
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

    def get_labels(self, l, label = "", is_truth=False, label_samples = ""):
        """
        Get the used labels.
        """
        
        if is_truth:
            label = label if label != "" else "Kriging level {} (truth)".format(l)
            label_samples = label_samples if label_samples != "" else "Samples Kriged truth"
        else:
            label = label if label != "" else "Kriging level {}".format(l)
            label_samples = label_samples if label_samples != "" else "Samples level {}".format(l)
        label_exact = "Exact level {}".format(l)

        return label, label_samples, label_exact

    def get_standards(self, l, label, color, marker, is_truth, show_exact, label_samples):
        # Setting the labels used
        label, label_samples, label_exact = self.get_labels(l,label,is_truth, label_samples)

        if is_truth:
            l += 1

        if show_exact:
            if self.plot_truth and not is_truth:
                show_exact = False
            color_exact = self.colors[l+1]
        else:
            color_exact = 0

        # color and marker of this level
        if color == "":
            color = self.colors[l]
        if marker == "":
            marker = self.markers[3] # we only display one set of samples, so better be consistent!
        

        return l, label, label_samples, label_exact, color, marker, show_exact, color_exact

    ###########################################################################
    ## Plotting                                                             ###
    ###########################################################################

    def plot_exact_new_figure(self):
        from matplotlib import cm
        from matplotlib.colors import LightSource
        cmap = cm.summer
        # cmap = cm.cividis
        # cmap = cm.plasma
        if "Rosenbrock" in self.solver.name:
            ls = LightSource(azdeg=45, altdeg=45)  # Customize light properties
            if self.d >2:
                ls = LightSource(azdeg=45, altdeg=65)  # Customize light propert    ies
        else:
            ls = LightSource(azdeg=-45, altdeg=45)  # Customize light properties

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        y_exact, _ = self.solver.solve(self.X_pred, l = -1) #l argument possible
        y_exact = y_exact.reshape(self.X_plot[0].shape)
        rgb = ls.shade(y_exact, cmap)
        surf = ax.plot_surface(*self.X_plot, y_exact, facecolors=rgb, shade=True, linewidth=0.05)
        # surf = ax.plot_surface(*self.X_plot, y_exact, shade = True, lightsource=ls, cmap = cmap, linewidth=0.2)
        surf.set_edgecolor((0.95,0.95,0.95,1))
        # fix_colors(ax.plot_surface(*self.X_plot, y_exact))
        fig.suptitle(f"{self.d}d {self.solver.name}", y = 0.9)
        if self.d > 2:
            fig.suptitle(f"{self.d}d {self.solver.name} (projected)", y = 0.9)
        
        ax.set_zlabel("Z")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        fig.tight_layout()
        
        " figure with transform! "
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        y_1 = self.solver.solve(self.X_pred, l = 2)[0].reshape(self.X_plot[0].shape) #l argument possible
        y_2 = self.solver.solve(self.X_pred, l = 3)[0].reshape(self.X_plot[0].shape) #l argument possible
        y_exact, _ = self.solver.solve(self.X_pred, l = -1) #l argument possible
        y_exact = y_exact.reshape(self.X_plot[0].shape)

        rgb = ls.shade(y_exact, cmap)
        surf = fix_colors(ax.plot_surface(*self.X_plot, y_1, shade=True, linewidth=0.05, alpha = .4, label = "Low fidelity"))
        surf = fix_colors(ax.plot_surface(*self.X_plot, y_2, shade=True, linewidth=0.05, alpha = .4, label = "Medium fidelity"))
        # surf = fix_colors(ax.plot_surface(*self.X_plot, y_exact, linewidth=0.05, alpha = .45, label = "High fidelity"))
        surf = fix_colors(ax.plot_surface(*self.X_plot, y_exact, facecolors=rgb, shade=True, linewidth=0.05, alpha = .45, label = "High fidelity"))
        # surf = ax.plot_surface(*self.X_plot, y_exact, shade = True, lightsource=ls, cmap = cmap, linewidth=0.2)
        surf.set_edgecolor((0.95,0.95,0.95,1))
        # fix_colors(ax.plot_surface(*self.X_plot, y_exact))
        fig.suptitle(f"{self.d}d {self.solver.name}", y = 0.9)
        if self.d > 2:
            fig.suptitle(f"{self.d}d {self.solver.name} (projected)", y = 0.9)
        
        ax.set_zlabel("Z")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        handles, labels = ax.get_legend_handles_labels()
        order = [2,1,0]
        leg = fig.legend([handles[idx] for idx in order],[labels[idx] for idx in order]) 


        for i, lh in enumerate(leg.legendHandles):
            lh.set_alpha(0.8)
            if order[i] != 2:
                lh.set_color(self.colors[order[i]])
            else:
                lh.set_color( (0.0, 0.5, 0.4))

        fig.tight_layout()

    def init_1d_fig(self):
        fig, self.axes = plt.subplots(1 + self.plot_double_axes, 1, figsize = (12, 4 + 4 * self.plot_double_axes))
        fig.suptitle("{}".format(self.figure_title))
        if not self.plot_double_axes:
            self.axes = [self.axes]

        for ax in self.axes:
            ax.colors = []
        fig.canvas.manager.window.move(0,0)
        self.fig = fig

    def plot_1d_ax(self, ax, predictor, l, X_sample = None, y_sample = None, show_exact: bool = False, is_truth : bool = False, label="", color = "", marker = "", label_samples = ""):
        """
        plot 1d function
        """

        # plot on the inset axis if it is there! Do this by 'recursion'
        if hasattr(ax,'axin'):
            ax.axin.colors = ax.colors
            self.plot_1d_ax(ax.axin,predictor,l,X_sample,y_sample,show_exact,is_truth,color,marker)


        l, label, label_samples, label_exact, color, marker, show_exact, color_exact = self.get_standards(l, label, color, marker, is_truth, show_exact, label_samples)

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
            ax.scatter(X_sample, y_sample, marker=marker, s = self.s_standard + self.truth_s_increase if is_truth else self.s_standard, color = color, linewidth=2, facecolor = "none" if is_truth else color, label = label_samples, zorder = 5)
            ax.colors.append(color)

        if show_exact and self.plot_exact_possible:
            # NOTE this only works for test functions.
            # exact result of the level we try to predict
            y_exact, _ = self.solver.solve(self.X_pred, l = -1) #l argument possible
            y_exact = y_exact.reshape(self.X_plot[0].shape)
            ax.plot(*self.X_plot, y_exact, '--', label = label_exact, color = color_exact, alpha = 0.5) 
            ax.colors.append(color_exact)

    def plot_1d(self, predictor, l, X_sample = None, y_sample = None, show_exact: bool = True, is_truth : bool = False, label="", color = "", marker = "", label_samples = ""):
        if l >= self.l_hifi and show_exact:
            show_exact = self.plot_exact_possible
        else:
            show_exact = False

        self.plot_1d_ax(self.axes[0],predictor,l,X_sample,y_sample,show_exact,is_truth,label,color,marker, label_samples)

        if l >= self.l_hifi and self.plot_double_axes:
            self.plot_1d_ax(self.axes[1],predictor,l,X_sample,y_sample,show_exact,is_truth,label,color,marker, label_samples)


    def init_2d_fig(self):
        "initialise figure"
        fig = plt.figure(figsize=(15.3,10))
        fig.suptitle("{}".format(self.figure_title))

        axes = []
        # first expand horizontally, then vertically; depending on plotting options
        ncols = 2
        nrows = 2
        ind = 1

        # prediction surfaces
        ax = fig.add_subplot(nrows, ncols, ind, projection="3d")
        ax.set_title("prediction surface")
        axes.append(ax)

        # will be formatted per line as sublist [min_std, min, max, max_std] 
        # with eg min_std being the minimum of line - variance
        ax.lims = [] 
        ind += 1

        # 'exact' / truth surfaces
        ax = fig.add_subplot(nrows, ncols, ind, projection="3d")
        ax.set_title("exact surface")
        axes.append(ax)
        ax.lims = []
        ind += 1

        # contours
        ax = fig.add_subplot(nrows, ncols, ind)
        ax.set_title("prediction contour")
        ax.set_aspect("equal")
        axes.append(ax)
        ind += 1

        # 'exact' / truth contours
        ax = fig.add_subplot(nrows, ncols, ind)
        ax.set_title("exact contour")
        ax.set_aspect("equal")
        axes.append(ax)

        # keep track of a list of plotted colors.
        for ax in axes:
            ax.colors = []
            ax._plot_ref = {}

        fig.canvas.manager.window.move(0,0)
        fig.canvas.draw()
        plt.show(block=False)
        
        self.fig = fig
        self.axes = axes


    def plot_2d_ax(self,ax,predictor, l, X_sample = None, y_sample = None, show_exact: bool = False, is_truth : bool = False, label="", color = "", marker = "", label_samples = ""):
        """
        There are 4 desired axes / modes of plotting:
        1) Normal surfaces (+ exact if possible), (with or without samples)
        2) Normal contours (+ exact if possible), (with or without samples)
        3) Predicted + truth surfaces (+ exact if possible), (with or without samples)
        4) Predicted + truth contours (+ exact if possible), (with or without samples)
        """

        # plot on the inset axis if it is there! Do this by 'recursion'
        if hasattr(ax,'axin'):
            ax.axin.colors = ax.colors
            self.plot_2d_ax(ax.axin,predictor,l,X_sample,y_sample,show_exact,is_truth,color,marker)


        l, label, label_samples, label_exact, color, marker, show_exact, color_exact = self.get_standards(l, label, color, marker, is_truth, show_exact, label_samples)

        " STEP 1: data preparation "
        # retrieve predictions from the provided predictor
        y_hat, mse = predictor.predict(self.X_pred)

        # reshape to X_plot shape
        y_hat = y_hat.reshape(self.X_plot[0].shape)
        std = np.sqrt(mse.reshape(self.X_plot[0].shape))

        # for the non-main lines, lower the resolution for faster plotting
        y_sub = (y_hat - self.std_mult * std)[::self.sub_skip,::self.sub_skip]


        # ADDITIONALLY get (and later plot) the exact level solution simultaneously
        y_exact = 0
        if show_exact and self.plot_exact_possible:
            y_exact, _ = self.solver.solve(self.X_pred)
            y_exact = y_exact.reshape(self.X_plot[0].shape)


        " STEP 2: plotting "
        if ax.name == "3d":
            # if the axis is 3d, we will plot a surface
            alpha_main = 0.35
            alpha_std = 0.05

            fix_colors(ax.plot_surface(*self.X_plot, y_hat, alpha=alpha_main, color=color, label=label))
            ax.plot_surface(*self.X_plot_sub, y_sub, alpha=alpha_std, color=color)
            ax.plot_surface(*self.X_plot_sub, y_sub, alpha=alpha_std , color=color)
            ax.colors.append(color)
            ax.lims.append([np.min(y_hat - self.std_mult * std), np.min(y_hat), np.max(y_hat), np.max(y_hat + self.std_mult * std)])

            if X_sample is not None and y_sample is not None:
                # add sample locations
                ax.scatter(*X_sample[:, self.d_plot].T, y_sample, marker = marker, s = self.s_standard + self.truth_s_increase if is_truth else self.s_standard, color = color, linewidth=2, facecolor = "none" if is_truth else color, label = label_samples) # needs s argument, otherwise smaller in 3d!
                ax.colors.append(color)

            if show_exact and self.plot_exact_possible and l >= self.l_hifi:
                fix_colors(ax.plot_surface(*self.X_plot, y_exact, alpha=0.5, label = label_exact, color = color_exact))
                ax.colors.append(color_exact)
        else:
            # then we are plotting a contour, 2D data (only X, no y)
            # there is no difference between plotting the normal and truth here
            CS = ax.contour(*self.X_plot, y_hat, colors=color) # contour is not animated!
            CS.collections[-1].set_label(label)
            ax.colors.append(color)
            ax.CS = CS # Used for giving only the last contour inline labelling

            if X_sample is not None:
                ax.scatter(*X_sample[:, self.d_plot].T, marker = marker, s = self.s_standard + self.truth_s_increase if is_truth else self.s_standard, color = color, linewidth=2, facecolor = "none" if is_truth else color, label= label_samples)
                ax.colors.append(color)

            if show_exact and self.plot_exact_possible and l >= self.l_hifi:
                CS = ax.contour(*self.X_plot, y_exact, colors=color_exact)
                CS.collections[-1].set_label(label_exact)
                ax.colors.append(color_exact)
                ax.CS = CS # Used for giving only the last contour inline labelling
                

    def plot_2d(self, predictor, l, X_sample = None, y_sample = None, show_exact: bool = False, is_truth : bool = False, label="", color = "", marker = "", label_samples = ""):
        """
        Plotting for surfaces and possibly contours of the predicted function in 2d.
        Exact function values plotted if available.
        @param label: define a custom label for the prediction surface. If given, all other labels will not be used. 
        """      
        for i, ax in enumerate(self.axes):
            if i == 1 or i == 3: # right two plots, here we only want the truth / +exact
                show_exact = self.plot_exact_possible
                if l >= self.l_hifi:
                    self.plot_2d_ax(ax,predictor,l,X_sample,y_sample,show_exact,is_truth,label,color,marker,label_samples)
            else:
                if not is_truth:
                    self.plot_2d_ax(ax,predictor,l,X_sample,y_sample,show_exact,is_truth,label,color,marker,label_samples)

    def plot_kriged_truth_best(self, ax, mf_model):
        X_opt, z_opt = mf_model.K_truth.X_opt, mf_model.K_truth.z_opt

        if hasattr(ax,'axin'):
            self.plot_kriged_truth_best(ax.axin, mf_model)

        l, _,_,_,color,_,_,_ = self.get_standards(mf_model.l_hifi, "", "", "", True, "", "")
        label = "Best prediction truth"
        if self.d == 1:
            ax.scatter(*X_opt[:, self.d_plot].T,z_opt, marker = "X", s = 100, zorder = 10, facecolor = "orange", edgecolors= color, label = label, alpha = 0.7)
            ax.colors.append(color)
        elif not ax.name == "3d":
            ax.scatter(*X_opt[:, self.d_plot].T, marker = "X", s = 100, zorder = 10, facecolor = "orange", edgecolors= color, label = label, alpha = 0.7)
            ax.colors.append(color)

    def plot_kriged_truth(self, mf_model : MultiFidelityKrigingBase):
        """
        Use to plot the fully sampled hifi truth Kriging with the prediction core.ordinary_kriging.
        """
        if self.plot_truth:
            self.plot(mf_model.K_truth, mf_model.l_hifi, mf_model.X_truth, mf_model.Z_truth, is_truth=True) #  color = 'black',
            
            for ax in self.axes:
                self.plot_kriged_truth_best(ax, mf_model)

    def set_zoom_inset(self, axes_nrs, x_rel_range : list,  inset_rel_limits = [[]]):
        """
        axes_nrs (list): the axis numbers on which we want have an (single) inset.
        inset_rel_limits (list of list): each list comprised of the location of the inset relative to the main axis,
            formatted as [x0, y0, width, height] met x0 y0 = 0,0 bottom left
        xlim: give the xlims per ax
        y_zoom_centres: only the y_zoom_centre is given. The y_lims are inferred in the same ratio as the limits.
        """
        for i, ax in enumerate(self.axes):
            if i in axes_nrs and ax.name != '3d':
                j = axes_nrs.index(i)
                # set the location of the inset
                if np.any(inset_rel_limits):
                    limit = inset_rel_limits[j]
                else:
                    if self.d == 1:
                        limit = [0.5, 0.58, 0.3, 0.4]
                    else:
                        # underneath the legend (if it is on the right)
                        limit = [1.1, 0.05, 0.4, 0.4]
                
                # assign the inset axes as a variable to the ax, such that we can access it adhoc
                ax.axin = ax.inset_axes(limit) 
                ax.axin.x_rel_range = x_rel_range[j]
                ax.axin.x_zoom_centre = [0] * self.d # will be set by draw_current_levels as best point
                ax.axin.y_zoom_centre = 0
                ax.axin.limit = limit


    def plot_predicted_points(self, ax, l, X_plot_est, Z_plot_est, label = ""):
        """
        Plot the predicted points.
        """

        if hasattr(ax,"axin"):
            self.plot_predicted_points(ax.axin, l, X_plot_est, Z_plot_est)

        color_predicted = 'black'
        ax.colors.append(color_predicted)
        label_predicted = label if label != "" else "Extrapolated points level {}".format(l)
        marker_predicted = "+"

        # 3D scatter plot
        if ax.name == "3d" or self.d == 1:
            # plot predicted points
            ax.scatter(
                *X_plot_est[:, self.d_plot].T,
                Z_plot_est,
                c=color_predicted,
                marker=marker_predicted,
                label = label_predicted, 
                s = 70,
                zorder = 5,
                animated=False,
            )

        else: # the 2D scatter plot
            # plot predicted points
            ax.scatter(
                *X_plot_est[:, self.d_plot].T,
                c=color_predicted,
                marker=marker_predicted,
                label = label_predicted, 
                s = 70,
                zorder = 5,
                animated=False,
            )

    def plot_best(self, ax, X_s, Z_s, best):
        if hasattr(ax,"axin"):
            self.plot_best(ax.axin, X_s, Z_s, best)
            ax.axin.x_zoom_centre = X_s[best, self.d_plot][0]
            if self.d == 1:
                ax.axin.y_zoom_centre = Z_s[best]
            else:
                ax.axin.y_zoom_centre = X_s[best, self.d_plot][1]

        color_best = 'black'
        ax.colors.append(color_best)
        
        if ax.name == "3d" or self.d == 1:
            # plot best point
            ax.scatter(*X_s[best, self.d_plot].T, Z_s[best], s = 300, marker = "*", color = color_best, zorder = 6, facecolor="none", label="Current best sample")
        else:
            # plot best point
            ax.scatter(*X_s[best, self.d_plot].T, s = 300, marker = "*", color = color_best, zorder = 6, facecolor="none", label="Current best sample")
    
    def plot_optima(self, ax, mf_model):
        if hasattr(ax,'axin'):
            self.plot_optima(ax.axin, mf_model)

        X_opt, z_opt = mf_model.solver.get_optima()
        color_opt = "y"
        label_opt = "Analytical optima" if X_opt.shape[0] > 1 else "Analytical optimum"
        if self.d == 1:
            ax.scatter(*X_opt[:, self.d_plot].T,z_opt, marker = "x", s = 100, zorder = 10, c = color_opt, label = label_opt)
            ax.colors.append(color_opt)
        elif not ax.name == "3d":
            ax.scatter(*X_opt[:, self.d_plot].T, marker = "x", s = 100, zorder = 10, c = color_opt, label = label_opt)
            ax.colors.append(color_opt)


    def draw_current_levels(self, mf_model : MultiFidelityKrigingBase, K_mf_extra = None):
        """
        @param X: !! 3D array, different in format, it is now an array of multiple X in the standard format
                    for the last level, this contains only the sampled locations.
        @param Z: list of the sampled z (or y) values; these can be different than the predictions when noise is involved.
        @param K_mf: list of the Kriging predictors of each level
        @param X_unique: unique X present in X_l. At these locations we estimate our prediction points.
        """
        if self.counter % self.plot_once_every == 0:
            # check correlations and RMSE levels
            RMSE = RMSE_norm_MF(mf_model, no_samples=True)
            mf_model.print_stats(RMSE)

            t_start = time.time()

            X_mf = mf_model.X_mf # contains the hifi sampled level too
            Z_mf = mf_model.Z_mf # contains the hifi sampled level too
            K_mf = mf_model.K_mf
            L = mf_model.L
            self.l_hifi = mf_model.l_hifi

            # reset axes for live plotting purposes
            axes = self.axes
            for ax in axes:
                if hasattr(ax,'axin'):
                    x_rel_range = ax.axin.x_rel_range
                    x_zoom_centre = ax.axin.x_zoom_centre
                    y_zoom_centre = ax.axin.y_zoom_centre
                    limit = ax.axin.limit

                    ax.clear()

                    ax.axin = ax.inset_axes(limit) 
                    ax.axin.colors =[]
                    ax.axin.x_rel_range = x_rel_range
                    ax.axin.x_zoom_centre = x_zoom_centre
                    ax.axin.y_zoom_centre = y_zoom_centre
                    ax.axin.limit = limit
                else:
                    ax.clear()
                ax.colors =[]
            
            if self.plot_NRMSE_text:
                ax = self.axes[-1]
                xlims = ax.get_xlim()
                ax.text(0.05, 0.9, f"NRMSE = {round(RMSE[-1],2)}%",
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=ax.transAxes,
                    fontsize = 'large', zorder = 10)
                ax.set_xlim(xlims)

            " plot known kriging levels"
            has_prediction = isinstance(mf_model,ProposedMultiFidelityKriging) and mf_model.l_hifi + 1 == mf_model.number_of_levels
            for l in range(mf_model.number_of_levels - 1 if has_prediction else mf_model.number_of_levels):
                if mf_model.d == 1 or not has_prediction: # for 1d the added points is not too confusing
                    self.plot(K_mf[l], l, X_mf[l], Z_mf[l]) 
                else:
                    self.plot(K_mf[l], l)

            " plot prediction kriging "
            if has_prediction:
                l = self.l_hifi
                
                if K_mf_extra is None: # very crowded plot otherwise!
                    # prediction line, this is re-interpolated if we used noise and might not cross the sample points exactly
                    # includes sample points, but not the predicted points!
                    self.plot(K_mf[l], l, X_mf[l], Z_mf[l], label="Extrapolated level {}".format(l))
                    
                    # set up to plot our prediction`s` estimated part of points seperately in black
                    X_plot_est = mf_model.return_unique_exc(X_exclude=X_mf[-1])
                    Z_plot_est = K_mf[l].predict(self.transform_X(X_plot_est))[0]

                    # plot the methods prediction points
                    for ax in self.axes:
                        self.plot_predicted_points(ax, l, X_plot_est, Z_plot_est)
                else:
                    self.plot(K_mf[l], l, label="Extrapolated level {}".format(l))

            " plot MFK if available, for 1d only"
            if hasattr(mf_model,'try_use_MFK') and mf_model.try_use_MFK and mf_model.trained_MFK and self.d == 1:
                self.plot(mf_model, len(self.colors) - 1, show_exact = False, label="MFK of Le Gratiet (2014)")


            " plot best point "
            # set up to plot best out of all the *sampled* hifi locations
            # NOTE not per definition the best location of all previous levels too
            ind_best = mf_model.get_best_sample(arg=True)

            for ax in self.axes:
                self.plot_best(ax, X_mf[-1], Z_mf[-1], ind_best)

            " plot 'full' Kriging level in case of linearity check"
            # Is per definition a prediction!
            if K_mf_extra != None:
                l = mf_model.number_of_levels + 1
                self.plot(K_mf_extra, l, K_mf_extra.X_s, K_mf_extra.Z_s, label=K_mf_extra.name, label_samples="Samples "+K_mf_extra.name)

                X_plot_est = mf_model.return_unique_exc(X_exclude=K_mf_extra.X_s)
                Z_plot_est = K_mf_extra.predict(self.transform_X(X_plot_est))[0]
                
                # best should stay like this due to how linearity_check is written (this is an OK instance, but .y contains all predicted points too)
                best = np.argmin(K_mf_extra.Z_s) 
                
                for ax in self.axes:
                    self.plot_predicted_points(ax, l, X_plot_est, Z_plot_est, label="Extrapolated points of\n"+K_mf_extra.name)

            if hasattr(mf_model, "K_truth"):
                self.plot_kriged_truth(mf_model)

            " plot optima if known "
            if self.plot_exact_possible:
                for ax in self.axes:
                    self.plot_optima(ax, mf_model)

            self.set_axis_props(mf_model)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events() 
            plt.show(block=False)

            if self.make_video:
                # image_from_plot = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                # image_from_plot = image_from_plot.reshape(tuple(int(i*self.dpi_scale) for i in self.fig.canvas.get_width_height()[::-1]) + (3,))
                # self.frames.append(image_from_plot)
                if self.save_svg:
                    path = os.path.join(self.video_path, 'image_{}.svg'.format(self.counter))
                    if self.d == 1 and self.fig._suptitle.get_text() == "":
                        w, h = self.fig.get_size_inches()
                        bbox = Bbox([[0,0],[w,h - 0.5]])
                        self.fig.savefig(path, bbox_inches=bbox)
                    else:
                        self.fig.savefig(path, bbox_inches='tight')
                
                path = os.path.join(self.video_path, 'image_{}.png'.format(self.counter))
                # self.fig.savefig(path)
                # img = Image.fromarray(np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(tuple(int(i*self.fig.canvas._dpi_ratio) for i in self.fig.canvas.get_width_height()[::-1]) + (3,)))
                if self.d == 1 and self.fig._suptitle.get_text() == "":
                    w, h = self.fig.get_size_inches()
                    bbox = Bbox([[0,0],[w,h - 0.5]])
                    self.fig.savefig(path, bbox_inches=bbox)
                else:
                    w, h = self.fig.canvas.get_width_height()
                    try:
                        img = Image.fromarray(np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((int(h*self.fig.canvas._dpi_ratio),int(w*self.fig.canvas._dpi_ratio),3)))
                    except:
                        try:
                            img = Image.fromarray(np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((int(h*self.fig.canvas._dpi_ratio) + 1,int(w*self.fig.canvas._dpi_ratio),3)))
                        except:
                            img = Image.fromarray(np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((int(h*self.fig.canvas._dpi_ratio),int(w*self.fig.canvas._dpi_ratio) + 1,3)))
                    img.save(path)
                    # .savefig('my_plot.png')
                    # self.frames.append(self.fig.canvas.tostring_rgb())
            
            print("##############################")
            print("### Plotting took {:.4f} s ###".format(time.time() - t_start))
            print("##############################")
            ### Plotting took 2.8344 s ###

        self.counter += 1


    ###########################################################################
    ## Formatting                                                           ###
    ###########################################################################

    def set_axis_props(self, mf_model):
        """ set axes properties like color and legend """
        
        if self.show_legend:
            for ax_nr, ax in enumerate(self.axes):
                is_line_or_surface = [] # encodes for items that get flushed up the legend
                is_line_or_surface_colors = []
                is_other = []
                counter = 0

                # !! deze hele klotezooi is nodig om de kleuren te corrigeren, locatie is eigenlijk sidebusiness
                leg = ax.legend() # create a naive legend
                handles, labels = ax.get_legend_handles_labels()

                if mf_model.d == 1:
                    for i, lh in enumerate(leg.legendHandles):
                        if isinstance(lh,mpl.patches.Rectangle) or isinstance(lh,mpl.lines.Line2D):# or 'level 2' in lh._label:
                            is_line_or_surface.append(i)
                            is_line_or_surface_colors.append(lh._color)
                        else:
                            # still if the label color is exactly the same as of line or surface, we want to add it after!
                            loc = -1
                            for jj, j in enumerate(is_line_or_surface_colors):
                                if lh._original_edgecolor == j:
                                    loc = jj + 1 + counter
                                    break
                            if loc >= 0:
                                is_line_or_surface.insert(loc,i)
                                counter += 1
                            else:
                                is_other.append(i)
                else: 
                    for i, lh in enumerate(leg.legendHandles):
                        if isinstance(lh,mpl.patches.Rectangle) or isinstance(lh,mpl.lines.Line2D) or 'level 2' in lh._label:
                            is_line_or_surface.append(i)
                        else:
                            if len(is_line_or_surface) > 0 and ax.colors[i]==ax.colors[is_line_or_surface[-1]]:# and not 'best' in lh._label: #NOTE als best sample aan het eind moet
                                is_line_or_surface.append(i)
                            else:
                                is_other.append(i)            

                # reorder, surfaces / lines flushed forwards! 
                order = is_line_or_surface + is_other

                # changing the order of the last four entries (exact, predicted, best, optima)
                if hasattr(mf_model,'try_use_MFK') and mf_model.try_use_MFK and mf_model.trained_MFK and self.d == 1:
                    # then we plot MFK too, use last 5
                    order_sub = order[-5:]
                    del order[-5:]
                    order += order_sub[2:4] + [order_sub[1]] + [order_sub[0]] + [order_sub[-1]]
                elif self.d == 1:
                    order_sub = order[-4:]
                    del order[-4:]
                    order_pred = order_sub[1:3]
                    del order_sub[1:3]
                    order += order_pred + order_sub

                if self.d == 2:
                    pass

                if (mf_model.d != 1) and ax_nr == 1 or ax_nr == 3:
                    loc = 'upper left'
                    bb = (1.08, 1.0)
                else:
                    if mf_model.d == 1 and ax_nr == 0:
                        bb = (-0.08, 0.49)
                        loc = 'center right'
                    else:
                        bb = (-0.08, 1.0)
                        loc = 'upper right'
                
                extra = []
                extra_text = []
                if ax_nr == 0 or ax_nr == 1: # works for both 1d as 2d plotting
                    extra_text = [u"\nVariance displayed with\n\u00B1 {} standard deviations".format(self.std_mult)]
                    extra = [Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)]

                leg = ax.legend([handles[idx] for idx in order] + extra,[labels[idx] for idx in order] + extra_text,loc = loc, bbox_to_anchor=bb)

                # now reset the color and alpha! (calling ax.legend resets this)
                # NOTE this is only fucked for 3d surface plots
                if ax.name == '3d':
                    ax.set_zlabel("Z")
                    for i in range(len(order)):
                        lh = leg.legendHandles[i]
                        without_facecolor = False
                        try:
                            if not any(lh.get_facecolor()):
                                without_facecolor = True
                        except:
                            pass

                        if not without_facecolor:
                            lh.set_alpha(1)
                            lh.set_color(ax.colors[order[i]])
                    for t in leg.texts:
                        t.set_alpha(1)
                else:
                    try:
                        ax.clabel(ax.CS, inline=1, fontsize=10)
                    except:
                        pass

        for ax in self.axes:
            ax.set_xlabel(self.axis_labels[0])
            ax.set_ylabel('y')
 
            if mf_model.d > 1:
                ax.set_ylabel(self.axis_labels[1])

            # reset x and y lim. For axin outside the plot, the bounds get fucked each timestep
            ax.set_xlim(self.lb[self.d_plot[0]],self.ub[self.d_plot[0]])

            # formatting for inset axis!
            if hasattr(ax,'axin'):

                # sub region of the original image to display
                # we basically set different limits on the original data
                # forrester moet ongeveer gefocust rond x = 0.62, y = -1.3

                height = ax.get_ylim()[1] - ax.get_ylim()[0] # is not set yet!!
                width = ax.get_xlim()[1] - ax.get_xlim()[0]
                rel_width = ax.axin.limit[2]
                rel_height = ax.axin.limit[3]
                
                # dx = ax.axin.get_xlim()[1] - ax.axin.get_xlim()[0]
                dx = width * ax.axin.x_rel_range
                dy = dx * height/width * rel_height/rel_width
                
                ax.axin.set_xlim(ax.axin.x_zoom_centre - dx / 2, ax.axin.x_zoom_centre + dx / 2)
                ax.axin.set_ylim(ax.axin.y_zoom_centre - dy / 2, ax.axin.y_zoom_centre + dy / 2)
        
                # no tick labels
                ax.axin.set_xticklabels([])
                ax.axin.set_yticklabels([])    

                # set the connecting lines and boxes
                ax.indicate_inset_zoom(ax.axin)#, edgecolor="black")

        " Setting / sharing the z-axis ranges for the 3d plots "
        # first limit creating
        # goal is to make the main line still vieable, even if there are large stds involved (in which we are not always interested)
        lim = []
        mult = 2
        for ax in self.axes:
            if ax.name == '3d':
                # one lim per lineset (main +- std)
                lims = np.array(ax.lims)
                lims_extr = np.min(lims[:,:2],axis = 0)
                lims_extr = np.append(lims, np.max(np.array(ax.lims)[:,2:],axis = 0))

                # last lim of either prediction, truth, or even exact (last lim presumed to be most precise)
                range_last_main = lims[-1,2] - lims[-1,1]
                range_last_std = lims[-1,3] - lims[-1,0]
                range_last_avg = (range_last_main + range_last_std) / 2

                # do not take more than 2x (mult) the average (main and +- std) of the last data range
                # but, always show the full range_last_std
                if lims_extr[-1] - lims_extr[0] <= mult * range_last_avg:
                    lim.append([lims_extr[0], lims_extr[-1]])
                else:
                    lim.append([lims[-1,0], lims[-1,-1]])

        if np.any(lim):
            zlim = np.array([np.min(lim, axis = 0)[0], np.max(lim, axis = 0)[1]])

            # take 5% extra as in matplotlib standards
            zlim = zlim * 1.05

            # limit setting 
            for ax in self.axes:
                if ax.name == '3d':
                    ax.set_zlim(zlim)
        
        if self.d == 1 and self.plot_double_axes:
            maxx = -1000000000
            minn = 1000000000
            for line in self.axes[0].get_lines():
                y = line.get_ydata()
                minn = np.min([np.min(y), minn])
                maxx = np.max([np.max(y), maxx])
            lim0 = self.axes[0].get_ylim()
            lim1 = self.axes[1].get_ylim()
            self.axes[0].set_ylim([np.min([lim1[0], minn * 1.05]),np.max([lim1[1], maxx * 1.05])]) 


        if not self.tight:
            self.fig.tight_layout() 
            self.tight = True


    def render_video(self):
        if self.make_video:
            print("Creating video of plots", end = '\r')
            
            img_paths_list = sorted(glob.glob(self.video_path + os.path.sep + "*.png"), key=os.path.getmtime)
            start = time.time()
            # with imageio.get_writer(os.path.join(self.video_path,"movie.gif"), mode="I", duration=0.5) as writer:
            with imageio.get_writer(os.path.join(self.video_path,"movie.mp4"), mode="I", fps = 2) as writer:
                for img_path in img_paths_list:
                    # img = Image.open(io.BytesIO(byte_img))
                    # img.save(os.path.join(self.video_path,"test.png"))

                    # byte_img = svg2png(url = svg_path)
                    # img = imageio.imread(io.BytesIO(byte_img))
                    
                    img = imageio.imread(img_path)
                    writer.append_data(img)

                # repeat the last one to better indicate ending
                for i in range(5):
                    writer.append_data(img) # type:ignore
            print(f"Gif creation time: {time.time() - start:.4f}")

            # copy the end-state input file
            shutil.copy(self.setup_file_path, self.video_path)


            
            
