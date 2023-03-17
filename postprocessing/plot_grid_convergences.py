# pyright: reportGeneralTypeIssues=false, reportUnboundVariable = false
import numpy as np

import matplotlib.pyplot as plt

import sys
import os
from utils.formatting_utils import correct_formatX

from core.mfk.mfk_base import MultiFidelityKrigingBase

def plot_grid_convergence(mf_model : MultiFidelityKrigingBase):
    if not 'EVA' in mf_model.solver.name: # type:ignore
        print("This plotting solution is EVA / containing a timetrace specific.")
        sys.exit()

    " Prepare plotting grid "
    n_plots_x = int(np.ceil(np.sqrt(len(mf_model.X_truth))))
    n_plots_y = int(np.ceil(len(mf_model.X_truth) / n_plots_x))

    # sort the X and Z list for 'grid-like' plotting
    # sort on increasing y (x1), then on increasing x (x0) within that sublist
    inds = np.argsort(-mf_model.X_truth[:, 1]) 
    for j in range(n_plots_y):
        sl = range(j * n_plots_x, j * n_plots_x + n_plots_x)
        ind = np.argsort(mf_model.X_truth[inds[sl], 0]) 
        inds[sl] = inds[ind + j * n_plots_x]

    # resort the lists
    X_plot_list = [mf_model.X_truth[i] for i in inds]

    " Setting up the plot "
    # sharex and sharey for easy visual comparison
    fig, axes = plt.subplots(n_plots_y, n_plots_x, constrained_layout=False, sharex=True, sharey=True, figsize = (11.29 * 100/80,8.27 * 100/80)) # A4 sized!
    axes_twin = np.array([[ax.twiny() for ax in axlist] for axlist in axes])

    # create space for suptitle, secondary axis label and legend on top
    fig.subplots_adjust(
        top=0.85,
        bottom=0.06,
        left=0.08,
        right=0.99,
        hspace=0.2,
        wspace=0.08)
    fig.suptitle("Convergence plots", x = 0.51)

    # remove the xticks labels, keep of upper row
    for axlist in axes_twin[1:]:
        for ax in axlist:
            ax.xaxis.set_ticklabels([])
        
    # set upper row labels to the values of L
    for ax in axes_twin[0]:
        ax.set_xticks([0,1,2])
        ax.xaxis.set_ticklabels(mf_model.L)


    " Plot the data "
    nlvl = len(mf_model.L)
    ZZ = []
    for l in range(nlvl):
        Z, _, TT = mf_model.solver.solve(correct_formatX(X_plot_list, mf_model.d), mf_model.L[l], get_time_trace = True)
        ZZ.append(Z)
        for i, x in enumerate(X_plot_list):
            ax = axes[int(np.floor(i / n_plots_x)), i % n_plots_x]  
            
            # solve per level and add to plot directly
            ax.plot(TT[i][:,0], TT[i][:,1], label = "Refinement level = {}".format(mf_model.L[l]))


    ### secondary twin axes operations
    ZZ = np.array(ZZ)
    for i, x in enumerate(X_plot_list):
        # plot the convergence on the twin-y axis
        ax_twin = axes_twin[int(np.floor(i / n_plots_x)), i % n_plots_x]  
        ax_twin.plot(np.arange(nlvl), ZZ[:,i], c="black", label = "Objective value (max Fy)")


    " Plot the NURBS and axes set titles "
    for i, x in enumerate(X_plot_list):
        ax = axes[int(np.floor(i / n_plots_x)), i % n_plots_x]  
        with np.printoptions(precision=4, suppress=True):
            ax.set_title("x={}".format(x), fontsize = 11)

        # get the background bounds when all bounds are known.
        xb = ax.get_xbound()
        yb = ax.get_ybound()

        # select the background image
        path = os.path.join(mf_model.solver.get_output_path(x, mf_model.L[-1])[0], "NURBS.png")
        img = plt.imread(path)

        # show background image
        ax.imshow(
            img,
            # aspect=img.shape[1]/img.shape[0],
            aspect = "auto",
            interpolation="nearest",
            alpha=0.5,
            extent=[xb[0], xb[1], yb[0], yb[1]],
        )


    " Make legend "
    over_t_text = "Plotted over simulation time"
    over_L_text = "Plotted over fidelity refinement factor"
    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = ax_twin.get_legend_handles_labels()

    fig.text(0.51, 0.01, 'Time [s]', ha='center')
    fig.text(0.51, 0.90, 'Fidelity level L [-]', ha='center') # NOTE hardcoded!
    fig.text(0.01, 0.5, 'Body force in y-direction [N/m]', va='center', rotation='vertical')

    legend1  = fig.legend(handles, labels, loc='upper left', title = over_t_text)
    fig.add_artist(legend1)
    fig.legend(h2, l2, loc='upper right', title = over_L_text)

    for i, ax in enumerate(axes.flat):
        ax.set_zorder(axes_twin.flat[i].get_zorder()+1)
        ax.set_frame_on(False)

def plot_grid_convergence_Z(model, Z_list):
    """
    This function builds a list over levels L for each coordinate x in X list and plots them in a grid.
    This function only works properly for fully nested DoE`s.
    """

    # Initiate lists of x-y plotting date for each point x in the lowest DoE
    X_plot_list = np.array([model.L for i in range(model.X_truth.shape[0])])  # levels L will be the x-axis
    Z_plot_list = np.array(Z_list).T # transverse the list such that each x now on the rows

    n_plots_x = int(np.ceil(np.sqrt(len(X_plot_list))))
    n_plots_y = int(np.ceil(len(X_plot_list) / n_plots_x))

    # sort the X and Z list for 'grid-like' plotting
    # sort on increasing y (x1), then on increasing x (x0) within that sublist
    inds = np.argsort(-model.X_truth[:, 1])
    for j in range(n_plots_y):
        sl = range(j * n_plots_x, j * n_plots_x + n_plots_x)
        ind = np.argsort(model.X_truth[inds[sl], 0])
        inds[sl] = inds[ind + j * n_plots_x]

    # resort the lists
    X_plot_list = [X_plot_list[i] for i in inds]
    Z_plot_list = [Z_plot_list[i] for i in inds]

    # sharex and sharey for easy visual comparison
    fig, axes = plt.subplots(n_plots_y, n_plots_x, sharex=True, sharey=True)
    fig.suptitle("Convergence plots")

    for i in range(len(X_plot_list)):
        ax = axes[int(np.floor(i / n_plots_x)), i % n_plots_x]  
        x = model.X_truth[inds[i]]
        ax.set_title("x={}".format(x))
        ax.plot(X_plot_list[i], Z_plot_list[i], c="black")

    # set the background when all y-bounds of the rows are known.
    for i in range(model.X_truth.shape[0]):
        ax = axes[int(np.floor(i / n_plots_x)), i % n_plots_x]  
        x = model.X_truth[inds[i]]
        xb = ax.get_xbound()
        yb = ax.get_ybound()

        # select highest level for background image
        path = os.path.join(model.solver.get_output_path(x, model.L[-1])[0], "NURBS.png")
        img = plt.imread(path)

        # show background image
        ax.imshow(
            img,
            aspect="auto",
            interpolation="nearest",
            alpha=0.5,
            extent=[xb[0], xb[1], yb[0], yb[1]],
        )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left')
    fig.tight_layout()


def plot_grid_convergence_tt(model, TT):
    """
    This function for each coordinate x in X plots the full timetraces in a grid, e.g. multiple lines per subplot.
    This function only works properly for fully nested DoE`s.
    @param TT: list per level L of numpy array per x with columns (times_array, value_array)
    """

    # Initiate lists of x-y plotting date for each point x in the lowest DoE
    X_plot_list = [[] for t in range(model.X_truth.shape[0])]
    Z_plot_list = [[] for z in range(model.X_truth.shape[0])]

    # initialize the lowest level of the doe
    for i, tl in enumerate(TT[0]):
        X_plot_list[i].append(tl[:,0])
        Z_plot_list[i].append(tl[:,1])

    # This only makes sense for fully nested sample locations over l
    # for each level present,
    for l in range(1, len(model.L)):
        # find the location of the lower DoE corresponding index and add
        for i in range(model.X_truth.shape[0]):
            X_plot_list[i].append(TT[l][i][:, 0])
            Z_plot_list[i].append(TT[l][i][:, 1])

    n_plots_x = int(np.ceil(np.sqrt(len(X_plot_list))))
    n_plots_y = int(np.ceil(len(X_plot_list) / n_plots_x))

    # sort the X and TT for 'grid-like' plotting
    # sort on increasing y (x1), then on increasing x (x0) within that sublist
    inds = np.argsort(-model.X_truth[:, 1])
    for j in range(n_plots_y):
        sl = range(j * n_plots_x, j * n_plots_x + n_plots_x)
        ind = np.argsort(model.X_truth[inds[sl], 0])
        inds[sl] = inds[ind + j * n_plots_x]

    X_plot_list = [X_plot_list[i] for i in inds]
    Z_plot_list = [Z_plot_list[i] for i in inds]

    fig, axes = plt.subplots(n_plots_y, n_plots_x, sharex=True, sharey=True)
    fig.suptitle("Convergence plots")

    for i in range(model.X_truth.shape[0]):
        ax = axes[int(np.floor(i / n_plots_x)), i % n_plots_x]  
        x = model.X_truth[inds[i]]
        ax.set_title("x={}".format(x))
        for l in range(len(model.L)): 
            ax.plot(X_plot_list[i][l], Z_plot_list[i][l], label = "Refinement level = {}".format(model.L[l]))

    # set the background when all y-bounds of the rows are known.
    for i in range(len(X_plot_list)):
        ax = axes[int(np.floor(i / n_plots_x)), i % n_plots_x]  
        x = model.X_truth[inds[i]]
        xb = ax.get_xbound()
        yb = ax.get_ybound()
        path = os.path.join(model.solver.get_output_path(x, model.L[-1])[0], "NURBS.png")
        img = plt.imread(path)
        ax.imshow(
            img,
            aspect="auto",
            interpolation="nearest",
            alpha=0.5,
            extent=[xb[0], xb[1], yb[0], yb[1]],
        )
    handles, labels = ax.get_legend_handles_labels()  
    fig.legend(handles, labels, loc='upper left')
    fig.tight_layout()