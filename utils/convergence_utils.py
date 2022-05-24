import numpy as np
import matplotlib.pyplot as plt
import os


def plot_grid_convergence(X_list, Z_list, L, solver):
    """
    This function builds a list over levels L for each coordinate x in X list and plots them in a grid.
    This function only works properly for fully nested DoE`s.
    """

    # Initiate lists of x-y plotting date for each point x in the lowest DoE
    X_plot_list = [[] for i in range(X_list[0].shape[0])]  # levels L will be the x-axis
    Z_plot_list = [[] for z in range(X_list[0].shape[0])]  # object z will be the y-axis

    # initialize the lowest level of the doe
    for i, z in enumerate(Z_list[0]):
        X_plot_list[i].append(L[0])
        Z_plot_list[i].append(z)

    # This only makes sense for fully nested sample locations over l
    # for each level present,
    for l in range(1, len(X_list)):
        # find the location of the lower DoE corresponding index and add
        w = np.where(np.all(X_list[l] == X_list[l - 1], axis=1))[0]
        for i in w:
            X_plot_list[i].append(L[l])
            Z_plot_list[i].append(Z_list[l][i])

    n_plots_x = int(np.ceil(np.sqrt(len(X_plot_list))))
    n_plots_y = int(np.ceil(len(X_plot_list) / n_plots_x))

    # sort the X and Z list for 'grid-like' plotting
    # sort on increasing y (x1), then on increasing x (x0) within that sublist
    inds = np.argsort(-X_list[0][:, 1])
    for j in range(n_plots_y):
        sl = range(j * n_plots_x, j * n_plots_x + n_plots_x)
        ind = np.argsort(X_list[0][inds[sl], 0])
        inds[sl] = inds[ind + j * n_plots_x]

    X_plot_list = [X_plot_list[i] for i in inds]
    Z_plot_list = [Z_plot_list[i] for i in inds]

    fig, axes = plt.subplots(n_plots_y, n_plots_x, sharex=True, sharey="row")
    fig.suptitle("Convergence plots")

    for i in range(len(X_plot_list)):
        ax = axes[int(np.floor(i / n_plots_x)), i % n_plots_x]
        x = X_list[0][inds[i]]
        ax.set_title("x={}".format(x))
        ax.plot(X_plot_list[i], Z_plot_list[i], c="black")

    # set the background when all y-bounds of the rows are known.
    for i in range(len(X_plot_list)):
        ax = axes[int(np.floor(i / n_plots_x)), i % n_plots_x]
        x = X_list[0][inds[i]]
        xb = ax.get_xbound()
        yb = ax.get_ybound()
        path = os.path.join(solver.get_output_path(x, L[-1])[0], "NURBS.png")
        img = plt.imread(path)
        ax.imshow(
            img,
            aspect="auto",
            interpolation="nearest",
            alpha=0.5,
            extent=[xb[0], xb[1], yb[0], yb[1]],
        )

    fig.tight_layout()


def plot_grid_convergence_tt(X_list, TT, L, solver):
    """
    This function for each coordinate x in X plots the full timetraces in a grid, e.g. multiple lines per subplot.
    This function only works properly for fully nested DoE`s.
    @param TT: list per level L of numpy array per x with columns (times_array, value_array)
    """

    # Initiate lists of x-y plotting date for each point x in the lowest DoE
    X_plot_list = [[] for t in range(X_list[0].shape[0])]  # levels L will be the x-axis
    Z_plot_list = [[] for z in range(X_list[0].shape[0])]

    # initialize the lowest level of the doe
    for i, tl in enumerate(TT[0]):
        X_plot_list[i].append(tl[:,0])
        Z_plot_list[i].append(tl[:,1])

    # This only makes sense for fully nested sample locations over l
    # for each level present,
    for l in range(1, len(X_list)):
        # find the location of the lower DoE corresponding index and add
        w = np.where(np.all(X_list[l] == X_list[l - 1], axis=1))[0]
        for i in w:
            X_plot_list[i].append(TT[l][i][:, 0])
            Z_plot_list[i].append(TT[l][i][:, 1])

    n_plots_x = int(np.ceil(np.sqrt(len(X_plot_list))))
    n_plots_y = int(np.ceil(len(X_plot_list) / n_plots_x))

    # sort the X and TT for 'grid-like' plotting
    # sort on increasing y (x1), then on increasing x (x0) within that sublist
    inds = np.argsort(-X_list[0][:, 1])
    for j in range(n_plots_y):
        sl = range(j * n_plots_x, j * n_plots_x + n_plots_x)
        ind = np.argsort(X_list[0][inds[sl], 0])
        inds[sl] = inds[ind + j * n_plots_x]

    X_plot_list = [X_plot_list[i] for i in inds]
    Z_plot_list = [Z_plot_list[i] for i in inds]

    fig, axes = plt.subplots(n_plots_y, n_plots_x, sharex=True, sharey="row")
    fig.suptitle("Convergence plots")

    for i in range(len(X_plot_list)):
        ax = axes[int(np.floor(i / n_plots_x)), i % n_plots_x]
        x = X_list[0][inds[i]]
        ax.set_title("x={}".format(x))
        for l in range( len(X_list)): 
            ax.plot(X_plot_list[i][l], Z_plot_list[i][l], label = "L = {}".format(L[l]))

    # set the background when all y-bounds of the rows are known.
    for i in range(len(X_plot_list)):
        ax = axes[int(np.floor(i / n_plots_x)), i % n_plots_x]
        x = X_list[0][inds[i]]
        xb = ax.get_xbound()
        yb = ax.get_ybound()
        path = os.path.join(solver.get_output_path(x, L[-1])[0], "NURBS.png")
        img = plt.imread(path)
        ax.imshow(
            img,
            aspect="auto",
            interpolation="nearest",
            alpha=0.5,
            extent=[xb[0], xb[1], yb[0], yb[1]],
        )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')
    fig.tight_layout()