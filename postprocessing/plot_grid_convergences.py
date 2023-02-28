if not 'EVA' in setup.solver_str: # type:ignore
    print("This plotting solution is EVA / containing a timetrace specific.")
    sys.exit()

for l in range(nlvl):
    solver = get_solver(setup)
    z, _, tt = solver.solve(mf_model.X_truth,mf_model.L[l], get_time_trace = True)
    Z.append(z)
    TT.append(tt)















import numpy as np
import matplotlib.pyplot as plt
import os


def plot_grid_convergence(model, Z_list):
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