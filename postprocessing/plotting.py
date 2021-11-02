import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def plot_kriging(setup, X, y, X_new, y_hat, mse):
    if setup.live_plot:
        if setup.d == 1 or len(setup.d_plot) == 1:
            plot_1d(setup, X, y, X_new, y_hat, mse)
        else:
            plot_2d(setup, X, y, X_new, y_hat, mse)


def plot_2d(setup, X, y, X_new, y_hat, mse):
    d_to_plot0 = setup.d_plot[0]
    d_to_plot1 = setup.d_plot[1]
    d = setup.d

    L = int(round(np.power(X.shape[0], 1 / d)))
    P1 = X[:: L ** (d - 2), d_to_plot0].reshape(-1,L)
    P2 = X[:: L ** (d - 2), d_to_plot1].reshape(-1,L)
    Z = y[:: L ** (d - 2)].reshape(-1,L)

    L = int(round(np.power(X_new.shape[0], 1 / d)))
    P1_new = X_new[:: L ** (d - 2), d_to_plot0].reshape(-1,L)
    P2_new = X_new[:: L ** (d - 2), d_to_plot1].reshape(-1,L)
    Z_hat = y_hat[:: L ** (d - 2)].reshape(-1,L)
    std = np.sqrt(mse[:: L ** (d - 2)]).reshape(-1,L)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(P1, P2, Z, alpha=0.5)

    ax.plot_surface(P1_new, P2_new, Z_hat, alpha=0.5)
    ax.plot_surface(P1_new, P2_new, Z_hat - std, alpha=0.5)
    ax.plot_surface(P1_new, P2_new, Z_hat + std, alpha=0.5)

    ax.set_title("{}".format(setup.solver_str))
    ax.set_xlabel(setup.search_space[0][d_to_plot0])
    ax.set_ylabel(setup.search_space[0][d_to_plot1])
    ax.set_zlabel("Z")


def plot_1d(setup, X, y, X_new, y_hat, mse):
    L = int(np.power(X.shape[0], 1 / setup.d))
    d_to_plot = setup.d_plot[0]
    P1 = X[:, d_to_plot].reshape(-1,L)

    L = int(math.sqrt(len(X_new[:, 0])))
    P1_new = X_new[:, d_to_plot].reshape(-1,L)

    fig = plt.figure()
    ax = fig.add_subplot()

    Z = y.reshape(P1.shape)
    ax.plot(P1, Z, alpha=0.5)

    Z_hat = y_hat.reshape(P1_new.shape)
    std = np.sqrt(mse).reshape(P1_new.shape)
    ax.plot(P1_new, Z_hat, alpha=0.5)
    ax.plot(P1_new, Z_hat - std, alpha=0.5)
    ax.plot(P1_new, Z_hat + std, alpha=0.5)

    ax.set_title("{}".format(setup.solver_str))
    ax.set_xlabel(setup.search_space[0][d_to_plot])
    ax.set_ylabel("Z")
