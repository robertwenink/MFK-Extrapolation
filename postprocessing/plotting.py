import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def plot2d(X, y, X_new, y_hat, mse):
    # TODO change to the d-th root sort out of the works for all dimensions as well
    L = int(math.sqrt(len(X[:, 0])))
    P1 = X[:, 0].reshape(L, L)
    P2 = X[:, 1].reshape(L, L)

    L = int(math.sqrt(len(X_new[:, 0])))
    P1_new = X_new[:, 0].reshape(L, L)
    P2_new = X_new[:, 1].reshape(L, L)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    Z = y.reshape(P1.shape)
    ax.plot_surface(P1, P2, Z, alpha=0.5)

    Z_hat = y_hat.reshape(P1_new.shape)
    std = np.sqrt(mse).reshape(P1_new.shape)
    ax.plot_surface(P1_new, P2_new, Z_hat, alpha=0.5)
    ax.plot_surface(P1_new, P2_new, Z_hat-std, alpha=0.5)
    ax.plot_surface(P1_new, P2_new, Z_hat+std, alpha=0.5)

    ax.set_title("{}".format(__name__))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
