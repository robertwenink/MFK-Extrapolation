import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


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
    X = np.ones((len(Xx),d))*(lb+(ub-lb)/2)
    X[:,d_plot] = Xx
    return X

def plot_2d(setup, X, y, predictor):
    d_plot = setup.d_plot
    d = setup.d
    n_per_d = setup.n_per_d

    fig = plt.figure()
    
    X_new = grid_plot(setup)
    y_hat, mse = predictor.predict(X_new)
    
    X_predict = X_new[:,d_plot].reshape(n_per_d,n_per_d,-1)
    y_hat = y_hat.reshape(n_per_d,n_per_d)
    std = np.sqrt(mse.reshape(n_per_d,n_per_d))
    
    if setup.type_of_plot == "Surface":
        ax = fig.add_subplot(111, projection="3d")
        # ax.plot_surface(P1, P2, Z, alpha=0.5)
        ax.plot_surface(X_predict[:,:,0], X_predict[:,:,1], y_hat, alpha=0.5)
        ax.plot_surface(X_predict[:,:,0], X_predict[:,:,1], y_hat - std, alpha=0.5)
        ax.plot_surface(X_predict[:,:,0], X_predict[:,:,1], y_hat + std, alpha=0.5)
        ax.set_zlabel("Z")
    elif setup.type_of_plot == "Contour":
        ax = fig.add_subplot()
        ax.contour(X_predict[:,:,0], X_predict[:,:,1], y_hat)

    # depict sample data. 
    ax.scatter(X[:, d_plot[0]],X[:, d_plot[1]],y)

    ax.set_title("{}".format(setup.solver_str))
    ax.set_xlabel(setup.search_space[0][d_plot[0]])
    ax.set_ylabel(setup.search_space[0][d_plot[1]])

def plot_1d(setup, X, y, predictor):
    pass
# def plot_1d(setup, X, y, X_predict, y_hat, mse):
#     L = int(np.power(X.shape[0], 1 / setup.d))
#     d_to_plot = setup.d_plot[0]
#     P1 = X[:, d_to_plot].reshape(-1,L)
# 
#     L = int(math.sqrt(len(X_predict[:, 0])))
#     P1_predict = X_predict[:, d_to_plot].reshape(-1,L)
# 
#     fig = plt.figure()
#     ax = fig.add_subplot()
# 
#     Z = y.reshape(P1.shape)
#     ax.plot(P1, Z, alpha=0.5)
# 
#     Z_hat = y_hat.reshape(P1_predict.shape)
#     std = np.sqrt(mse).reshape(P1_predict.shape)
#     ax.plot(P1_predict, Z_hat, alpha=0.5)
#     ax.plot(P1_predict, Z_hat - std, alpha=0.5)
#     ax.plot(P1_predict, Z_hat + std, alpha=0.5)
# 
#     ax.set_title("{}".format(setup.solver_str))
#     ax.set_xlabel(setup.search_space[0][d_to_plot])
#     ax.set_ylabel("Z")
