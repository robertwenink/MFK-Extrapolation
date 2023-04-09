import numpy as np
from pyDOE2 import lhs
import inspect
import sys
import matplotlib.pyplot as plt
from core.ordinary_kriging.kernel import dist_matrix


def get_doe(setup):
    for name,obj in inspect.getmembers(sys.modules[__name__], inspect.isfunction): 
        if name == setup.doe:
            return obj
    return lambda *args, **kwargs: print("No correct DoE provided!") 

def get_doe_name_list():
    """Helper function for the GUI"""
    name_list = []
    for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isfunction):
        if name not in ["lhs", "get_doe_name_list", "get_doe"]:
            name_list.append(name)
    return name_list


def grid(setup, n=None, n_per_d : int = 10):
    """
    Build a grid/ full factorial DoE and sample n points in each dimension d.
    Be aware that each column/ row is linearly dependent and thus correlation matrix R is (nearly) invertable.
    """
    if n is not None and n_per_d == 10:
        n_per_d = int(round(n ** (1 / setup.d)))
    d = setup.d
    lb = setup.search_space[1]
    ub = setup.search_space[2]
    lin = np.linspace(lb, ub, n_per_d)
    lis = [lin[:, i] for i in range(d)]
    res = np.meshgrid(*lis)
    X = np.stack(res, axis=-1).reshape(-1, d)
    return X

def grid_unit_hypercube(d, n=None, n_per_d : int = 10):
    """
    Build a grid/ full factorial DoE and sample n points in each dimension d for a unit hypercube.
    Be aware that each column/ row is linearly dependent and thus correlation matrix R is (nearly) invertable.
    """
    if n is not None and n_per_d == 10:
        n_per_d = int(round(n ** (1 / d)))
    lin = np.linspace(0, 1, n_per_d)
    lis = [lin for i in range(d)]
    res = np.meshgrid(*lis)
    X = np.stack(res, axis=-1).reshape(-1, d)
    return X


def LHS(setup, n=None, n_per_d : int = 10, random_state = 1):
    """
    LHS
    """
    if n is not None:
        samples = n
    else:
        samples = n_per_d * setup.d

    # sampling plan from 0 to 1
    X = lhs(setup.d, samples=samples, criterion='maximin',iterations=min(int(2*samples), 150), random_state = random_state) # type: ignore

    # scale data according to the search space bounds
    X = (
        np.multiply(X, setup.search_space[2] - setup.search_space[1])  # type: ignore
        + setup.search_space[1]
    )
    return X

def LHS_subset(setup, X_superset, x_init, amount):
    """
    Greedy selection via Morris-Mitchels / maximin criterium. See Forrester 2008 as well.
    """ 
    assert amount <= X_superset.shape[0]

    ind = np.argwhere((x_init == X_superset).all(axis=-1))[0].item()
    ind_list = [ind]

    dist_mat = dist_matrix(X_superset, X_superset)
        
    bounds = setup.search_space[1:]
    low = np.min(X_superset - bounds[0],axis=1)
    up = np.min(bounds[1] - X_superset,axis=1)
    # *2 because bound is "in between" with ghost point, distance we take into account should be twice as large
    min_to_bound = np.min([low,up],axis=0)*2 
    	
    while len(ind_list) < amount:
        # add distances of a point to current chose points, take one with highest minimum distance.
        sub = dist_mat[ind_list] # Euclidian distance
        s=np.min(sub,axis=0) # axis = 0, dist matrix is symmetric

        s2 = np.min([s,min_to_bound],axis=0)
        ind_next = np.argmax(s2)
        ind_list.append(ind_next)
    return X_superset[ind_list]

def plot_doe():
    plot_doe.d = 2
    plot_doe.search_space = np.array([[np.inf] * plot_doe.d,[0] * plot_doe.d, [1] * plot_doe.d])
    fig = plt.figure()
    X = LHS(plot_doe,20)
    X_lo = X.T
    plt.scatter(*X_lo,label='low-fidelity superset')
    x_init = X[10]
    X_hi = LHS_subset(plot_doe, X, x_init, 5).T
    plt.scatter(*X_hi,label='high-fidelity subset')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.scatter(x_init[0],x_init[1],color='r',label="x initial")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()

if __name__=="__main__":
    plot_doe()