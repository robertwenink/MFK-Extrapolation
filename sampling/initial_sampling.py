import numpy as np
from pyDOE2 import lhs
import inspect
import sys


def get_doe(setup):
    for name,obj in inspect.getmembers(sys.modules[__name__], inspect.isfunction): 
        if name == setup.doe:
            return obj

def get_doe_name_list():
    """Helper function for the GUI"""
    name_list = []
    for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isfunction):
        if name not in ["lhs", "get_doe_name_list", "get_doe"]:
            name_list.append(name)
    return name_list


def grid(setup, n=None, n_per_d=None):
    """
    Build a grid and sample n points in each dimension d.
    Be aware that each column/ row is linearly dependent and thus correlation matrix R is (nearly) invertable.
    """
    if n is not None and n_per_d is None:
        n_per_d = int(round(n ** (1 / setup.d)))
    d = setup.d
    lb = setup.search_space[1]
    ub = setup.search_space[2]
    lin = np.linspace(lb, ub, n_per_d)
    lis = [lin[:, i] for i in range(d)]
    res = np.meshgrid(*lis)
    X = np.stack(res, axis=-1).reshape(-1, d)
    return X


def LHS(setup, n=None, n_per_d=None):
    """
    LHS
    """
    if n is not None:
        samples = n
    elif n_per_d is not None:
        samples = n_per_d ** setup.d

    # sampling plan from 0 to 1
    X = lhs(setup.d, samples=samples)

    # scale data according to the search space bounds
    X = (
        np.multiply(X, setup.search_space[2] - setup.search_space[1])
        + setup.search_space[1]
    )
    return X


