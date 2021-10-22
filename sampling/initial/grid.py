"""
Build a grid and sample n points in each dimension d
"""
import numpy as np

def generate(setup,n=3):
    """
    param n: number of points per dimension
    """
    
    lb = setup.search_space[1]
    ub = setup.search_space[2]
    lin = np.linspace(lb, ub, n)
    lis = [lin[:,i] for i in range(setup.d)]
    res = np.meshgrid(*lis)
    X = np.stack(res, axis=-1).reshape(-1, setup.d)
    return X

