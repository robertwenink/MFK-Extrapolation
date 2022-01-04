import numpy as np
from models.kriging.method.OK import *
# Power describing the speed of convergence. Higher is faster
CONVERGENCE_SPEED = 2

def solve_sq(l, p=2):
    """return sum of series reciprocals up to number l.
    p=2; Basel problem converges to pi**2 / 6"""
    n = np.arange(
        1, l ** CONVERGENCE_SPEED + 2
    )  # +2, because then indexing from l=0 and less crazy first convergence step
    mod = 0
    return (np.pi ** 2 / 6 + mod) / (np.sum(np.divide(1, n ** 2)) + mod)


def solve_sq_inverse(l, p=2):
    return 1 / solve_sq(l, p)


def solve_ah(l):
    """alternating harmonic"""
    n = np.arange(2, l ** CONVERGENCE_SPEED + 2)
    n[::2] *= -1
    harmonic = 1 + np.sum(np.divide(1, n))
    return 1 + (np.log(2) - harmonic) * 2


def forrester2008(x):
    # last term is not from forrester2008
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4) # + np.sin(8 * 2 * np.pi * x)


def conv_modifier(solver_v, x):
    """this function provides a wrapper for the solver, introducing irregular convergence.
    mode =  0 ('original' Forrester 2008)
            1 (added sinus)
            2 (non-linear sinus, i.e. shifting over the levels in period)
    """
    noise = True
    mode = 3
    non_linear_power = CONVERGENCE_SPEED * 2 

    noise = 0.1 * noise * (np.random.standard_normal(size=x.shape)-0.5)

    if mode == 1:
        return solver_v + 2 * (1 - solver_v) * (np.sin(x * 2 * np.pi) + noise)
    elif mode == 2:
        return solver_v + (1 - solver_v) * (0.5 + np.sin(np.sqrt(solver_v+1)/np.sqrt(2) * x * 2 * np.pi) + noise)
    elif mode == 3:
        return solver_v + (1 - solver_v) * (0.5 + np.sin(solver_v * x * 2 * np.pi) + noise)
    else: 
        return solver_v * np.ones_like(x)


def mf_forrester2008(x, l, solver):
    x = correct_formatX(x)
    conv = conv_modifier(solver(l), x)
    A = conv
    linear_gain_mod = 2
    B = 10 * abs(conv - 1) * linear_gain_mod # using abs is a non-linear effect
    C = -5 * abs(conv - 1) * linear_gain_mod # using abs is a non-linear effect
    ret = A * forrester2008(x) + B * (x - 0.5) + C
    return ret.ravel(), sampling_costs(l) * x.shape[0]


def sampling_costs(l):
    return l ** 4