############################
# CONVERGENCE BASE MODELS
############################
 
import numpy as np

# Power describing the speed of convergence. Higher is faster
CONVERGENCE_SPEED = 2

def conv_stable_up(l):
    """return sum of series reciprocals up to number l.
    p=2; Basel problem converges to pi**2 / 6"""
    p = 2
    n = np.arange(
        1, l ** CONVERGENCE_SPEED + 2
    )  # +2, because then indexing from l=0 and less crazy first convergence step
    mod = 0
    return (np.pi ** 2 / 6 + mod) / (np.sum(np.divide(1, n ** 2)) + mod)


def conv_stable_down(l):
    return 1 / conv_stable_up(l)


def conv_alternating(l):
    """alternating harmonic"""
    n = np.arange(2, l ** CONVERGENCE_SPEED + 2)
    n[::2] *= -1
    harmonic = 1 + np.sum(np.divide(1, n))
    return 1 + (np.log(2) - harmonic) * 2








