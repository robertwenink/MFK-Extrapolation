import numpy as np
from utils.data_formatting import correct_formatX
from models.kriging.method.OK import *

# Power describing the speed of convergence. Higher is faster
CONVERGENCE_SPEED = 2

############################
# CONVERGENCE BASE MODELS
############################

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
    return 1 / conv_stable_up(l, p)


def conv_alternating(l):
    """alternating harmonic"""
    n = np.arange(2, l ** CONVERGENCE_SPEED + 2)
    n[::2] *= -1
    harmonic = 1 + np.sum(np.divide(1, n))
    return 1 + (np.log(2) - harmonic) * 2


def get_mf_solver(setup,Solver):
    """
    @param setup: required for choosing the desired mf mode.
    @param setup: solver class; we want to use and override the solve() of the base class.
    """

    # https://stackoverflow.com/questions/21060073/dynamic-inheritance-in-python
    class MF_Solver(Solver):

        def __init__(self,setup):
            self.conv_noise = setup.conv_noise
            self.conv_mod = setup.conv_mod

            # selection of base convergence model
            if "Stable up" in setup.conv_type:
                self.conv_base = conv_stable_up
            elif "Stable down" in setup.conv_type:
                self.conv_base = conv_stable_down
            elif "Alternating" in setup.conv_type:
                self.conv_base = conv_alternating
            
            
        ############################
        # CONVERGENCE MODIFIER
        ############################

        def conv_modifier(self, conv_base, x):
            """this function provides a wrapper for the convergence type, introducing irregular convergence.
            mode =  0 ('original' function, but extended for domain x)
                    1 (added sinus)
                    2 (non-linear sinus, i.e. shifting over the levels in period)
                    3 (more non-linear variant)
            """

            non_linear_power = CONVERGENCE_SPEED * 2 

            noise = 0.1 * self.conv_noise * (np.random.standard_normal(size=x.shape)-0.5)

            if self.conv_mod == 1:
                return conv_base + 2 * (1 - conv_base) * (np.sin(x * 2 * np.pi) + noise)
            elif self.conv_mod == 2:
                return conv_base + (1 - conv_base) * (0.5 + np.sin(np.sqrt(conv_base+1)/np.sqrt(2) * x * 2 * np.pi) + noise)
            elif self.conv_mod == 3:
                return conv_base + (1 - conv_base) * (0.5 + np.sin(conv_base * x * 2 * np.pi) + noise)
            else: 
                return conv_base * np.ones_like(x)

        def sampling_costs(self,l):
            return l ** 4

        def solve(self,X,l=20):
            if l == 20:
                print("WARNING: not providing an level for the MF solver!")
            X = self.check_formatX(X)
            
            conv = self.conv_modifier(self.conv_base(l), np.sum(X,axis=1))
            A = conv
            
            # TODO, use other optimization problems and add them, i.e. linear, cosine, cube, etc.
            # np.sum(X,axis=1) is the linear function
            # TODO 
            linear_gain_mod = 2
            B = 10 * (conv - 1) * linear_gain_mod # using abs is a non-linear effect
            C = -5 * (conv - 1) * linear_gain_mod # using abs is a non-linear effect
            s = super().solve(X)
            
            # A, B, C = correct_formatX(A),correct_formatX(B),correct_formatX(C)
            # su = correct_formatX(np.sum(X,axis=1))
            res = A * s + B * (np.sum(X,axis=1) - 0.5) + C
            return res, self.sampling_costs(l) * X.shape[0]

    return MF_Solver(setup)

# original example mf wrapper
def mf_forrester2008(x, l, solver):
    x = correct_formatX(x)
    conv = conv_modifier(solver(l), x)
    A = conv
    linear_gain_mod = 2
    B = 10 * abs(conv - 1) * linear_gain_mod # using abs is a non-linear effect
    C = -5 * abs(conv - 1) * linear_gain_mod # using abs is a non-linear effect
    ret = A * forrester2008(x) + B * (x - 0.5) + C
    return ret.ravel(), sampling_costs(l) * x.shape[0]







