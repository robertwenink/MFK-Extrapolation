"""
This file contains all internal solvers used for testing, plus the abstract classes Solver and TestFunction.
"""
# pyright: reportGeneralTypeIssues=false
import numpy as np
from typing import Callable, Any

from abc import ABC, abstractmethod
from core.sampling.DoE import LHS
from utils.formatting_utils import correct_formatX
from utils.selection_utils import create_X_infill
from core.sampling.solvers.convergence_base_models import *

# create seeded random number generator
rng = np.random.default_rng(12345)


class Solver(ABC):
    max_d = 10
    min_d = 1

    input_parameter_list = ["xi"]
    output_parameter_list = ["z"]

    name = "Solver"

    def __init__(self, setup=None):
        pass

    @abstractmethod
    def solve(self, X):
        raise NotImplementedError("Implement the solve method")

    def check_format_X(self, X, d_req=None):
        """
        First make sure X is of the correct format.
        X always is of the form [[..,..],[..,..],..]
        """
        X = correct_formatX(X, dim=d_req)

        if d_req != None:
            assert X.shape[1] == d_req, "Dimension should be {}".format(d_req)

        return X

    def check_format_y(self, y):
        assert isinstance(y, np.ndarray), "y should be an np array"
        assert y.ndim == 1, "y should be 1 dimensional"
        return y

    def get_preferred_search_space(self, d):
        """
        This defines and passes the names and ranges of the design space variables to the GUI and further backend.
        """
        return [["x{}".format(i) for i in range(d)], [0] * d, [1] * d]


class TestFunction(Solver):
    """
    Abstract Class for testfunctions.
    X.shape[0] the amount of samples
    X.shape[1] is the number of dimensions of the sample.
    """

    name = "TestFunction"
    objective_function = lambda z: np.min(z)

    @abstractmethod
    def get_preferred_search_space(self, d):
        pass

    @abstractmethod
    def get_optima(self):
        return [0], [0]

    def __init__(self, setup=None):
        """
        setup must be None; we do not always require the use of a setup,
        as this class is used in creation of the setup object/ GUI for retrieving min_d, max_d.
        """
        
        if setup is not None:
            self.mf = setup.mf
            self.d = setup.d
            self.lb = setup.search_space[1]
            self.ub = setup.search_space[2]
            self.bound_range = self.ub - self.lb

            # NOTE X_infill not very scalable but sure, only done once
            self.solver_noise = False # for fake first solving
            X = LHS(setup, n_per_d=40)
            Z = self.solve(X)[0]
            self.value_metrics = [np.min(Z), np.max(Z), np.mean(Z)]

            self.solver_noise = setup.solver_noise
            if self.solver_noise == True: # bcs the one from setup is usually a boolean
                self.solver_noise = 0.02

            if self.mf:
                # keep this for plotting text
                self.conv_type = setup.conv_type

                # selection of base convergence model
                if "Stable up" in self.conv_type:
                    conv_base_func = conv_stable_up
                elif "Stable down" in self.conv_type:
                    conv_base_func = conv_stable_down
                elif "Alternating" in self.conv_type:
                    conv_base_func = conv_alternating
                else:
                    raise Exception("Convergence base type not correctly linked.")

                # TODO, use other optimization problems and add them, i.e. linear, cosine, cube, etc.
                # np.sum(X,axis=1) is the linear function
                self.conv_mod_func = self.mf_conv_modifier(
                    conv_base_func, setup.conv_mod, lambda X: np.sum(X, axis=1)
                )

    def sampling_costs(self, l):
        return (l + 1) ** 4

    def mf_conv_modifier(self, conv_base_func, conv_mod, X_acc_func):
        """this function provides a wrapper for the multi-fidelity convergence type, introducing irregular convergence.
        @param conv_base_func: the base 1d convergence method retrieveing a scalar for some level l.
        @param conv_mod: hardness of modifier
        @param X_acc_func: function used to accumulate X over its dimensions.
            X needs to b accumulated over dimensions in order to be used in the sinoid.
        mode =  0 ('original' function, but extended for domain x)
                1 (added sinus)
                2 (non-linear sinus, i.e. shifting over the levels in period)
                3 (more non-linear variant)
        @returns self.conv_mod_func(X,l)
        """
        nr_of_periods = 2
        if conv_mod == 1:
            return lambda X, l: conv_base_func(l) + 2 * (1 - conv_base_func(l)) * (
                np.sin(X_acc_func(X) / (np.sum(self.bound_range)) * nr_of_periods * 2 * np.pi)
            )
        elif conv_mod == 2:
            return lambda X, l: conv_base_func(l) + (1 - conv_base_func(l)) * (
                np.sin(
                    # bij problemen ooit met negatieve power: np.sign(conv_base_func(l))*np.abs((conv_base_func(l)+0j)**(1/2)))
                    (conv_base_func(l))**(1/8)
                    * X_acc_func(X) / (np.sum(self.bound_range)) * nr_of_periods
                    * 2
                    * np.pi
                )
            )
        elif conv_mod == 3:
            return lambda X, l: conv_base_func(l) + (1 - conv_base_func(l)) * (
                np.sin((conv_base_func(l))**(1/4) * X_acc_func(X) / (np.sum(self.bound_range)) * nr_of_periods * 2 * np.pi)
            )
        else:
            return lambda X, l: conv_base_func(l) * np.ones(X.shape[0])
    
    @staticmethod
    def _modify(func : Callable) -> Any:
        """decorator to add noise and multi fidelity functionality"""

        def wrapper(self, X, l=None, ratio_abs_rel_noise = 0.5):
            """
            set l = None for no convergence noise, additionally l = -1 for no constant noise too

            noise_level is the flat amount of noise we apply.
            
            Noise cannot be applied *value because then at y = 0 there will be 0 noise.
            At the same time, equal amounts of noise (although more relastic), might not realistic either
            because you would expect large values or very turbulent simulations to display more noise.
            
            @param ratio_abs_rel_noise: absolute noise / relative noise
                value of 1 means only absolute noise, value of 0 only relative noise

            There are 2 types of noise levels
            1) convergence-dependent noise levels
                    actual noise level is reduced using the convergence function 
                    such that higher fidelities experience less noise
                    l = 0 knows no reduction and the function is normalized as such
            2) constant noise levels
                    This noise is present in all fidelity levels. It is defined as 1/10th of the supplied noise_level

            These noises can then be applied in two ways to the solver data:                    
            1) relative: wrt to own value
            2) absolute: wrt to mean, median, or max value;
            median is chosen now because for some functions the response can get extremely steep around the domain borders 
            while elsewhere the response surface is quite flat
            
            """
            cost = 1

            # use base solver
            val = func(self, X)
            self.check_format_y(val)

            if self.mf and l != None and l != -1:
                cost = self.sampling_costs(l) * X.shape[0]

                " convergence profile "
                X = self.check_format_X(X)
                conv = self.conv_mod_func(X, l)
                conv0 = self.conv_mod_func(X, 0)
                self.check_format_y(conv)

                " transformation "
                # using abs(conv - 1) is a non-linear effect, but is required for mimicking Forrester/Meliani (in which case it is not non-linear??)
                data_range_multiplier = (self.value_metrics[1] - self.value_metrics[0]) / 20 # is ~20 for forrester
                linear_gain_mod = 2 * data_range_multiplier
                A = conv
                B = 10 * (conv - 1) * linear_gain_mod
                C = -5 * (conv - 1) * linear_gain_mod

                val = A * val + B * (np.sum(X, axis=1) / self.d - 0.5 * np.max(self.bound_range)) + C

                if self.solver_noise:
                    " convergence noise "
                    # normalization wrt lowest conv level
                    # norm = v - min / (max - min)
                    norm_conv = abs(conv - 1) / abs(conv0 - 1)

                    # get the convergence corrected noise_level
                    conv_noise = norm_conv * self.solver_noise * (
                        # if run with very high numbers, absolute mean is ~0.8
                        # https://en.wikipedia.org/wiki/Half-normal_distribution
                        # 1 * np.sqrt(2) / np.sqrt(np.pi) = 0.7978845608028655
                        # ik wil dat normaliseren naar 1 zodat het noise_level 
                        # precies overeenkomt met de eenzijdige expectation
                        rng.standard_normal(size = conv.shape) / 0.8
                    )
                    
                    # add the relative part of noise
                    val *= 1 + (1 - ratio_abs_rel_noise) * conv_noise

                    # absolute part of noise, this part is equal for all values
                    # value metrics: min, max, mean, median; used here: median
                    val += ratio_abs_rel_noise * self.value_metrics[2] * conv_noise
                
            # TODO NOTE assume toplevel without noise for less stochastic comparison!
            if self.solver_noise and l != -1 and not l is None:
                " noise regardless of fidelity level execpt for the truth / lvl 2 "
                # provide noise, present at every fidelity level and relative to the noise value
                # defined as 1/10th of the noise_level
                #  TODO wel of niet /0.8
                const_noise = self.solver_noise / 10 * (rng.standard_normal(size=val.shape)) / 0.8           

                # relative part of noise, this part is bigger for larger values
                val *= 1 + (1 - ratio_abs_rel_noise) * const_noise
                
                # absolute part of noise, this part is equal for all values
                # value metrics: min, max, mean, median; used here: median
                val += ratio_abs_rel_noise * self.value_metrics[2] * const_noise


            return val, cost

        return wrapper


class Forrester2008(TestFunction):
    max_d = 1
    min_d = 1
    name = "Forrester2008"

    def __init__(self, setup=None):
        super().__init__(setup)

    @TestFunction._modify
    def solve(self, X):
        """
        Wolfram alpha: min{(6 x - 2)^2 sin(12 x - 4)|0<=x<=1}≈-6.02074 at x≈0.757249
        """
        X = self.check_format_X(X, d_req=1)

        # last term is not from forrester2008
        res = (6 * X - 2) ** 2 * np.sin(12 * X - 4)  # + np.sin(8 * 2 * np.pi * X)
        return res.ravel()

    @staticmethod
    def get_preferred_search_space(d):
        return [["x"], [0], [1]]

    def get_optima(self):
        return correct_formatX([[0.757249]],self.d), [-6.02074]

class Branin_Perdikaris(TestFunction):
    max_d = 2
    min_d = 2
    name = "Branin_Perdikaris"

    def hifi(self, x1, x2):
        return ((-1.275 * x1**2)/np.pi**2 + 5*x1/np.pi + x2 -6)**2 + (10 - 5/(4*np.pi))*np.cos(x1) + 10
    
    def mefi(self, x1, x2):
        return 10*np.sqrt(self.hifi(x1-2,x2-2)) + 2*(x1 - 0.5) - 3*(3*x2 -1) -1

    def lofi(self, x1, x2):
        return self.mefi(1.2*(x1+2), 1.2*(x2+2)) - 3 * x2 + 1

    def solve(self, X, l = None):
        X = self.check_format_X(X)
        x1 = X[:, 0]
        x2 = X[:, 1]

        if l == 1:
            return self.lofi(x1,x2), 1
        if l == 2:
            return self.mefi(x1,x2), 1
        if l == 3 or l == None:
            return self.hifi(x1,x2), 1

    @staticmethod
    def get_preferred_search_space(d):
        return [["x0", "x1"], [-5, 0], [10, 15]]
    
    def get_optima(self):
        # return correct_formatX([[-np.pi,12.275],[np.pi,2.275],[9.42478,2.475]],self.d), 0.397887
        return correct_formatX([[-np.pi,12.275]],self.d), [0.397887]

class Branin(TestFunction):
    max_d = 2
    min_d = 2
    name = "Branin"

    def __init__(self, setup=None): 
        super().__init__(setup)

    @TestFunction._modify
    def solve(self, X):
        """
        https://www.sfu.ca/~ssurjano/branin.html
        x1 ∈ [-5, 10], x2 ∈ [0, 15].
        a = 1, b = 5.1 ⁄ (4π2), c = 5 ⁄ π, r = 6, s = 10 and t = 1 ⁄ (8π)
        global minima: f(x*)=0.397887, at x*=(-pi,12.275),(pi,2.275) and (9.42478,2.475)

       adaption of Forroster 2008: use + 5x1, we use a parabola instead to maintain the same optimum location.
        """
        X = self.check_format_X(X, d_req=2)

        x1 = X[:, 0]
        x2 = X[:, 1]
        a = 1
        b = 5.1 / (4 * np.pi ** 2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        return (a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s) + (x1 + np.pi)**2 / 10
        # MINIMIZE ((x2 - 5.1 / (4 * pi ^2) * x2 + 5 / pi * x1 - 6) ** 2 + 10 * ((1 - 1 / (8 * pi)) * cos(x1) + 1) + 5*x1)

    @staticmethod
    def get_preferred_search_space(d):
        return [["x0", "x1"], [-5, 0], [10, 15]]

    def get_optima(self):
        # return correct_formatX([[-np.pi,12.275],[np.pi,2.275],[9.42478,2.475]],self.d), 0.397887
        return correct_formatX([[-np.pi,12.275]],self.d), [0.397887]


class Runge(TestFunction):
    """
    https://en.wikipedia.org/wiki/Runge%27s_phenomenon
    """
    name = "Runge"
    
    def __init__(self, setup=None): 
        super().__init__(setup)
    
    @TestFunction._modify
    def solve(self, X, offset=0.0):
        X = self.check_format_X(X)

        y = 1 / (1 + np.sum((X - offset) ** 2, axis=1))
        return y


class Stybtang(TestFunction):
    name = "Stybtang"
    
    def __init__(self, setup=None): 
        super().__init__(setup)
    
    @TestFunction._modify
    def solve(self, X):
        """
        STYBLINSKI-TANG FUNCTION
        https://www.sfu.ca/~ssurjano/stybtang.html

        The function is usually evaluated on the hypercube xi ∈ [-5, 5], for all i = 1, …, d.

        global minimum: f(x*)=-39.16599*d, at x*=(-2.903534,...,-2.903534)
        """
        X = self.check_format_X(X)
        y = np.sum(np.power(X, 4) - 16 * np.power(X, 2) + 5 * X, axis=1) / 2
        return y

    @staticmethod
    def get_preferred_search_space(d):
        return [["x{}".format(i) for i in range(d)], [-5] * d, [5] * d]

    def get_optima(self):
        return correct_formatX([[-2.903534]*self.d],self.d), [-39.16599*self.d]


class Curretal88exp(TestFunction):
    """
    https://www.sfu.ca/~ssurjano/curretal88exp.html
    The function is evaluated on the square xi ∈ [0, 1], for all i = 1, 2.
    """

    max_d = 2
    min_d = 2
    name = "Curretal"

    def __init__(self, setup=None): 
        super().__init__(setup)

    @TestFunction._modify
    def solve(self, X):
        X = self.check_format_X(X, d_req=2)

        x1 = X[:, 0]
        x2 = X[:, 1]

        fact1 = 1 - np.exp(-1 / (2 * x2))
        fact2 = 2300 * np.power(x1, 3) + 1900 * np.power(x1, 2) + 2092 * x1 + 60
        fact3 = 100 * np.power(x1, 3) + 500 * np.power(x1, 2) + 4 * x1 + 20

        return fact1 * fact2 / fact3

    def solve_LF(self, X):
        # Cheap Currin function
        X1 = X + 0.05
        X2 = np.maximum(X + np.array([0.05, -0.05]), np.array([-np.Inf, 0]))
        X3 = X + np.array([-0.05, 0.05])
        X4 = np.maximum(X - 0.05, np.array([-np.Inf, 0]))
        res = 0.25 * (self.solve(X1) + self.solve(X2) + self.solve(X3) + self.solve(X4))
        return res

    @staticmethod
    def get_preferred_search_space(d):
        return [["x{}".format(i) for i in range(d)], [0] * d, [1] * d]


class Rastrigin(TestFunction):
    name = "Ratrigin"

    def __init__(self, setup=None): 
        super().__init__(setup)

    @TestFunction._modify
    def solve(self, X):
        """
        https://www.sfu.ca/~ssurjano/rastr.html
        The Rastrigin function has several local minima. It is highly multimodal, but locations of the minima are regularly distributed.
        The function is usually evaluated on the hypercube xi ∈ [-5.12, 5.12], for all i = 1, …, d.
        global minimum: f(x*)=0, at x*=[0]*d
        """
        X = self.check_format_X(X)
        d = X.shape[0]
        y = 10 * d + np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X), axis=1)
        return y

    @staticmethod
    def get_preferred_search_space(d):
        return [["x{}".format(i) for i in range(d)], [-5.12] * d, [5.12] * d]

    def get_optima(self):
        return correct_formatX([[0]*self.d],self.d), [0]


class Rosenbrock(TestFunction):
    name = "Rosenbrock"

    def __init__(self, setup=None): 
        super().__init__(setup)

    @TestFunction._modify
    def solve(self, X):
        """
        https://www.sfu.ca/~ssurjano/rosen.html
        Rosenbrock function(Any order, usually 2D and 10D, sometimes larger dimension is tested)
        with global minima: 0 at x = [1] * dimension
        The function is usually evaluated on the hypercube xi ∈ [-5, 10], for all i = 1, …, d,
        although it may be restricted to the hypercube xi ∈ [-2.048, 2.048], for all i = 1, …, d.
        """
        X = self.check_format_X(X)
        y = np.sum(
            (1 - X[:, :-1]) ** 2 + 100 * ((X[:, 1:] - X[:, :-1] ** 2) ** 2), axis=1
        )
        return y

    @staticmethod
    def get_preferred_search_space(d):
        return [["x{}".format(i) for i in range(d)], [-2.048] * d, [2.048] * d]

    def get_optima(self):
        return correct_formatX([[1]*self.d],self.d), [0]

class Hartmann6(TestFunction):
    """
    6-dimensional Hartmann function.
    https://www.sfu.ca/~ssurjano/hart6.html
    Global minimum f(x*) = -3.32237 at x* = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)"""
    
    name = "Hartmann6"
    max_d = 6
    min_d = 6

    def __init__(self, setup=None): 
        super().__init__(setup)
    
    alpha = np.array([1, 1.2, 3, 3.2]).T

    A = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )

    P = 1e-4 * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 2047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )

    @TestFunction._modify
    def solve(self, X):
        X = self.check_format_X(X, 6)
        # TODO
        print("Hartmann6 not implemented yet")

    def get_preferred_search_space(self, d):
        return [["x{}".format(i) for i in range(d)], [0] * d, [1] * d]


" Functions really only used for testing purposes, with a simple analytic solution "


class XLinear(TestFunction):
    @TestFunction._modify
    def solve(self, X):
        X = self.check_format_X(X)

        y = np.sum(X, axis=1)
        return y


class XSquared(TestFunction):
    @TestFunction._modify
    def solve(self, X, offset=0.25):
        X = self.check_format_X(X)

        y = np.sum((X - offset) ** 2, axis=1)
        return y


class XCubed(TestFunction):
    @TestFunction._modify
    def solve(self, X, offset=0.25):
        X = self.check_format_X(X)

        y = np.sum((X - offset) ** 3, axis=1)
        return y


class XCosine(TestFunction):
    @TestFunction._modify
    def solve(self, X):
        X = self.check_format_X(X)
        y = np.cos(np.sum(X, axis=1))
        return y
