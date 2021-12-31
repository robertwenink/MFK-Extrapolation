"""
This file contains all internal solvers used for testing, plus the abstract classes Solver and TestFunction.
"""

import math
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from abc import ABC, abstractmethod
from utils.data_formatting import correct_formatX


class Solver(ABC):
    max_d = 10
    min_d = 1

    def __init__(self):
        pass

    @abstractmethod
    def solve(self):
        pass

    def get_preferred_search_space(self, d):
        pass


class TestFunction(ABC):
    """
    Abstract Class for testfunctions.
    X.shape[0] the amount of samples
    X.shape[1] is the number of dimensions of the sample.
    """

    input_parameter_list = ["xi"]
    output_parameter_list = ["z"]
    objective_function = lambda z: np.min(z)

    def check_formatX(self, X, d_req=None):
        """
        First make sure X is of the correct format.
        X always is of the form [[..,..],[..,..],..]
        """
        X = correct_formatX(X)

        if d_req != None:
            assert X.shape[1] == d_req, "Dimension should be {}".format(d_req)

        return X


class Forrester2008(TestFunction, Solver):
    max_d = 1
    min_d = 1

    def solve(self, X):
        X = self.check_formatX(X, d_req=1)

        # last term is not from forrester2008
        res = (6 * X - 2) ** 2 * np.sin(12 * X - 4)  # + np.sin(8 * 2 * np.pi * X)
        return res.ravel() 

    def get_preferred_search_space(self, d):
        return [["x"], [0], [1]]


class Branin(TestFunction, Solver):
    max_d = 2
    min_d = 2

    def solve(self, X):
        """
        https://www.sfu.ca/~ssurjano/branin.html
        x1 ∈ [-5, 10], x2 ∈ [0, 15].
        a = 1, b = 5.1 ⁄ (4π2), c = 5 ⁄ π, r = 6, s = 10 and t = 1 ⁄ (8π)
        global minima: f(x*)=0.397887, at x*=(-pi,12.275),(pi,2.275) and (9.42478,2.475)
        """
        X = self.check_formatX(X, d_req=2)

        x = X[:, 0]
        y = X[:, 1]
        a = 1
        b = 5.1 / (4 * np.pi ** 2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        return (a * (y - b * x ** 2 + c * x - r) ** 2 + s * (1 - t) * np.cos(x) + s) + x

    def get_preferred_search_space(self, d):
        return [["x0", "x1"], [-5, 0], [10, 15]]


class Runge(TestFunction, Solver):
    """
    https://en.wikipedia.org/wiki/Runge%27s_phenomenon
    """

    def solve(self, X, offset=0.0):
        X = self.check_formatX(X)

        y = 1 / (1 + np.sum((X - offset) ** 2, axis=1))
        return y


class Stybtang(TestFunction, Solver):
    def solve(self, X):
        """
        STYBLINSKI-TANG FUNCTION
        https://www.sfu.ca/~ssurjano/stybtang.html

        The function is usually evaluated on the hypercube xi ∈ [-5, 5], for all i = 1, …, d.

        global minimum: f(x*)=-39.16599*d, at x*=(-2.903534,...,-2.903534)
        """
        X = self.check_formatX(X)
        y = np.sum(np.power(X, 4) - 16 * np.power(X, 2) + 5 * X, axis=1) / 2
        return y

    def get_preferred_search_space(self, d):
        return [["x{}".format(i) for i in range(d)], [-5] * d, [5] * d]


class Curretal88exp(TestFunction, Solver):
    """
    https://www.sfu.ca/~ssurjano/curretal88exp.html
    The function is evaluated on the square xi ∈ [0, 1], for all i = 1, 2.
    """

    max_d = 2
    min_d = 2

    def solve(self, X):
        X = self.check_formatX(X, d_req=2)

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

    def get_preferred_search_space(self, d):
        return [["x{}".format(i) for i in range(d)], [0] * d, [1] * d]


class Rastrigin(TestFunction, Solver):
    def solve(self, X):
        """
        https://www.sfu.ca/~ssurjano/rastr.html
        The function is usually evaluated on the hypercube xi ∈ [-5.12, 5.12], for all i = 1, …, d.
        global minimum: f(x*)=0, at x*=[0]*d
        """
        X = self.check_formatX(X)
        d = X.shape[0]
        y = 10 * d + np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X), axis=1)
        return y

    def get_preferred_search_space(self, d):
        return [["x{}".format(i) for i in range(d)], [-5.12] * d, [5.12] * d]


class Rosenbrock(TestFunction, Solver):
    def solve(self, X):
        """
        https://www.sfu.ca/~ssurjano/rosen.html
        Rosenbrock function(Any order, usually 2D and 10D, sometimes larger dimension is tested)
        with global minima: 0 at x = [1] * dimension
        The function is usually evaluated on the hypercube xi ∈ [-5, 10], for all i = 1, …, d,
        although it may be restricted to the hypercube xi ∈ [-2.048, 2.048], for all i = 1, …, d.
        """
        X = self.check_formatX(X)
        y = np.sum(
            (1 - X[:, :-1]) ** 2 + 100 * ((X[:, 1:] - X[:, :-1] ** 2) ** 2), axis=1
        )
        return y

    def get_preferred_search_space(self, d):
        return [["x{}".format(i) for i in range(d)], [-2.048] * d, [2.048] * d]


class Hartmann6(TestFunction, Solver):
    """
    6-dimensional Hartmann function.
    https://www.sfu.ca/~ssurjano/hart6.html
    Global minimum f(x*) = -3.32237 at x* = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)"""

    max_d = 6
    min_d = 6

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

    def solve(self, X):
        X = self.check_formatX(X, 6)
        # TODO

    def get_preferred_search_space(self, d):
        return [["x{}".format(i) for i in range(d)], [0] * d, [1] * d]


" Functions really only used for testing purposes, with a simple analytic solution "


class XLinear(TestFunction, Solver):
    def solve(self, X):
        X = self.check_formatX(X)

        y = np.sum(X, axis=1)
        return y


class XSquared(TestFunction, Solver):
    def solve(self, X, offset=0.25):
        X = self.check_formatX(X)

        y = np.sum((X - offset) ** 2, axis=1)
        return y


class XCubed(TestFunction, Solver):
    def solve(self, X, offset=0.25):
        X = self.check_formatX(X)

        y = np.sum((X - offset) ** 3, axis=1)
        return y


class XCosine(TestFunction, Solver):
    def solve(self, X):
        X = self.check_formatX(X)
        y = np.cos(np.sum(X, axis=1))
        return y
