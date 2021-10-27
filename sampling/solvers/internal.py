"""
This file contains all internal solvers used for testing, plus the abstract classes Solver and TestFunction.
"""

import math
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from abc import ABC, abstractmethod


class Solver(ABC):
    max_d = 10
    min_d = 1

    def __init__(self):
        pass

    @abstractmethod
    def solve(self):
        pass

    def get_preferred_search_space(d):
        pass


class TestFunction(ABC):
    """
    Abstract Class for testfunctions.
    X.shape[0] the amount of samples
    X.shape[1] is the number of dimensions of the sample.
    """

    input_parameter_list = ["xi"]
    output_parameter_list = ["z"]
    objective_function = lambda z: min(z)

    def plot2d(self, X=None):
        if X is not None:
            X = self.correctFormatX(X, 2)
            # TODO change to the d-th root sort out of the works for all dimensions as well
            L = int(math.sqrt(len(X[:, 0])))
            P1 = X[:, 0].reshape(L, L)
            P2 = X[:, 1].reshape(L, L)
        else:
            # refer to p1 p2 instead of x,y to avoid confusion with X of data
            p1 = p2 = np.arange(-5, 5, 0.05)
            P1, P2 = np.meshgrid(p1, p2)
            X = np.hstack((np.reshape(P1, (-1, 1)), np.reshape(P2, (-1, 1))))
            print("Created new X")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        zs = self.solve(X)
        Z = zs.reshape(P1.shape)

        ax.plot_surface(P1, P2, Z)
        ax.set_title("{}".format(self.__class__.__name__))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.show()

    def correctFormatX(self, X, d_req=None):
        """
        First make sure X is an np.array.
        X always is of the form [[..,..],[..,..],..]
        """
        X = np.array(X)

        # if only a single datapoint
        if len(X.shape) < 2:
            X = np.array([X])

        if d_req != None:
            assert X.shape[1] == d_req, "Dimension should be {}".format(d_req)

        return X


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
        X = self.correctFormatX(X, d_req=2)

        x = X[:, 0]
        y = X[:, 1]
        a = 1
        b = 5.1 / (4 * np.pi ** 2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        return (a * (y - b * x ** 2 + c * x - r) ** 2 + s * (1 - t) * np.cos(x) + s) + x

    def get_preferred_search_space(d):
        return [["x0", "x1"], [-5, 0], [10, 15]]


class BraninNoise(Branin):
    """Branin function including standard normal noise."""

    def solve(self, X):
        noise_free = super().solve(X)
        y = noise_free + np.random.standard_normal(size=noise_free.shape)
        return y


class Paulson(TestFunction, Solver):
    """
    Made-up function of https://github.com/capaulson/pyKriging/blob/master/pyKriging/testfunctions.py
    """

    max_d = 2
    min_d = 2

    def solve(self, X, hz=5):
        X = self.correctFormatX(X, d_req=2)

        x = X[:, 0]
        y = X[:, 1]
        return 0.5 * np.sin(x * hz) + 0.5 * np.cos(y * hz)

    def paulson1(self, X, hz=10):
        X = self.correctFormatX(X, d_req=2)

        x = X[:, 0]
        y = X[:, 1]
        return (np.sin(x * hz)) / ((x + 0.2)) + (np.cos(y * hz)) / ((y + 0.2))


class Runge(TestFunction, Solver):
    """
    https://en.wikipedia.org/wiki/Runge%27s_phenomenon
    """

    def solve(self, X, offset=0.0):
        X = self.correctFormatX(X)

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
        X = self.correctFormatX(X)
        y = np.sum(np.power(X, 4) - 16 * np.power(X, 2) + 5 * X, axis=1) / 2
        return y

    def get_preferred_search_space(d):
        return [["x{}".format(i) for i in range(d)], [-5] * d, [5] * d]


class Curretal88exp(TestFunction, Solver):
    """
    https://www.sfu.ca/~ssurjano/curretal88exp.html
    The function is evaluated on the square xi ∈ [0, 1], for all i = 1, 2.
    """

    max_d = 2
    min_d = 2

    def solve(self, X):
        X = self.correctFormatX(X, d_req=2)

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

    def get_preferred_search_space(d):
        return [["x{}".format(i) for i in range(d)], [0] * d, [1] * d]


class Rastrigin(TestFunction, Solver):
    def solve(self, X):
        """
        https://www.sfu.ca/~ssurjano/rastr.html
        The function is usually evaluated on the hypercube xi ∈ [-5.12, 5.12], for all i = 1, …, d.
        global minimum: f(x*)=0, at x*=[0]*d
        """
        X = self.correctFormatX(X)
        d = X.shape[0]
        y = 10 * d + np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X), axis=1)
        return y

    def get_preferred_search_space(d):
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
        X = self.correctFormatX(X)
        y = (1 - X[:, :-1]) ** 2 + 100 * ((X[:, 1:] - X[:, :-1] ** 2) ** 2)
        return y

    def get_preferred_search_space(d):
        return [["x{}".format(i) for i in range(d)], [-5] * d, [10] * d]


# TODO multifidelity test cases
# https://www.sfu.ca/~ssurjano/multi.html


class Hartmann3(TestFunction, Solver):
    max_d = 3
    min_d = 3

    def solve(self, X):
        pass


class Hartmann6(TestFunction, Solver):
    max_d = 6
    min_d = 6

    def solve(self, X):
        pass


class XLinear(TestFunction, Solver):
    def solve(self, X):
        X = self.correctFormatX(X)

        y = np.sum(X, axis=1)
        return y


class XSquared(TestFunction, Solver):
    def solve(self, X, offset=0.25):
        X = self.correctFormatX(X)

        y = np.sum((X - offset) ** 2, axis=1)
        return y


class XCubed(TestFunction, Solver):
    def solve(self, X, offset=0.25):
        X = self.correctFormatX(X)

        y = np.sum((X - offset) ** 3, axis=1)
        return y


class XCosine(TestFunction, Solver):
    def solve(self, X):
        X = self.correctFormatX(X)
        y = np.cos(np.sum(X, axis=1))
        return y
