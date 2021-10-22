"""
This file contains all internal solvers used for testing, plus the abstract classes Solver and TestFunction.
"""

import math
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from abc import ABC
from abc import abstractmethod


class Solver(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def solve(self):
        pass

class TestFunction(ABC):
    """
    Abstract Class for testfunctions.
    X.shape[0] the amount of samples
    X.shape[1] is the number of dimensions of the sample.
    """

    def plot2d(self, X=None):
        if X is not None:
            X = self.correctFormatX(X, 2)
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

class Linear(TestFunction, Solver):
    def solve(self, X):
        X = self.correctFormatX(X)

        y = np.array([], dtype=float)
        for i in range(X.shape[0]):
            y = np.append(y, np.sum(X[i]))
        return y


class Squared(TestFunction, Solver):
    def solve(self, X, offset=0.25):
        X = self.correctFormatX(X)

        offset = np.ones(X.shape[1]) * offset
        y = np.array([], dtype=float)
        for i in range(X.shape[0]):
            y = np.append(y, (np.sum((X[i] - offset) ** 2) ** 0.5))
        return y


class Cubed(TestFunction, Solver):
    def solve(self, X, offset=0.25):
        X = self.correctFormatX(X)

        offset = np.ones(X.shape[1]) * offset
        y = np.array([], dtype=float)
        for i in range(X.shape[0]):
            y = np.append(y, (np.sum((X[i] - offset) ** 3) ** (1 / 3.0)))
        return y


class Branin(TestFunction, Solver):
    def solve(self, X):
        """
        https://www.sfu.ca/~ssurjano/branin.html
        x1 ∈ [-5, 10], x2 ∈ [0, 15].
        a = 1, b = 5.1 ⁄ (4π2), c = 5 ⁄ π, r = 6, s = 10 and t = 1 ⁄ (8π)
        minima at ..
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

class BraninNoise(TestFunction, Solver):
    def solve(self, X):
        X = self.correctFormatX(X, d_req=2)

        x = X[:, 0]
        y = X[:, 1]
        X1 = 15 * x - 5
        X2 = 15 * y
        a = 1
        b = 5.1 / (4 * np.pi ** 2)
        c = 5 / np.pi
        d = 6
        e = 10
        ff = 1 / (8 * np.pi)
        noiseFree = (
            a * (X2 - b * X1 ** 2 + c * X1 - d) ** 2 + e * (1 - ff) * np.cos(X1) + e
        ) + 5 * x
        withNoise = []
        for i in noiseFree:
            withNoise.append(i + np.random.standard_normal() * 15)
        return np.array(withNoise)

class Paulson(TestFunction, Solver):
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
    def solve(self, X, offset=0.0):
        X = self.correctFormatX(X)

        offset = np.ones(X.shape[1]) * offset
        y = np.array([], dtype=float)
        for i in range(X.shape[0]):
            y = np.append(y, (1 / (1 + np.sum((X[i] - offset) ** 2))))
        return y

class Stybtang(TestFunction, Solver):
    def solve(self, X):
        """
        STYBLINSKI-TANG FUNCTION
        https://www.sfu.ca/~ssurjano/stybtang.html
        """
        X = self.correctFormatX(X)
        y = np.sum(np.power(X, 4) - 16 * np.power(X, 2) + 5 * X, axis=1) / 2
        return y

class Curretal88exp(TestFunction, Solver):
    def solve(self, X):
        X = self.correctFormatX(X, d_req=2)

        x1 = X[:, 0]
        x2 = X[:, 1]

        fact1 = 1 - np.exp(-1 / (2 * x2))
        fact2 = 2300 * np.power(x1, 3) + 1900 * np.power(x1, 2) + 2092 * x1 + 60
        fact3 = 100 * np.power(x1, 3) + 500 * np.power(x1, 2) + 4 * x1 + 20

        return fact1 * fact2 / fact3


class Cosine(TestFunction, Solver):
    def solve(self, X):
        X = self.correctFormatX(X)
        y = np.cos(np.sum(X, axis=1))
        return y

class Rastrigin(TestFunction, Solver):
    def solve(self, X):
        """
        2D Rastrigin function:
            with global minima: 0 at x = [0, 0]
        :param x:
        :return:
        """
        X = self.correctFormatX(X, d_req=2)

        y = (
            20
            + X[:, 0] ** 2
            + X[:, 1] ** 2
            - 10 * (np.cos(2 * np.pi * X[:, 0]) + np.cos(2 * np.pi * X[:, 1]))
        )
        return y

class Rosenbrock(TestFunction, Solver):
    def solve(self, X):
        """
        Rosenbrock function(Any order, usually 2D and 10D, sometimes larger dimension is tested)
        with global minima: 0 at x = [1] * dimension
        :param x:
        :return:
        """
        X = self.correctFormatX(X)

        y = [0.0] * X.shape[0]
        for i in np.arange(0, X.shape[1] - 1):
            y += (1 - X[:, i]) ** 2 + 100 * ((X[:, i + 1] - X[:, i] ** 2) ** 2)

        return y

class Hartmann3(TestFunction, Solver):
    def solve(self, X):
        pass

class Hartmann6(TestFunction, Solver):
    def solve(self, X):
        pass
