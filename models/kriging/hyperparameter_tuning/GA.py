""" Adapted from: Copyright at end of file """
from numba import njit
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

import tqdm
from itertools import repeat
import os
import functools
from multiprocessing.dummy import Pool as ThreadPool

from scipy.optimize import minimize
from pyDOE2 import lhs

from utils.formatting_utils import correct_format_hps


class Tuner:
    def __init__(
        self,
        function,
        hps_init,
        hps_constraints,
        progress_bar=True,
    ):

        """
        @param function <Callable> - the given objective function to be minimized

        @param hps_constraints <numpy array/None> - provide an array of tuples
        of length two as boundaries for each hps; the length of the array must be equal dimension.
        For example, np.array([0,100],[0,200]) determines lower boundary 0 and upper boundary 100 for first hps.

        @param convergence_curve <True/False> - Plot the convergence curve or not. Default is True.
        @progress_bar <True/False> - Show progress bar or not. Default is True.
        """

        self.assert_input(function, hps_init, hps_constraints)

        self.set_hps_to_tune(hps_constraints)
        self.hps_init = hps_init
        self.f = function
        self.progress_bar = progress_bar

        self.report = []

        self.pop_s = min(int(2 ** (2 + 2 * self.dim)), 1000)
        # print("Population size = {}".format(self.pop_s))

    def set_hps_to_tune(self, hps_constraints):
        """
        select which hyper parameters we are going to tune.
        if the upper and lowerbounds of the hyper parameter constraints are equal, we willnot tune them
        """
        self.hps_tune_indices = []
        self.dim = 0

        for i in range(hps_constraints.shape[0]):
            if hps_constraints[i][0] != hps_constraints[i][1]:
                self.hps_tune_indices.append(True)
                self.dim += 1
            else:
                self.hps_tune_indices.append(False)
        self.hps_constraints = hps_constraints[self.hps_tune_indices]

    def assert_input(self, function, hps_init, hps_constraints):
        # input hpss' boundaries
        assert (
            type(hps_constraints).__module__ == "numpy"
        ), "\n hps_constraints must be numpy array"

        assert hps_constraints.ndim == 2, "hps_constraints must be of dimension 2"

        for i in hps_constraints:
            assert (
                len(i) == 2
            ), "\n boundary for each hps must be a tuple of length two."
            assert (
                i[0] <= i[1]
            ), "\n lower_boundaries must be smaller than upper_boundaries [lower,upper]"

        # check correctness  hps_init wrt boundaries
        assert (
            hps_init.ndim == 1
        ), "\n please define hps_init as an 1d array of shape (..,)"

        # input function
        assert callable(function), "function must be callable"

    def initialize_population(self):
        self.hps_pop = np.zeros((self.pop_s, self.hps_init.shape[0])) + self.hps_init
        self.hps_pop[:, self.hps_tune_indices] = 0

        # new initial Population
        pop = np.zeros((self.pop_s, self.dim + 2))

        p = np.arange(0, self.pop_s - 1)
        i = np.arange(self.dim)
        i, p = np.meshgrid(i, p)

        # pop[p, i] = self.hps_constraints[i, 0] + np.random.random(
        #     size=(p.shape[0], self.dim)
        # ) * (self.hps_constraints[i, 1] - self.hps_constraints[i, 0])

        pop[p, i] = self.hps_constraints[i, 0] + lhs(
            self.dim,
            samples=self.pop_s-1,
            criterion="maximin",
            iterations=20,
        ) * (self.hps_constraints[i, 1] - self.hps_constraints[i, 0])

        # replace worst individual with hps_init, hps_init might differ from pop[0,..]
        pop[-1, : self.dim] = self.hps_init[self.hps_tune_indices]

        return pop

    def reform_pop(self, hps):
        """
        Return the hyperparameters to full form, including the non-tuned hps.
        """
        new_hps = self.hps_pop[: ((hps.ndim - 1) * (hps.shape[0] - 1) + 1), :]
        new_hps[:, self.hps_tune_indices] = hps
        return new_hps

    def hillclimbing(self, individuals):
        sys.stdout.write("\rRunning hillclimbing ...")
        sys.stdout.flush()
        # NOTE sequentially is faster than using multithreading due to overhead.
        for i in range(individuals.shape[0]):
            if individuals[i][self.dim + 1] == 0:  # tune only those not tuned before
                hps0 = individuals[i][: self.dim]
                result = minimize(
                    self.sim, hps0, bounds=self.hps_constraints  # , method="L-BFGS-B"
                )
                individuals[i][: self.dim] = result.x
                individuals[i][self.dim] = result.fun
                individuals[i][self.dim + 1] = result.success
        sys.stdout.write(" " * 100)
        return individuals

    def sim(self, hps):
        """Transform the population back to full form and negate the results to obtain a maximization problem"""
        return -self.f(self.reform_pop(hps))

    def progress(self, count, total, status=""):
        """
        Print the tuning progress.
        This is a relatively expensive function due to stdout.
        """
        if count % (int(total / 100)) == 0:
            bar_len = 50
            filled_len = int(round(bar_len * count / float(total)))

            percents = round(100.0 * count / float(total), 1)
            bar = "|" * filled_len + "_" * (bar_len - filled_len)

            sys.stdout.write("\r%s %s%s %s" % (bar, percents, "%", status))
            sys.stdout.flush()

    def finalize(self, pop):
        # Sort
        pop = pop[pop[:, self.dim].argsort()]

        if pop[0, self.dim] < self.best_function:
            self.best_function = pop[0, self.dim]
            self.best_hps = pop[0, : self.dim]

        # Report
        self.best_hps = self.reform_pop(self.best_hps).ravel()
        self.output_dict = {
            "hps": self.best_hps,  # return the full hp!
            "function": -self.best_function,
        }

        if self.progress_bar == True:
            # reset the bar to an empty line
            sys.stdout.write("\r%s" % (" " * 100))


class MultistartHillclimb(Tuner):
    def __init__(
        self,
        function,
        hps_init,
        hps_constraints,
        convergence_curve=True,  # TODO niet belangrijk hier
        progress_bar=True,
    ):
        super().__init__(function, hps_init, hps_constraints)
        self.pop_s *= 2  # two times as large initial pop as GA based alg.
        self.run()

    def run(self):
        # inits
        pop = self.initialize_population()

        pop = self.hillclimbing(pop)
        pop = pop[pop[:, self.dim].argsort()]

        self.best_hps = pop[0, : self.dim]
        self.best_function = pop[0, self.dim]

        self.finalize(pop)


class GeneticAlgorithm(Tuner):

    """
    Elitist genetic algorithm for solving problems with
    continuous hpss.

    run(): implements the genetic algorithm

    outputs:
            output_dict:  a dictionary including the best set of hpss
        found and the value of the given function associated to it.
        {'hps': , 'function': }

            report: a list including the record of the progress of the
            algorithm over iterations

    """

    def __init__(
        self,
        function,
        hps_init,
        hps_constraints,
        convergence_curve=True,
        progress_bar=True,
    ):
        self.__name__ = "Genetic Algorithm"
        super().__init__(function, hps_init, hps_constraints, progress_bar)

        self.param = {
            "max_num_iteration": None,  # stoping criteria of the genetic algorithm (GA)
            "mutation_probability": 0.1,
            "elit_ratio": 0.05,
            "crossover_probability": 0.3,
            "parents_portion": 0.3,
            "crossover_type": "uniform",  # <string> - Default is 'uniform'; 'one_point' or 'two_point' are other options
            "max_iteration_without_improv": None,
        }

        self.assert_GA_parameters()

        # convergence_curve
        self.convergence_curve = convergence_curve
        if convergence_curve == True:
            plt.close("GA")
            self.fig, self.ax = plt.subplots()
            self.fig.set_label("GA")

        # SET max numbers of iteration
        if self.param["max_num_iteration"] == None:
            self.iterate = 1000
        else:
            self.iterate = int(self.param["max_num_iteration"])

        self.hillclimb_interval = int(self.iterate / 20)  # each 5 percent

        self.stop_mniwi = False
        if self.param["max_iteration_without_improv"] == None:
            if self.hillclimb_interval:
                self.mniwi = 2 * self.hillclimb_interval
            else:
                self.mniwi = self.iterate + 1
        else:
            self.mniwi = int(self.param["max_iteration_without_improv"])

        # if all correct, run
        self.run()

    def run(self):
        # inits
        pop = self.initialize_population()

        # do hillclimbing
        pop = self.hillclimbing(pop)
        pop = pop[pop[:, self.dim].argsort()]

        self.best_hps = pop[0, : self.dim]
        self.best_function = pop[0, self.dim]

        t = 0
        counter = 0
        while t < self.iterate:

            if self.progress_bar == True:
                self.progress(t, self.iterate, status="GA is running...")

            # Normalizing objective function
            minobj = pop[0, self.dim]
            if minobj < 0:
                normobj = pop[:, self.dim] + abs(minobj)
            else:
                normobj = pop[:, self.dim]

            maxnorm = np.amax(normobj)
            normobj = maxnorm - normobj + 1

            # Calculate probability
            sum_normobj = np.sum(normobj)
            prob = normobj / sum_normobj
            cumprob = np.cumsum(prob)

            # select random parents while keeping elitists.
            # works for the elitists because we sorted earlier.
            index = np.searchsorted(cumprob, np.random.random())
            pop[self.num_elit : self.par_s] = pop[index]

            par_count = 0
            while par_count == 0:
                ef_par_list = np.random.random(self.par_s) <= self.prob_cross
                par_count = np.sum(ef_par_list)

            ef_par = pop[: self.par_s][ef_par_list]

            for k in range(self.par_s, self.pop_s, 2):
                r1 = np.random.randint(0, par_count)
                r2 = np.random.randint(0, par_count)
                pvar1 = ef_par[r1, : self.dim]
                pvar2 = ef_par[r2, : self.dim]

                ch1, ch2 = cross(pvar1, pvar2, self.dim, self.c_type)

                ch1 = mut(ch1, self.hps_constraints, self.prob_mut, self.dim)
                ch2 = mutmidle(
                    ch2, pvar1, pvar2, self.hps_constraints, self.prob_mut, self.dim
                )
                pop[k : k + 2, : self.dim] = ch1, ch2
                pop[k : k + 2, self.dim + 1] = 0

            # Fitness of whole population at once.
            if (t + 1) % self.hillclimb_interval == 0:
                # pop[:self.par_s, :] = self.hillclimbing(pop[:self.par_s, :])
                pop = self.hillclimbing(self.rerandomize_population(pop))
                # pop[self.par_s:, :] = self.hillclimbing(pop[self.par_s:, :])
            else:
                pop[self.par_s :, self.dim] = self.sim(pop[self.par_s :, : self.dim])

            # Sort
            pop = pop[pop[:, self.dim].argsort()]
            if pop[0, self.dim] < self.best_function:
                if not np.allclose(pop[0, self.dim], self.best_function):
                    counter = 0
                else:
                    counter += 1
                self.best_function = pop[0, self.dim].copy()
                self.best_hps = pop[0, : self.dim].copy()
            else:
                counter += 1

            if counter > self.mniwi:
                self.stop_mniwi = True
                break

            # Report
            self.report.append(-pop[0, self.dim])
            t += 1

            #############################################################
            #############################################################

        self.finalize(pop)

        if self.stop_mniwi == True:
            sys.stdout.write(
                "\r\tGA terminated due to not improving for a maximum number of iterations.\n"
            )

        if self.convergence_curve == True:
            self.ax.plot(self.report, label="Objective function")
            self.ax.set_xlabel("Iteration")
            self.ax.set_ylabel("log likelihood")
            self.fig.suptitle(self.__name__)
            plt.draw()

    def rerandomize_population(self, pop):
        """With hillclimbing, we will have many points converged to the same solution, i.e. are np.allclose;
        This happens mainly in the elitists/parents. We filter those, and introduce new random samples.
        Multistart in ga form"""
        d = np.arange(self.dim)
        for i in range(1, pop.shape[0]):
            if np.isclose(pop[i], pop[i - 1]).all():
                pop[i, d] = self.hps_constraints[d, 0] + np.random.random(
                    size=(1, self.dim)
                ) * (self.hps_constraints[d, 1] - self.hps_constraints[d, 0])
        return pop

    def assert_GA_parameters(self):
        # input algorithm's parameters
        assert (
            self.param["parents_portion"] <= 1 and self.param["parents_portion"] >= 0
        ), "parents_portion must be in range [0,1]"

        self.par_s = int(self.param["parents_portion"] * self.pop_s)
        if (self.pop_s - self.par_s) % 2 != 0:
            self.par_s += 1

        self.prob_mut = self.param["mutation_probability"]

        assert (
            self.prob_mut <= 1 and self.prob_mut >= 0
        ), "mutation_probability must be in range [0,1]"

        self.prob_cross = self.param["crossover_probability"]
        assert (
            self.prob_cross <= 1 and self.prob_cross >= 0
        ), "mutation_probability must be in range [0,1]"

        assert (
            self.param["elit_ratio"] <= 1 and self.param["elit_ratio"] >= 0
        ), "elit_ratio must be in range [0,1]"

        trl = self.pop_s * self.param["elit_ratio"]
        if trl < 1 and self.param["elit_ratio"] > 0:
            self.num_elit = 1
        else:
            self.num_elit = int(trl)

        assert (
            self.par_s >= self.num_elit
        ), "\n number of parents must be greater than number of elits"

        self.c_type = self.param["crossover_type"]
        assert (
            self.c_type == "uniform"
            or self.c_type == "one_point"
            or self.c_type == "two_point"
        ), "\n crossover_type must 'uniform', 'one_point', or 'two_point' Enter string"


@njit(cache=True, fastmath=True)
def mut(x, hps_constraints, prob_mut, dim):
    for i in range(dim):
        ran = np.random.random()
        if ran < prob_mut:
            x[i] = hps_constraints[i, 0] + np.random.random() * (
                hps_constraints[i, 1] - hps_constraints[i, 0]
            )
    return x


@njit(cache=True, fastmath=True)
def mutmidle(x, p1, p2, hps_constraints, prob_mut, dim):
    for i in range(dim):
        ran = np.random.random()
        if ran < prob_mut:
            if p1[i] < p2[i]:
                x[i] = p1[i] + np.random.random() * (p2[i] - p1[i])
            elif p1[i] > p2[i]:
                x[i] = p2[i] + np.random.random() * (p1[i] - p2[i])
            else:
                x[i] = hps_constraints[i, 0] + np.random.random() * (
                    hps_constraints[i, 1] - hps_constraints[i, 0]
                )
    return x


@njit(cache=True, fastmath=True)
def cross(x, y, dim, c_type):

    ofs1 = x.copy()
    ofs2 = y.copy()

    if c_type == "one_point":
        ran = np.random.randint(0, dim)
        for i in range(0, ran):
            ofs1[i] = y[i]
            ofs2[i] = x[i]

    if c_type == "two_point":
        ran1 = np.random.randint(0, dim)
        ran2 = np.random.randint(ran1, dim)

        for i in range(ran1, ran2):
            ofs1[i] = y[i]
            ofs2[i] = x[i]

    if c_type == "uniform":
        for i in range(0, dim):
            ran = np.random.random()
            if ran < 0.5:
                ofs1[i] = y[i]
                ofs2[i] = x[i]

    return ofs1, ofs2
