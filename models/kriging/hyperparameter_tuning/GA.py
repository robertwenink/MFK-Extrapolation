""" Adapted from: Copyright at end of file """
from numba import njit
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

class geneticalgorithm:

    """
    An implementation of elitist genetic algorithm for solving problems with
    continuous, integers, or mixed variables.

    Implementation and output:
        methods:
                run(): implements the genetic algorithm
        outputs:
                output_dict:  a dictionary including the best set of variables
            found and the value of the given function associated to it.
            {'variable': , 'function': }

                report: a list including the record of the progress of the
                algorithm over iterations

    """

    def __init__(
        self,
        function,
        other_function_arguments,
        dimension,
        variable_boundaries,
        algorithm_parameters={
            "max_num_iteration": 2001,
            "population_size": 100,
            "mutation_probability": 0.1,
            "elit_ratio": 0.05,
            "crossover_probability": 0.5,
            "parents_portion": 0.3,
            "crossover_type": "uniform",
            "max_iteration_without_improv": None,
        },
        convergence_curve=True,
        progress_bar=True,
    ):
        """
        @param function <Callable> - the given objective function to be minimized
        NOTE: This implementation minimizes the given objective function.
        (For maximization multiply function by a negative sign: the absolute
        value of the output would be the actual objective function)

        @param dimension <integer> - the number of decision variables

        @param variable_boundaries <numpy array/None> - provide an array of tuples
        of length two as boundaries for each variable; the length of the array must be equal dimension. 
        For example, np.array([0,100],[0,200]) determines lower boundary 0 and upper boundary 100 for first variable.

        @param algorithm_parameters:
            @ max_num_iteration <int> - stoping criteria of the genetic algorithm (GA)
            @ population_size <int>
            @ mutation_probability <float in [0,1]>
            @ elit_ration <float in [0,1]>
            @ crossover_probability <float in [0,1]>
            @ parents_portion <float in [0,1]>
            @ crossover_type <string> - Default is 'uniform'; 'one_point' or
            'two_point' are other options
            @ max_iteration_without_improv <int> - maximum number of
            successive iterations without improvement. If None it is ineffective

        @param convergence_curve <True/False> - Plot the convergence curve or not
        Default is True.
        @progress_bar <True/False> - Show progress bar or not. Default is True.

        """
        self.__name__ = geneticalgorithm

        # input function
        assert callable(function), "function must be callable"
        self.f = function

        # NOTE own additions
        self.other_function_arguments = other_function_arguments
        self.hps_shape = variable_boundaries.shape
        variable_boundaries = variable_boundaries.reshape(-1,2)

        self.dim = int(dimension)

        # input variables' boundaries
        assert (
            type(variable_boundaries).__module__ == "numpy"
        ), "\n variable_boundaries must be numpy array"

        assert (
            len(variable_boundaries) == self.dim
        ), "\n variable_boundaries must have a length equal dimension"

        for i in variable_boundaries:
            assert (
                len(i) == 2
            ), "\n boundary for each variable must be a tuple of length two."
            assert (
                i[0] <= i[1]
            ), "\n lower_boundaries must be smaller than upper_boundaries [lower,upper]"
        self.var_bound = variable_boundaries
        
        # convergence_curve
        self.convergence_curve = convergence_curve

        # progress_bar
        self.progress_bar = progress_bar

        # input algorithm's parameters
        self.param = algorithm_parameters
        self.pop_s = int(self.param["population_size"])

        assert (
            self.param["parents_portion"] <= 1 and self.param["parents_portion"] >= 0
        ), "parents_portion must be in range [0,1]"

        self.par_s = int(self.param["parents_portion"] * self.pop_s)
        trl = self.pop_s - self.par_s
        if trl % 2 != 0:
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

        if self.param["max_num_iteration"] == None:
            self.iterate = 0
            for i in range(0, self.dim):
                self.iterate += (
                    (self.var_bound[i][1] - self.var_bound[i][0])
                    * 50
                    * (100 / self.pop_s)
                )
            self.iterate = int(self.iterate)
            if (self.iterate * self.pop_s) > 10000000:
                self.iterate = 10000000 / self.pop_s
        else:
            self.iterate = int(self.param["max_num_iteration"])

        self.c_type = self.param["crossover_type"]
        assert (
            self.c_type == "uniform"
            or self.c_type == "one_point"
            or self.c_type == "two_point"
        ), "\n crossover_type must 'uniform', 'one_point', or 'two_point' Enter string"

        self.stop_mniwi = False
        if self.param["max_iteration_without_improv"] == None:
            self.mniwi = self.iterate + 1
        else:
            self.mniwi = int(self.param["max_iteration_without_improv"])


    def run(self):
        # Initial Population
        pop = np.array([np.zeros(self.dim + 1)] * self.pop_s)
        solo = np.zeros(self.dim + 1)

        for p in range(0, self.pop_s):
            i = np.arange(self.dim)
            solo[i] = self.var_bound[i,0] + np.random.random(size = self.dim) * (
                self.var_bound[i,1] - self.var_bound[i,0]
            )

            pop[p] = solo

        # Fitness of whole population at once.
        pop[:,self.dim] = self.sim(pop[:,:-1])

        # Report
        obj = pop[0,-1]
        self.report = []
        self.test_obj = obj
        self.best_variable = solo[:-1]
        self.best_function = obj

        t = 1
        counter = 0
        while t <= self.iterate:

            if self.progress_bar == True:
                self.progress(t, self.iterate, status="GA is running...")

            # Sort
            pop = pop[pop[:, self.dim].argsort()]

            if pop[0, self.dim] < self.best_function:
                counter = 0
                self.best_function = pop[0, self.dim].copy()
                self.best_variable = pop[0, : self.dim].copy()
            else:
                counter += 1

            # Report
            self.report.append(-pop[0, self.dim])

            # Normalizing objective function
            normobj = np.zeros(self.pop_s)

            minobj = pop[0, self.dim]
            if minobj < 0:
                normobj = pop[:, self.dim] + abs(minobj)
            else:
                normobj = pop[:, self.dim].copy()

            maxnorm = np.amax(normobj)
            normobj = maxnorm - normobj + 1

            # Calculate probability
            sum_normobj = np.sum(normobj)
            prob = np.zeros(self.pop_s)
            prob = normobj / sum_normobj
            cumprob = np.cumsum(prob)

            # Select parents
            par = np.array([np.zeros(self.dim + 1)] * self.par_s)

            for k in range(0, self.num_elit):
                par[k] = pop[k].copy()
            for k in range(self.num_elit, self.par_s):
                index = np.searchsorted(cumprob, np.random.random())
                par[k] = pop[index].copy()

            ef_par_list = np.array([False] * self.par_s)
            par_count = 0
            while par_count == 0:
                for k in range(0, self.par_s):
                    if np.random.random() <= self.prob_cross:
                        ef_par_list[k] = True
                        par_count += 1

            ef_par = par[ef_par_list].copy()

            # New generation
            pop = np.array([np.zeros(self.dim + 1)] * self.pop_s)

            for k in range(0, self.par_s):
                pop[k] = par[k].copy()

            for k in range(self.par_s, self.pop_s, 2):
                r1 = np.random.randint(0, par_count)
                r2 = np.random.randint(0, par_count)
                pvar1 = ef_par[r1, :self.dim].copy()
                pvar2 = ef_par[r2, :self.dim].copy()

                ch1,ch2 = cross(pvar1, pvar2, self.dim, self.c_type)

                ch1 = mut(ch1, self.var_bound, self.prob_mut, self.dim)
                ch2 = mutmidle(ch2, pvar1, pvar2, self.var_bound, self.prob_mut, self.dim)
                solo[: self.dim] = ch1.copy()
                pop[k] = solo.copy()
                solo[: self.dim] = ch2.copy()
                pop[k + 1] = solo.copy()
            
            # Fitness of whole population at once.
            pop[self.par_s:,self.dim] = self.sim(pop[self.par_s:,:-1])
                
            #############################################################
            t += 1
            if counter > self.mniwi:
                pop = pop[pop[:, self.dim].argsort()]
                if pop[0, self.dim] >= self.best_function:
                    t = self.iterate
                    if self.progress_bar == True:
                        self.progress(t, self.iterate, status="GA is running...")
                    t += 1
                    self.stop_mniwi = True

        # Sort
        pop = pop[pop[:, self.dim].argsort()]

        if pop[0, self.dim] < self.best_function:

            self.best_function = pop[0, self.dim].copy()
            self.best_variable = pop[0, : self.dim].copy()
        
        # Report
        self.report.append(-pop[0, self.dim])
        self.output_dict = {
            "variable": self.best_variable.reshape(self.hps_shape[:-1]),
            "function": -self.best_function,
        }
        if self.progress_bar == True:
            show = " " * 100
            sys.stdout.write("\r%s" % (show))
        sys.stdout.write("\r The best solution found:\n %s" % (self.best_variable))
        sys.stdout.write("\n\n Objective function:\n %s\n" % (-self.best_function))
        sys.stdout.flush()
        re = np.array(self.report)
        if self.convergence_curve == True:
            plt.plot(re)
            plt.xlabel("Iteration")
            plt.ylabel("Objective function")
            plt.title("Genetic Algorithm")
            plt.draw()

        if self.stop_mniwi == True:
            sys.stdout.write(
                "\nGA is terminated due to the"
                + " maximum number of iterations without improvement was met!\n"
            )
    

    def sim(self, X):
        temp = X.reshape((-1,*self.hps_shape[:-1]))
        obj = -self.f(temp, *self.other_function_arguments)
        return obj

    def progress(self, count, total, status=""):
        """
        This is a relatively expensive function due to stdout.
        """
        if  count % (int(total/100))==0:
            bar_len = 50
            filled_len = int(round(bar_len * count / float(total)))

            percents = round(100.0 * count / float(total), 1)
            bar = "|" * filled_len + "_" * (bar_len - filled_len)

            sys.stdout.write("\r%s %s%s %s" % (bar, percents, "%", status))
            sys.stdout.flush()

@njit(cache=True, fastmath=True)
def mut(x,var_bound,prob_mut,dim):
    for i in range(dim):
        ran = np.random.random()
        if ran < prob_mut:
            x[i] = var_bound[i,0] + np.random.random() * (
                var_bound[i,1] - var_bound[i,0]
            )
    return x

@njit(cache=True, fastmath=True)
def mutmidle(x, p1, p2,var_bound,prob_mut,dim):
    for i in range(dim):
        ran = np.random.random()
        if ran < prob_mut:
            if p1[i] < p2[i]:
                x[i] = p1[i] + np.random.random() * (p2[i] - p1[i])
            elif p1[i] > p2[i]:
                x[i] = p2[i] + np.random.random() * (p1[i] - p2[i])
            else:
                x[i] = var_bound[i,0] + np.random.random() * (
                    var_bound[i,1] - var_bound[i,0]
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

"""

Copyright 2020 Ryan (Mohammad) Solgi

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to use, 
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
Software, and to permit persons to whom the Software is furnished to do so, 
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.

"""
