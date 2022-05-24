"""
Classes corresponding to external solvers will be defined in this file.
The external solvers are required to:
1) be callable from the command line by keyword (use argparse i.e. --in1 1.2 --in2 82.34 --... )
    - a list of input parameters to be provided here in its corresponding class
2) provide output to a file in json format
    - a list of output parameters to be provided here in its corresponding class
    - these parameters can be derived quantities, as a result of postprocessing in this module
3) accept an output path when being called from the command line

"""

from sampling.solvers.internal import Solver
from sampling.solvers.NURBS import *
from utils.filter_utils import filter_spiked_signal

from abc import ABC, abstractmethod
import traceback

import subprocess
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
import matplotlib.pyplot as plt

from itertools import groupby
import os
import shutil
import shlex
import time
import pandas as pd
import glob


class ExternalSolver(Solver):
    def __init__(self):
        # !! system might still use python3 when python 2 is present, or mac/linux
        try:
            subprocess.run(
                "python3 --version",
                check=True,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            self.python_string = "python3 "
        except:
            subprocess.run(
                "python --version",
                check=True,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            self.python_string = "python "

    @property
    def solver_path(self):
        raise NotImplementedError

    def get_output_path(self, x):
        """
        function that creates the output filepath dependent on the inputs"""
        pass

    def process_outputs(self):
        """Process the outputs received by the external solver to comply with the defines output parameters of this class. \n\
            For instance: the solver provides a time trace of the pressure while we are interested in the maximum pressure"""
        pass


def run_cmd(cmd, output_path):
    """Function that makes a cmd call to the external solver, returns the process and output log file."""

    fname = os.path.join(output_path, "output.log")
    f = open(fname, "w+")

    cmd_argslist = shlex.split(cmd)
    p = subprocess.Popen(cmd_argslist, stdout=f, stderr=subprocess.PIPE)

    return p, f


class EVA(ExternalSolver):
    """
    EVA accepts the arguments:
      resultpath: RELATIVE path to cwd.
      grid_refinement: float scaling the grid density.
    """

    # input_parameter_list = ["x0", "x1"]
    output_parameter_list = ["p"]
    min_d = 2
    max_d = 2

    solver_path = r'"C:\Users\RobertWenink\OneDrive - Delft University of Technology\Documents\TUDelft\Master\Afstuderen\IRA\EVA\EVA\main.py"'

    def __init__(self, setup=None):
        super().__init__()

        from sampling.solvers.NURBS import interpolating_curve, create_nurbs

        if setup != None:
            # then we are not setting up a virtual class for the GUI anymore.ABC()
            
            self.filter_span_frac, self.fitler_delta = 0.01, 0.04
            
            assert os.path.exists(
                "input.tex"
            ), "Put an input.tex corresponding to EVA in your working directory.\n"

            # create base identifier of the optimization run depending on name provided in the input.tex of EVA.
            optimization_id = []
            with open("input.tex") as f_data:
                for k, g in groupby(f_data, lambda x: not x.startswith(("'"))):
                    if not k:
                        optimization_id.append(
                            np.array(
                                [
                                    [str(x) for x in d.split()]
                                    for d in g
                                    if len(d.strip())
                                ]
                            )
                        )

            optimization_id = np.asarray(optimization_id[0])
            self.optimization_id = optimization_id[0, 1]

            self.results_path = os.path.normpath(
                ".\{}_Results".format(setup.solver_str)
            )

            # Check whether the specified path exists or not
            self.base_path = os.path.join(self.results_path, self.optimization_id)

            if not os.path.exists(self.base_path):
                # Create a new directory because it does not exist
                os.makedirs(self.base_path)

    def get_output_path(self, x, refinement):
        """
        function that creates the output filepath dependent on the inputs
        """
        # create path for results folder, dependent on:
        # 1) name provided in input.tex
        # 2) variable values
        # complete path structure will be: results_path/optimization_id/run_id
        run_id = "_".join(format(i, ".4f") for i in x)
        run_id = run_id.replace(".", "")

        output_path = os.path.join(self.base_path, "{:.1f}".format(refinement), run_id)
        return output_path, run_id

    def read_results(self, output_path):
        """Reads the results of EVA.
        @returns: 
        - objective function: max acceleration
        - cost of sampling
        - dataframe of the time series data """
        data = pd.read_csv(
            os.path.join(output_path, "Excel", "Body_forces", "Force_body_y.csv")
        )
        data = data.iloc[1: , :]
        # times = data.iloc[1:, 0]  # t
        # forces = data.iloc[1:, 1]  # N/m bcs 2D -> 3D

        # apply moving average to reduce numerical noise
        data.iloc[: , 1] = filter_spiked_signal(data, self.filter_span_frac, self.fitler_delta)

        # NOTE HARDCODED but constant conversion so of no influence to optimization
        # NOTE mass = 1 means that we are just assessing the forces.
        mass = 1  # kg/m

        acc_tt = data.iloc[: , 1] / mass  # this is the full timetrace of acceleration
        max_acc = np.max(acc_tt)

        # time found in file with name like simulation_time_8.18.txt
        p_string = output_path + "\simulation_time*.txt"

        # NOTE should always exist if finished succesfully, else this returns an error!
        p = glob.glob(p_string)[0]
        f = open(p)

        cost_in_seconds = float(
            f.readlines()[1]
        )  # seconds are on second line, without text

        return max_acc, cost_in_seconds, data.to_numpy()

    def inspect_results(self, output_path):
        plt.close("body_forces")

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(plt.imread(os.path.join(output_path, "NURBS.png")))

        path = os.path.join(output_path, "Excel", "Body_forces", "Force_body_y.csv")
        df = pd.read_csv(path)
        df = df.iloc[1: , :]
        x = df.iloc[:, 0]

        ax2.plot(x, df.iloc[:, 1], label="original")

        # moving averages
        ax2.plot(
            x,
            df.rolling(5, min_periods=1).mean().iloc[:, 1],
            label="Simple Moving average over 5 timesteps",
        )
        ax2.plot(
            x,
            df.ewm(alpha=0.15, adjust=False).mean().iloc[:, 1],
            label="Exponential Moving Average, alpha = 0.3",
        )
        ax2.plot(
            x,
            df.rolling(10, min_periods=1).mean().iloc[:, 1],
            label="Simple Moving average over 10 timesteps",
        )
        ax2.plot(
            x,
            filter_spiked_signal(df , self.filter_span_frac, self.fitler_delta),
            label="filter_spiked_signal",
        )
        # set non overlapping ticks and legend
        ax2.set_xticks(x[:: int(len(x) / 10)])
        ax2.set_xticklabels(ax2.get_xticks(), rotation=45, ha="right")
        ax2.legend()

        fig.set_label("body_forces")
        fig.suptitle(os.path.split(output_path)[-1])

    def solve(self, X, refinement=None, get_time_trace=False):

        start_time = time.time()

        # select level if no refinement defined
        if refinement == None:
            l = 2
            refinements = [1, 1.5, 2, 3, 4]
            refinement = refinements[l]

        # copy unadapted input.tex for future reference
        if not os.path.exists(self.base_path):
            shutil.copyfile("input.tex", os.path.join(self.base_path, "input_copy.tex"))

        # setup multithreading bcs we dont want to overload system, and have more uniform running times between batches and single runs.
        num = min(
            int(cpu_count() / 2), X.shape[0]
        )  # defaults to the cpu count of machine otherwise
        # num = 1
        tp = ThreadPool(num)

        counter = 0
        batch_size = X.shape[0]

        def worker(cmd, output_path, run_id):
            p, f = run_cmd(cmd, output_path)
            print("Started run {} with pid {}".format(run_id, p.pid))

            # p.communicate() implies waiting for the process to finish!
            _, error = p.communicate()

            if error:
                print("ret> ", p.returncode)
                print("Error> error ", error.strip())
                print("Unsuccesfully finished run {} with pid {}".format(run_id, p.pid))
            else:
                print("Succesfully finished run {} with pid {}".format(run_id, p.pid))
                counter += 1

            print(
                "Succesfully completed {} out of {} runs.".format(counter, batch_size)
            )

        # retrieve EVA solution, this should be done in parallel in the case of an initial DoE.
        for x in X:
            output_path, run_id = self.get_output_path(x, refinement)
            p_string = output_path + "\simulation_time*.txt"
            p = glob.glob(p_string)
            if len(p) == 0:
                # NOTE do not run if already been run before and completed
                # NOTE we base this on the filepath. Thus, we create bands of the same value/ seemingly input related noise depending on formatting .4f!!!

                # remove all contents of the folder to avoid conflicts with previously unfinished runs
                files = glob.glob(output_path + "/*")

                if len(files) > 0:
                    print("WARNING: removing files of a previous unsuccesful run.")
                    shutil.rmtree(output_path)
                    if os.path.exists(output_path + ".png"):
                        os.remove(output_path + ".png")

                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                # create the NURBS parameterized figure
                interpolating_curve(x, output_path)
                shutil.copyfile(
                    output_path + ".png", os.path.join(output_path, "NURBS.png")
                )
                shutil.copyfile(
                    os.path.normpath(".\Reconstruct_plot.py"),
                    os.path.join(output_path, "Reconstruct_plot.py"),
                )

                # create command and call
                cmd = self.python_string + '{} "{}" {}'.format(
                    EVA.solver_path, output_path, refinement
                )
                tp.apply_async(worker, (cmd, output_path, run_id))

        # https://stackoverflow.com/questions/26774781/python-multiple-subprocess-with-a-pool-queue-recover-output-as-soon-as-one-finis
        tp.close()
        tp.join()

        # Gathering results. Do this after all processes have been concluded to maintain exact order!!
        Z = []
        costs = []
        acc_tt_list = []

        for x in X:
            output_path, run_id = self.get_output_path(x, refinement)
            z, cost, acc_tt = self.read_results(output_path)

            Z.append(z)
            costs.append(cost)
            acc_tt_list.append(acc_tt)

        print(
            "--- Finished sampling batch in {:.2f} minutes ---".format(
                (time.time() - start_time) / 60
            )
        )

        # inspect best sample from batch.
        self.inspect_results(self.get_output_path(X[np.argmin(Z)], refinement)[0])

        if get_time_trace:
            return np.array(Z), np.sum(costs), acc_tt_list

        return np.array(Z), np.sum(costs)