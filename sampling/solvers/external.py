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

from abc import ABC, abstractmethod
import traceback
import subprocess
from itertools import groupby
import os
from shutil import copyfile
import shlex
import time

class ExternalSolver(Solver):

    def __init__(self):
        # !! system might still use python3 when python 2 is present, or mac/linux
        try:
            subprocess.run("python3 --version", check = True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            self.pstring = "python3 "
        except:
            subprocess.run("python --version", check = True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            self.pstring = "python "

    @property
    def solver_path(self):
        raise NotImplementedError

    def run_cmd(self, cmd, output_path):
        """Function that makes a cmd call to the external solver, returns the process and output log file."""

        fname = os.path.join(output_path, "output.log")
        f = open(fname,'w+')

        cmd_argslist = shlex.split(self.pstring + cmd)
        p = subprocess.Popen(cmd_argslist,stdout=f,stderr=subprocess.PIPE)
        
        return p, f
            
    def create_output_path(self):
        """
        function that creates the output filepath dependent on the inputs"""
        pass

    def process_outputs(self):
        """Process the outputs received by the external solver to comply with the defines output parameters of this class. \n\
            For instance: the solver provides a time trace of the pressure while we are interested in the maximum pressure"""
        pass


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
                                [[str(x) for x in d.split()] for d in g if len(d.strip())]
                            )
                        )

            optimization_id = np.asarray(optimization_id[0])
            self.optimization_id = optimization_id[0, 1]

            self.results_path = os.path.normpath(".\{}_Results".format(setup.solver_str))

            # Check whether the specified path exists or not
            self.base_path = os.path.join(self.results_path, self.optimization_id)

            if not os.path.exists(self.base_path):
                # Create a new directory because it does not exist 
                os.makedirs(self.base_path)

    def read_results(self,path):
        return np.random.rand(),np.random.rand()

    def solve(self, X):
        Z = []
        costs = []
        processes = []

        refinement = 1

        # copy unadapted input.tex for future reference
        copyfile("input.tex", os.path.join(self.base_path,"input.tex"))
        
        for x in X:
            # create path for results folder, dependent on:
            # 1) name provided in input.tex
            # 2) variable values
            # complete path structure will be: results_path/optimization_id/run_id
            run_id = "_".join(format(i, ".4f") for i in x)
            run_id = run_id.replace(".", "")

            output_path = os.path.join(self.base_path, "{:.1f}".format(refinement), run_id)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            # create the NURBS parameterized figure
            interpolating_curve(x, output_path)
            copyfile(output_path+".png", os.path.join(output_path,"NURBS.png"))

            # retrieve EVA solution, this should be done in parallel in the case of an initial DoE.
            p = os.path.join(output_path,"Data")
            if not os.path.exists(p): 
                # NOTE do not run if already been run before, then there would be a Data folder!!
                # NOTE we base this on the filepath. Thus, we create bands of the same value/ seemingly input related noise depending on formatting .4f!!!
        
                cmd = '{} "{}" {}'.format(EVA.solver_path, output_path, refinement)
                p,f = self.run_cmd(cmd, output_path)
                processes.append((p,f))
                print("Started run {} with pid {}".format(run_id,p.pid))

        for p, f in processes:
            # p.communicate() implies waiting for the process to finish!
            output,error = p.communicate()
            # if output: # output is written to file.
            #     print("ret> ",p.returncode)
            #     print("OK> output ",output)
            if error:
                print("ret> ",p.returncode)
                print("Error> error ",error.strip())
            f.close()
            print("Finished process {}".format(p.pid))
            
            z, cost = self.read_results(output_path)

            Z.append(z)
            costs.append(cost)
                
        return np.array(Z), np.array(costs)
