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

from abc import ABC, abstractmethod

class ExternalSolver(Solver):
    input_parameter_list = ['a', 'b', 'c']
    output_parameter_list = ['z']

    min_d = 1
    max_d = len(input_parameter_list)


    def solve(self):
        pass

    def run_cmd(self):
        """Function that makes a cmd call to the external solver"""
        pass

    def create_output_filename(self):
        """function that creates the output filename dependent on the inputs"""
        pass

    def process_outputs(self):
        """Process the outputs received by the external solver to comply with the defines output parameters of this class.
        for instance: the solver provides a time trace of the pressure while we are interested in the maximum pressure"""


    
    
