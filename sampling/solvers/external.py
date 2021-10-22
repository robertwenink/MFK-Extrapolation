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
