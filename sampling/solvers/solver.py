from abc import ABC
from abc import abstractmethod

import sampling.solvers.external as external
import sampling.solvers.internal as internal
from sampling.solvers.multifidelity import get_mf_solver

import inspect

CLSMEMBERS = inspect.getmembers(external, inspect.isclass)
CLSMEMBERS += inspect.getmembers(internal, inspect.isclass)


def get_solver_name_list():
    name_list = []
    for name, obj in CLSMEMBERS:
        if name not in ["Solver", "TestFunction", "ABC", "ExternalSolver"]:
            if "external" in str(obj):
                name_list.append("external: " + name)
            else:
                name_list.append("internal: " + name)
    return name_list


def get_solver(setup=None, name=None):
    """
    Retrieve the solver corresponding to the input file name
    """
    if setup is not None:
        solver_name = setup.solver_str
    elif name is not None:
        solver_name = name
    else:
        raise Exception("solver_name not defined")

    
    # not efficient but small list
    for name, obj in CLSMEMBERS:
        if name == solver_name:
            solver = obj
    
    if setup is not None and setup.mf:
        return get_mf_solver(setup,solver)

    return solver()
