from preprocessing.input import Input
from sampling.initial import grid
from sampling.solvers.solver import get_solver

setup = Input(2)
setup.X = grid.generate(setup, 3)


test = get_solver(setup)()
test.plot2d(setup.X)


if setup.SAVE_DATA:
    "update the input file explicitly by setting the corresponding value of data_dict."
    setup.data_dict['X'] = setup.X.tolist()
    setup.create_input_file()
