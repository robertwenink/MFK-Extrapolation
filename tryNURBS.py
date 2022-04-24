from sampling.DoE import grid_unit_hypercube
from sampling.solvers.NURBS import *

X = grid_unit_hypercube(2, n_per_d = 16) 
for x in X:
    # for i in range(4-len(x)):
    #     x = np.append(x,0) 
    # x[3] = x[1]
    # create_nurbs(x)
    interpolating_curve(x)