from geomdl import BSpline, NURBS
from geomdl.visualization import VisMPL
from geomdl import knotvector

import numpy as np
import matplotlib.pyplot as plt
from sampling.DoE import grid_unit_hypercube

# ship parameters
B = 1
h = 1


def create_nurbs(x0, x1, x2, x3):
    """
    Create a nurbs shape variation for a combination of inputs and create a figure from it.
    @param x0,x1,x2,x3 : between 0 and 1
    
    The following control point parameterization creates smooth continuous, one-to-one variations:
    0.4 <= x0 <= 1.0
    0.0 <= x1, x2 <= 1.0
    0.6 <= x3 <= 0.95
    """

    # Create the curve instance
    crv = NURBS.Curve() # NURBS = generalisatie van b-splines, geeft hetzelfde resultaat.
    
    # Set degree
    crv.degree = 3
        
    # Scale x0 from [0, 1] -> [0.4, 1.0]
    x0 = x0 * (1 - 0.8) + 0.8
    x1 = x1 * (1 - 0.1) + 0.1
    x2 = x2 * (1 - 0.1) + 0.1
    x3 = x3 * (0.95 - 0.6) + 0.6
    


    # Set control points
    crv.ctrlpts = [[0, 1], [0, x0], [x1, x2], [x3, 0], [1, 0]]
    
    # Set knot vector
    crv.knotvector = knotvector.generate(crv.degree, crv.ctrlpts_size)
    
    # # # Set the visualization component of the curve and render
    # crv.vis = VisMPL.VisCurve2D()
    # crv.render()
    
    def plot_filled_curve(crv):
        # Prepare points
        evalpts = np.array(crv.evalpts)
    
        # Plot points together on the same graph
        fig = plt.figure(figsize=(2*B, 2*h))
        plt.axis('off')
        plt.fill_between(evalpts[:, 0] * B, evalpts[:, 1] * h,1)
    
        # to remove all plotting borders
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
        # save figure
        name = "{:.3f}_{:.3f}_{:.3f}_{:.3f}".format(x0, x1, x2, x3)
        name = name.replace(".", "")
        plt.savefig("./NURBS/{}.png".format(name), bbox_inches = 'tight', pad_inches = 0, transparent=True)
        # plt.show()
        plt.close()
    
    plot_filled_curve(crv)

# create_nurbs(1,0,1,0.95)
# create_nurbs(1,1,0,0.95)    
# create_nurbs(0.5,0.5,0.5,0.65)    

X = grid_unit_hypercube(4, n_per_d = 3) 
for x in X:
    create_nurbs(*x)
