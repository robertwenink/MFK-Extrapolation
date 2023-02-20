from geomdl import BSpline, NURBS
from geomdl.visualization import VisMPL
from geomdl import knotvector
from geomdl import fitting

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# ship parameters
make_meters = np.array(True)
B = 3.4/2 * (10 - ~make_meters * 9) # half width; in deci-meters
D = 4.3 * (10 - ~make_meters * 9)
h = D/2
vrijboord = 0.5 * (10 - ~make_meters * 9)
angle_ded = 10

def create_curve_figure(crv, x=None, path=None):
    # Prepare points
    evalpts = np.array(crv.evalpts)

    # Plot points together on the same graph
    mydpi = 100
    fig = plt.figure(figsize=(B, h + vrijboord), dpi = mydpi)

    plt.fill_between(
        evalpts[:, 0] * B, evalpts[:, 1] * h, h + vrijboord, color="#1f77b4"
    )
    # plt.fill_between(
    #     2 * B - evalpts[:, 0] * B, evalpts[:, 1] * h, h + vrijboord, color="#1f77b4"
    # )
    # to remove all plotting borders

    plt.axis("off")
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator()) # type: ignore    
    plt.gca().yaxis.set_major_locator(plt.NullLocator()) # type: ignore    
    
    def get_concat_h(im1, im2):
        dst = Image.new('RGBA', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst
    
    # save figure
    if path == None:
        assert np.any(x != None), "\nPlease provide an input vector x if no path is given."
        name = "_".join(format(i, ".3f") for i in x)
        name = name.replace(".", "")
        path = "./NURBS/{}".format(name)

    plt.savefig(path+".png", bbox_inches="tight", pad_inches=0, transparent=True)        
    plt.close()

    
    # reopen and mirror to get perfect mirror, matplotlib has overlap at centre point otherwise!
    img = Image.open(path+".png")
    mirror_img = ImageOps.mirror(img)
    get_concat_h(img, mirror_img).save(path+".png")

def interpolating_curve(x, path=None):
    # Scale variables
    assert x.shape[0] == 2
    x0, x3 = x

    # define two diagonal lines to fix the angle at the end
    # if we dont do this we might 'sway' outside the breadth
    #NOTE this is the bilge corner
    r0, r1 = 0.01, 0.02
    min_ang = 0 / 180 * np.pi # 0 is recht naar beneden!
    max_ang = (90-angle_ded) / 180 * np.pi 
    max_diff = max_ang - min_ang
    x0_x0 = np.sin(x0 * max_diff + min_ang) * r0
    x0_y0 = 1 - np.cos(x0 * max_diff + min_ang) * r0
    # x0_x1 = np.sin(x0 * 1/2 *np.pi) * r1
    # x0_y1 = 1 - np.cos(x0 * 1/2 *np.pi) * r1

    # diagonal
    # x1_x = x1 * (0.7 - 0.3) + 0.3
    # x1_y = x1 * (0.7 - 0.3) + 0.3

    # square
    # x1 = x1 * (0.9 - 0.5) + 0.5
    # x2 = x2 * (0.6 - 0.2) + 0.2

    #NOTE this is the keel corner
    r0, r1 = 0.02, 0.01
    min_ang = angle_ded / 180 * np.pi  # 0 is naar links horizontaal
    max_ang = (90-angle_ded) / 180 * np.pi
    max_diff = max_ang - min_ang
    # x3_x0 = 1 - np.cos(x3 * max_diff + min_ang) * r0
    # x3_y0 = np.sin(x3 * max_diff + min_ang) * r0
    x3_x1 = 1 - np.cos(x3 * max_diff + min_ang) * r1
    x3_y1 = np.sin(x3 * max_diff + min_ang) * r1

    # Set control points
    # points = [[0, 1], [x0_x0, x0_y0], [x0_x1, x0_y1], [x3_x0, x3_y0], [x3_x1, x3_y1], [1, 0]]
    points = [[0, 1], [x0_x0, x0_y0], [x3_x1, x3_y1], [1, 0]]
    degree = 3  # cubic curve

    # Do global curve interpolation
    curve = fitting.interpolate_curve(points, degree)

    # Plot the interpolated curve
    # curve.delta = 0.01
    # curve.vis = VisMPL.VisCurve2D()
    # curve.render()

    create_curve_figure(curve, x, path)


def create_nurbs(x, path=None):
    """
    Create a nurbs shape variation for a combination of inputs and create a figure from it.
    @param x0,x1,x2,x3 : between 0 and 1

    The following control point parameterization creates smooth continuous, one-to-one variations:
    0.4 <= x0 <= 1.0
    0.0 <= x1, x2 <= 1.0
    0.6 <= x3 <= 0.95
    """
    x0, x1, x2, x3, *_ = x

    # Create the curve instance
    crv = (
        NURBS.Curve()
    )  # NURBS = generalisatie van b-splines, geeft hetzelfde resultaat.

    # Set degree
    crv.degree = 3

    # define two diagonal lines to fix the angle at the end
    # if we dont do this we might 'sway' outside the breadth
    r0, r1 = 0.01, 0.02
    x0_x0 = np.sin(x0 * 1 / 2 * np.pi) * r0
    x0_y0 = 1 - np.cos(x0 * 1 / 2 * np.pi) * r0
    x0_x1 = np.sin(x0 * 1 / 2 * np.pi) * r1
    x0_y1 = 1 - np.cos(x0 * 1 / 2 * np.pi) * r1

    # diagonal
    x1_x = x1 * (0.7 - 0.3) + 0.3
    x1_y = x1 * (0.7 - 0.3) + 0.3

    # # square
    up = 0.8
    low = 0.2
    x1 = x1 * (up - low) + low
    x2 = x2 * (up - low) + low

    # lower corner, angle constrained between 0 and 70 deg
    r0, r1 = 0.02, 0.01
    max_ang = 70 / 180 * np.pi
    min_ang = 0 / 180 * np.pi
    max_diff = max_ang - min_ang
    x3_x0 = 1 - np.cos(x3 * max_diff + min_ang) * r0
    x3_y0 = np.sin(x3 * max_diff + min_ang) * r0
    x3_x1 = 1 - np.cos(x3 * max_diff + min_ang) * r1
    x3_y1 = np.sin(x3 * max_diff + min_ang) * r1

    # Set control points
    crv.ctrlpts = [
        [0, 1],
        [x0_x0, x0_y0],
        [x0_x1, x0_y1],
        [x3_x0, x3_y0],
        [x3_x1, x3_y1],
        [1, 0],
    ]

    # Set knot vector
    crv.knotvector = knotvector.generate(crv.degree, crv.ctrlpts_size)

    # # # Set the visualization component of the curve and render
    crv.vis = VisMPL.VisCurve2D()
    crv.render()

    create_curve_figure(crv, x, path)
