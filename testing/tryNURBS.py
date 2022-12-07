from core.sampling.DoE import grid_unit_hypercube
from core.sampling.solvers.NURBS import *
import matplotlib.pyplot as plt
import os


X = grid_unit_hypercube(2, n_per_d=4)


fig, axes = plt.subplots(4, 4, sharex=True, sharey=True)
fig.suptitle("NURBS variations")

if not os.path.exists(os.path.dirname(__file__)+"./NURBS/"):
    os.mkdir(os.path.dirname(__file__)+"./NURBS/")

for i, x in enumerate(X):
    name = "_".join(format(i, ".3f") for i in x)
    name = name.replace(".", "")
    path = os.path.dirname(__file__)+"./NURBS/{}".format(name)
    interpolating_curve(x, path)

for i, x in enumerate(X):
    name = "_".join(format(j, ".3f") for j in x)
    name = name.replace(".", "")
    path = os.path.dirname(__file__)+"./NURBS/{}.png".format(name)
    
    ax = axes[3 - int(np.floor(i / 4)), i % 4]
    ax.set_title("x=[{}]".format(name.replace("_", ", ")))

    img = plt.imread(path)
    ax.imshow(img, aspect="equal", interpolation="nearest")

    ax.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
        right=False,
        left=False,
        labelleft=False,
    )  # labels along the bottom edge are off

fig.tight_layout()
plt.show()