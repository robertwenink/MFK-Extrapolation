import matplotlib.pyplot as plt

from core.sampling.solvers.internal import TestFunction
from core.kriging.kernel import dist_matrix
from core.sampling.solvers.solver import get_solver
from utils.error_utils import RMSE_norm_MF, RMSE_focussed

class ConvergencePlotting():
    """
    Plot the convergence of the simulation (towards the known optimum; requires the solver to be a testfunction).
    Do this with the following criteria:
    1) difference to optimum value
    2) spatial difference to (any) optimum location
    
    Do this for both:
    A) The current best sample (and/or current best predicted)
    B) The next max EI location (x_new)

    Further plot the following criteria that relate to the accuracy of the surrogate model:
    1) RMSE
    2) RMSE_focussed
    """

    def __init__(self, setup, *args, **kwargs) -> None:
        
        self.d = setup.d
        self.solver = get_solver(setup)
        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        if isinstance(self.solver,TestFunction):
            self.X_opt, self.Z_opt = self.solver.get_optima(self.d)

        self.iteration_number = []

        self.values_best = []
        self.values_x_new = []

        self.distances_best = []
        self.distances_x_new = []

        self.RMSEs = []
        self.RMSE_focussed = []

        self.init_figure()
        

    def init_figure(self):
        fig, ax = plt.subplots()
        ax2=ax.twinx()
        ax3=ax.twinx()

        fig.subplots_adjust(right=0.75)
        ax3.spines.right.set_position(("axes", 1.2))

        # 1) difference to optimum value
        p1, = ax.plot([], [], color=self.colors[0], marker="o")
        ax.set_xlabel("Iteration number")
        ax.set_ylabel("Objective value [-]")

        # 2) spatial difference to (any) optimum location
        # using a twin object on the same plot for two y-axes!
        p2, = ax2.plot([], [], color=self.colors[1], marker="o")
        ax2.set_ylabel("Euclidian parameter distance [-]")
        
        # 3) RMSE
        p3, = ax3.plot([], [], color=self.colors[2], marker="o")
        ax3.set_ylabel("RMSE [-]")
        
        ax.yaxis.label.set_color(p1.get_color())
        ax2.yaxis.label.set_color(p2.get_color())
        ax3.yaxis.label.set_color(p3.get_color())

        tkw = dict(size=4, width=1.5)
        ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
        ax2.tick_params(axis='y', colors=p2.get_color(), **tkw)
        ax3.tick_params(axis='y', colors=p3.get_color(), **tkw)
        ax.tick_params(axis='x', **tkw)

        ax.legend(handles=[p1, p2, p3])
        # plt.show()

    def plot_convergence(self, model):
        """
        Plot the convergence stats (and calculate the stats needed). 
        @param model: either a multi or single-fidelity model
        """

    def update_data(self, x_best, x_new, value_best, value_x_new, RMSE, RMSE_focussed):

        # values
        self.values_best.append(value_best)
        self.values_x_new.append(value_x_new)

        # distances
        self.distances_best.append(min(dist_matrix(self.X_opt,x_best)))
        self.distances_x_new.append(min(dist_matrix(self.X_opt,x_new)))

        # RMSE
        self.RMSEs.append(RMSE)
        self.RMSE_focussed.append(RMSE_focussed)
