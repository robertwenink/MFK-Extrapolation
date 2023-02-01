# pyright: reportGeneralTypeIssues=false
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import time

import core.kriging.mf_kriging as mf # like this because circular import!
from core.sampling.solvers.internal import TestFunction
from core.kriging.kernel import dist_matrix
from core.sampling.solvers.solver import get_solver
from utils.error_utils import RMSE_norm_MF
from utils.error_utils import RMSE_focussed as RMSE_focussed_func
from utils.selection_utils import get_best_prediction, get_best_sample

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
        self.colors[2:3] = []

        if isinstance(self.solver,TestFunction):
            self.X_opt, self.Z_opt = self.solver.get_optima()

        self.iteration_numbers = []

        self.values_best = []
        self.values_x_new = []

        self.distances_best = []
        self.distances_x_new = []

        self.RMSEs = []
        self.RMSE_focussed = []
        self.RMSE_focuss_percentage = 20
        self.main_font_size = 12

        self.init_figure()

        
    def set_state(self, setup):
        """ (re)set the state"""
        if hasattr(setup, 'convergence_plotting_dict'):
            for key in setup.convergence_plotting_dict:
                setattr(self, key, list(setup.convergence_plotting_dict[key]))

    def get_state(self):
        """ retrieve current state """
        attribute_dict = {'iteration_numbers','values_best','values_x_new','distances_best','distances_x_new','RMSEs','RMSE_focussed'}
        state = {k: self.__dict__[k] for k in set(list(attribute_dict))}
        return state


    def init_figure(self):
        """ 
        initialise the figure 
        """

        self.fig, (ax_opt, ax_rmse) = plt.subplots(1,2, figsize = (11.5,6))
        
        " Plots for value and distance convergence "
        ax_opt.set_title("Value and distance\nconvergence to the optimum")
        ax_dist=ax_opt.twinx()
        
        # 1) difference to optimum value
        p1_best, = ax_opt.plot([], [], color=self.colors[0], label = "Current best sample")
        p1_ei, = ax_opt.plot([], [], "--", color=self.colors[0], label = "Next sample (best EI)")
        ax_opt.set_xlabel("Iteration number [-]", fontsize = self.main_font_size)
        ax_opt.set_ylabel("Objective value [-]", fontsize = self.main_font_size)

        # 2) spatial difference to (any) optimum location
        # using a twin object on the same plot for two y-axes!
        p2_best, = ax_dist.plot([], [], color=self.colors[1], label = "Current best sample")
        p2_ei, = ax_dist.plot([], [], "--", color=self.colors[1], label = "Next sample (best EI)")
        ax_dist.set_ylabel("Euclidian parameter distance [-]", fontsize = self.main_font_size)
                
        # set distinguishable axis properties for twin axes        
        ax_opt.yaxis.label.set_color(p1_best.get_color())
        ax_dist.yaxis.label.set_color(p2_best.get_color())

        tkw = dict(size=4, width=1)
        ax_opt.tick_params(axis='y', colors=p1_best.get_color(), **tkw)
        ax_dist.tick_params(axis='y', colors=p2_best.get_color(), **tkw)


        " RMSE plots "
        ax_rmse.set_title("RMSE convergence")
        p3_full, = ax_rmse.plot([], [], color=self.colors[2], label = "Full predicted surrogate")
        p3_focussed, = ax_rmse.plot([], [], color=self.colors[3], label = "Focussed (opt + {}%)\npredicted surrogate".format(self.RMSE_focuss_percentage))
        ax_rmse.set_ylabel("RMSE with respect to kriged truth [%]", fontsize = self.main_font_size)
        ax_rmse.set_xlabel("Iteration number [-]", fontsize = self.main_font_size)


        # set legends and layout
        # ax_opt.legend(handles=[p1_best, p2_best, p1_ei, p2_ei], loc='upper right', bbox_to_anchor = (-0.25, 1.0))
        ax_opt.legend(loc='upper left', bbox_to_anchor = (-0.12, -0.15), title = "Objective value")
        ax_dist.legend(loc='upper right', bbox_to_anchor = (1.12, -0.15), title = "Distance to optimum")
        ax_rmse.legend(loc='upper center', bbox_to_anchor = (0.5, -0.15))
        
        self.axes = [ax_opt, ax_dist, ax_rmse]
        self.artists = [p1_best, p1_ei, p2_best, p2_ei, p3_full, p3_focussed]

        for ax in self.axes:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # screen_width = get_monitors()[0].width
        self.fig.canvas.manager.window.move(1201,0) # type: ignore ; to the right
        self.fig.canvas.manager.window.move(0,800) # type: ignore ; underneath
        self.fig.tight_layout()

        self.fig.canvas.draw()
        plt.show(block=False)
        self.fig.tight_layout()

    def update_convergence_data(self, model, x_new, ei = None):
        """
        Update the convergence data.
        """
        if np.any(x_new):
            RMSE = RMSE_norm_MF(model, no_samples=True)
            RMSE_focussed = RMSE_focussed_func(model, self.RMSE_focuss_percentage)
            x_best, value_best = get_best_sample(model)
            x_new, value_x_new = get_best_prediction(model, x_new)

            self._update_data(x_best, x_new, value_best, value_x_new, RMSE, RMSE_focussed)

    def plot_convergence(self):
        """
        Plot the convergence stats (and calculate the stats needed). 
        @param model: either a multi or single-fidelity model
        """

        self._update_plot()
        

    def _update_data(self, x_best, x_new, value_best, value_x_new, RMSE, RMSE_focussed):
        """
        Update the data of the plot
        """
        self.iteration_numbers.append(len(self.values_best))

        # values
        self.values_best.append(value_best)
        self.values_x_new.append(value_x_new)

        # distances
        self.distances_best.append(min(dist_matrix(self.X_opt,x_best)))
        self.distances_x_new.append(min(dist_matrix(self.X_opt,x_new)))

        # RMSE
        self.RMSEs.append(RMSE[-1])  # TODO alles weergeven?
        self.RMSE_focussed.append(RMSE_focussed[-1]) # TODO alles weergeven?

    def _update_plot(self):
        
        # self.artists = [p1_best, p1_ei, p2_best, p2_ei, p3_full, p3_focussed]
        for artist in self.artists:
            artist.set_xdata(self.iteration_numbers)
        
        self.artists[0].set_ydata(self.values_best)
        self.artists[1].set_ydata(self.values_x_new)

        self.artists[2].set_ydata(self.distances_best)
        self.artists[3].set_ydata(self.distances_x_new)
        self.artists[4].set_ydata(self.RMSEs)
        self.artists[5].set_ydata(self.RMSE_focussed)
        
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view(True,True,True)
            # ax.redraw_in_frame()
            
        self.fig.canvas.draw()
        self.fig.canvas.flush_events(); 
        plt.show(block=False)
