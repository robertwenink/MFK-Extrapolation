# pyright: reportGeneralTypeIssues=false, reportUnboundVariable = false
import numpy as np
from pyDOE2 import lhs
from scipy.optimize import minimize

from core.ordinary_kriging.OK import OrdinaryKriging
from core.sampling.solvers.internal import TestFunction # like this because circular import!
from utils.formatting_utils import correct_formatX

import core.routines.EGO as ego
import core.mfk.mfk_base as mf

class KrigingBase():
    def __init__(self, setup, *args, **kwargs) -> None:
                
        self.lb = setup.search_space[1]
        self.ub = setup.search_space[2]
        self.bounds = np.array(setup.search_space[1:])

    def find_best_point(self, prediction_function, criterion = 'SBO'):
        """
        Optimizer for getting the best EI (criterion = 'EI') or prediction in the current self
        @param prediction_function
        """

        if criterion == "EI" and isinstance(self, ego.EfficientGlobalOptimization):
            _, y_min = self.get_best_sample() 
            # y_min = np.min(self.Z_pred)
            
            def obj_k(x): 
                y_pred, mse_pred = prediction_function(x)
                return -self.EI(y_min, y_pred, np.sqrt(mse_pred/2)) # TODO hier /2 gedaan!
        else: # then just use the surrogates prediction (surrogate based optimization: SBO)
            def obj_k(x): 
                y_pred, mse_pred_or_cost = prediction_function(np.atleast_2d(x))
                return float(y_pred)

        success = False
        n_start = 40
        n_optim = 1  # in order to have some success optimizations with SLSQP
        n_max_optim = 20

        while not success and n_optim <= n_max_optim:
            opt_all = []
            
            # x_start = self._sampling(n_start) # TODO should be a LHS
            x_start = self.bounds[0, :] + lhs(
                self.d,
                samples=n_start,
                criterion="maximin",
                iterations=20,
                random_state=0,
            ) * (self.bounds[1, :] - self.bounds[0, :])

            # iterate and minimize over each of the points in the lhs
            for ii in range(n_start):

                try:
                    opt_all.append(
                        # minimize returns dict with x, fun, success
                        minimize(
                            obj_k,
                            x_start[ii, :],
                            method="SLSQP",
                            bounds=self.bounds.T,
                            options={"maxiter": 200},
                        )
                    )

                except ValueError:  # in case "x0 violates bound constraints" error
                    if self.printing:
                        print("warning: `x0` violates bound constraints")
                        print("x0={}".format(x_start[ii, :]))
                        print("bounds={}".format(self.bounds))
                        opt_all.append({"success": False})

            opt_all = np.asarray(opt_all)
            opt_success = opt_all[[opt_i["success"] for opt_i in opt_all]]
            obj_success = np.array([opt_i["fun"] for opt_i in opt_success])
            success = obj_success.size != 0
            if not success:
                n_optim += 1

        if n_optim >= n_max_optim:
            # self.log("Internal optimization failed at EGO iter = {}".format(k))
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!! WARNING: INTERNAL EI OPTIMIZATION FAILED !!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return np.atleast_2d(0), 0
        ind_min = np.argmin(obj_success)
        opt = opt_success[ind_min]
        x_best = np.atleast_2d(opt["x"])

        return x_best, -obj_success[ind_min] if criterion == "EI" else obj_success[ind_min]


    def get_best_prediction(self, model, x_best = None, try_use_solver = False):
        """ 
        @param x_best: location of interest (i.e. best prediction or highest EI)
        returns X_best, y_best 
        """
        
        if try_use_solver and isinstance(self.solver, TestFunction):
            prediction_function = self.solver.solve
        else:
            prediction_function = model.predict

        if np.all(x_best == None):
            x_best, y_best = self.find_best_point(prediction_function)
        elif np.all(x_best != None):
            y_best, _ = prediction_function(x_best)
        
        return x_best, y_best

    def get_best_sample(self, arg = False):
        """
        returns the best sample as X, y
        if arg = True, return only the sample index 
        """
        if isinstance(self, mf.MultiFidelityKrigingBase):
            best_ind = np.argmin(self.Z_mf[-1])
            if not arg:
                return correct_formatX(self.X_mf[-1][best_ind], self.d), self.Z_mf[-1][best_ind] 

        elif isinstance(self, OrdinaryKriging):
            best_ind = np.argmin(self.y)
            if not arg:
                return correct_formatX(self.X[best_ind], self.d), self.y[best_ind] 

        return best_ind # type: ignore

    def get_best_extrapolation(self, arg = False):
        best_ind = np.argmin(self.Z_pred)
        if not arg:
            return correct_formatX(self.X_unique[best_ind], self.d), self.Z_pred[best_ind] 
        return best_ind
