# pyright: reportGeneralTypeIssues=false, reportUnboundVariable = false
import numpy as np
from beautifultable import BeautifulTable

from core.mfk.kriging_base import KrigingBase

from core.ordinary_kriging.kernel import get_kernel
from core.ordinary_kriging.OK import OrdinaryKriging
from core.sampling.solvers.solver import get_solver
from core.sampling.DoE import LHS_subset
from pyDOE2 import lhs

from utils.selection_utils import isin_indices
from utils.formatting_utils import correct_formatX
from utils.correlation_utils import check_correlations

class MultiFidelityKrigingBase(KrigingBase):

    def __init__(self, setup, max_cost = None, initial_nr_samples = 3, max_nr_levels : int = 3, printing = True, *args, **kwargs):
        """
        @param kernel (list): list containing kernel function, (initial) hyperparameters, hyper parameter constraints
        @param max_cost: maximum cost available for sampling. 
        @param max_nr_levels: maximum number of Kriging levels that is added to the MF model. Is considered to be the level of the high-fidelity + 1.
        @param L: fidelity input list, is being set using set_L() if other values are desired.
        """

        super().__init__(setup, *args, **kwargs)
        
        self.printing = printing
        self.kernel = get_kernel(setup.kernel_name, setup.d, setup.noise_regression) 
        self.d = setup.d
        self.solver = get_solver(setup)
        self.max_cost = max_cost
        self.initial_nr_samples = initial_nr_samples

        self.number_of_levels = 0
        self.max_nr_levels = max_nr_levels
        self.l_hifi = max_nr_levels - 1

        # parallel capability, used for sampling the hifi in assumption verifications
        # NOTE this should be either the actual pc or the max amount of initial hifi samples desired.
        self.pc = 3 

        # list of Kriging models, one per level
        self.K_mf = []

        # multifidelity X and Z, each entry and np.ndarray
        self.X_mf = [] # NOTE gaat bij input readen ervanuit dat de lijsten van verschillende lengtes zijn (wat eigenlijk altijd zo is)!
        self.X_unique = np.array([[]])
        self.Z_mf = []
        
        # cost variables: total cost per level; expected cost per sample per level 
        self.costs_per_level = {}
        self.costs_expected = {}
        self.costs_expected_nested = {}
        self.costs_total = 0

        self.L = np.arange(max_nr_levels) + 1

        self.tune_counter = 0
        self.tune_lower_every = 1
        self.tune_prediction_every = 1


    def set_L(self, L : list):
        """
        Set the levels passed to the solver for fidelity (and cost) calculation.
        """
        self.L = L
        self.max_nr_levels = len(L)
    
    def set_L_costs(self, C):
        """Deviate from the standard implemented cost model and set standard costs per level in L."""
        assert len(C) == len(self.L), "Provide a costs list the length of L!"
        self.C = C

        # reset all costs if possible
        try:
            for l in range(self.number_of_levels):
                self.costs_per_level[l] = C[l] * self.X_mf[l].shape[0]
                self.costs_expected[l] = C[l]
                self.costs_expected_nested[l] = sum([self.costs_expected[i] for i in range(l+1)])
        except:
            pass

    def update_costs(self, cost, l):
        """
        Updates the total costs and expected cost per sample for level l.
        Prints the cost.
        """

        try:
            self.costs_per_level[l] += cost
        except:
            self.costs_per_level[l] = cost
            self.costs_expected[l] = cost
            self.costs_expected_nested[l] = cost

        # set total cost per level if expected costs are set
        if hasattr(self,'C'):
            self.costs_per_level[l] = self.C[l] * self.X_mf[l].shape[0]
        
        # TODO cheap fix for not using lowest level reference method.
        if not self.proposed:
            self.costs_per_level[0] = 0
        

        # expected cost per sample at a level
        self.costs_expected[l] = self.costs_per_level[l] / self.X_mf[l].shape[0]
        self.costs_expected_nested[l] = sum([self.costs_expected[i] for i in range(l+1)])
        self.costs_total = sum(self.costs_per_level.values())

    def print_stats(self,RMSE_list):
        """Print insightfull stats. Best called after adding a high-fidelity sample."""
        if self.printing:
            table = BeautifulTable()
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Total costs: {}".format(self.costs_total))
            row0 = [self.X_mf[l].shape[0] for l in range(self.number_of_levels)]
            row1 = [self.costs_per_level[l] for l in range(self.number_of_levels)]
            row2 = [self.costs_expected[l] for l in range(self.number_of_levels)]
            row3 = ["{:.4f} %".format(RMSE) for RMSE in RMSE_list] 
            if len(RMSE_list) == 2:
                row3 += [""] 

            table.rows.append(row0)
            table.rows.append(row1)
            table.rows.append(row2)
            table.rows.append(row3)
            table.rows.header = ["Number of samples","Costs","Expected costs","RMSE wrt kriged truth"]
            table.columns.header = ["Level {}".format(l) for l in range(self.number_of_levels)]
            table.set_style(BeautifulTable.STYLE_MARKDOWN) # type: ignore
            print(table)
                
    def sample(self, l, X_new):
        """
        Function that does all required actions for sampling if the level already exists.
        Only samples new locations.
        x_new is the value / an np.ndarray of to sample location(s)
        """

        # check per sample if not already sampled at this level
        # NOTE broadcasting way
        inds = isin_indices(X_new, self.X_mf[l], inversed = True)
        
        if np.any(inds):
            Z_new, cost = self.solver.solve(X_new[inds], self.L[l])

            self.X_mf[l] = np.append(self.X_mf[l], X_new[inds], axis=0)
            self.Z_mf[l] = np.append(self.Z_mf[l], Z_new)

            self.X_unique = self.return_updated_unique(self.X_unique, X_new[inds])
            self.update_costs(cost, l)
    
    def sample_nested(self, l, X_new):
        """
        Function that does all required actions for nested core.sampling.
        """

        for i in range(not self.proposed, l+1):
            self.sample(i, X_new)
        
        if not self.proposed and l == 1:
            # for fairness, use only 1 level for MFK reference
            # TODO easy shortcut solution
            self.Z_mf[0] = self.Z_mf[1]
            self.X_mf[0] = self.X_mf[1]

        if l == self.max_nr_levels - 1 and hasattr(self, 'X_truth'):
            # NOTE has to be after sampling underlying levels, for the correlation checks
            self.sample_truth(X_new)
            self.create_update_K_truth()

    def sample_new(self, l, X_new):
        """
        Same as function sample, but for sampling a new level.
        Does not take
        """
        if self.number_of_levels < self.max_nr_levels:
            
            if self.printing:
                string = f"Sampling new level {l:d}!"
                print(f"{'':=>{len(string)}}")
                print(string)
                print(f"{'':=>{len(string)}}")

            Z_new, cost = self.solver.solve(X_new, self.L[l])
            self.X_mf.append(X_new)
            self.Z_mf.append(Z_new)

            self.X_unique = self.return_updated_unique(self.X_unique, X_new)
            
            self.number_of_levels += 1
            self.update_costs(cost,l)
        else:
            raise RuntimeError("WARNING trying to sample new level while max number of levels has already been reached")

    def sample_initial_hifi(self, setup):
        """
        Select best sampled (!) point of previous level and sample there again.
        Then find a LHS subset with (n_subset = parallel capability) and solve.
        """

        # select best sampled (!) point of previous level and sample again there
        x_new = self.X_mf[-1][np.argmin(self.Z_mf[-1])]
        l = self.l_hifi

        # select other points for cross validation, in LHS style
        X_hifi = correct_formatX(LHS_subset(setup, self.X_unique, x_new, self.initial_nr_samples),setup.d)

        self.sample_new(l,X_hifi)

        # sample around initial samples to improve noise estimates, no training
        x_new = correct_formatX(x_new, self.d)
        self.sample_around(1, x_new, initial = True)
        X_hifi_sub = correct_formatX(X_hifi[isin_indices(X_hifi, x_new, inversed = True)],self.d) # from return_unique_exc
        self.sample_around(1, X_hifi_sub, initial = False)

        # calls to train
        self.sample_nested(l, X_hifi)


    def sample_around(self, l, X_new, bounds_percentage = 0.01, initial = False):
        """ 
        This function samples around some already known point to get a better noise estimate already from the start
        @param initial (bool): for the initial sample we would like a higher periferal sample amount
        """
        if self.solver.name == 'EVA':
            bounds_percentage = 0.5

        # only do this for the proposed method! not fair otherwise
        if self.proposed:
            for x in X_new:
                bounds_range = (self.bounds[1, :] - self.bounds[0, :]) * bounds_percentage / 100
                lb = x - bounds_range
                ub = x + bounds_range
                nr_samples = min(self.d * (1 + initial),6)
                
                X_noise = lb + lhs(
                        self.d,
                        samples=nr_samples,
                        criterion="maximin",
                        iterations=20,
                        random_state=self.randomstate,
                    ) * (ub - lb)
                
                # simply remove samples that are outside the original bounds
                
                X_noise = X_noise[~(((X_noise > self.bounds[1, :]) | (X_noise < self.bounds[0, :])).any(1))]

                self.sample_nested(l, X_noise, train = False)

    def sample_truth(self, X = None):
        """
        Sample the hifi truth on X_unique or X and compare it with our prediction.
        @param X: optional X to provide, otherwise the current self.X_unique will be used. 
            X does not have to be a subset of X_unique.
        """

        if not hasattr(self,'X_truth'):
            self.X_truth = self.X_unique
            
        if X is not None:
            # X does not have to be in X_unique for testing purposes
            # So, create an unique X_truth tracking where the high fidelity has been sampled.
            self.X_truth = self.return_updated_unique(self.X_truth, X)

        # sample or re-retrieve the 'truth'
        self.Z_truth = self.solver.solve(self.X_truth,self.L[-1])[0]

        # TODO select only those indices where the truth is known s.t. we can always calc this!
        if self.Z_mf[0].shape[0] == self.Z_mf[1].shape[0] == self.Z_truth.shape[0]:
            check_correlations(self.Z_mf[0], self.Z_mf[1], self.Z_truth)


    def set_unique(self):
        """
        Finds the unique sample locations over all levels, from the list containing each level`s X 2d nd.array.

        @return uniques including X_exclude, uniques excluding X_exclude; are equal when X_exclude = []
        """
        # # https://www.peterbe.com/plog/fastest-way-to-uniquify-a-list-in-python-3.6
        res = [item for sublist in self.X_mf for item in sublist]
        res = np.unique(res,axis=0)

        # return and reformat X
        self.X_unique = correct_formatX(res,self.d)  

    def return_updated_unique(self, X_unique, X_new):
        """
        If an element of X_new is not in X_unique, add it.
        """

        if not np.any(X_unique):
            return X_new

        res = np.unique(np.vstack([X_unique, X_new]), axis = 0)
        return correct_formatX(res,self.d)  

    def return_unique_exc(self, X_exclude=[[]]):
        """
        @param X_exclude: X we want to exclude from our list of uniques. 
        Often these are the sampled locations at the highest level (points we do not need to predict anymore). 
        Useful for plotting.
        """

        res_exc = correct_formatX(self.X_unique[isin_indices(self.X_unique, X_exclude, inversed = True)],self.d)
        return res_exc

    def create_OKlevel(self, X_l, y_l = [], train=True, tune=False, hps_init=None, hps_noise_ub = False, R_diagonal=0, name = "", append : bool = True, add_empty : bool = False):
        """
        This method clusters and provides all the functionality required for setting up a new kriging level.
        @param X_l: initial X for new level l
        @param y_l: initial y matching X, if not provide then sample. Level will be taken as len(self.X_mf).
        @param tune (bool): tune at initialisation. If False, hps_init should be defined.
        @param hps_init (np.ndarray): hyperparameter value array. Last defines the noise. 
                If not provided, try to take the hyperparameters of the previous level.
        @param hps_noise_ub: provide an upperbound to the noise hyperparameter. Used such that no infinite noise can be used (in the prediction).
        @param R_diagonal (np.ndarray): added noise or variance per sample.
        @param add_empty: create an empty Kriging level, used in case of (re)setting the state of the model.
        @param name: name of the Kriging level, used for display.
        """
        
        if hps_init is None and self.number_of_levels >= 1:
            try:
                hps_init = self.K_mf[-1].hps
            except:
                tune = True

        kernel = self.kernel
        if hps_noise_ub and (hps_init is not None):
            # noise upperbound should be larger than previous tuned noise!
            kernel[-1][-1,1] = min(max(hps_init[-1] * 2, kernel[-1][-1,1]),0.5)

        name = "Level {}".format(self.number_of_levels) if name == "" else name

        ok = OrdinaryKriging(kernel, self.d, hps_init=hps_init, name = name)

        if tune == False and not add_empty:
            if hps_init is None:
                if self.printing:
                    print("hps_init is not defined! Tune set to True")
                tune = True

        if not np.any(y_l) and not add_empty:
            if append:
                self.sample_new(self.number_of_levels, X_l)
                y_l = self.Z_mf[self.number_of_levels - 1]
            else:
                y_l, _ = self.solver.solve(X_l, self.L[self.number_of_levels])

        if train and not add_empty:
            retuning = True if add_empty else False
            ok.train(X_l, y_l, tune, retuning, R_diagonal)
        
        if append:
            self.K_mf.append(ok)
            self.number_of_levels += 1

        return ok



    " Functions to be implemented by derived classes "
    # def create_OKlevel(self, X_l, y_l = [], *args, **kwargs):
    #     raise NotImplementedError("Create new level not implemented")

    def create_update_K_truth(self):
        """
        Create or update the Kriging model of the full truth. 
        This function is called after sampling the truth or whenever a sample at the highest level is sampled.
        """
        raise NotImplementedError("Creation / Updating of K_truth model not implemented")

    def train(self, *args, **kwargs):
        """
        Update the data and train / tune the model.
        """
        raise NotImplementedError("Training function not implemented")


    def get_state(self):
        """
        Gets the current state of the data of the MF Kriging model.
        """
        # TODO deze functie wordt niet aangeroepen!! maar die van MFK wrap: inherentence probleem
        state = {k: self.__dict__[k] for k in set(list(self.__dict__.keys())) 
        - set({'options','printer','D_all','F','p','q','optimal_rlf_value','ij','supports','X','X_norma','best_iteration_fail','nb_ill_matrix','sigma2_rho','training_points','y','nt','theta0','noise0',
        'kernel','K_mf','solver','K_truth','X_infill','tune_prediction_every','tune_lower_every','tune_counter','lambs','lamb1','lamb2','pdf'})}

        # works for both smt as org
        K_mf_list = []
        for K in self.K_mf:
            if isinstance(K, OrdinaryKriging):
                K_mf_list.append(K.get_state())
        state['K_mf_list'] = K_mf_list

        if hasattr(self,'K_truth'):
            state['K_truth'] = self.K_truth.get_state()

        if hasattr(self,'K_pred'):
            state['K_pred'] = self.K_pred.get_state()

        return state
     
    def set_state(self, data_dict):
        """
        Sets a (saved) state of the data of the MF Kriging model
        """
        raise NotImplementedError("set_state not implemented for this class")

