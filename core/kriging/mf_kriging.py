import sys
import numpy as np
from core.kriging.OK import OrdinaryKriging


from core.sampling.DoE import LHS_subset
from utils.formatting_utils import correct_formatX


class MultiFidelityKriging():

    def __init__(self, kernel, solver, max_cost = None, max_nr_levels : int = 3, *args, **kwarg):
        """
        @param kernel (list): list containing kernel function, (initial) hyperparameters, hyper parameter constraints
        @param max_cost: maximum cost available for sampling. 
        @param max_nr_levels: maximum number of Kriging levels that is added to the MF model. Is considered to be the level of the high-fidelity + 1.
        @param L: fidelity input list TODO
        """

        super().__init__(*args, **kwarg) # possibly useful later if inhereting from something other than Object!

        self.kernel = kernel
        self.solver = solver
        self.max_cost = max_cost #1500e5
        self.max_nr_levels = max_nr_levels
        self.l_hifi = max_nr_levels - 1

        # parallel capability, used for sampling the hifi in assumption verifications
        # NOTE this should be either the actual pc or the max amount of initial hifi samples desired.
        self.pc = 3 

        # list of Kriging models, one per level
        self.K_mf = []

        # multifidelity X and Z, each entry and np.ndarray
        self.X_mf = [] # TODO miss ook dict maken!
        self.Z_mf = []
        
        # cost variables: total cost per level; expected cost per sample per level 
        self.costs_tot = {}
        self.costs_exp = {}

        # TODO
        self.L = np.arange(max_nr_levels) + 1

    def set_L(self, L : list):
        self.L = L
        self.max_nr_levels = len(L)

    def add_level(self, X_l, y_l = [], tune=False, R_diagonal=0, hps_init=None, hps_noise_ub = False, train=True):
        """
        This method clusters and provides all the functionality required for setting up a new kriging level.
        @param X_l: initial X for new level l
        @param y_l: initial y matching X, if not provide then sample. Level will be taken as len(self.X_mf).
        @param tune (bool): tune at initialisation. If False, hps_init should be defined.
        @param hps_init (np.ndarray): hyperparameter value array. Last defines the noise. 
                If not provided, try to take the hyperparameters of the previous level.
        @param hps_noise_ub: provide an upperbound to the noise hyperparameter. Used such that no infinite noise can be used (in the prediction).
        @param R_diagonal (np.ndarray): added noise or variance per sample.
        """

        if hps_init is None and len(self.X_mf) >= 1:
            hps_init = self.K_mf[-1].hps

        if hps_noise_ub and (hps_init is not None):
            # noise upperbound should be larger than previous tuned noise!
            kernel = self.kernel[-1][-1,1]
            kernel = hps_init[-1] * 2
        else:
            kernel = self.kernel

        ok = OrdinaryKriging(kernel, hps_init=hps_init)

        if tune == False:
            if hps_init is None:
                raise NameError(
                    "hps_init is not defined! When tune=False, hps_init must be defined"
                )

        if not y_l:
            self.sample_new(len(self.X_mf), X_l)


        if train:
            ok.train(X_l, y_l, tune, R_diagonal)

        return ok #TODO toevoegen aan K_mf ipv return?

    def update_costs(self, cost, l):
        """
        Updates the total costs and expected cost per sample for level l.
        Prints the cost.
        """

        # total cost per level
        self.costs_tot[l] += cost

        # expected cost per sample at a level
        self.costs_exp[l] = self.costs_tot[l] / self.X_mf[l].shape[0]

        # print current total costs (all levels)
        if l == self.max_nr_levels - 1:
            print("Current cost: {} ; {} hifi samples".format(sum(self.costs_tot.values()),self.X_mf[-1].shape[0]))
        else:
            print("Current cost: {}".format(sum(self.costs_tot.values())))

    
    def sample(self, l, X_new):
        """
        Function that does all required actions for sampling if the level already exists.
        x_new is the value / an np.ndarray of to sample location(s)
        """

        # check if not already sampled at this level
        # TODO for loop, parallel sampling.
        if np.all(np.isin(X_new,self.X_mf[l],invert=True)):
            Z_new, cost = self.solver.solve(X_new, self.L[l])

            self.X_mf[l] = np.append(self.X_mf[l], X_new, axis=0)
            self.Z_mf[l] = np.append(self.Z_mf[l], Z_new)
            
            self.update_costs(cost, l)


    def sample_new(self, l, X_new):
        """
        Same as function sample, but for sampling a new level.
        """
        Z_new, cost = self.solver.solve(X_new, self.L[l])
        self.X_mf.append(X_new)
        self.Z_mf.append(Z_new)
        self.update_costs(cost,l)


    def sample_nested(self, l, X_new):
        """
        Function that does all required actions for nested core.sampling.
        """

        # Sampled HIFI location should be sampled at all lower levels as well, i.e. nested DoE; this because of Sf and optimally making use of the hifi point.
        # TODO do this for all levels? i.e. only the for loop?
        # NOTE leave for loop out to only sample the highest level
        sampled_levels = [l]

        self.sample(l, X_new)

        if l == self.max_nr_levels:
            for i in range(l):
                self.sample(i, X_new)
                sampled_levels.append(i)
        
        # retrain the sampled levels with the newly added sample.
        # NOTE this is proposed method specific actually (where we dont want to train the top level perse)
        for i in sampled_levels:
            if i != self.max_nr_levels - 1:
                self.K_mf[i].train(self.X_mf[i], self.Z_mf[i])


    def sample_initial_hifi(self, setup, X_unique):
        """
        Select best sampled (!) point of previous level and sample there again.
        Then find a LHS subset with (n_subset = parallel capability) and solve.
        """

        # select best sampled (!) point of previous level and sample again there
        x_b = self.X_mf[-1][np.argmin(self.Z_mf[-1])]
        l = self.l_hifi

        # select other points for cross validation, in LHS style
        X_hifi = correct_formatX(LHS_subset(setup, X_unique, x_b, self.pc),setup.d)

        self.sample_new(l,X_hifi)
        self.sample_nested(l, X_hifi)


    def set_state(self, setup):
        """
        Sets a (saved) state of the data of the MF Kriging model
        """
        if hasattr(setup, "X_mf"):
            self.X_mf = setup.X_mf

            if hasattr(setup, "Z_mf"):
                self.X_mf = setup.Z_mf
                self.costs_tot = setup.costs_tot
                self.costs_exp = setup.costs.exp


    def get_state(self):
        """
        Gets the current state of the data of the MF Kriging model
        """
        state = {}
        
        if hasattr(self, "X_mf"):
            state["X_mf"] = self.X_mf

            if hasattr(self, "Z_mf"):
                state["Z_mf"] = self.Z_mf
                state["costs_tot"] = self.costs_tot
                state["costs_exp"] = self.costs_exp

        return state




class ProposedMultiFidelityKriging(MultiFidelityKriging):

    def __init__(self, *args, **kwarg):
        super().__init__(*args, **kwarg)

        # list of predicted z values at the highest/ new level
        self.Z_pred = []

class MultiFidelityEGO():
    # initial EI
    ei = 1
    ei_criterion = 2 * np.finfo(np.float32).eps
