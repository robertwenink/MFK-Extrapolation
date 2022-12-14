import numpy as np
from core.kriging.OK import OrdinaryKriging


from core.sampling.DoE import LHS_subset
from utils.formatting_utils import correct_formatX
from utils.error_utils import RMSE_norm_MF
from utils.correlation_utils import check_correlations


class MultiFidelityKriging():

    def __init__(self, kernel, d, solver, max_cost = None, max_nr_levels : int = 3, *args, **kwarg):
        """
        @param kernel (list): list containing kernel function, (initial) hyperparameters, hyper parameter constraints
        @param max_cost: maximum cost available for sampling. 
        @param max_nr_levels: maximum number of Kriging levels that is added to the MF model. Is considered to be the level of the high-fidelity + 1.
        @param L: fidelity input list TODO
        """

        super().__init__(*args, **kwarg) # possibly useful later if inhereting from something other than Object!

        self.kernel = kernel
        self.d = d
        self.solver = solver
        self.max_cost = max_cost #1500e5

        self.number_of_levels = 0
        self.max_nr_levels = max_nr_levels
        self.l_hifi = max_nr_levels - 1

        # parallel capability, used for sampling the hifi in assumption verifications
        # NOTE this should be either the actual pc or the max amount of initial hifi samples desired.
        self.pc = 3 

        # list of Kriging models, one per level
        self.K_mf = []

        # multifidelity X and Z, each entry and np.ndarray
        self.X_mf = [] # TODO miss ook dict maken!
        self.X_unique = np.array([])
        self.Z_mf = []
        
        # cost variables: total cost per level; expected cost per sample per level 
        self.costs_tot = {}
        self.costs_exp = {}

        # TODO
        self.L = np.arange(max_nr_levels) + 1

    def set_L(self, L : list):
        self.L = L
        self.max_nr_levels = len(L)


    def create_level(self, X_l, y_l = [], train=True, tune=False, hps_init=None, hps_noise_ub = False, R_diagonal=0, append : bool = True):
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
        
        if hps_init is None and self.number_of_levels >= 1:
            hps_init = self.K_mf[-1].hps

        if hps_noise_ub and (hps_init is not None):
            # noise upperbound should be larger than previous tuned noise!
            kernel = self.kernel
            kernel[-1][-1,1] = hps_init[-1] * 2
        else:
            kernel = self.kernel

        ok = OrdinaryKriging(kernel, self.d, hps_init=hps_init)

        if tune == False:
            if hps_init is None:
                raise NameError(
                    "hps_init is not defined! When tune=False, hps_init must be defined"
                )

        if not np.any(y_l):
            if append:
                self.sample_new(self.number_of_levels, X_l)
                y_l = self.Z_mf[self.number_of_levels]
            else:
                y_l, _ = self.solver.solve(X_l, self.L[self.number_of_levels])

        if train:
            ok.train(X_l, y_l, tune, R_diagonal)
        
        if append:
            self.K_mf.append(ok)
            self.number_of_levels += 1

        return ok

    def update_costs(self, cost, l):
        """
        Updates the total costs and expected cost per sample for level l.
        Prints the cost.
        """

        # total cost per level
        try:
            self.costs_tot[l] += cost
        except:
            self.costs_tot[l] = cost
            self.costs_exp[l] = cost

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
        Only samples new locations.
        x_new is the value / an np.ndarray of to sample location(s)
        """

        # check per sample if not already sampled at this level

        # NOTE looping way
        # TODO for loop, parallel sampling.
        #      in EVA this is done separately, might be better to put it here!
        # inds = []
        # for i, x_new in enumerate(X_new):
        #     # has x not already been sampled before?
        #     if ~np.any(np.all(x_new == self.X_mf[l], axis=1)):
        #         inds.append(i)

        # NOTE broadcasting way
        inds = ~(X_new == self.X_unique[:, None]).all(axis=-1).any(axis=0)
        
        if np.any(inds):
            Z_new, cost = self.solver.solve(X_new[inds], self.L[l])

            self.X_mf[l] = np.append(self.X_mf[l], X_new[inds], axis=0)
            self.Z_mf[l] = np.append(self.Z_mf[l], Z_new)

            self.X_unique = self.return_updated_unique(self.X_unique, X_new[inds])
            self.update_costs(cost, l)


    def sample_new(self, l, X_new):
        """
        Same as function sample, but for sampling a new level.
        """
        Z_new, cost = self.solver.solve(X_new, self.L[l])
        self.X_mf.append(X_new)
        self.Z_mf.append(Z_new)

        self.X_unique = self.return_updated_unique(self.X_unique, X_new)
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


    def sample_initial_hifi(self, setup):
        """
        Select best sampled (!) point of previous level and sample there again.
        Then find a LHS subset with (n_subset = parallel capability) and solve.
        """

        # select best sampled (!) point of previous level and sample again there
        x_b = self.X_mf[-1][np.argmin(self.Z_mf[-1])]
        l = self.l_hifi

        # select other points for cross validation, in LHS style
        X_hifi = correct_formatX(LHS_subset(setup, self.X_unique, x_b, self.pc),setup.d)

        self.sample_new(l,X_hifi)
        self.sample_nested(l, X_hifi)


    def sample_truth(self, X = None, use_known_X : bool = True):
        """
        Sample the hifi truth on X_unique or X and compare it with our prediction.
        @param X: optional X to provide, otherwise the current self.X_unique will be used. 
            X does not have to be a subset of X_unique.
        @param use_known_X: reuse the X where the truth has been sampled if possible, 
            such that there is no additional sampling required.
        """
        
        if X is None:
            if not hasattr(self,'X_truth'):
                self.X_truth = self.X_unique
            X = self.X_truth
        else:
            # X does not have to be in X_unique for testing purposes
            # So, create an unique X_truth tracking where the high fidelity has been sampled.
            self.X_truth = self.return_updated_unique(self.X_truth, X)


        # sample or retrieve the 'truth'
        self.Z_truth = self.solver.solve(X,self.L[-1])[0]
        
        # check correlations and RMSE levels
        RMSE_norm_MF(X, self.Z_truth, self.K_mf)
        check_correlations(self.Z_mf[0], self.Z_mf[1], self.Z_truth)



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

            # TODO save whole state, including tuned hyperparameters etc.
            # just pickle the whole object?



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

        res_exc = np.array([item for item in self.X_unique if item not in X_exclude])

        return correct_formatX(res_exc,self.d)
        

class ProposedMultiFidelityKriging(MultiFidelityKriging):

    def __init__(self, *args, **kwarg):
        super().__init__(*args, **kwarg)

        # list of predicted z values at the highest/ new level
        self.Z_pred = np.array([])
        self.mse_pred = np.array([])

class MultiFidelityEGO():
    # initial EI
    ei = 1
    ei_criterion = 2 * np.finfo(np.float32).eps
