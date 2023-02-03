# pyright: reportGeneralTypeIssues=false,
import numpy as np
from beautifultable import BeautifulTable
import scipy.stats as scistats
# from scipy import stats

from copy import deepcopy, copy


from core.kriging.OK import OrdinaryKriging
from core.kriging.kernel import get_kernel
from core.sampling.DoE import LHS_subset, LHS
from core.sampling.solvers.solver import get_solver

import core.proposed_method as wp # like this because circular import!
from core.sampling.solvers.internal import TestFunction

from utils.formatting_utils import correct_formatX
from utils.correlation_utils import check_correlations
from utils.selection_utils import isin_indices, isin, get_best_prediction, get_best_sample, create_X_infill

from smt.applications import MFK


class MultiFidelityKrigingBase(object):

    def __init__(self, setup, max_cost = None, initial_nr_samples = 3, max_nr_levels : int = 3, *args, **kwargs):
        """
        @param kernel (list): list containing kernel function, (initial) hyperparameters, hyper parameter constraints
        @param max_cost: maximum cost available for sampling. 
        @param max_nr_levels: maximum number of Kriging levels that is added to the MF model. Is considered to be the level of the high-fidelity + 1.
        @param L: fidelity input list, is being set using set_L() if other values are desired.
        """

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
        self.X_unique = np.array([])
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

        self.lb = setup.search_space[1]
        self.ub = setup.search_space[2]
        self.n_infill_per_d= 601 # TODO is duplicate met in EGO class gedifinieerde var
        self.X_infill = create_X_infill(setup.d, self.lb, self.ub, self.n_infill_per_d)

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

        # expected cost per sample at a level
        self.costs_expected[l] = self.costs_per_level[l] / self.X_mf[l].shape[0]
        self.costs_expected_nested[l] = sum([self.costs_expected[i] for i in range(l+1)])
        self.costs_total = sum(self.costs_per_level.values())

    def print_stats(self,RMSE_list):
        """Print insightfull stats. Best called after adding a high-fidelity sample."""
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

        for i in range(l+1):
            self.sample(i, X_new)

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
        self.sample_nested(l, X_hifi)

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

        res_exc = np.array([item for item in self.X_unique if item not in X_exclude])

        return correct_formatX(res_exc, self.d)

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
            #TODO this is done for the complete MF_KRIGING!!!!
            # noise upperbound should be larger than previous tuned noise!
            kernel[-1][-1,1] = min(max(hps_init[-1] * 2, kernel[-1][-1,1]),0.5)

        name = "Level {}".format(self.number_of_levels) if name == "" else name

        ok = OrdinaryKriging(kernel, self.d, hps_init=hps_init, name = name)

        if tune == False and not add_empty:
            if hps_init is None:
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
        state = {k: self.__dict__[k] for k in set(list(self.__dict__.keys())) 
        - set({'options','printer','D_all','F','p','q','optimal_rlf_value','ij','supports','X','X_norma','best_iteration_fail','nb_ill_matrix','sigma2_rho','training_points','y','nt','theta0','noise0',
        'kernel','K_mf','solver','K_truth','X_infill','tune_prediction_every','tune_lower_every','tune_counter'})}

        # works for both smt as org
        K_mf_list = []
        for K in self.K_mf:
            if isinstance(K, OrdinaryKriging):
                K_mf_list.append(K.get_state())
        state['K_mf_list'] = K_mf_list

        if hasattr(self,'K_truth'):
            state['K_truth'] = self.K_truth.get_state()

        return state
     
    def set_state(self, data_dict):
        """
        Sets a (saved) state of the data of the MF Kriging model
        """
        raise NotImplementedError("set_state not implemented for this class")


class MFK_wrap(MFK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_state(self):
        # not: ,'F_all','p_all','q_all','y_norma_all'
        state = {k: self.__dict__[k] for k in set(list(self.__dict__.keys())) 
        - set({'options','printer','D_all','F','p','q','optimal_rlf_value','ij','supports','X','X_norma','best_iteration_fail','nb_ill_matrix','sigma2_rho','training_points','y','nt','theta0','noise0',
        'kernel','K_mf','solver','K_truth','X_infill','tune_prediction_every','tune_lower_every','tune_counter'})}
            

        # print(dict_types(state))
        return state

    def set_state(self, data_dict):
        for key in data_dict:
            if key in ['optimal_noise_all','q_all','p_all','optimal_theta']:
                setattr(self, key, list(data_dict[key]))
            elif key in ['optimal_par']: # quite redundant now
                setattr(self, key, list(data_dict[key]))
            else:    
                setattr(self, key, data_dict[key])

    def predict(self, X):
        return (self.predict_values(X).reshape(-1,), self.predict_variances(X).reshape(-1,))

class MFK_org(MultiFidelityKrigingBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def sample_nested(self, l, X_new):
        """
        Function that does all required actions for nested core.sampling.
        """

        super().sample_nested(l, X_new)

        sampled_levels = [l]
        for i in range(l):    
            sampled_levels.append(i)

        # retrain the sampled levels with the newly added sample.
        # NOTE this is proposed method specific actually (where we dont want to train the top level perse)
        tune = False
        if self.tune_counter % self.tune_lower_every == 0:
            tune = True

        self.retrain(levels = sampled_levels, tune = tune)

        self.tune_counter += 1

    def retrain(self, levels, tune = False):
        for i in levels:
            if i != self.max_nr_levels - 1: # because we first should do the weighing procedure again (K_mf is not initialised!)
                self.K_mf[i].train(self.X_mf[i], self.Z_mf[i], tune = tune)
                
    def create_update_K_truth(self):
        if hasattr(self,'X_truth') and hasattr(self,'Z_truth'):
            if not hasattr(self,'K_truth'):
                print("Creating Kriging model of truth", end = '\r')
                self.K_truth = self.create_OKlevel(self.X_truth, self.Z_truth, tune = True, append = False, name = "Truth")
            else:
                print("Updating Kriging model of truth", end = '\r')
                self.K_truth.train(self.X_truth, self.Z_truth, tune = True, retuning = True)

    def set_state(self, data_dict):
        """
        Sets a (saved) state of the data of the MF Kriging model
        
        Of each model in K_mf we need to retrieve the dictionary too, init them, then set the dictionary. 
        'kernel' is a (numba) function so cant be get or set and needs to be re-initialised -> init the classes like normal, then assign data
        idem for 'solver'
        """

        for key in data_dict:
            if key not in ['K_mf_list','number_of_levels','K_truth']:
                setattr(self, key, data_dict[key])

        # init the new levels
        for l in range(data_dict['number_of_levels']):
            k = self.create_OKlevel([],add_empty=True)
            k.set_state(data_dict['K_mf_list'][l])

        if 'K_truth' in data_dict:
            self.K_truth = self.create_OKlevel([],append = False, add_empty=True) 
            self.K_truth.set_state(data_dict['K_truth'])

class MFK_smt(MFK, MultiFidelityKrigingBase):
    """Provide a wrapper to smt`s MFK, to interface with the own code."""

    def __init__(self, *args, **kwargs):
        self.MFK_kwargs = kwargs['MFK_kwargs']
        super().__init__(**self.MFK_kwargs)
        MultiFidelityKrigingBase.__init__(self, *args, **kwargs)

    def sample_nested(self, l, X_new):
        """
        Function that does all required actions for nested core.sampling.
        """

        super().sample_nested(l, X_new)
        self.train()

    def create_update_K_truth(self, data_dict = None):
        if hasattr(self,'X_truth') and hasattr(self,'Z_truth'):
            # overloaded function
            if not hasattr(self,'K_truth'):
                print("Creating Kriging model of truth", end = '\r')
                kwargs = copy(self.MFK_kwargs)
                kwargs['n_start'] *= 2 # truth is not tuned often so rather not have any slipups
                self.K_truth = MFK_wrap(**kwargs)
                self.K_truth.X_infill = self.X_infill
                self.K_truth.name = "K_truth"

            if data_dict != None:
                self.K_truth.set_state(data_dict)
            else:
                self.set_training_data(self.K_truth, [*self.X_mf[:self.max_nr_levels-1], self.X_truth], [*self.Z_mf[:self.max_nr_levels-1], self.Z_truth])
                self.K_truth.train()
                self.K_truth.X_opt, self.K_truth.z_opt = get_best_prediction(self.K_truth)
                print("Succesfully trained Kriging model of truth", end = '\r')

    
    def set_training_data(self, model : MFK, X_mf : list, Z_mf : list):
        """
        Set the training data for 'model'; data found in X_mf and Z_mf

        For model leading to K_truth, last entry of X_mf Z_mf should contain full data.
        """

        for i, X in enumerate(X_mf):
            X = correct_formatX(X,self.d)
            y = Z_mf[i].reshape(-1,1) # smt uses second dimension (..,1) instead of (..,)
            # print(f"{X.shape:} {y.shape:}")
            if i == len(X_mf) - 1:
                # if top level
                model.set_training_values(X, y)
            else:
                model.set_training_values(X, y, name = i)

    def train(self):
        """
        Update the training data and train.
        """
        if self.X_mf[-1].shape[0] >= len(self.X_mf):
            if hasattr(self,'not_maximum_level'):
                super().__init__(**self.MFK_kwargs)
                del self.not_maximum_level
            self.set_training_data(self, self.X_mf, self.Z_mf)
        else:
            print("WARNING: TRAINING WITHOUT TOPLEVEL")
            self.set_training_data(self, self.X_mf[:self.max_nr_levels - 1], self.Z_mf[:self.max_nr_levels - 1])
            self.not_maximum_level = True
        super().train()
        self.setK_mf()

    def setK_mf(self):
        """
        After each model update this function needs to be called to set the K_mf list, 
        such that the code does not change with respect to stacked Kriging models (original 'MFK')
        """
        newK_mf = True if self.K_mf == [] else False
        if newK_mf:
            K_mf = []
        else:
            K_mf = self.K_mf

        for l in range(self.number_of_levels - isinstance(self, ProposedMultiFidelityKriging)):
            obj = ObjectWrapper(self, l)

            # K.predict = lambda x: self._predict_l(x,l) 
            if newK_mf:
                K_mf.append(obj)
            else:
                K_mf[l] = obj

        # # to test
        # for i, K in enumerate(K_mf):
        #     print(K.predict(self.X_mf[-1]))
        #     print(self._predict_intermediate_values(self.X_mf[-1], i + 1))
        #     print(self._predict_l(self.X_mf[-1], i))
        #     print("---------------------------------------")

        self.K_mf = K_mf

    def _predict_l(self, X_new, l):
        """
        Predicts and returns the prediction and associated mean square error
        
        Returns
        -------
        - y_hat : the Kriging mean prediction
        - mse_var : the Kriging mean squared error
        """
        X_new = correct_formatX(X_new, self.d)

        # call for the value at that level
        # if lvl == highest level then equivalent to calling sm.predict_values(x)
        lvl = l + 1 # lvl is 1-indexed, bcs stupid!! 
        y_hat = self._predict_intermediate_values(X_new, lvl).reshape(-1,)

        # call for the MSE, contains all MSE for each level (over the columns)
        MSE, sigma2_rhos = self.predict_variances_all_levels(X_new)
        
        # std = np.sqrt(MSE[:,l].reshape(-1,1))        
        MSE_lvl = MSE[:,l].reshape(-1,)

        # for the HF only
        # y = sm.predict_values(x)
        # var = sm.predict_variances(x)

        return y_hat, MSE_lvl 

    def predict_top_level(self, X):
        """
        Return the toplevel regardless of the fake level in place.
        """
        z,mse = self.predict_values(X).reshape(-1,), self.predict_variances(X).reshape(-1,)
        return z, mse

    
    def set_state(self, data_dict):
        # and then set all attributes
        for key in data_dict:
            if key not in ['K_mf_list','K_truth']:
                setattr(self, key, data_dict[key])

        # only does something when there is a X_truth and Z_truth
        if 'K_truth' in data_dict:
            self.create_update_K_truth(data_dict['K_truth'])

        # K_mf for the first two MFK levels in case of ProposedMFK_EGO! otherwise full
        self.setK_mf()



# pass the sm object to the K_mf list and map the 'predict(x,l)' function to 'predict(x)'
# https://stackoverflow.com/questions/1443129/completely-wrap-an-object-in-python
class ObjectWrapper(MFK_smt):
    def __init__(self, baseObject, l_fake):
        # TODO dit neemt ook de oude K_mf mee dus je krijgt een zieke nesting van objecten! ofwel gebruik de MFK_smt set_state
        self.__dict__ = deepcopy({k: baseObject.__dict__[k] for k in set(list(baseObject.__dict__.keys())) - set({'K_truth','X_truth','Z_truth'})})
        self.K_mf = []
        # fake hps for proposed method
        self.hps = None # TODO zou nice zijn hier de echte hps, geconverteerd naar mijn format, te hebben
        self.l_fake = l_fake
        # TODO sigma_hat!

    def predict(self, X):
        """
        Predicts and returns the prediction and associated mean square error
        
        Returns
        -------
        - y_hat : the Kriging mean prediction
        - mse_var : the Kriging mean squared error
        """
        return self._predict_l(X, self.l_fake)
    
    def corr(self, X, X_other, hps):
        # NOTE from mfk.py line 419
        # dx = self._differences(X, Y=self.X_norma_all[lvl])
        dx = self._differences(X, Y=X_other)
        d = self._componentwise_distance(dx) 
        r_ = self._correlation_types[self.options["corr"]](
            self.optimal_theta[self.l_fake], d
        ).reshape(X.shape[0], X_other.shape[0])
        return r_


" Say what class to inherit from!! "
smt = True
class ProposedMultiFidelityKriging(MFK_smt if smt else MFK_org, MultiFidelityKrigingBase):
    """
    eigenlijk zou 'MultiFidelityKrigingBase' de implementatie van LaGratiet moeten zijn!

     ik moet dus in 'MultiFidelityKrigingBase' de implemenatie van SMT importeren en zorgen dat dezelfde data interface bestaat.
     dan is dus de integratie van LaGratiet in mijn proposed method ook straightforward omdat K1 dan al LaGratiet MF-Kriging is!!
     wel interesant te bedenken: werkt voor de proposed method onafhankelijke noise estimates beter?
    """

    def __init__(self, *args, **kwargs):
        """
        param smt (bool): if True use the MFK solution of smt to build the prediction and other models. 
            Otherwise, use separate Original Kriging models for the truth and other models / levels.
        """
        super().__init__(*args, **kwargs)

        # list of predicted z values at the highest/ new level
        self.Z_pred = np.array([])
        self.mse_pred = np.array([])

    def prepare_proposed(self, setup):
        # doe = get_doe(setup)
        X_l = LHS(setup, n_per_d = 10)
        
        if isinstance(self, MFK_smt):
            self.sample_new(0, X_l)
            self.sample_new(1, X_l) 

        elif isinstance(self, MFK_org):
            tune = True

            hps = None
            if not tune:
                # then try to use some old hps
                if setup.d == 2:
                    # hps for Branin
                    hps = np.array([-1.42558281e+00, -2.63967644e+00, 2.00000000e+00, 2.00000000e+00, 1.54854970e-04])
                elif setup.d == 1:
                    # hps for Forrester
                    hps = np.array([1.26756467e+00, 2.00000000e+00, 9.65660595e-04])
                else:
                    tune = True
        
            self.create_OKlevel(X_l, tune=tune, hps_init = hps)
            self.create_OKlevel(X_l, tune=tune)
        
        else:
            print("Not initialised as a form of MFK.\nNo levels created: exiting!")
            import sys
            sys.exit()

        " level 2 / hifi initialisation "
        # do we want to sample the complete truth (yes!)
        self.sample_truth()

        # sampling the initial hifi
        self.sample_initial_hifi(setup) 
        # self.setK_mf() # already called indirectly!

        self.K_mf = self.K_mf[:2] # might be needed in case of MFK
        self.number_of_levels = 2

        # and add the (reinterpolated!) predictive level
        if isinstance(self, MFK_smt): 
            # Universal Kriging does not always provide a representative correlation function for our goal due to the trend!
            # so, first tune an OK on lvl 1
            K_mf_new = self.create_OKlevel(self.X_mf[1], self.Z_mf[1], append = True, tune = True, hps_noise_ub = True) # sigma_hat is not known yet!

            # do an initial prediction
            Z_pred, mse_pred, _ = wp.weighted_prediction(self)

            K_mf_new.train(self.X_unique, Z_pred, tune = True, retuning = False, R_diagonal = mse_pred / K_mf_new.sigma_hat)
        else:
            # do an initial prediction
            Z_pred, mse_pred, _ = wp.weighted_prediction(self)

            K_mf_new = self.create_OKlevel(self.X_unique, Z_pred, append = True, tune = True, hps_noise_ub = True, R_diagonal= mse_pred / self.K_mf[-1].sigma_hat)
        K_mf_new.reinterpolate()


    def set_state(self, data_dict):
        super().set_state(data_dict)
        
        if isinstance(self, MFK_smt):
            # then add the highest level (which is an OK object for now)
            self.number_of_levels = 2
            k = self.create_OKlevel([],add_empty=True)
            k.set_state(data_dict['K_mf_list'][0])
            if len(data_dict['K_mf_list']) > 1:
                print("WARNING: K_mf_list should be of length 1; only containing the predicted level.")
             


class EfficientGlobalOptimization():
    """This class performs the EGO algorithm on a given layer"""

    def __init__(self, setup, *args, **kwargs):
        # initial EI
        self.ei = 1
        self.ei_criterion = 2 * np.finfo(np.float32).eps
        lb = setup.search_space[1]
        ub = setup.search_space[2]
        n_infill_per_d = 601
        self.X_infill = create_X_infill(setup.d, lb, ub, n_infill_per_d)


    @staticmethod
    def EI(y_min, y_pred, sigma_pred):
        """
        Expected Improvement criterion, see e.g. (Jones 2001, Jones 1998)
        param y_min: the sampled value belonging to the best X at level l
        param y_pred: (predicted) points at level l, locations X_pred
        param sigma_pred: (predicted) variance of points at level l, locations X_pred
        """

        u = np.where(sigma_pred == 0, 0, (y_min - y_pred) / sigma_pred)  # normalization
        EI = sigma_pred * (u * scistats.norm.cdf(u) + scistats.norm.pdf(u))
        return EI


class MultiFidelityEGO(ProposedMultiFidelityKriging, MFK_smt, EfficientGlobalOptimization):
    
    def __init__(self, setup, *args, **kwarg):
        super().__init__(setup, *args, **kwarg)
        # Ik snap die inheritance structuur niet, onderstaand lijkt goed uit te leggen
        # https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way
        EfficientGlobalOptimization.__init__(self, setup, *args, **kwarg) 

    def optimize(self, pp = None, cp = None):
        self.max_cost = np.inf
        while np.sum(self.costs_total) < self.max_cost:
            " output "
            if pp != None:
                pp.draw_current_levels(self)
            if cp != None:
                cp.plot_convergence()

            # predict and calculate Expected Improvement
            y_pred, sigma_pred = self.K_mf[-1].predict(self.X_infill) 
            _, y_min = get_best_sample(self) 

            ei = self.EI(y_min, y_pred, np.sqrt(sigma_pred))

            # select best point to sample
            print("Maximum EI: {:4f}".format(np.max(ei)))
            x_new = correct_formatX(self.X_infill[np.argmax(ei)], self.d)

            # terminate if criterion met, or when we revisit a point
            # NOTE level 2 is reinterpolated, so normally 0 EI at sampled points per definition!
            # however, we use regulation (both through R_diagonal of the proposed method as regulation constant for steady inversion)
            # this means the effective maximum EI can be higher than the criterion!
            if np.all(ei < self.ei_criterion) or isin(x_new,self.X_mf[-1]):
                break


            l = self.level_selection(x_new)
            print("Sampling on level {} at {}".format(l,x_new))
            
            self.sample_nested(l, x_new)

            # weigh each prediction contribution according to distance to point.
            Z_pred, mse_pred, _ = wp.weighted_prediction(self)

            # build new kriging based on prediction
            # NOTE, we normalise mse_pred here with the previous known process variance, since otherwise we would arrive at a iterative formulation.
            # NOTE for top-level we require re-interpolation if we apply noise
            if self.tune_counter % self.tune_prediction_every == 0:
                tune = True
            else: 
                tune = False

            # if we sample on level 2, we really should first retune for correcting i.e. sigma_hat, without R_diag to get the correct base, just like at init
            if l == 2:
                self.K_mf[-1].train(self.X_unique, Z_pred, tune=tune)
            
            # train the prediction model, including heteroscedastic noise
            self.K_mf[-1].train(
                self.X_unique, Z_pred, tune=tune#, R_diagonal= mse_pred / self.K_mf[-1].sigma_hat
            )
            self.K_mf[-1].reinterpolate()

            if cp != None:
                # plot convergence here and not before break, otherwise on data reload will append the same result over and over
                cp.update_convergence_data(self, x_new, ei) # pass these, we do not want to recalculate!!



        " Print end-results "
        s_width = 71
        print("┏" + "━" * s_width + "┓")
        print("┃ Best found point \t\tx = {}, f(x) = {: .4f} \t┃".format(get_best_sample(self)[0][0],get_best_sample(self)[1]))
        if hasattr(self,'K_truth') and hasattr(self.K_truth,'X_opt'):
            print("┃ Best (truth) predicted point \tx = {}, f(x) = {: .4f} \t┃".format(self.K_truth.X_opt[0], self.K_truth.z_opt))
        if isinstance(self.solver, TestFunction):
            X_opt, Z_opt = self.solver.get_optima()
            X_opt, Z_opt = np.atleast_2d(X_opt), np.atleast_2d(Z_opt)
            ind = np.argmin(Z_opt)
            print("┃ Exact optimum at point \tx = {}, f(x) = {: .4f} \t┃".format(X_opt[ind], Z_opt[ind].item()))
            print("┃ Search resolutions of \tx = {}\t\t\t┃".format((self.ub - self.lb)/(self.n_infill_per_d-1)))
        print("┗" + "━" * s_width + "┛")
             
    def level_selection(self, x_new):
        # select level l on which we will sample, based on location of ei and largest source of uncertainty there (corrected for expected cost).
        # source of uncertainty is from each component in proposed method which already includes approximated noise levels.
        # in practice this means: only for small S1 will we sample S0, otherwise S1. Nested DoE not required.
        # For convenience: we will now threat the proposed method as a two-level MF (0 & 1) and 2
        # this especially makes sense since the proposed method operates on the premise that sampling 
        # an extra level (level 0) is worth it compared to the gains of the prediction!

        # 1) check of de kriging variances gelijk zijn, aka of de predicted variances van de kriging toplevel 
        # ongeveer overeenkomen met de prediction gebaseerd op lagere levels. Dus punt x_new virtueel aan X_unique voor weighted_prediction toevoegen en vergelijken.
        # 1) variances van Kriging unknown scheiden
        # 2) meenemen in de weighing
        # 3) scalen mbv cost_exp naar Meliani, i.e. argmax(sigma_red[l]/cost_exp[l])

        sigma_red = self._std_reducable_proposed(x_new) # kan op x_new 0 zijn
        # s0_red, s1_red, s2_red = self._mse_reducable_meliani_smt(x_new)
        
        maxx = 0
        l = 0
        # dit is het normaal, als de lagen gelinkt zijn! is hier niet, dit werkt alleen voor meliani.
        for i in range(1,3):
            value = sigma_red[i] / self.costs_expected_nested[i]**2
            maxx = max(maxx,value)
            # NOTE gives preference to highest level, which is good:
            # if no reductions expected anywhere lower, we should sample the highest (our only 'truth')
            if value == maxx:
                l = i
                
        # check per sample if not already sampled at this level
        # NOTE broadcasting way
        if isin(x_new,self.X_mf[l]):
            l += 1 # then already sampled! given we stop when we resample the highest level, try to higher the level we are on!
            print("Highered level, sampling l = {} now!".format(l))
        
        return l

    def _std_reducable_proposed(self, x_new):
        """
        Calculate the reducable variance at x_new, tailored for the proposed method (lvl 2) + smt (lvl 0 & 1)

        There are 3 sources of mse prediction information that we want to translate 
        to a variance reduction at lvl 2 by sampling level l:
        1) weighed prediction mse
        2) kriging prediction mse (can be spurious!!)
        3) smt mfk prediction mse

        Of any end-ansers for projections to lvl2, only take the lowest expected reduction. 
        This improves stability primarily because the prediction level might not always be perfectly tuned, 
        which would lead to unnecessarily selecting the highest level. 
        This discrepency is the result of a mixture of different model types (UK & OK) and different (tuner) codes.
        """
        

        " 1) weighed prediction mse "
        z_pred_weighed, mse_pred_weighed, Ef_weighed = wp.weighted_prediction(self, X_test=x_new)


        " 2) kriging prediction mse (can be spurious!!) " 
        # noise corrected not true at lvl 2 due to reinterpolation!
        z_pred_model, mse_pred_model = self._mse_reducable_predict(2, x_new, noise_correct=False) 

        " 3) smt mfk prediction mse "
        # with propagate_uncertainty, mse is as-is; 
        # this is important to the proposed methods split-up since each level contributes independently!
        std_pred_smt = np.sqrt(self._mse_reducable_meliani_smt(x_new, propagate_uncertainty = False))


        " Results selection "
        # Now, calculate the weighed result
        # s2 = s1 + Ef_weighed * (s1 + s0) + weighted(Sf) * (z1 - z0)
        # last term does not provide an expected reduction, so we only require a weighed Ef
        # we assume a fully nested DoE
        s0_exp_red = Ef_weighed * std_pred_smt[0]
        s1_exp_red = std_pred_smt[1] + Ef_weighed * (std_pred_smt[0] + std_pred_smt[1])
        
        # now various options for lvl 2:
        options = [np.sqrt(mse_pred_model).item(), np.sqrt(mse_pred_weighed).item()]
        if len(std_pred_smt) == 3:
            options.append(std_pred_smt[2])
        print(f"MSE lvl 2 options are: mse_pred_model, mse_pred_weighed, mse_pred_smt\n\t{options}")
        # s2_exp_red = min(options)
        s2_exp_red = np.sqrt(mse_pred_weighed).item()


        print("Z_pred_weighed: {:.4f}; Z_pred_model: {:.4f}".format(z_pred_weighed.item() ,z_pred_model.item()))
        print("std_pred_weighed: {:.8f}; std_pred_model: {:8f}".format(np.sqrt(mse_pred_weighed).item(), np.sqrt(mse_pred_model).item()))

        s_exp_red = [s0_exp_red, s1_exp_red, s2_exp_red]
        print(s_exp_red)
        return s_exp_red 
        
    ##################### OLD
    def _mse_reducable_predict(self, l, x_new, noise_correct = True):
        """
        Tries to find the part of the mse that is reducable by sampling and correcting for noise at the sampled points.
        This time, by taking the variance at sampled points from the predictor and averaging those.
        Seems a more logical and reasonable solution.
        """
        _, mse = self.K_mf[l].predict(self.X_mf[l])
        z_new, mse_new = self.K_mf[l].predict(x_new)

        if noise_correct:
            return z_new, np.clip(0,mse_new - np.average(mse)).item()
        else:
            return z_new, mse_new

    def _std_reducable2(self, x_new):
        ret = []
        for l in range(self.number_of_levels - 1, -1, -1):
            predictor = self.K_mf[l]
            if isinstance(predictor,MFK):
                # ret += self._std_reducable2_smt(predictor, l, x_new)
                ret1 = self._mse_reducable_meliani_smt(x_new)[:2] # NOTE only take first two
                ret1.reverse()
                ret += ret1
                break
            else:
                ret.append(self._std_reducable_org(predictor, x_new))
        
        ret.reverse()
        return np.array(ret).clip(min=0)



    def _std_reducable2_org(self, predictor, x_new):
        """
        Finds the part of the mse that is reducable by i.e. sampling.
        Does this by predicting at location x_new, training as if it was sampled, predicting.
        What is left is the base variance present even at the new sample (without tuning).
        Comparing gives the expected reducable variance.

        NOTE does not work with proximate samples!
        """
        # noise_hp = predictor.hps[-1]

        # get the full noise prediction
        y_add, s0 = predictor.predict(x_new)
        s0 = np.sqrt(s0) 

        R_diagonal_old = 0
        X_old = predictor.X
        y_old = predictor.y
        X = np.append(X_old, x_new, axis=0)
        y = np.append(y_old, y_add)
        
        R_diagonal_old = predictor.R_diagonal
        if np.any(R_diagonal_old != 0):
            R_diagonal = np.append(R_diagonal_old,0)
        else:
            R_diagonal = R_diagonal_old
        
        # do a fake training by training on extended dataset with extend y = expectation, then retaining on old dataset
        predictor.train(X, y, R_diagonal = R_diagonal)
        y_updated, s0_updated = predictor.predict(x_new)
        
        # reset (retrain actually) to the previous state
        predictor.train(X_old, y_old, R_diagonal = R_diagonal_old)

        s_reducable = (np.sqrt(s0) - np.sqrt(s0_updated)).item()

        # mse_reducable = np.sqrt(max(mse_reducable,0))
        # return float(mse_reducable), s0
        print("s = {} of which reducable {}".format(s0,s_reducable))
        return s_reducable


    def _mse_reducable_meliani_smt(self, x_new, propagate_uncertainty = True):
        # NOTE TODO melianis level selection / calculation of sigma2 reduction does not include noise at sampled points!! (quite certain)
        # heel zeker als je kijkt binnen predict_variances_all_levels
        
        # NOTE optie propagate_uncertainty doet MSE[:, i] = MSE[:, i] + sigma2_rho * MSE[:, i - 1]
        # delta is zijn eigen GP met N(0,sigma) -> 'simple Kriging'
        # hierbij is MSE[:, i] = delta_k,0 op level 0 en delta_k,l = MSE[:, l]
        # omdat de levels van le gratiet gelinkt zijn is de onzekerheid opbouwend en er is ook de eigen kriging onzekerheid
        # als gratiet (check dit) de discrepancy onzekerheid neemt aangenomen dat de constante (!) multiplicative factor klopt 
        # (en dus de mean van onderliggende klopt), dan is de disceprepancy onzekerheid simpelweg de onzekerheid op het bovenliggende nivuea
        # ofwel in de vergelijking: MSE[:, i] = MSE[:, i] + sigma2_rho * MSE[:, i - 1], 
        # als we dan kijken naar de *verschillen tussen levels* zoals ik dat doe:
        # sd1_added = sd1 + Ef * sd0 ~ s2 + s1 + Ef * (s1 + s0) -- (s2 assumed no var) --> s1 + Ef * (s1 + s0)

        
        self.options["propagate_uncertainty"] = propagate_uncertainty # standard option, calculates the cumulative 
        MSE_contribution, sigma2_rhos = self.predict_variances_all_levels(x_new)

        # the std2 contribution of level k to the highest fidelity
        # sigma_con = []
        # for l in range(MSE.shape[0]):
        #     sigma_con.append()

        # sigma_red is the sum of all sigma_con`s up to level k (where we sample)
        # NOTE I think Meliani is incorrect. Yes we do sample lower levels, and yes they do have a uncertainty contribution
        # However: when you sample a higher level, lower levels cannot provide extra info anymore to that level (see Kennedy & O`Hagan)
        # TODO Korondi 2021 gebruiken!!!!! die doet het wel goed, i.e. gebruikt alleen de sigma_con!


        return MSE_contribution.flatten()
        
