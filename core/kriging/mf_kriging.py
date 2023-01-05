# pyright: reportGeneralTypeIssues=false,
import numpy as np
from beautifultable import BeautifulTable

from core.kriging.OK import OrdinaryKriging
from core.sampling.DoE import LHS_subset
from utils.formatting_utils import correct_formatX
from utils.error_utils import RMSE_norm_MF
from utils.correlation_utils import check_correlations
from utils.formatting_utils import correct_formatX

class MultiFidelityKriging():

    def __init__(self, kernel, d, solver, max_cost = None, max_nr_levels : int = 3, *args, **kwarg):
        """
        @param kernel (list): list containing kernel function, (initial) hyperparameters, hyper parameter constraints
        @param max_cost: maximum cost available for sampling. 
        @param max_nr_levels: maximum number of Kriging levels that is added to the MF model. Is considered to be the level of the high-fidelity + 1.
        @param L: fidelity input list, is being set using set_L() if other values are desired.
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
        self.X_mf = [] # NOTE gaat bij input readen ervanuit dat de lijsten van verschillende lengtes zijn (wat eigenlijk altijd zo is)!
        self.X_unique = np.array([])
        self.Z_mf = []
        
        # cost variables: total cost per level; expected cost per sample per level 
        self.costs_per_level = {}
        self.costs_expected = {}
        self.costs_expected_nested = {}
        self.costs_total = 0

        self.L = np.arange(max_nr_levels) + 1

    def set_L(self, L : list):
        self.L = L
        self.max_nr_levels = len(L)


    def create_level(self, X_l, y_l = [], train=True, tune=False, hps_init=None, hps_noise_ub = False, R_diagonal=0, name = "", append : bool = True, add_empty : bool = False):
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
            hps_init = self.K_mf[-1].hps

        kernel = self.kernel
        if hps_noise_ub and (hps_init is not None):
            #TODO this is done for the complete MF_KRIGING!!!!
            # noise upperbound should be larger than previous tuned noise!
            kernel[-1][-1,1] = min(max(hps_init[-1] * 2, kernel[-1][-1,1]),0.5)

        name = "Level {}".format(self.number_of_levels) if name == "" else name

        ok = OrdinaryKriging(kernel, self.d, hps_init=hps_init, name = name)

        if tune == False and not add_empty:
            if hps_init is None:
                raise NameError(
                    "hps_init is not defined! When tune=False, hps_init must be defined"
                )

        if not np.any(y_l) and not add_empty:
            if append:
                self.sample_new(self.number_of_levels, X_l)
                y_l = self.Z_mf[self.number_of_levels]
            else:
                y_l, _ = self.solver.solve(X_l, self.L[self.number_of_levels])

        if train and not add_empty:
            retuning = True if add_empty else False
            ok.train(X_l, y_l, tune, retuning, R_diagonal)
        
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
            self.costs_per_level[l] += cost
        except:
            self.costs_per_level[l] = cost
            self.costs_expected[l] = cost
            self.costs_expected_nested[l] = cost

        # expected cost per sample at a level
        self.costs_expected[l] = self.costs_per_level[l] / self.X_mf[l].shape[0]
        self.costs_expected_nested[l] = sum([self.costs_expected[i] for i in range(l+1)])
        self.costs_total = sum(self.costs_per_level.values())

    def print_stats(self,RMSE):
        """Print insightfull stats. Best called after adding a high-fidelity sample."""
        table = BeautifulTable()
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Total costs: {}".format(self.costs_total))
        row0 = [self.X_mf[l].shape[0] for l in range(self.number_of_levels)]
        row1 = [self.costs_per_level[l] for l in range(self.number_of_levels)]
        row2 = [self.costs_expected[l] for l in range(self.number_of_levels)]

        table.rows.append(row0)
        table.rows.append(row1)
        table.rows.append(row2)
        table.rows.append(RMSE)
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
        inds = ~(X_new == self.X_mf[l][:, None]).all(axis=-1).any(axis=0)
        
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

        for i in range(l):
            self.sample(i, X_new)
            sampled_levels.append(i)

        if l == self.max_nr_levels - 1:
            # NOTE hase to be after sampling underlying levels, for the correlation checks
            self.sample_truth(X_new) 
        
        # retrain the sampled levels with the newly added sample.
        # NOTE this is proposed method specific actually (where we dont want to train the top level perse)
        for i in sampled_levels:
            if i != self.max_nr_levels - 1: # because we first should do the weighing procedure again.
                self.K_mf[i].train(self.X_mf[i], self.Z_mf[i], tune = True)



    def sample_initial_hifi(self, setup):
        """
        Select best sampled (!) point of previous level and sample there again.
        Then find a LHS subset with (n_subset = parallel capability) and solve.
        """

        # select best sampled (!) point of previous level and sample again there
        x_new = self.X_mf[-1][np.argmin(self.Z_mf[-1])]
        l = self.l_hifi

        # select other points for cross validation, in LHS style
        X_hifi = correct_formatX(LHS_subset(setup, self.X_unique, x_new, self.pc),setup.d)

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
        
        # check correlations and RMSE levels
        RMSE = RMSE_norm_MF(self.X_truth, self.Z_truth, self.K_mf)
        self.print_stats(RMSE)

        # TODO select only those indices where the truth is known! This is needed during EGO.
        if self.Z_mf[0].shape[0] == self.Z_mf[1].shape[0] == self.Z_truth.shape[0]:
            check_correlations(self.Z_mf[0], self.Z_mf[1], self.Z_truth)

        if not hasattr(self,'K_truth'):
            print("Creating Kriging model of truth", end = '\r')
            self.K_truth = self.create_level(self.X_truth, self.Z_truth, tune = True, append = False, name = "Truth")
        else:
            print("Updating Kriging model of truth", end = '\r')
            self.K_truth.train(self.X_truth, self.Z_truth, tune = True, retuning = True)


    def get_state(self):
        """
        Gets the current state of the data of the MF Kriging model.
        all keys available: ['kernel', 'd', 'solver', 'max_cost', 'number_of_levels', 'max_nr_levels', 'l_hifi', 'pc', 'K_mf', 'X_mf', 'X_unique', 'Z_mf', 'costs_tot', 'costs_exp', 'L', 'Z_pred', 'mse_pred', 'X_truth', 'Z_truth']
        keys to avoid: kernel, K_mf (moet per Kriging object weer worden ge-init)
        """
        state = {}
        # MFK: dict_keys(['kernel', 'd', 'solver', 'max_cost', 'number_of_levels', 'max_nr_levels', 'l_hifi', 'pc', 'K_mf', 'X_mf', 'X_unique', 'Z_mf', 'costs_tot', 'costs_exp', 'L', 'Z_pred', 'mse_pred', 'X_truth', 'Z_truth'])
        
        K_mf_list = []
        for K in self.K_mf:
            K_mf_list.append(K.get_state())

        state = {k: self.__dict__[k] for k in set(list(self.__dict__.keys())) - set({'kernel','K_mf','solver','K_truth'})}

        if hasattr(self,'K_truth'):
            state['K_truth'] = self.K_truth.get_state()

        state['K_mf_list'] = K_mf_list

        return state


    def set_state(self, data_dict):
        """
        Sets a (saved) state of the data of the MF Kriging model
        
        Of each model in K_mf we need to retrieve the dictionary too, init them, then set the dictionary. 
        'kernel' is a (numba) function so cant be get or set and needs to be re-initialised -> init the classes like normal, then assign data
        idem for 'solver'
        """

        # NOTE not set explicilty like in OK, quite long list
        for key in data_dict:
            if key not in ['K_mf_list','number_of_levels']:
                if 'K_truth' in key:
                    self.K_truth = self.create_level([],append = False, add_empty=True, name = data_dict['K_truth']['name']) 
                    self.K_truth.set_state(data_dict['K_truth'])
                else:
                    setattr(self, key, data_dict[key])

        # init the new levels
        for l in range(data_dict['number_of_levels']):
            k = self.create_level([],add_empty=True)
            k.set_state(data_dict['K_mf_list'][l])



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
    """
    TODO eigenlijk zou 'MultiFidelityKriging' de implementatie van LaGratiet moeten zijn!
     ik moet dus in 'MultiFidelityKriging' de implemenatie van SMT importeren en zorgen dat dezelfde data interface bestaat.
     dan is dus de integratie van LaGratiet in mijn proposed method ook straightforward omdat K1 dan al LaGratiet MF-Kriging is!!
     wel interesant te bedenken: werkt voor de proposed method onafhankelijke noise estimates beter?
    """
    def __init__(self, *args, **kwarg):
        super().__init__(*args, **kwarg)

        # list of predicted z values at the highest/ new level
        self.Z_pred = np.array([])
        self.mse_pred = np.array([])

class EfficientGlobalOptimization(object):
    """This class performs the EGO algorithm on a given layer"""

    def __init__(self, *args, **kwarg):
        super().__init__(*args, **kwarg)

        # initial EI
        self.ei = 1
        self.ei_criterion = 2 * np.finfo(np.float32).eps

class MultiFidelityEGO(ProposedMultiFidelityKriging, MultiFidelityKriging, EfficientGlobalOptimization):
    
    def __init__(self, *args, **kwarg):
        super().__init__(*args, **kwarg)
        pass

    def level_selection(self, x_new, Ef_weighed):
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

        K0 = self.K_mf[0]
        K1 = self.K_mf[1]
        K2 = self.K_mf[2] # prediction!

        # this is an ESTIMATE of the base noise, actual noise can deviate due to correlations and regularization
        # get the part of the noise that could be reduced by sampling it!
        s0_red, s0 = _std_reducable(K0, x_new)
        s1_red, s1 = _std_reducable(K1, x_new)
        s2_red, s2 = _std_reducable(K2, x_new) # kan op x_new 0 zijn
        print(s0_red,s1_red,s2_red)
        
        # NOTE sigma_red should contain the amount of reduction at the top/prediction level we get by sampling it!
        # this will be different between mine and meliani`s
        # s2 = s1 + weighted(Ef) * (s1 - s0) + weighted(Sf) * (z1 - z0)
        # last term does not provide an expected reduction, so we only require a weighed Ef
        s1_exp_red = s1_red + Ef_weighed * abs(s1 + s0)
        sigma_red = [s1_exp_red, s1_exp_red, s2_red]
        print(sigma_red)
        maxx = 0
        ind_max = 0
        # dit is het normaal, als de lagen gelinkt zijn! is hier niet, dit werkt alleen voor meliani.
        for i in range(1,3):
            #TODO is dit de goede formule?
            value = sigma_red[i]**2/self.costs_expected_nested[i]
            maxx = max(maxx,value)
            # NOTE gives preference to highest level, which is good:
            # if no reductions expected anywhere lower, we should sample the highest (our only 'truth')
            if value == maxx:
                ind_max = i
        
        return ind_max



def _std_reducable(predictor, x_new):
    """
    Finds the part of the mse that is reducable by i.e. sampling.
    If this number is low, near zero, or negative -> sampling not usefull!
    Does this by re-calculating mu_hat sigma_hat and mse using a correlation matrix R where only the noise part is added.
    """
    y = predictor.y
    z0_new, s0 = predictor.predict(x_new)
    s0 = np.sqrt(s0)
    noise_hp = 1 + predictor.hps[-1]

    if np.isclose(noise_hp, 0.0):
        return s0, s0

    R = predictor.corr(predictor.X, predictor.X, predictor.hps)    
    np.fill_diagonal(R, np.ones(predictor.X.shape[0]) + noise_hp + predictor.regularization) # reset to 1s only on diag
    R_in = np.linalg.pinv(R)

    r = predictor.corr(predictor.X, x_new, predictor.hps)

    """Estimated variance sigma squared of the Gaussian process, according to (Jones 2001)"""
    t = y - np.sum(np.dot(R_in, y)) / np.sum(R_in) # y - mu_hat
    sigma_hat = np.dot(t.T, np.dot(R_in, t)) / y.shape[0]

    """Mean squared error of the predictor, according to (Jones 2001)"""
    # hiervan willen we de mse allen op de diagonaal, i.e. de mse van een punt naar een ander onsampled punt is niet onze interesse.
    # t0 = 1 - np.diag(np.dot(rtR_in , r)) # to make this faster we only need to calculates the columns/rows corresponding to the diagonals.
    t = 1 - np.sum(np.multiply(np.dot(r.T,R_in), r.T), axis=-1)
    mse_noise_only = np.abs(sigma_hat * (t + t ** 2 / np.sum(R_in)))

    mse_reducable = s0 - mse_noise_only

    # return mse_reducable
    mse_reducable = np.sqrt(max(mse_reducable,0))
    return float(mse_reducable), s0
    # return np.sqrt(abs(mse_reducable)) * np.sign(mse_reducable)
