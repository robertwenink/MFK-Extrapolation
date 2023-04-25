
# pyright: reportGeneralTypeIssues=false
import numpy as np
from copy import deepcopy, copy

from smt.applications import MFK
from core.mfk.mfk_ok import MFK_org
from core.sampling.DoE import LHS
from utils.formatting_utils import correct_formatX

from core.mfk.mfk_base import MultiFidelityKrigingBase

class MFK_wrap(MFK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_state(self):
        # not: ,'F_all','p_all','q_all','y_norma_all'
        state = {k: self.__dict__[k] for k in set(list(self.__dict__.keys())) 
        - set({'options','printer','D_all','F','p','q','optimal_rlf_value','ij','supports','X','X_norma','best_iteration_fail','nb_ill_matrix','sigma2_rho','training_points','y','nt',
        'kernel','K_mf','solver','K_truth','K_pred','X_infill','tune_prediction_every','tune_lower_every','tune_counter','lambs','lamb1','lamb2','pdf'})}
        
        if hasattr(self, 'K_truth'):
            state['K_truth'] = self.K_truth.get_state()

        if hasattr(self, 'K_pred'):
            state['K_pred'] = self.K_pred.get_state()

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

    def predict(self, X, redo_Ft = True):
        y, mse = self.predict_values(X).reshape(-1,), self.predict_variances_all_levels(X, redo_Ft)[0][:, -1].reshape(-1,)
        return y, mse
    
    # TODO this is only called for sf functions. Double of the function in Object_wrapper
    def corr(self, X, X_other):
        # NOTE from mfk.py line 419
        dx = self._differences(X, Y=X_other)
        d = self._componentwise_distance(dx) 
        r_ = self._correlation_types[self.options["corr"]](
            self.optimal_theta[0], d 
        ).reshape(X.shape[0], X_other.shape[0])
        return r_

class MFK_smt(MFK_wrap, MultiFidelityKrigingBase):
    """Provide a wrapper to smt`s MFK, to interface with the own code."""

    def __init__(self, *args, **kwargs):
        self.MFK_kwargs = kwargs['MFK_kwargs']
        super().__init__(**self.MFK_kwargs)
        MultiFidelityKrigingBase.__init__(self, *args, **kwargs)


    def prepare_initial_surrogate(self, setup, X_l = None):
        # doe = get_doe(setup)
        if X_l is None:
            X_l = LHS(setup, n_per_d = 10)
        
        if isinstance(self, MFK_org):
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

        elif isinstance(self, MFK_smt):
            self.sample_new(0, X_l)
            self.sample_new(1, X_l) 
        else:
            if self.printing:
                print("Not initialised as a form of MFK.\nNo levels created: exiting!")
            import sys
            sys.exit()

        " level 2 / hifi initialisation "
        # do we want to sample the complete truth? (yes!)

        if self.printing:
            print(f"{'':=>25}")
            print("Sampling the full truth!")
            print(f"{'':=>25}")

        self.sample_truth()

        # sampling the initial hifi
        self.sample_initial_hifi(setup) 
        # self.setK_mf() # already called indirectly!


    def sample_nested(self, l, X_new, train = True):
        """
        Function that does all required actions for nested core.sampling.
        """

        super().sample_nested(l, X_new)
        if train:
            self.train()

    def create_update_K_truth(self, data_dict = None):
        if hasattr(self,'X_truth') and hasattr(self,'Z_truth'):
            # overloaded function
            if not hasattr(self,'K_truth'):
                if self.printing:
                    print("Creating Kriging model of truth")
                kwargs = deepcopy(self.MFK_kwargs)
                kwargs['n_start'] *= 2 # truth is not tuned often so rather not have any slipups
                kwargs['optim_var'] = self.optim_var
                self.K_truth = MFK_wrap(**kwargs)
                self.K_truth.name = "K_truth"

            if data_dict != None:
                self.K_truth.set_state(data_dict)
            else:
                self.set_training_data(self.K_truth, [*self.X_mf[:self.max_nr_levels-1], self.X_truth], [*self.Z_mf[:self.max_nr_levels-1], self.Z_truth])
                self.K_truth.train() 
                self.K_truth.X_opt, self.K_truth.z_opt = self.get_best_prediction(self.K_truth, x_centre_focus = self.K_truth.X_opt if hasattr(self.K_truth,'X_opt') else None)
                if self.printing:
                    print("Succesfully trained Kriging model of truth")#, end = '\r')

    
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
        TRAIN_MFK_WHEN_SF_PROPOSED = True
        # TODO this is only valid when no method weighing is done!!
        if not (self.proposed and self.use_single_fidelities and not TRAIN_MFK_WHEN_SF_PROPOSED):# or True: # TODO
            if self.X_mf[-1].shape[0] >= len(self.X_mf):
                # then just train the full model
                if hasattr(self,'not_maximum_level'):
                    # when changing the amount of used levels, the model should be re-initialised
                    super().__init__(**self.MFK_kwargs)
                    del self.not_maximum_level
                self.set_training_data(self, self.X_mf, self.Z_mf)
                
                # if we have all levels, we want optim_var to be true for resampling purposes! 
                if hasattr(self,'proposed') and self.proposed == True:
                    # NOTE no reinterpolation if proposed method, 
                    # bcs we do not used method weighing and we need the lower fidelities to retain their variances.
                    # reinterpolation on the other hand does not fucntion properly in smt if lower levels retain their variances.
                    self.options['optim_var'] = False 
                else:
                    self.options['optim_var'] = self.optim_var

            else:
                if self.printing:
                    print("WARNING: TRAINING WITHOUT TOPLEVEL")
                self.set_training_data(self, self.X_mf[:self.max_nr_levels - 1], self.Z_mf[:self.max_nr_levels - 1])
                self.not_maximum_level = True
                self.options['optim_var'] = False # explicitly set to false, the medium level is the 'high' fidelity here!!
            
            super().train()
        self.setK_mf() # ALWAYS call setK_mf bcs K_mf is used in weighed prediction, i.e. create_update_K_pred is only called AFTER weighed prediction -> = Fault!
        

    def setK_mf(self, only_rebuild_top_level = False, retrain = False):
        """
        After each model update this function needs to be called to set the K_mf list, 
        such that the code does not change with respect to stacked Kriging models (original 'MFK')

        In case this is the proposed method, we are tracking two things:
        1) The 'regular' mfk model
        2) The proposed mfk model; 
            our model then has, similar to K_truth, a Z_pred and X_pred (next to mse_pred)
            if proposed exists, we init K_mf using it

        @param only_rebuild_top_level: only retrain the top level. This implies lower levels are/should be already up-to-date!
            in terms of training only actually important while using single fidelity levels, since they are created/trained in this function.
        @param retrain (unused): was used to retrain each step to check if that would improve results.
        """

        # in case of proposed
        USE_SF_LEVELS_FOR_PROPOSED = self.use_single_fidelities
        
        K_mf = []

        nlvl = 3 if self.proposed else self.nlvl

        if USE_SF_LEVELS_FOR_PROPOSED and self.proposed:
            # then top-level will be K_pred, which should already be formulated (setK_mf called after create_update_K_pred!)
            if only_rebuild_top_level:
                # just re-use the previously defined and trained levels
                for l in range(nlvl - 1):
                    K_mf.append(self.K_mf[l])
            else:
                if self.printing:
                    print("Training single-fidelity lower levels.")
                for l in range(nlvl - 1):
                    if len(self.K_mf) != 0 and not retrain: #K_mf is defined in the kriging_base
                        sf_model = self._create_train_sf_model(l, old_model=self.K_mf[l])
                    else:
                        sf_model = self._create_train_sf_model(l)
                    K_mf.append(sf_model)

            # always add current K_pred model
            if hasattr(self,'K_pred'):
                # K_pred not defined on first call (it uses the lower levels of K_mf to be build)
                K_mf.append(self.K_pred)
        else:
            for l in range(nlvl):
                base_object = self
                if l == 2 and self.proposed:
                    if hasattr(self, 'K_pred'):
                        base_object = self.K_pred
                        l = base_object.nlvl - 1 # needed to get correct predict_l in case of SF K_pred
                        obj = ObjectWrapper(base_object, l)
                else:
                    obj = ObjectWrapper(base_object, l)

                K_mf.append(obj) #type:ignore

        # # to test
        # for i, K in enumerate(K_mf):
        #     print(K.predict(self.X_mf[-1]))
        #     print(self._predict_intermediate_values(self.X_mf[-1], i + 1))
        #     print(self._predict_l(self.X_mf[-1], i))
        #     print("---------------------------------------")

        self.K_mf = K_mf

    def _create_train_sf_model(self, l, old_model = None):
        """
        For creating and training a single fidelity level Kriging model (implemented as a MFK with just one level)
        """

        # initialise
        if old_model is None:
            kwargs = deepcopy(self.MFK_kwargs)
            kwargs['optim_var'] = False # required False!
            kwargs['eval_noise'] = True
            kwargs['noise0'] = [[0.0]] # stupid that this is required
            kwargs['n_start'] *= 2
            sf_model = MFK_wrap(**kwargs)
            sf_model.name = f"sf_model_{l}"
        else:
            sf_model = old_model
            # self.options['noise0'] = [[0.0]]


        # train
        sf_model.set_training_values(self.X_mf[l], self.Z_mf[l])
        sf_model.train()

        return sf_model

    def _predict_l(self, X_new, l, redo_Ft = True):
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
        MSE, sigma2_rhos = self.predict_variances_all_levels(X_new, redo_Ft)
        
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
            if key not in ['K_mf_list','K_truth','K_pred']:
                setattr(self, key, data_dict[key])

        # only does something when there is a X_truth and Z_truth
        if 'K_truth' in data_dict:
            self.create_update_K_truth(data_dict['K_truth'])

        if 'K_pred' in data_dict:
            self.create_update_K_pred(data_dict['K_pred'])
        else:
            # K_mf for the first two MFK levels in case of ProposedMFK_EGO! otherwise full
            self.setK_mf()



# pass the sm object to the K_mf list and map the 'predict(x,l)' function to 'predict(x)'
# https://stackoverflow.com/questions/1443129/completely-wrap-an-object-in-python
class ObjectWrapper(MFK_smt):
    def __init__(self, baseObject : MFK_smt, l_fake):
        
        # direct onderstaande neemt ook de oude K_mf mee dus je krijgt een zieke nesting van objecten! ofwel gebruik de MFK_smt set_state
        # self.__dict__ = deepcopy({k: baseObject.__dict__[k] for k in set(list(baseObject.__dict__.keys())) - set({'K_truth','X_truth','Z_truth'})})

        MFK_wrap.__init__(self, **deepcopy(baseObject.MFK_kwargs))
        for key, item in baseObject.get_state().items():
            if key not in ['K_truth', 'K_pred']: # dont add the full dicts of these sub-models
                self.__setattr__(key,deepcopy(item))

        self.K_mf = []
        self.l_fake = l_fake

    def predict(self, X, redo_Ft = True):
        """
        Predicts and returns the prediction and associated mean square error
        
        Returns
        -------
        - y_hat : the Kriging mean prediction
        - mse_var : the Kriging mean squared error
        """
        return self._predict_l(X, self.l_fake, redo_Ft)
    
    def corr(self, X, X_other):
        # NOTE from mfk.py line 419
        # dx = self._differences(X, Y=self.X_norma_all[lvl])
        dx = self._differences(X, Y=X_other)
        d = self._componentwise_distance(dx) 
        r_ = self._correlation_types[self.options["corr"]](
            # self.optimal_theta[self.l_fake], d NOTE poging tot normalizeren
            # max(self.optimal_theta), d * (10/2) ** 2
            self.optimal_theta[self.l_fake], d # TODO normalize with the average distance between real datapoints
        ).reshape(X.shape[0], X_other.shape[0])
        return r_

