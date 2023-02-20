
# pyright: reportGeneralTypeIssues=false,
import numpy as np
from copy import deepcopy, copy

from smt.applications import MFK
from utils.formatting_utils import correct_formatX

from core.mfk.mfk_base import MultiFidelityKrigingBase

class MFK_wrap(MFK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_state(self):
        # not: ,'F_all','p_all','q_all','y_norma_all'
        state = {k: self.__dict__[k] for k in set(list(self.__dict__.keys())) 
        - set({'options','printer','D_all','F','p','q','optimal_rlf_value','ij','supports','X','X_norma','best_iteration_fail','nb_ill_matrix','sigma2_rho','training_points','y','nt',
        'kernel','K_mf','solver','K_truth','K_pred','X_infill','tune_prediction_every','tune_lower_every','tune_counter'})}
        
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

    def predict(self, X):
        return self.predict_values(X).reshape(-1,), self.predict_variances(X).reshape(-1,)

class MFK_smt(MFK_wrap, MultiFidelityKrigingBase):
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
                print("Creating Kriging model of truth")
                kwargs = deepcopy(self.MFK_kwargs)
                kwargs['n_start'] *= 2 # truth is not tuned often so rather not have any slipups
                kwargs['optim_var'] = True
                self.K_truth = MFK_wrap(**kwargs)
                self.K_truth.name = "K_truth"

            if data_dict != None:
                self.K_truth.set_state(data_dict)
            else:
                self.set_training_data(self.K_truth, [*self.X_mf[:self.max_nr_levels-1], self.X_truth], [*self.Z_mf[:self.max_nr_levels-1], self.Z_truth])
                self.K_truth.train()
                self.K_truth.X_opt, self.K_truth.z_opt = self.get_best_prediction(self.K_truth)
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
        
        if not hasattr(self,'mse_pred') and not hasattr(self,'Z_pred'): # bcs setK_mf called in create_update_K_pred
            self.setK_mf()

    def setK_mf(self):
        """
        After each model update this function needs to be called to set the K_mf list, 
        such that the code does not change with respect to stacked Kriging models (original 'MFK')

        In case this is the proposed method, we are tracking two things:
        1) The 'regular' mfk model
        2) The proposed mfk model; 
            our model then has, similar to K_truth, a Z_pred and X_pred (next to mse_pred)
            if proposed exists, we init K_mf using it
        """
        K_mf = []

        nlvl = 3 if hasattr(self, 'K_pred') else self.nlvl
        for l in range(nlvl):
            base_object = self
            if l == 2 and hasattr(self, 'K_pred'):
                base_object = self.K_pred
                l = base_object.nlvl - 1 # needed to get correct predict_l in case of SF K_pred
            obj = ObjectWrapper(base_object, l)

            K_mf.append(obj)

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
        # TODO dit neemt ook de oude K_mf mee dus je krijgt een zieke nesting van objecten! ofwel gebruik de MFK_smt set_state
        # self.__dict__ = deepcopy({k: baseObject.__dict__[k] for k in set(list(baseObject.__dict__.keys())) - set({'K_truth','X_truth','Z_truth'})})
        MFK_wrap.__init__(self, **deepcopy(baseObject.MFK_kwargs))
        for key, item in baseObject.get_state().items():
            if key not in ['K_truth', 'K_pred']: # dont add the full dicts of these sub-models
                self.__setattr__(key,deepcopy(item))

        self.K_mf = []
        # fake hps for proposed method
        self.hps = None # TODO zou nice zijn hier de echte hps, geconverteerd naar mijn format, te hebben
        self.l_fake = l_fake

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
            # self.optimal_theta[self.l_fake], d TODO
            # max(self.optimal_theta), d * (10/2) ** 2
            self.optimal_theta[self.l_fake], d # TODO normalize with the average distance between real datapoints
        ).reshape(X.shape[0], X_other.shape[0])
        return r_
