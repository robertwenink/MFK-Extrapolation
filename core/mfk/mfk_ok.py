from core.mfk.mfk_base import MultiFidelityKrigingBase

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
                print("Creating Kriging model of truth")
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
