# pyright: reportGeneralTypeIssues=false
import time
import numpy as np
import scipy.integrate
from copy import copy, deepcopy

from smt.applications import MFK
from numba import njit 

from core.sampling.DoE import LHS
from core.mfk.mfk_smt import MFK_smt, MFK_wrap
from core.mfk.mfk_ok import MFK_org

from utils.formatting_utils import correct_formatX
from utils.selection_utils import isin_indices

" Say what class to inherit from!! "
# use MFK_org or MFK_smt
class ProposedMultiFidelityKriging(MFK_smt):

    def __init__(self, *args, **kwargs):
        """
        param smt (bool): if True use the MFK solution of smt to build the prediction and other models. 
            Otherwise, use separate Original Kriging models for the truth and other models / levels.
        """
        super().__init__(*args, **kwargs)
        self.proposed = True

        self.method_weighing = True
        self.try_use_MFK = True
        self.trained_MFK = False
        self.use_het_noise = False

        self.variance_weighing = True
        self.distance_weighing = True
        self.use_uncorrected_Ef = False

        # to find precision, use Y = np.exp(-(lamb1 ** 2 + lamb2 ** 2) / 2) / (2 * np.pi) and use the double trapz/simpon integral part
        self.lambs = np.arange(-5, 5, 0.01)
        self.lamb1, self.lamb2 = np.meshgrid(self.lambs, self.lambs)

        # scipy.stats.norm.pdf(lamb) == np.exp(-x**2 / 2) / np.sqrt(2*np.pi)
        # bivariate : https://mathworld.wolfram.com/BivariateNormalDistribution.html
        rho = 0.0   # pearson correlation coefficient between variance of lvl 0 and 1 !!! In this work, we assume full independence, so 0
        self.pdf = np.exp(-(self.lamb1 ** 2 - 2 * rho * self.lamb1 * self.lamb2 + self.lamb2 ** 2) / (2 * (1-rho**2))) / (2 * np.pi * np.sqrt(1-rho**2))


    def prepare_initial_surrogate(self, setup, X_l = None):
        super().prepare_initial_surrogate(setup, X_l)

        # and add the (reinterpolated!) predictive level
        if hasattr(self, 'proposed') and self.proposed == False:
            pass
        elif isinstance(self, MFK_smt): 
            # Universal Kriging does not always provide a representative correlation function for our goal due to the trend!
            # so, first tune an OK on lvl 1
            # K_mf_new = self.create_OKlevel(self.X_mf[1], self.Z_mf[1], append = True, tune = True, hps_noise_ub = True) # sigma_hat is not known yet!

            # do an initial prediction (based on lvl index -1 = lvl 1)
            self.weighted_prediction()
            self.create_update_K_pred()

            # K_mf_new.train(self.X_unique, Z_pred, tune = True, retuning = False, R_diagonal = mse_pred / K_mf_new.sigma_hat)
        else:
            # do an initial prediction
            Z_pred, mse_pred, _ = self.weighted_prediction()

            K_mf_new = self.create_OKlevel(self.X_unique, Z_pred, append = True, tune = True, hps_noise_ub = True, R_diagonal= mse_pred / self.K_mf[-1].sigma_hat)
            K_mf_new.reinterpolate()

        # auxiliary help variable
        self.prepare_succes = True

    def create_update_K_pred(self, data_dict = None):
        # options
        USE_SF_PRED_MODEL = self.use_single_fidelities # base Proposed model on single fidelity levels?
        USE_HET_NOISE_FOR_EI = self.use_het_noise # use heteroscedastic noise addition?
        DO_MANUAL_REINTERPOLATE = False

        if hasattr(self,'mse_pred') and hasattr(self,'Z_pred'):
            # NOTE voor nu alles met use_het_noise uit! 
            # Het lijkt alsof smt geen geen noise regression meer doet over punten die al variance bij zich dragen
            # bovendien wordt de noise van nabijliggende 'neppe' punten niet beperkt door 'echte' punten 
            # maar wordt de noise van echte punten juist tot het niveau van de neppe punten gebracht omdat die noise 'gemeten, dus waarheid' is.
            # dit is logisch normaliter maar werkt voor mij counter-productief!
            # in effect wordt de noise wel geregressed, de level selection gebaseerd op variance is nog informatief, 
            # maar de EI wordt niet met behulp hiervan berekend. Om dit wel te verkrijgen:
            # 1) train met: 
            #    - ['eval_noise'] = True
            #    - ['optim_var'] = True -> re-interpolation
            # 2) train met:
            #    - ['eval_noise'] = False
            #    - ['optim_var'] = False
            #    - ['use_het_noise'] = True (voegt eigenlijk alleen maar de noise in de noise-output toe!)
            # zo hebben we én een noise-regressed mean, en de variance output voor EI (consistent met level selection)

            if not hasattr(self,'K_pred'):
                if self.printing:
                    print("Creating Kriging model of pred")
                kwargs = copy(self.MFK_kwargs)

                # we are going to use heteroscedastic noise evaluation at the top level!
                kwargs['eval_noise'] = True
                # kwargs['use_het_noise'] = True
                kwargs['theta_bounds'] = [0.01,20]
                kwargs['theta0'] = [kwargs['theta_bounds'][0]]
                kwargs['print_training'] = False

                self.K_pred = MFK_wrap(**kwargs)
                self.K_pred.d = self.d
                self.K_pred.name = "K_pred"
                self.K_pred.MFK_kwargs = kwargs

            if data_dict != None:
                self.K_pred.set_state(data_dict)
            else:
                if USE_SF_PRED_MODEL:
                    self.set_training_data(self.K_pred, [self.X_unique], [self.Z_pred])
                    self.K_pred.options['noise0'] = [[0.0]]
                else:
                    self.set_training_data(self.K_pred, [*self.X_mf[:self.max_nr_levels-1], self.X_unique], [*self.Z_mf[:self.max_nr_levels-1], self.Z_pred])
                    # self.K_pred.options['noise0'] = [[0.0]] * 3

                # NOTE voor uitleg van settings, bekijk: smt_debug_optimvar.py
                self.K_pred.options['eval_noise'] = True
                self.K_pred.options['optim_var'] = self.optim_var and self.use_single_fidelities and not DO_MANUAL_REINTERPOLATE
                self.K_pred.options['use_het_noise'] = False
                self.K_pred.train()

                # tested: all 0!
                # _,mse_test = self.K_pred.predict(self.X_unique)
                # print(f"mse_test: {mse_test}")

                if USE_HET_NOISE_FOR_EI:
                    if USE_SF_PRED_MODEL:
                        current_variance = self.K_pred.predict_variances(self.X_unique).reshape(-1,)
                        # self.K_pred.options['noise0'] = [self.mse_pred + current_variance]
                        self.K_pred.options['noise0'] = [self.mse_pred]
                    else:
                        MSE = self.K_pred.predict_variances_all_levels(self.X_unique)[0]
                        # TODO kijken of self.mse_pred + noise op samples locatie werkt!
                        # TODO dit per sample individueel doen, niet aggregated (evaluated noises kunnen bijv wel anders zijn!)
                        # tegelijkertijd: de minimum mogelijke noise is in principe met de minste hoeveelheid Kriging variance.
                        mse_medium = np.min(MSE[:,1].reshape(-1,1))
                        mse_hifi = np.min(MSE[:,2].reshape(-1,1)) # MSE from model with evaluated noise
                        mse_pred_hifi = np.min(self.mse_pred[np.nonzero(self.mse_pred)])  # extrapolation mag nooit belangrijker zijn dan een sample dus neem de min!
                        min_mse = np.min([mse_medium,mse_hifi,mse_pred_hifi])

                        mse_pred_adapted = deepcopy(self.mse_pred)
                        mse_pred_adapted[np.nonzero(self.mse_pred == 0)] = min_mse

                        self.K_pred.options['noise0'] = [MSE[:,0].reshape(-1,1), MSE[:,1].reshape(-1,1), mse_pred_adapted.reshape(-1,1)] # was self.mse_pred!
                        # self.K_pred.options['noise0'] = [[0], [0], [self.mse_pred]]

                    self.K_pred.options['eval_noise'] = False # NOTE als dit False is, doet optim_var niks
                    self.K_pred.options['optim_var'] = False # NOTE True would have no effect with optim_var no
                    self.K_pred.options['use_het_noise'] = True

                    self.K_pred.train()

                if DO_MANUAL_REINTERPOLATE and self.optim_var and not self.use_single_fidelities:
                    # TODO afmaken en checken
                    # manual reinterpolation = misschien handig
                    values = self.K_pred.predict_values(self.X_unique)
                    self.K_pred.set_training_values(self.X_unique, values)
                    self.K_pred.options['use_het_noise'] = False
                    if USE_SF_PRED_MODEL:
                        self.K_pred.options['noise0'] = [[0.0]]
                    else:
                        self.K_pred.options['noise0'] = [[0.0]] * 3
                    self.K_pred.options["eval_noise"] = False 
                    self.K_pred.train()


                self.K_pred.X_opt, self.K_pred.z_opt = self.get_best_prediction(self.K_pred)
                if self.printing:
                    print("Succesfully trained Kriging model of prediction")#, end = '\r')
            
            # update the K_pred model in K_mf!
            self.setK_mf(only_rebuild_top_level=True)

    def check_validity(self, Ef_array, w_bool, mu = 0.02):
        if np.sum(w_bool) <= 1:
            print("Validity check: not enough useable data")
        else:    
            E_mean = np.dot(Ef_array, w_bool) / np.sum(w_bool)

            lh = np.abs(E_mean-Ef_array) * w_bool
            if self.printing:
                print(lh)
            if np.all(lh <= mu * abs(E_mean)):
                print("Validity check: Valid!")
            else:
                print("Validity check: Maybe not valid!")
        print(f"Validity check: {len(w_bool)-np.sum(w_bool)} unlucky samples")

    def weighted_prediction(self, X_s = [], Z_s = [], assign : bool = True, X_test = np.array([])):
        """
        Function that weights the results of the function 'Kriging_unknown' for multiple samples
        at the (partly) unknown level.

        In case convergence is linear over levels and location,
        the "Kriging_unknown" prediction is exact if we have exact data.
        In case convergence is not linear, we get discrepancies;
        something we try to solve by sampling and weighing the solutions.

        @param X_s = X[-1]: Locations at which we have sampled at the new level.
        @param X_unique: all unique previously sampled locations in X.
        @param Z_s: list hifi level sample locations
        @param K_mf: Kriging models of the levels, should only contain *known* levels.
        @param assign: Assign the results Z_pred and mse_pred as property to the mf_model.
        @param X_test: only used when assessing one single point in isolation instead of complete X_unique.

        X_s, Z_s and assign are only used when using a reduced dataset in linearity check
        """ 

        # data pre-setting
        X_unique, K_mf =  self.X_unique, self.K_mf
        if X_test.size != 0:
            X_unique = np.append(X_unique,X_test, axis = 0)

        # 
        if not (np.any(X_s) and np.any(Z_s)):
            X_s, Z_s = self.X_mf[-1],  self.Z_mf[-1]
        
        # In case there is only one sample, nothing to weigh
        if len(Z_s) == 1:
            Z_pred, mse_pred, Sf2_weighed, Ef_weighed, D_w = self.Kriging_unknown_z(X_s, X_unique, Z_s, K_mf) # NOTE niet echt D_Sf2
            D_w = np.array(D_w, dtype=bool)
        else:
            " Collect new results "
            # NOTE if not in X_unique, we could just add a 0 to all previous,
            # might be faster but more edge-cases
            D_z, D_mse, D_Sf2, D_Ef, D_w = [], [], [], [], []
            for i in range(X_s.shape[0]):
                Z_p, mse_p, Sf2, Ef, w = self.Kriging_unknown_z(X_s[i], X_unique, Z_s[i], K_mf)
                D_z.append(Z_p), D_mse.append(mse_p), D_Sf2.append(Sf2), D_Ef.append(Ef), D_w.append(w) # type: ignore
            D_z, D_mse, D_Sf2, D_Ef, D_w = np.array(D_z), np.array(D_mse), np.array(D_Sf2), np.array(D_Ef), np.array(D_w, dtype=bool)

            " Weighing "
            # preparation: get columns with sampled points
            idx = isin_indices(X_unique, X_s) # 1d
            mask = np.array([isin_indices(X_unique, correct_formatX(xs, self.d)) for xs in X_s]) # locaties in de #samples x 1d array

            # 1) variance based: predictions based on fractions without large variances Sf involved are more reliable
            if self.variance_weighing:
                # relative weighing of distance scaling wrt variance scaling
                if np.all(~D_w):
                    # if there is nothing
                    D_w = ~D_w

                if np.all(D_w) and self.method_weighing:
                    c_var = np.ones_like(D_mse) # do nothing, we trust all our samples equally -> only distance weighing!
                else:
                    D_w_local = D_w if self.method_weighing else np.ones_like(D_w, dtype=bool) # because of lin is returned in kriging unknown z now

                    mse_mean = np.mean(D_mse[D_w_local, :], axis = 1) # select using boolean weight (only use samples not completely w = 0)
                    divisor = np.max(mse_mean[mse_mean <= np.mean(mse_mean)])
                    exp_fac = D_mse/divisor - 0
                    # print(exp_fac)
                    c_var = np.exp(-exp_fac) # min omdat bij uitschieters alleen de meest bizarre uitschieter gefilterd wordt terwijl de rest miss ook wel kak is.
                    # print("c_var before clipping")
                    # print(c_var)
                    c_var = c_var.clip(max = 1)
                    # We basically do not want to include the other method in variance weighing. If all are other method neither.
                    # so we take the mean over the c_var`s of those of the extrapolation, in effect distance weighing becomes the prevalent
                    c_var[~D_w_local,:] = np.mean(c_var[D_w_local], axis = 0)
                    if self.printing:
                        print(f"c_var means = {np.mean(c_var,axis=1)}, met X_s = {X_s}")
            else:
                c_var = np.ones_like(D_mse)

            # 2) distance based: take the (tuned) Kriging correlation function
            c_dist_corr = K_mf[-1].corr(X_s, X_unique)#, K_mf[-1].hps) NOTE this means not usable for OK anymore
            c_dist_org = c_dist_corr * c_var

            if self.distance_weighing:
                # NOTE one would say retrain/ retune on only the sampled Z_s (as little as 2 samples), to get the best/most stable weighing.
                # However, we cannot tune on such little data, the actual scalings theta are best represented by the (densely sampled) previous level.
                c_dist = np.zeros_like(c_dist_org)

                " distance weighing "
                # er is geen interpolerende oplossing momenteel.
                # daarom: sequentially huidige locatie eraf halen, column opnieuw schalen naar 1, herhalen voor volgend punt
                selection = c_dist_org[:, idx]
                c_dist_inter = deepcopy(c_dist_org)

                # rescale such that rows at samples are one again
                for k in range(c_dist_org.shape[0]):
                    # scale all samples back to 1, scale rest of row too
                    z = np.where(mask[k,:] == True)[0] # get column of the sample
                    c_dist_inter[k,:] /= c_dist_inter[k,z]

                # only use linearly consistent row operations! so, rowwise +- and constant multiply
                # for each sample, make the other rows 0 (then other sample row operations will keep this zero, multiplications too)
                for i, col in enumerate(selection.T): # NOTE col selects the row otherwise!!!!
                    col = c_dist_inter[:, idx][:,i] # NOTE col selects the row!!!!
                    j = np.where(col == 1.0)[0]
                    aftrek = col[:,np.newaxis] * c_dist_inter[j,:]
                    aftrek[j,:] = 0

                    c_dist_inter -= aftrek

                    # rescale such that rows at samples are one again
                    for k in range(c_dist_org.shape[0]):
                        # scale all samples back to 1, scale rest of row too
                        z = np.where(mask[k,:] == True)[0] # get column of the sample
                        c_dist_inter[k,:] /= c_dist_inter[k,z]

                # clip to max min (i.e. negative values should not have influence anymore)
                # NOTE max destroys some of the correlation information. The balancing between samples is still done by dividing with the sum
                # print("c_dist_inter before clipping")
                # print(c_dist_inter)
                c_dist_inter = c_dist_inter.clip(min = 0)# , max=1) 

                # rescale: contributions of all samples should add up to 1
                c_dist_inter /= np.sum(c_dist_inter, axis=0)

                c_dist = c_dist_inter

                col_indices = np.where(np.all(c_dist == 0, axis=0))[0]
                if col_indices.any():
                    print("WARNING: THERE ARE COLUMNS WITH 0 ONLY")
                    print(col_indices)

                sums = np.sum(c_dist[:, idx], axis=0)
                if not np.all(np.isclose(sums, 1)):
                    print("WARNING: SAMPLE COLUMNS DONT SUM TO 1")
                    print(sums)

                c_dist_samples = c_dist[mask]
                if not np.all(np.isclose(c_dist_samples,1)):
                    print("WARNING: NOT ONLY SAMPLES ARE 1 / SAMPLES NOT 1!")
                    print(c_dist_samples)
            else:            
                c_dist = np.ones_like(c_dist_corr)
                c_dist = c_dist_corr

            " reweigh c_dist "
            # Although interpolating and ruling out influence of samples that should not have influence,
            #       in high variance environments there is no 'averaging' effect anymore over sample contributions, thereby deteriorating NRMSE
            # In this way, it can be that distance weighing is essential in non-linear cases, but can be bad if each sample in principle has a good estimate (besides noise)
            # Therefore, re-add part of a ones matrix for this averaging effect + avoiding all zeros situations
            dist_ratio = 10/100 # higher is more parts non-reduced solution
            c_dist += dist_ratio * c_dist_corr
            c_dist /= np.sum(c_dist, axis=0)

            # moet wel echt * zijn op eoa manier, want anders als het een heel onbetrouwbaar 'maar dicht bij' sample is, weegt ie alsnog zwaar mee
            c = (c_dist * c_var) 
            c /= np.sum(c, axis=0)

            np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, linewidth=1000)
            
            # print(idx)
            # print(c)

            if self.printing:
                if np.min(D_Sf2) * 1000 < np.max(D_Sf2):
                    print("FF kijken hier")
                if np.min(D_Sf2) > np.max(D_Ef**2):
                    print("WARNING: np.min(D_Sf) > np.max(D_Ef)")

            " Scale to sum to 1; correct for sampled locations "
            # Contributions coefficients should sum up to 1.
            # Furthermore, sampled locations should always be the only contribution for themselves (so set to 1).
            #  if we dont do this, EI does not work!
            #  however, we might get very spurious results -> we require regression!
            c[:, idx] = 0  # set all correlations at the location of the samples to zero
            c = c + mask  # and then in corresponding index/correlation row set to 1 to retrieve sample exactly

            # Scale for Z
            b = np.sum(c, axis=0)
            c_z = np.divide(c, b, out=np.zeros_like(c), where= b!=0)
            Z_pred = np.sum(np.multiply(D_z, c_z), axis=0)

            # Scale for mse
            # NOTE noise is added to the samples regardless of what this function returns
            c_mse = c_z - mask  # always 0 variance for samples, the Kriging model can add variance independently if it wants
            mse_pred = np.sum(np.multiply(D_mse, c_mse), axis=0)

            Ef_weighed = np.dot(D_Ef, c_z)
            Sf2_weighed = np.dot(D_Sf2, c_z)

            self.check_validity(D_Ef, D_w)
        # print_metrics(Z_pred, mse_pred, Ef_weighed, D_Sf2, K_mf[1].optimal_par[0]["sigma2"]) # type:ignore

        # assign the found values to the mf_model, in order to be easily retrieved.
        if assign:
            self.Z_pred = Z_pred
            self.mse_pred = mse_pred
            if self.printing:
                print(f"Max mse_pred = {max(self.mse_pred)}")

        if X_test.shape[0] == 0:
            return Z_pred, mse_pred, Ef_weighed, np.sqrt(Sf2_weighed)
        else:
            # in case we want to return only one item. Above, x_test is appended to x_unique for easier implementation.
            # this means we select the last item here to retrieve x_test
            if len(Z_s) == 1:
                return Z_pred[-1], mse_pred[-1], Ef_weighed, np.sqrt(Sf2_weighed)
            else:
                # bcs for multiple samples, Ef is weighed per sample, there is not a single Ef for all
                return Z_pred[-1], mse_pred[-1], Ef_weighed[-1], np.sqrt(Sf2_weighed[-1])
    
    
    def set_state(self, data_dict):
        super().set_state(data_dict)
        
        if isinstance(self, MFK_smt) and 'K_mf_list' in data_dict:
            # then add the highest level (which is an OK object for now)
            self.number_of_levels = 2
            k = self.create_OKlevel([],add_empty=True)
            k.set_state(data_dict['K_mf_list'][0])
            if len(data_dict['K_mf_list']) > 1:
                if self.printing:
                    print("WARNING: K_mf_list should be of length 1; only containing the predicted level.")



    def Kriging_unknown_z(self, x_b, X_unique, z_pred, K_mf):
        """
        Assume the relative differences at one location are representative for the
        differences elsewhere in the domain, if we scale according to the same convergence.
        Use this to predict a location, provide an correction/expectation based on chained variances.

        @param x_b: a (presumably best) location at which we have/will sample(d) the new level.
        @param X_unique: all unique previously sampled locations in X.
        @param z_pred: new level`s sampled location value(s).
        @param K_mf: Kriging models of the levels, should only contain known levels.
        @return extrapolated predictions Z2_p, corresponding mse S2_p (squared!)
        """

        K0 = K_mf[0]
        K1 = K_mf[1]

        # get values corresponding to x_b of previous 2 levels
        # NOTE using predict might be less reliable than using the actual sample
        #   because predict is not necessarily equal to the sample. 
        #   So, if noise/mean estimates are not very well we could get terrible results.
        x_b = correct_formatX(x_b, self.d)
        z0_b, s0_b = K0.predict(x_b)
        z1_b, s1_b = K1.predict(x_b)
        s0_b, s1_b = np.sqrt(s0_b), np.sqrt(s1_b)  # NOTE apparent noise due to smt nugget

        # NOTE puur voor test scenario
        # z0_b, s0_b = 1, 0.1
        # z1_b, s1_b = 1.0001, 0.1
        # z_pred = np.array([2])
        
        # using the actual samples! # NOTE VEEL VEEL SLECHTER, hebben echt noise regression nodig
        # mask = isin_indices(X_unique, correct_formatX(x_b,self.d))
        # z0_b = self.Z_mf[0][mask]
        # z1_b = self.Z_mf[1][mask]

        # get all other values of previous 2 levels
        Z0, S0 = K0.predict(X_unique)
        Z1, S1 = K1.predict(X_unique)
        S0, S1 = np.sqrt(S0), np.sqrt(S1)

        # indexing at 1 always returns a smt MFK object!
        Z1_is_alt = False 
        if self.trained_MFK and self.try_use_MFK: # isinstance(self, MFK) and hasattr(self,'nlvl') and self.nlvl == 3
            # use the MFK solution if available! 
            # predict_top_level could return either a l1 or l2 prediction.
            Z1_alt, S1_alt = self.predict_top_level(X_unique)
            S1_alt = np.sqrt(S1_alt)
        else:
            Z1_alt, S1_alt = Z1, S1
            Z1_is_alt = True
        
        # variables for function below
        Ef_uncorrected = ((z_pred - z1_b) / (z1_b - z0_b)).item()

        # @njit
        def exp_f(lamb1, lamb2, pdf):
            """get expectation of the function according to
            integration of standard normal gaussian function lamb"""
            c1 = z_pred - z1_b
            c2 = z1_b - z0_b
            div = c2 + lamb1 * s1_b + lamb2 * s0_b

            f = np.divide((c1 - lamb1 * s1_b), div, out=np.ones_like(lamb1) * 0, where=div!=0) # liever wat kleiner (trekt naar Z1 toe) dan heel groot, dus 0
            # f = np.zeros_like(lamb1)
            # mask = np.nonzero(div)
            # f[mask] = np.divide((c1 - lamb1[mask] * s1_b), div[mask])

            return f * pdf, f
        
        # start = time.time()
        # evaluate expectation
        Y, f = exp_f(self.lamb1, self.lamb2, self.pdf)
        
        Ef = scipy.integrate.simpson(
            scipy.integrate.simpson(Y, self.lambs, axis=0), self.lambs, axis=0
        )  # axis=0 because the last axis (default) are the dimensions.

        # mid = time.time()
        # print(f"double integral took: {mid - start}")

        # Ef = Ef_uncorrected
        @njit
        def var_f(pdf):
            # Var X = E[(X-E[X])**2]
            # down below: random variable - exp random variable, squared, times the pdf of rho (N(0,1));
            # we integrate this function and retrieve the expectation of this expression, i.e. the variance.
            return (f - Ef) ** 2 * pdf

        Y = var_f(self.pdf)
        Sf = np.sqrt(
            scipy.integrate.simpson(scipy.integrate.simpson(Y, self.lambs, axis=0), self.lambs, axis=0)
        )  # NOTE not a normal gaussian variance, but we use it as such
        # print(f"second double integral took: {time.time() - mid}")
        
        # scaling according to reliability of Ef based on ratio Sf/Ef, and correlation with surrounding samples
        # important for: suppose the scenario where a sample at L1 lies between samples of L2. 
        # Then it would not be reasonable to discard all weighted information and just take the L1 value even if Sf/Ef is high.

        # we want to linearly decrease lin between some bounds. Lets say untill Sf = 1 * Ef we find this acceptable, after which we will decrease lin (or w) to 0 at Sf = 5 * Ef
        # so with in this example a = 1 and b = 5
        a, b = 1, 5
        if Sf <= a * abs(Ef):
            lin = 1
        elif Sf >= b * abs(Ef):
            lin = 0
        else:
            c = Sf / abs(Ef)
            lin = 1 - (c - a) / (b - a) 

        # alleen terugvallen wanneer de laag eronder of MFK ook daadwerkelijk wat kan toevoegen
        # w = lin if isinstance(K_mf[1],MFK) and K_mf[1].nlvl == 3 else 1 # this means effectively only method weighing with MFK result!
        if self.method_weighing:
            w = lin
        else:
            w = 1 # if no method weighing
        # NOTE
        # this is the point at which we can integrate other MF methods!!!!
        # e.g. this is exactly what other Mf methods do: combine multiple fidelities and weigh their contributions automatically.
        # then, we can use Sf not to switch back to L1, but to switch back to the other model if our prediction turns out to be unreliable.
        # bcs there is no hi-fi info available for the other MF method I presume it will use L1 values further away from our L2 values, just like we want!

        # retrieve the (corrected) prediction + std
        # NOTE this does not retrieve z_pred at x_b if sampled at kriged locations.

        if self.printing:
            print(f"({x_b}) Ef org: {Ef_uncorrected:6f} vs Ef corrected: {Ef:6f}; Sf: {Sf:6f}; w: {w}")

        if self.use_uncorrected_Ef:
            Ef = Ef_uncorrected

        self.last_Ef = Ef

        Z2_p = w * (Ef * (Z1 - Z0) + Z1) + (1 - w) * Z1_alt

        # NOTE (Eb1+s_b1)*(E[Z1-Z0]+s_[Z1-Z0]), E[Z1-Z0] is just the result of the Kriging
        # with s_[Z1-Z0] approximated as S1 + S0 for a pessimistic always oppositely moving case
        S2_p = w * (S1 + abs((S1 + S0) * Ef) + abs(Z1 - Z0) * Sf ) + (1 - w) * S1_alt

        # get index in X_unique of x_b
        # ind = np.all(X_unique == x_b, axis=1)
        ind = isin_indices(X_unique, correct_formatX(x_b,self.d))

        # set Z2_p to z_pred at that index
        Z2_p[ind] = z_pred

        # set S2_p to 0 at that index (we will later include variance from noise.)
        S2_p[ind] = 0

        return Z2_p, S2_p ** 2, Sf ** 2, Ef, lin

