# pyright: reportGeneralTypeIssues=false
import numpy as np
from copy import copy, deepcopy

from smt.applications import MFK

from core.sampling.DoE import LHS
from core.mfk.mfk_smt import MFK_smt, MFK_wrap
from core.mfk.mfk_ok import MFK_org

from utils.formatting_utils import correct_formatX
from utils.selection_utils import isin_indices

" Say what class to inherit from!! "
# use MFK_org or MFK_smt
class ProposedMultiFidelityKriging(MFK_smt):
    """
    TODO interesant te bedenken: werkt voor de proposed method onafhankelijke noise estimates beter? -> houdt OK als optie
    """

    def __init__(self, *args, **kwargs):
        """
        param smt (bool): if True use the MFK solution of smt to build the prediction and other models. 
            Otherwise, use separate Original Kriging models for the truth and other models / levels.
        """
        super().__init__(*args, **kwargs)


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
            self.Z_pred, self.mse_pred, _ = self.weighted_prediction()
            self.create_update_K_pred()

            # K_mf_new.train(self.X_unique, Z_pred, tune = True, retuning = False, R_diagonal = mse_pred / K_mf_new.sigma_hat)
        else:
            # do an initial prediction
            Z_pred, mse_pred, _ = self.weighted_prediction()

            K_mf_new = self.create_OKlevel(self.X_unique, Z_pred, append = True, tune = True, hps_noise_ub = True, R_diagonal= mse_pred / self.K_mf[-1].sigma_hat)
            K_mf_new.reinterpolate()

        self.prepare_succes = True

    def create_update_K_pred(self, data_dict = None):
        # options
        USE_SF_PRED_MODEL = False # use as single fidelity?
        USE_HET_NOISE_FOR_EI = True # use heteroscedastic noise addition?

        if hasattr(self,'mse_pred') and hasattr(self,'Z_pred'):
            if hasattr(self,'K_pred'):
                # voor nu bij elke stap opnieuw creeeren om dat uit te sluiten
                del self.K_pred

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
                # kwargs['optim_var'] = True

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
                self.K_pred.options['optim_var'] = False 
                self.K_pred.options['use_het_noise'] = False
                self.K_pred.train()

                if USE_HET_NOISE_FOR_EI:
                    if USE_SF_PRED_MODEL:
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
                    self.K_pred.options['optim_var'] = False # NOTE True would have the same effect
                    self.K_pred.options['use_het_noise'] = True

                    self.K_pred.train()

                    # TODO afmaken en checken
                    # manual reinterpolation = misschien handig
                    values = self.K_pred.predict_values(self.X_unique)
                    self.K_pred.set_training_values(self.X_unique, values)
                    self.K_pred.options['use_het_noise'] = False
                    self.K_pred.options['noise0'] = [[0.0]] * 3
                    self.K_pred.options["eval_noise"] = False 
                    self.K_pred.train()


                self.K_pred.X_opt, self.K_pred.z_opt = self.get_best_prediction(self.K_pred)
                if self.printing:
                    print("Succesfully trained Kriging model of prediction", end = '\r')
            
            # update the K_pred model in K_mf!
            self.setK_mf()


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
            X_unique = X_test

        # 
        if not (np.any(X_s) and np.any(Z_s)):
            X_s, Z_s = self.X_mf[-1],  self.Z_mf[-1]
        
        # In case there is only one sample, nothing to weigh
        if len(Z_s) == 1:
            Z_pred, mse_pred, D_Sf, Ef_weighed = Kriging_unknown_z(X_s, X_unique, Z_s, K_mf) # NOTE niet echt D_sf
        else:
            " Collect new results "
            # NOTE if not in X_unique, we could just add a 0 to all previous,
            # might be faster but more edge-cases
            n = X_s.shape[0]
            D_z, D_mse, D_Sf, D_Ef = [], [], [], []
            for i in range(X_s.shape[0]):
                Z_p, mse_p, Sf, Ef = Kriging_unknown_z(X_s[i], X_unique, Z_s[i], K_mf)
                D_z.append(Z_p), D_mse.append(mse_p), D_Sf.append(Sf), D_Ef.append(Ef) # type: ignore
            D_z, D_mse, D_Sf, D_Ef = np.array(D_z), np.array(D_mse), np.array(D_Sf), np.array(D_Ef)

            " Weighing "
            # 1) distance based: take the (tuned) Kriging correlation function
            # NOTE one would say retrain/ retune on only the sampled Z_s (as little as 2 samples), to get the best/most stable weighing.
            # However, we cannot tune on such little data, the actual scalings theta are best represented by the (densely sampled) previous level.
            c_dist = K_mf[-1].corr(X_s, X_unique, K_mf[-1].hps)

             
            # get columns with sampled points
            idx = isin_indices(X_unique, X_s)
            mask = [isin_indices(X_unique, correct_formatX(xs, self.d)) for xs in X_s]

            # 2) variance based: predictions based on fractions without large variances Sf involved are more reliable
            
            # relative weighing of distance scaling wrt variance scaling
            c_dist_rel_weight = 5
            
            #    we want to keep the correlation/weighing the same if there is no variance,
            #    and otherwise reduce it.
            #    We could simply do: sigma += 1 and divide.
            #    However, sigma is dependend on scale of Z, so we should better use e^-sigma.
            #    This decreases distance based influence if sigma > 0.
            #    We take the variance of the fraction, S_f
            
            # NOTE normalize D_sf to D_Ef, hier geldt dat een hogere E_f automatisch een lagere S_f heeft, dus we normalizeren inverse
            D_sf_norm = D_Sf * (D_Ef ** 2) 
            c_var = np.exp(-(D_sf_norm)/(np.mean(D_sf_norm)*c_dist_rel_weight))

            # # moet wel echt * zijn op eoa manier, want anders als het een heel onbetrouwbaar 'maar dicht bij' sample is, weegt ie alsnog zwaar mee
            c = (c_dist.T * c_var).T 

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

        # print_metrics(Z_pred, mse_pred, Ef_weighed, D_Sf, K_mf[1].optimal_par[0]["sigma2"]) # type:ignore

        # assign the found values to the mf_model, in order to be easily retrieved.
        if assign:
            self.Z_pred = Z_pred
            self.mse_pred = mse_pred

        print(max(self.mse_pred))

        return Z_pred, mse_pred, Ef_weighed
    
    
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



def Kriging_unknown_z(x_b, X_unique, z_pred, K_mf):
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
    z0_b, s0_b = K0.predict(x_b)
    z1_b, s1_b = K1.predict(x_b)
    s0_b, s1_b = np.sqrt(s0_b), np.sqrt(s1_b)  # NOTE apparent noise due to smt nugget

    # get all other values of previous 2 levels
    Z0, S0 = K0.predict(X_unique)
    Z1, S1 = K1.predict(X_unique)
    S0, S1 = np.sqrt(S0), np.sqrt(S1)

    # indexing at 1 always returns a smt MFK object!
    if isinstance(K_mf[1], MFK) and K_mf[1].nlvl == 3:
        # use the MFK solution if available! 
        # predict_top_level could return either a l1 or l2 prediction.
        Z1_alt, S1_alt = K_mf[1].predict_top_level(X_unique)
        S1_alt = np.sqrt(S1_alt)
    else:
        Z1_alt, S1_alt = Z1, S1

    def exp_f(lamb1, lamb2):
        """get expectation of the function according to
        integration of standard normal gaussian function lamb"""
        c1 = z_pred - z1_b
        c2 = z1_b - z0_b
        div = c2 + lamb1 * s1_b + lamb2 * s0_b
        f = np.divide((c1 - lamb1 * s1_b), div, out=np.zeros_like(lamb1), where=div!=0)
        # f = (c1 - lamb1 * s1_b) / (
        #     c2 + lamb1 * s1_b + lamb2 * s0_b + np.finfo(np.float64).eps
        # )

        # scipy.stats.norm.pdf(lamb) == np.exp(-x**2 / 2) / np.sqrt(2*np.pi)
        # bivariate : https://mathworld.wolfram.com/BivariateNormalDistribution.html
        return f * np.exp(-(lamb1 ** 2 + lamb2 ** 2) / 2) / (2 * np.pi), f

    # to find precision, use Y = np.exp(-(lamb1 ** 2 + lamb2 ** 2) / 2) / (2 * np.pi) and use the double trapz part
    # 1000 steps; evaluates norm to 0.9999994113250346 ( met 1000x1000 evaluations!!)
    # 200 steps to 0.9999986775391365
    # 100 to       0.9999984361381121
    # 20 to        0.9999895499778308
    lambs = np.arange(-5, 5, 0.1)
    lamb1, lamb2 = np.meshgrid(lambs, lambs)

    # evaluate expectation
    Y, f = exp_f(lamb1, lamb2)
    Ef = np.trapz(
        np.trapz(Y, lambs, axis=0), lambs, axis=0
    )  # axis=0 because the last axis (default) are the dimensions.

    def var_f(lamb1, lamb2):
        # Var X = E[(X-E[X])**2]
        # down below: random variable - exp random variable, squared, times the pdf of rho (N(0,1));
        # we integrate this function and retrieve the expectation of this expression, i.e. the variance.
        return (f - Ef) ** 2 * np.exp(-(lamb1 ** 2 + lamb2 ** 2) / 2) / (2 * np.pi)

    Y = var_f(lamb1, lamb2)
    Sf = np.sqrt(
        np.trapz(np.trapz(Y, lambs, axis=0), lambs, axis=0)
    )  # NOTE not a normal gaussian variance, but we use it as such
    
    # scaling according to reliability of Ef based on ratio Sf/Ef, and correlation with surrounding samples
    # important for: suppose the scenario where a sample at L1 lies between samples of L2. 
    # Then it would not be reasonable to discard all weighted information and just take the L1 value even if Sf/Ef is high.

    # so:   if high Sf/Ef -> use correlation (sc = correlation)
    #       if low Sf/Ef -> use prediction (sc = 1)
    
    corr = K_mf[-1].corr(correct_formatX(x_b,X_unique.shape[1]), X_unique, K_mf[-1].hps).flatten()
    lin = np.exp(-1/2 * abs(Sf/Ef)) # = 1 for Sf = 0, e.g. when the prediction is perfectly reliable (does not say anything about non-linearity); 

    # w = 1 * lin + corr * (1 - lin) 
    # w = lin

    # alleen terugvallen wanneer de laag eronder of MFK ook daadwerkelijk wat kan toevoegen
    w = lin if isinstance(K_mf[1],MFK) and K_mf[1].nlvl == 3 else 1
    w = 1 # TODO no method weighing

    # print(f"W = {w}")
    # * np.exp(-abs(Sf/Ef))

    # NOTE
    # this is the point at which we can integrate other MF methods!!!!
    # e.g. this is exactly what other Mf methods do: combine multiple fidelities and weigh their contributions automatically.
    # then, we can use Sf not to switch back to L1, but to switch back to the other model if our prediction turns out to be unreliable.
    # bcs there is no hi-fi info available for the other MF method I presume it will use L1 values further away from our L2 values, just like we want!

    # retrieve the (corrected) prediction + std
    # NOTE this does not retrieve z_pred at x_b if sampled at kriged locations.
    Z2_p = w * (Ef * (Z1 - Z0) + Z1) + (1-w) * Z1_alt

    # NOTE (Eb1+s_b1)*(E[Z1-Z0]+s_[Z1-Z0]), E[Z1-Z0] is just the result of the Kriging
    # with s_[Z1-Z0] approximated as S1 + S0 for a pessimistic always oppositely moving case
    S2_p = w * (S1 + abs((S1 + S0) * Ef) + abs(Z1 - Z0) * Sf ) + (1 - w) * S1_alt

    # get index in X_unique of x_b
    ind = np.all(X_unique == x_b, axis=1)

    # set Z2_p to z_pred at that index
    Z2_p[ind] = z_pred

    # set S2_p to 0 at that index (we will later include variance from noise.)
    S2_p[ind] = 0

    # not Sf/Ef bcs for large Ef, Sf will already automatically be smaller by calculation!!
    return Z2_p, S2_p ** 2, Sf ** 2, Ef



def print_metrics(Z_pred, mse_pred, Ef_weighed,Sf_weighed,sigma_hat):
    print("┃ MEAN VALUES IN WEIGHED PREDICTION")
    print(f"┃ Z: {np.mean(Z_pred)}; mse: {np.mean(mse_pred)}\
    \n┃ Ef weighed: {np.mean(Ef_weighed)}; Sf non-weighed: {np.mean(Sf_weighed)}\n\
    ┃ Sigma_hat: {sigma_hat}")


