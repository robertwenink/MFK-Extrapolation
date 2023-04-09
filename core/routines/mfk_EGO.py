# pyright: reportGeneralTypeIssues=false, reportUnboundVariable = true
import numpy as np
import time 

from core.mfk.mfk_smt import MFK_smt
from core.mfk.mfk_ok import MFK_org
from core.mfk.proposed_mfk import ProposedMultiFidelityKriging
from core.routines.EGO import EfficientGlobalOptimization
from core.sampling.solvers.internal import TestFunction

from postprocessing.plot_live_metrics import ConvergencePlotting
from postprocessing.plotting import Plotting

from utils.selection_utils import isin
from utils.formatting_utils import correct_formatX



class MultiFidelityEGO(ProposedMultiFidelityKriging, MFK_smt, EfficientGlobalOptimization):
    
    def __init__(self, setup, proposed = True, optim_var = True, *args, **kwargs):
        self.optim_var = optim_var
        self.proposed = proposed
        if proposed:
            super().__init__(setup, *args, **kwargs)
        else:
            MFK_smt.__init__(self, setup, *args, **kwargs) 

        # Ik snap die inheritance structuur niet, onderstaand lijkt goed uit te leggen
        # https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way
        EfficientGlobalOptimization.__init__(self, setup, *args, **kwargs) 

        " OPTION FOR USING SINGLE FIDELITY LEVELS BEHIND K_PRED"
        self.use_single_fidelities = True

    def optimize(self, pp : Plotting = None, cp : ConvergencePlotting = None):
        self.max_cost = np.inf
        self.max_iter = 50

        n = 0
        while np.sum(self.costs_total) < self.max_cost:
            break
            start = time.time()
            prediction_function = self.K_mf[-1].predict
            
            " predict and calculate Expected Improvement "
            # There are three types of locations involved:
            # 1) The current best sample
            # 2) The current best extrapolation
            # 3) The best found EI
            
            # 2 and 3 should be corrected with the EI at 1
            # ei at 3 is in principle always > ei at 2 due to a small ofsett in x and thereby added kriging uncertainty (important in flat areas)
            # 1 does not have extra extrapolation variance, 2 and 3 do.
            # ei at 3 - ei at 2 should be > ei_criterion; otherwise: sample at level 2 at location 2!
            # but if ei at the next sample location - ei at 1 < ei_criterion -> stop! -> Do not use re-interpolation? We use it implicitly by doing this.

            # EI at sample
            x_min_sample, y_min_sample = self.get_best_sample() # this is a stable value!
            y_pred_sample, mse_pred_sample = prediction_function(x_min_sample)
            
            # only correct if not already corrected. ei_sample could for instance be negative then!
            if self.options['optim_var'] == False: 
                ei_sample = self.EI(y_min_sample, y_pred_sample, np.sqrt(mse_pred_sample)) # y_pred ipv min voor indieen geen reinterpolation!
            else:
                ei_sample = 0.0
            
            if self.proposed:
                # EI at best extrapolation
                x_min_extra, y_min_extra = self.get_best_extrapolation()
                y_pred_extra, mse_pred_extra = prediction_function(x_min_extra)

                # NOTE kan gelijk zijn aan sample! in dat geval automatisch ei_corrected = 0.0
                ei_extra = self.EI(y_min_sample, y_pred_extra, np.sqrt(mse_pred_extra)) 

                # EI at new location (best EI)
                x_best = x_min_sample if y_min_sample < y_min_extra else x_min_extra
            else:
                if 'x_min_extra' in locals():
                    x_best = x_min_extra # type:ignore
                elif 'x_ei_max' in locals():
                    x_best = x_ei_max # type:ignore
                else:
                    x_best = None

            x_ei_max, ei_max = super().find_best_point(prediction_function, criterion = 'EI', x_centre_focus = x_best)
            if self.printing:
                print(f"FINDING EI TOOK: {time.time() - start:4f}                                           ")
                print("MAXIMUM EI: {:4f}".format(np.max(ei_max)))

            if self.proposed:
                # correct both with ei_sample
                ei_max_corrected = ei_max - ei_sample
                ei_extra_corrected = ei_extra - ei_sample # type:ignore

            # used for giving preference over sampling a previous known point
                # we do not want to infinitely sample new low-fi points while sometimes its good to sample a new hifi
                cost_ratio = self.costs_expected_nested[2]/(self.costs_expected_nested[1])

                # check ei_criterion conditions
                if (ei_max_corrected - ei_extra_corrected) / cost_ratio**(1/2) <= self.ei_criterion: #ei_extra_corrected >= ei_max_corrected and 
                    ei = ei_extra_corrected
                    x_new = x_min_extra # type:ignore
                    if self.printing:
                        print(f"Using location of best extrapolation with corrected EI of {ei:6f}!")
                else:
                    ei = ei_max_corrected
                    x_new = x_ei_max
                    if self.printing:
                        print(f"Using location of max EI with corrected EI of {ei:6f}!")
            else:
                ei = ei_max - ei_sample
                x_new = x_ei_max
                if self.printing:
                     print(f"Corrected EI is {ei:6f}")


            " output "
            if pp != None:
                pp.draw_current_levels(self)
            if cp != None:
                if cp.iteration_numbers == []: # if not set yet in set_state, we do add the first data
                    cp.update_convergence_data(self, x_new, ei) # pass these, we do not want to recalculate!!
                cp.plot_convergence()

            
            " Terminate or continue loop based on EI "
            # terminate if criterion met, or when we revisit a point
            # NOTE level 2 is reinterpolated, so normally 0 EI at sampled points per definition!
            # however, we use regulation (both through R_diagonal of the proposed method as regulation constant for steady inversion)
            # this means the effective maximum EI can be higher than the criterion!
            if np.all(ei < self.ei_criterion):
                if self.printing:
                    print(f"Finishing due to reaching EI criterium with (corrected) EI of {ei:6f}!")
                break
            if isin(x_new,self.X_mf[-1]):
                if self.printing:
                    print("Finishing due to resampling previous hifi point!")
                break

            # NOTE original while loop criterium!!
            if n < self.max_iter:
                pass
            else:
                if self.printing:
                    print("Stopping because maximum number of iterations reached!")
                # check if the basis for virtual_x_new has already been sampled
                y_virtual_min, _ = prediction_function(x_best) # use the x_best, which is defined for both proposed as reference
                print(f"y_virtual_min: {y_virtual_min} vs y_min_sample: {y_min_sample}")
                if not isin(x_new, self.X_mf[-1]) and y_virtual_min <= y_min_sample: # second condition actually implies the first
                    self.sample_nested(2, x_new)
                    if self.proposed:
                        # weigh each prediction contribution according to distance to point.
                        self.weighted_prediction()
                        self.create_update_K_pred()
                    if self.printing:
                        print(f"Sampled at the best extrapolation to end with value {self.K_mf[-1].predict(x_new)[0]}!")
                break

            " Sampling and training procedures "
            l = self.level_selection(x_new)
            if self.printing:
                print("Sampling on level {} at {}".format(l,x_new))
            
            self.sample_nested(l, x_new)

            # TODO keep this? otherwise write about it.
            # sample around each high-fidelity sample for better noise predictions around it at the lower levels!
            # only sample when size 1, initial hifi subset is sampled seperately
            if l == self.max_nr_levels - 1:
                self.sample_around(1, x_new)

            if self.proposed:
                # weigh each prediction contribution according to distance to point.
                Z_pred, mse_pred, Ef_weighed, Sf_weighed = self.weighted_prediction()
                if np.all(Sf_weighed/Ef_weighed > 10):
                    if self.printing: print("High Sf/Ef ratio, trying to retrain lower levels!!")
                    self.setK_mf()
                    # then try to retrain the lower levels and redo the weighted prediction!!
                    Z_pred, mse_pred, Ef_weighed, Sf_weighed = self.weighted_prediction()
                    if np.any(Sf_weighed/Ef_weighed > 10) and self.printing:
                        print("Still bad Sf/Ef ratio .... continueing")

                self.create_update_K_pred()

            if isinstance(self, MFK_org):
                # build new kriging based on prediction
                # NOTE, we normalise mse_pred here with the previous known process variance, since otherwise we would arrive at a iterative formulation.
                # NOTE for top-level we require re-interpolation if we apply noise
                if self.tune_counter % self.tune_prediction_every == 0:
                    tune = True
                else: 
                    tune = False

                # if we sample on level 2, we really should first retune for correcting i.e. sigma_hat, without R_diag to get the correct base, just like at init
                if l == 2:
                    self.K_mf[-1].train(self.X_unique, self.Z_pred, tune=tune)
                
                # train the prediction model, including heteroscedastic noise
                self.K_mf[-1].train(
                    self.X_unique, self.Z_pred, tune=tune#, R_diagonal= mse_pred / self.K_mf[-1].sigma_hat
                )
                self.K_mf[-1].reinterpolate()

            if cp != None:
                # plot convergence here and not before break, otherwise on data reload will append the same result over and over
                cp.update_convergence_data(self, x_new, ei) # pass these, we do not want to recalculate!!
            
            n += 1



        " Print end-results "
        if self.printing:
            s_width = 34 + (4 + self.d * 6) + (9 + 8) + 2
            print("┏" + "━" * s_width + "┓")
            print(f"┃{f' Number of samples (low / high): {self.X_mf[1].shape[0]} / {self.X_mf[-1].shape[0]}':<{s_width}}┃")
            print(f"┃ Best found point \t\tx = {str(self.get_best_sample()[0][0]):<{4+self.d*6}}, f(x) = {f'{self.get_best_sample()[1]:.4f}':<8} ┃")
            if hasattr(self,'K_truth') and hasattr(self.K_truth,'X_opt'):
                print(f"┃ Best (truth) predicted point \tx = {str(self.K_truth.X_opt[0]):<{4+self.d*6}}, f(x) = {f'{self.K_truth.z_opt:.4f}':<8} ┃") # type: ignore
            if isinstance(self.solver, TestFunction):
                X_opt, Z_opt = self.solver.get_optima()
                X_opt_ex, Z_opt_ex = self.find_best_point(lambda x: self.solver.solve(x, l = -1))
                
                ind = np.argmin(Z_opt) # pakt gwn de eerste, dus = 0
                print(f"┃ Exact optimum at point \tx = {str(X_opt[ind]):<{4+self.d*6}}, f(x) = {f'{Z_opt[ind]:.4f}':<8} ┃")
                print(f"┃ Exact best prediction \tx = {str(X_opt_ex.flatten()):<{4+self.d*6}}, f(x) = {f'{float(Z_opt_ex):.4f}':<8} ┃")
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

        if self.proposed:
            sigma_red = self._std_reducable_proposed(x_new) # kan op x_new 0 zijn
        else:
            sigma_red = np.sqrt(self._mse_reducable_meliani_smt(x_new))
            if self.printing:
                print(f"Meliani level selection sigmas: {sigma_red}")
        
        maxx = 0
        l = 0
        # dit is het normaal, als de lagen gelinkt zijn! is hier niet, dit werkt alleen voor meliani.
        for i in range(1,3):
            if self.proposed:
                value = sigma_red[i]**2 / self.costs_expected_nested[i]**2 # NOTE hier kan optioneel ander gedrag worden gebruikt
            else:
                value = sigma_red[i]**2 / self.costs_expected_nested[i]**2

            maxx = max(maxx,value)
            # NOTE gives preference to highest level, which is good:
            # if no reductions expected anywhere lower, we should sample the highest (our only 'truth')
            if value == maxx:
                l = i
                
        # check per sample if not already sampled at this level
        # NOTE broadcasting way
        if isin(x_new,self.X_mf[l]):
            l += 1 # then already sampled! given we stop when we resample the highest level, try to higher the level we are on!
            if self.printing:
                print("Already sampled -> Highered level, sampling l = {} now!".format(l))
        
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
        z_pred_weighed, mse_pred_weighed, Ef_weighed, Sf_weighed = self.weighted_prediction(self, X_test=x_new, assign = False)


        " 2) kriging prediction mse (can be spurious!!) " 
        # noise corrected not true at lvl 2 due to reinterpolation!
        sigma_pred_model = []
        sigma_pred_model.append(self._sigma_reducable_predict(0, x_new, noise_correct=True)[1])
        sigma_pred_model.append(self._sigma_reducable_predict(1, x_new, noise_correct=True)[1])
        sigma_pred_model.append(self._sigma_reducable_predict(2, x_new, noise_correct=True)[1])

        " 3) smt mfk prediction mse "
        # with propagate_uncertainty, mse is as-is; 
        # this is important to the proposed methods split-up since each level contributes independently!
        if not (self.proposed and self.use_single_fidelities):
            std_meliani = np.sqrt(self._mse_reducable_meliani_smt(x_new, propagate_uncertainty = True)) 
            std_smt = np.sqrt(self._mse_reducable_meliani_smt(x_new, propagate_uncertainty = False))

        " Results selection "
        # Now, calculate the weighed result
        # s2 = s1 + Ef_weighed * (s1 + s0) + weighted(Sf) * (z1 - z0)
        # last term does not provide an expected reduction, so we only require a weighed Ef
        # we assume a fully nested DoE
        s0_exp_red = abs(Ef_weighed.item()) * sigma_pred_model[0]
        s1_exp_red = sigma_pred_model[1] + abs(Ef_weighed.item()) * (sigma_pred_model[0] + sigma_pred_model[1])

        z0, s0 = self._sigma_reducable_predict(0, x_new, noise_correct = False)
        z1, s1 = self._sigma_reducable_predict(1, x_new, noise_correct = False)
        s2 = (s1 + abs(Ef_weighed) * (s1 + s0) + Sf_weighed * abs(z1 - z0)).item()
        # now various options for lvl 2:
        # options = [np.sqrt(sigma_pred_model).item(), np.sqrt(mse_pred_weighed).item()]

        # s2_exp_red = min(options)
        s2_exp_red = np.sqrt(mse_pred_weighed).item()
        if self.printing:
            print(f"s2_exp_red {s2_exp_red} vs {s2}")


        # print("Z_pred_weighed: {:.4f}; Z_pred_model: {:.4f}".format(z_pred_weighed.item() ,z_pred_model.item()))
        # print("std_pred_weighed: {:.8f}; std_pred_model: {:8f}".format(np.sqrt(mse_pred_weighed).item(), np.sqrt(sigma_pred_model).item()))

        # TODO het grote vershcil met meliani is dat gratiet de onzerkerheid in het difference model mee neemt. (de sigma_delta`s)

        s_exp_red = np.array([s0_exp_red, s1_exp_red, s2_exp_red])

        extra_string = ""
        if not (self.proposed and self.use_single_fidelities) and len(std_meliani) == 3: #type: ignore
            # extra_string = f"\n\tmse_pred_smt_meliani: {std_pred_smt[2]:4f}"
            extra_string = f"\n\tsigma_meliani: \t\t{std_meliani}" #type: ignore
        
        if self.printing:
            print(f"MSE lvl 2 options are: \n\tsigma_pred_model:  \t{np.array(sigma_pred_model)}\n\tsigma_pred_weighed: \t{s_exp_red}{extra_string}")
        

        return s_exp_red 
        
    ##################### OLD
    def _sigma_reducable_predict(self, l, x_new, noise_correct = True):
        """
        Tries to find the part of the mse that is reducable by sampling and correcting for noise at the sampled points.
        This time, by taking the variance at sampled points from the predictor and averaging those.
        Seems a more logical and reasonable solution.
        """

        z_new, mse_new = self.K_mf[l].predict(x_new)

        if noise_correct:
            # _, mse = self.K_mf[l].predict(self.X_mf[l])
            _, mse = self.K_mf[l].predict(self.X_unique)
            return z_new, np.sqrt((mse_new - np.min(mse)).clip(min=0)).item()
        else:
            return z_new, np.sqrt(mse_new).item()


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
        
        # if we used reinterpolation, level selection at lower levels is nonsensical
        if self.options['optim_var'] == True:
            self.options['eval_noise'] = True
            self.options['optim_var'] = False
            self.train()
        self.options["propagate_uncertainty"] = propagate_uncertainty # standard option, calculates the cumulative 
        MSE_contribution, sigma2_rhos = self.predict_variances_all_levels(x_new)
        self.options["propagate_uncertainty"] = False # NEEEDS TO BE OFF otherwise reinterpolation will not work!!

        # the std2 contribution of level k to the highest fidelity
        # sigma_con = []
        # for l in range(MSE.shape[0]):
        #     sigma_con.append()

        # sigma_red is the sum of all sigma_con`s up to level k (where we sample)
        # NOTE I think Meliani is incorrect. Yes we do sample lower levels, and yes they do have a uncertainty contribution
        # However: when you sample a higher level, lower levels cannot provide extra info anymore to that level (see Kennedy & O`Hagan)
        # TODO Korondi 2021 gebruiken!!!!! die doet het wel goed, i.e. gebruikt alleen de sigma_con!


        return MSE_contribution.flatten()
        
