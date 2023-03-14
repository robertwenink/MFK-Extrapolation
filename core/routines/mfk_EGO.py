# pyright: reportGeneralTypeIssues=false, reportUnboundVariable = false
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



class MultiFidelityEGO(ProposedMultiFidelityKriging, EfficientGlobalOptimization):
    
    def __init__(self, setup, *args, **kwargs):
        super().__init__(setup, *args, **kwargs)
        # Ik snap die inheritance structuur niet, onderstaand lijkt goed uit te leggen
        # https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way
        EfficientGlobalOptimization.__init__(self, setup, *args, **kwargs) 

    def optimize(self, pp : Plotting = None, cp : ConvergencePlotting = None):
        self.max_cost = np.inf
        self.max_iter = 80

        virtual_x_new, virtual_ei, virtual_y_min = None, None, None
        n = 0
        while np.sum(self.costs_total) < self.max_cost and n < self.max_iter:
            
            start = time.time()
            
            " predict and calculate Expected Improvement "
            prediction_function = self.K_mf[-1].predict
            x_new, ei = ProposedMultiFidelityKriging.find_best_point(self, prediction_function, criterion = 'EI')
            # x_new =  correct_formatX(np.array([[0.9904, 0.4875]]), self.d) # TODO remove again [[-3.017 11.948]]
            if self.printing:
                print(f"FINDING EI TOOK: {time.time() - start:4f}")
                print("MAXIMUM EI: {:4f}".format(np.max(ei)))

            if np.all(virtual_x_new == None):
                virtual_x_new, virtual_ei = x_new, ei - self.ei_criterion * 2
                _, y_min = self.get_best_sample() # this is a stable value!
                virtual_y_min = y_min
            else:
                # misschien juist de oude EI (virtual_ei) op dat punt gebruiken?
                # het sterke aan herberekenen vind ik dat het hyperparameter onafhankelijk/gelijk is tussen iteraties
                # calculate new EI at previous point
                pass
                virtual_x_min, virtual_y_min = self.get_best_extrapolation()
                
                if self.printing:
                    print(f'Best extrapolation / point at {virtual_x_min} with obj {virtual_y_min}')
                
                y_pred, mse_pred = prediction_function(virtual_x_min)
                _, y_min = self.get_best_sample() # this is a stable value!
                virtual_ei = self.EI(y_min, y_pred, np.sqrt(mse_pred))

            cost_ratio = self.costs_expected_nested[2]/(self.costs_expected_nested[1])

            ei_corrected = ei - virtual_ei

            # idea: dynamic ei-criterion based on cost difference ratio!
            # if the ratio is low, then its okay to sample quickly at a higher level
            if ei_corrected <= self.ei_criterion or ei_corrected <= ei / cost_ratio ** (1/2):
                # then sample the best extrapolated point, set ei to the corrected (could terminate!)
                x_new = virtual_x_new 

                # dit corrigeert de EI indien er noise aanwezig is,
                # adhv de EI op het huidige beste punt! 
                # zo niet dan is het gelijk aan normale EI!
                ei = ei_corrected 
            else:
                # otherwise continue exploring
                virtual_x_new = x_new
                # virtual_ei = ei
            
            # NOTE als we geen het_noise gebruiken maar re-interpolationg is de corrected EI gelijk aan de ei
            if self.printing:
                print("MAXIMUM VIRTUAL CORRECTED EI: {:4f}".format(ei_corrected))


            " output "
            if pp != None:
                pp.draw_current_levels(self)
            if cp != None:
                if cp.iteration_numbers == []: # if not set yet in set_state, we do add the first data
                    cp.update_convergence_data(self, x_new, ei_corrected) # pass these, we do not want to recalculate!!
                cp.plot_convergence()


            # terminate if criterion met, or when we revisit a point
            # NOTE level 2 is reinterpolated, so normally 0 EI at sampled points per definition!
            # however, we use regulation (both through R_diagonal of the proposed method as regulation constant for steady inversion)
            # this means the effective maximum EI can be higher than the criterion!
            if np.all(ei < self.ei_criterion):
                if self.printing:
                    print("Finishing due to reaching EI criterium!")
                # check if the basis for virtual_x_new has already been sampled
                if not isin(virtual_x_new, self.X_mf[-1]) and virtual_y_min < y_min: # second condition actually implies the first
                    if self.printing:
                        print("Sampled at the best extrapolation to end!")
                    self.sample_nested(2, x_new)
                break
            if isin(x_new,self.X_mf[-1]):
                if self.printing:
                    print("Finishing due to resampling previous point!")
                break


            l = self.level_selection(x_new)
            if self.printing:
                print("Sampling on level {} at {}".format(l,x_new))
            
            self.sample_nested(l, x_new)

            # weigh each prediction contribution according to distance to point.
            self.Z_pred, self.mse_pred, _ = self.weighted_prediction()
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

        sigma_red = self._std_reducable_proposed(x_new) # kan op x_new 0 zijn
        # s0_red, s1_red, s2_red = self._mse_reducable_meliani_smt(x_new)
        
        maxx = 0
        l = 0
        # dit is het normaal, als de lagen gelinkt zijn! is hier niet, dit werkt alleen voor meliani.
        for i in range(1,3):
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
        z_pred_weighed, mse_pred_weighed, Ef_weighed = self.weighted_prediction(self, X_test=x_new)


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
        
        if self.printing:
            print(f"MSE lvl 2 options are: mse_pred_model, mse_pred_weighed, mse_pred_smt\n\t{options}")
        # s2_exp_red = min(options)
        s2_exp_red = np.sqrt(mse_pred_weighed).item()


        # print("Z_pred_weighed: {:.4f}; Z_pred_model: {:.4f}".format(z_pred_weighed.item() ,z_pred_model.item()))
        # print("std_pred_weighed: {:.8f}; std_pred_model: {:8f}".format(np.sqrt(mse_pred_weighed).item(), np.sqrt(mse_pred_model).item()))

        s_exp_red = [s0_exp_red, s1_exp_red, s2_exp_red]
        # print(s_exp_red)
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
            return z_new, (mse_new - np.average(mse)).clip(min=0).item()
        else:
            return z_new, mse_new


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
        
