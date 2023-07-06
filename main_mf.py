import numpy as np
from core.sampling.DoE import LHS

from core.sampling.solvers.internal import TestFunction
from core.sampling.solvers.solver import get_solver
from utils.correlation_utils import print_pearson_correlations
from utils.error_utils import RMSE_focussed, RMSE_norm_MF
from utils.formatting_utils import correct_formatX
np.set_printoptions(precision=4,linewidth = 150,sign=' ',suppress = True)
import sys
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11})

import warnings
warnings.simplefilter("ignore", RuntimeWarning) # Scipy optimize can throw x out of bounds, but this is really consequential

# np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)  # type: ignore
from copy import deepcopy, copy
# pyright: reportGeneralTypeIssues=false, reportOptionalCall=false

from utils.linearity_utils import check_linearity

from preprocessing.input import Input

# options to pick starting class from
from core.mfk.mfk_base import MultiFidelityKrigingBase
from core.mfk.mfk_smt import MFK_smt
from core.mfk.mfk_ok import MFK_org
from core.mfk.proposed_mfk import ProposedMultiFidelityKriging
from core.routines.mfk_EGO import MultiFidelityEGO

from postprocessing.plotting import Plotting
from postprocessing.plot_live_metrics import ConvergencePlotting

# new TODO 
# - linearity check voor proposed met smt afmaken
# - kijken of run 05396_00000 op L = 2 een normale tijd heeft (wss erg lang door laptop dicht tussendoor!)

# schrijven TODO list:
# 0) Het is frappant hoe erg mijn extrapolatie methode en de method van la gratiet op elkaar lijken!!
# 0a) 'three assumptions' in eigen methode moet de laatste weg, voor vergleijking met le gratiet method moet s0+s1 zijn niet -
#     bovendien, we hebben de variances per level als iid beschouwd, dat is tegenstrijdig
#     de afstand tussen twee variances is logischerwijs de som daarvan!! -> is weer normal!
# 2) integratie van MFK met eigen methode: bijv de re-interpolation en het-noise
# 4) beschrijven dat de weighing methode niet werkt met universal kriging omdat er dan niet perse correlaties meer bestaan
#       de trend beschrijft namelijk al de variabiliteit in de data en de correlatie hoeft niet meer nodig te zijn perse.
        # NOTE dit is dus niet perse waar. Als het GLS model al alle variabiliteit kan omschrijven, hoeft de Kriging correlation niks meer te doen! \
        # In termen van het Kriging model betekent dit dat twee punten met correlation 1 tov een ander punt een gelijke bijdrage behoren te hebben!
# 5) Beschrijven dat: In de proposed method procedure het model met de laagste 2 levels gescheiden houden in K_mf van de prediction!
#       Dit moet omdat de levels gelinkt zijn (er is bijv maar 1 sigma_hat) 
#       en het toplevel als de waarheid wordt genomen terwijl dat bij mij niet perse zo is omdat ik predicted points doorgeef.
#       dus: onderliggend mfk model lvl0 + lvl1 != mfk model lvl2_pred !!
#       als we dit niet doen zullen de laagste levels gaan overfitten en is de mse_pred ook -> 0 dus niks meer waard!
# 6) nieuwe EI selection procedure beschrijven!!
#       Onderdeel hiervan is de 'reducable' amount of noise, en dat meliani/korondi/gratiet dat niet doen.
# 8) discussie: beschrijven dat het interssant is te zien hoe bij noise de initiele surrogate (of eigenlijk vooral het eerste sample dat dicht bij elkaar light)
#           niet heel goed kan zijn, maar dat zodra de noise estimates beter worden 
#           door meer samples op de lower fidelities (dicht bij elkaar) de prediction vele malen beter wordt!!
# 9) combinatie eval_noise + heteroscedastic beschrijven voor proposed, reinterpolation voor de reference. 
#   Beschrijven dat alhoewel de aangepaste EI procedure voor beide wordt gebruikt, dit gezien de reinterpolatie voor de reference gelijk is aan het gebruikelijke.
# 11) No clear difference between using heteroscedastic noise evaluation or normal noise + reinterpolation has been observed.


# inits based on input settings
setup = Input(0)

conv_mods = [0, 1, 2, 3]
conv_types = ["Stable up", "Stable down", "Alternating"]
solver_noises = [0.0, 0.02, 0.05]

setup.conv_mod = conv_mods[1]
setup.conv_type = conv_types[1]
setup.solver_noise = solver_noises[0]

reuse_values = False
reload_endstate = False
# NOTE deze waardes aanpassen werkt alleen als reuse_values en reload_endstate uitstaat!
MFK_kwargs = {'print_global' : False,
                'print_training' : True,
                'print_prediction' : False,
                'eval_noise' : True,# always true
                # 'eval_noise' : setup.noise_regression, 
                'propagate_uncertainty' : False,  
                'optim_var' : True, # true: HF samples is forced to zero; = reinterpolation
                'hyper_opt' : 'Cobyla', # [‘Cobyla’, ‘TNC’] Cobyla standard
                'n_start': 30, # 10 = default, but I want it a bit more robust ( does not always tune to the same -> major influence to own result!)
                'corr' : 'squar_exp',
                }
  

# init plotting etc
pp = Plotting(setup, plotting_pause = 0.001, plot_once_every=1, fast_plot=False, make_video=True)
pp.plot_exact_new_figure()
# plt.show()
# pp.set_zoom_inset([0,3], x_rel_range = [0.05,0.2])
pp.plot_NRMSE_text = True
cp = ConvergencePlotting(setup)

plt.show()
sys.exit()


# NOTE zonder noise werkt zonder optim var beter voor reference
# mf_model = MFK_smt(setup, max_cost = 150000, initial_nr_samples = 1, **MFK_kwargs)# NOTE cant use one (1) because of GLS in smt!
mf_model = MultiFidelityEGO(setup, proposed = True, optim_var = True, initial_nr_samples = 2, max_cost = np.inf, MFK_kwargs = MFK_kwargs)
# mf_model = ProposedMultiFidelityKriging(setup, max_cost = 150000, initial_nr_samples = 1, MFK_kwargs = MFK_kwargs)

mf_model.distance_weighing = True
mf_model.variance_weighing = True

mf_model.method_weighing = False
mf_model.try_use_MFK = False

mf_model.use_uncorrected_Ef = False
mf_model.use_het_noise = False

# NOTE for EVA: refinement levels
mf_model.set_L([0.5, 1, 2])

if isinstance(get_solver(setup),TestFunction):
    mf_model.set_L([2, 3, None])
    # mf_model.set_L([1, 2, None])
    # mf_model.set_L([0, 2, None])
    mf_model.set_L_costs([1,10,1000])   


" level 0 and 1 : setting 'DoE' and 'solve' " 
used_endstate = hasattr(setup,'model_end') and hasattr(setup,'prepare_succes') and reload_endstate
if used_endstate:
    mf_model.set_state(deepcopy(setup.model_end))
    cp.set_state(setup)
elif hasattr(setup,'model') and hasattr(setup,'prepare_succes') and reuse_values:
    # deepcopy required!! copy gebruikt nog steeds references if possible, 
    # ofwel via set_attr linken we de dict setup.model direct aan de values van mf_model (en die worden geupdate!)
    mf_model.set_state(deepcopy(setup.model)) 
else: 
    mf_model.prepare_initial_surrogate(setup)  

setup.create_input_file(mf_model, cp if used_endstate else None, endstate = used_endstate)

pp.draw_current_levels(mf_model)

# provide full report
RMSE = RMSE_norm_MF(mf_model, no_samples=True)
X_RMSE = LHS(setup, n_per_d=80)
RMSE_focus = RMSE_focussed(mf_model, X_RMSE, 10)
mf_model.print_stats(RMSE, RMSE_focus)
print_pearson_correlations(mf_model)

plt.show()
sys.exit()

" sample from the predicted distribution in EGO fashion"
if isinstance(mf_model, MultiFidelityEGO):
    # mf_model.optimize(pp, cp, max_iter = 1)
    mf_model.optimize(pp, cp)
    # mf_model.optimize()   

" post processing "
setup.create_input_file(mf_model, cp, endstate = True)
cp.plot_convergence() 
pp.draw_current_levels(mf_model)

# report again fully
RMSE = RMSE_norm_MF(mf_model, no_samples=True)
X_RMSE = LHS(setup, n_per_d=80)
RMSE_focus = RMSE_focussed(mf_model, X_RMSE, 10)
mf_model.print_stats(RMSE, RMSE_focus)
print_pearson_correlations(mf_model)

# to print the best point easily
mf_model.solver.solve(correct_formatX(mf_model.get_best_sample()[0][0],mf_model.d),mf_model.L[-1])

try:
    pp.render_video()
except:
    pass

print(" Simulation finished ")
plt.show()


