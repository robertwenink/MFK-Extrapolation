import numpy as np

from core.sampling.solvers.internal import TestFunction
from core.sampling.solvers.solver import get_solver
np.set_printoptions(precision=3,linewidth = 150,sign=' ',suppress = True)
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
# - in proposed method de z_pred op sample locaties incrementen met de noise die we eerder vonden! De noise is namelijk niet 0 op sample locaties.
# 

# TODO list
# 6) DONE in proposed method voorkeur geven aan de correlation function van het proposed level als die bestaat! -> meer stabiliteit?
# 11) DONE In de proposed method procedure het model met de laagste 2 levels gescheiden houden in K_mf van de prediction!
#       Dit moet omdat de levels gelinkt zijn (er is bijv maar 1 sigma_hat) 
#       en het toplevel als de waarheid wordt genomen terwijl dat bij mij niet perse zo is omdat ik predicted points doorgeef.
#       dus: onderliggend mfk model lvl0 + lvl1 != mfk model lvl2_pred !!
#       als we dit niet doen zullen de laagste levels gaan overfitten en is de mse_pred ook -> 0 dus niks meer waard!

# optional TODO list
# 1) use validation dataset, seperate from K_truth! (K_truth can be unreliable too, i.e. based on model)

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
# 7) see train() in mfk_smt: OLS is only possible from 3 hifi samples onwards, independent of 2 or 3 levels !!
# 8) discussie: beschrijven dat het interssant is te zien hoe bij noise de initiele surrogate (of eigenlijk vooral het eerste sample dat dicht bij elkaar light)
#           niet heel goed kan zijn, maar dat zodra de noise estimates beter worden 
#           door meer samples op de lower fidelities (dicht bij elkaar) de prediction vele malen beter wordt!!
# 9) combinatie eval_noise + heteroscedastic beschrijven voor proposed, reinterpolation voor de reference. 
#   Beschrijven dat alhoewel de aangepaste EI procedure voor beide wordt gebruikt, dit gezien de reinterpolatie voor de reference gelijk is aan het gebruikelijke.
# 10) verandering van Sf weighing naar min ipv mean omdat het beste sample richtgevend moet zijn.
# 11) No clear difference between using heteroscedastic noise evaluation or normal noise + reinterpolation has been observed.


# inits based on input settings
setup = Input(0)

conv_mods = [0, 1, 2, 3]
solver_noises = [0.0, 0.02, 0.1]
conv_types = ["Stable up", "Stable down", "Alternating"]

setup.conv_mod = conv_mods[1]
setup.conv_type = conv_types[0] # 1 TODO
setup.solver_noise = solver_noises[1] # 0 TODO

reuse_values = False
reload_endstate = False
# NOTE deze waardes aanpassen werkt alleen als reuse_values en reload_endstate uitstaat!
MFK_kwargs = {'print_global' : False,
                'print_training' : True,
                'print_prediction' : False,
                'eval_noise' : True,# always true
                # 'eval_noise' : setup.noise_regression, 
                'propagate_uncertainty' : False,  
                'optim_var' : False, # true: HF samples is forced to zero; = reinterpolation
                'hyper_opt' : 'Cobyla', # [‘Cobyla’, ‘TNC’] Cobyla standard
                'n_start': 30, # 10 = default, but I want it a bit more robust ( does not always tune to the same -> major influence to own result!)
                'corr' : 'squar_exp',
                }

# mf_model = MFK_smt(setup, max_cost = 150000, initial_nr_samples = 1, **MFK_kwargs)# NOTE cant use one (1) because of GLS in smt!
mf_model = MultiFidelityEGO(setup, proposed = True, initial_nr_samples = 1, max_cost = np.inf, MFK_kwargs = MFK_kwargs)
# mf_model = ProposedMultiFidelityKriging(setup, max_cost = 150000, initial_nr_samples = 1, MFK_kwargs = MFK_kwargs)


# NOTE for EVA: refinement levels
mf_model.set_L([0.5, 1, 2])

if isinstance(get_solver(setup),TestFunction):
    mf_model.set_L([1, 2, None])
    mf_model.set_L_costs([1,10,1000])   


# init plotting etc
pp = Plotting(setup, plotting_pause = 0.001, plot_once_every=1, fast_plot=True, make_video=False)
pp.set_zoom_inset([0,3], x_rel_range = [0.05,0.2])
cp = ConvergencePlotting(setup)



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

do_check = False
if mf_model.X_mf[-1].shape[0] >= 3:
    if do_check and not reload_endstate and not check_linearity(mf_model, pp):
        print("WARNING Linearity check: NOT LINEAR enough, but continueing for now.")
    else:
        print("Linearity check: LINEAR enough!")
else:
    print("Too little hifi samples for reliable linearity check!")

# sys.exit()

" sample from the predicted distribution in EGO fashion"
if isinstance(mf_model, MultiFidelityEGO):
    mf_model.optimize(pp,cp)
    # mf_model.optimize()   

" post processing "
setup.create_input_file(mf_model, cp, endstate = True)
cp.plot_convergence() 
pp.draw_current_levels(mf_model)
pp.render_video()

print(" Simulation finished ")
plt.show()


