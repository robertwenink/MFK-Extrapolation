import numpy as np

from core.sampling.solvers.internal import TestFunction
from core.sampling.solvers.solver import get_solver
np.set_printoptions(precision=3,linewidth = 150,sign=' ')
import sys
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11})

# np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)  # type: ignore
from copy import deepcopy, copy
# pyright: reportGeneralTypeIssues=false, reportOptionalCall=false

from utils.linearity_utils import check_linearity

from preprocessing.input import Input

from core.mfk.mfk_base import MultiFidelityKrigingBase
from core.mfk.mfk_smt import MFK_smt
from core.mfk.mfk_ok import MFK_org
from core.mfk.proposed_mfk import ProposedMultiFidelityKriging
from core.routines.mfk_EGO import MultiFidelityEGO

from postprocessing.plotting import Plotting
from postprocessing.plot_live_metrics import ConvergencePlotting

import tkinter as tk
import ctypes
root = tk.Tk()
root.destroy()

# TODO list
# 1) (DONE) animate the plotting
# 2) (DONE) set_state get_state for new classes
# 3) RMSE and other analysis tools for base MFK too!! (important for comparison)
# 4) repeated experiments for testfunctions
# 5) 
#    a) (DONE) bij levelbepaling slechts 1 keer de MFK tunen! -> gebruik get/set_state method
#    b) (DONE) voor de lager gelegen methodes: als punten dicht bij elkaar liggen (en er noise rond punten wordt aangenomen) 
#       dan is een punt als waarheid aannemen tegenproductief in de zin dat noise moet vermeerderen aldaar. 
#       Ofwel, je krijgt een negatieve hoeveelheid noise vermindering -> FOUT (maar kan goed zijn als je alleen naar het top-level kijkt!)
#       OPLOSSING: Le Gratiet / Meliani beschrijft hoe je de sigma_red vind voor het model van Le Gratiet.
#       niet de oplossing!
#
# 6) DONE in proposed method voorkeur geven aan de correlation function van het proposed level als die bestaat! -> meer stabiliteit?
# 7) (DONE) results are quite similar. For the R_diagonal: compare and choose to use either sigma_hat of the OK class or optimal_par.sigma2 of the MFK
# 8) DONE corr is altijd 1 voor ObjectWrapper (zie schrijven todo 4 ook) -> eerst op level 1 tunen!
# 9) DONE mse_pred is negatief voor MFK in Kriging_unknown_z
# 10) DONE (doe ik niet) methode weighing alleen gebruiken als voorbij threshold (not done) én wanneer hoogste level MFK_smt gedefinieerd is (meer dan 3 samples) (Done). (krijgen nu wispelturige resultaten -> DONE smt geimplementeerd)
# 11) DONE In de proposed method procedure het model met de laagste 2 levels gescheiden houden in K_mf van de prediction!
#       Dit moet omdat de levels gelinkt zijn (er is bijv maar 1 sigma_hat) 
#       en het toplevel als de waarheid wordt genomen terwijl dat bij mij niet perse zo is omdat ik predicted points doorgeef.
#       dus: onderliggend mfk model lvl0 + lvl1 != mfk model lvl2_pred !!
#       als we dit niet doen zullen de laagste levels gaan overfitten en is de mse_pred ook -> 0 dus niks meer waard!
# 12) DONE implement efficient EI optimization
# 12a) implement some form of resolution procedure such that we do not infinitely focus on one spot.
#       - EI blijft nu tov het enige echte gesamplde punt! Ik moet juist alle predicted points meenemen! Daar zit uiteraard ook het echte punt tussen.
#       - omdat we zelf noise toevoegen zullen we altijd noise houden op welke plek dan ook! (zelfs na lvl 0 & 1 samplen op max EI locatie) 
#         de EI zal dus ook niet naar 0 gaan. In plaats daarvan zouden we de EI moeten vergelijken met de EI van de minimum extrapolated locatie.
#         Als die EI_new - EI_miny > 0, dan sampelen we daar op level 0 of 1 (op basis van level selection methode)
#         Anders sampelen we op de locatie van EI_miny (wat dus wordt verhoogd naar lvl 2)
#         PROBLEEM: hoe onderscheiden we de mse van een echt gesampled punt met die van een extrapolated? een vorm van EI criterion aanhouden en die resetten als lvl 2 is gesampled?
#         PROBLEEM: als de extrapolatie instabiel is, zal dit ook instabiel zijn omdat onze EI tov minimum extrapolated is -> zal betekenen dat uiteindelijk die minimum extrapolated gesampled wordt en we een correctie krijgen.
#         OPLOSSING: neem nog steeds het sampled punt als minimum, maar vergelijk consequtive EIs. Als EI2 < EI1 sample op EI1. als EI2> EI1, ga door.
#                   indien er dan gesampled is op lvl2, schuift simpelweg het minimum weer op en tada! 
#                   -> geen infinite sampling meer waar elke sampling wel variance reductie geeft en er infinitisimaal dichtbij ernaast gwn weer wordt gesampled.

# optional TODO list
# 1) use validation dataset, seperate from K_truth! (K_truth can be unreliable too, i.e. based on model)

# schrijven TODO list:
# 0) Het is frappant hoe erg mijn extrapolatie methode en de method van la gratiet op elkaar lijken!!
# 0a) 'three assumptions' in eigen methode moet de laatste weg, voor vergleijking met le gratiet method moet s0+s1 zijn niet -
#     bovendien, we hebben de variances per level als iid beschouwd, dat is tegenstrijdig
#     de afstand tussen twee variances is logischerwijs de som daarvan!! -> is weer normal!
# 1) DONE level selection procedure
# 2) integratie van MFK met eigen methode
# 3) branin aanpassing beschrijven
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

# inits based on input settings
setup = Input(0)
reuse_values = True	
reload_endstate = False
# NOTE deze waardes aanpassen werkt alleen als reuse_values en reload_endstate uitstaat!
MFK_kwargs = {'print_global' : False,
                'print_training' : False,
                'print_prediction' : False,
                # 'eval_noise' : False,
                'eval_noise' : setup.noise_regression,
                'propagate_uncertainty' : False, 
                'optim_var' : False, # true: HF samples is forced to zero; = reinterpolation
                'hyper_opt' : 'Cobyla', # [‘Cobyla’, ‘TNC’] Cobyla standard
                'n_start': 30, # 10 = default, but I want it a bit more robust ( does not always tune to the same -> major influence to own result!)
                'corr' : 'squar_exp',
                }
# mf_model = MFK_smt(setup, max_cost = 150000, initial_nr_samples = 1, **MFK_kwargs)# NOTE cant use one (1) because of GLS in smt! 
# mf_model = MultiFidelityEGO(setup, initial_nr_samples = 1, max_cost = 150000, MFK_kwargs = MFK_kwargs)
mf_model = ProposedMultiFidelityKriging(setup, max_cost = 150000, initial_nr_samples = 3, MFK_kwargs = MFK_kwargs)

# NOTE for EVA: refinement levels
mf_model.set_L([0.5, 1, 2])

if isinstance(get_solver(setup),TestFunction):
    mf_model.set_L([2, 3, None])
    mf_model.set_L_costs([1,9,10000])   

# init plotting etc
pp = Plotting(setup, plotting_pause = 0.001, plot_once_every=1, fast_plot=True, make_video=True)
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
    mf_model.prepare_proposed(setup)  

setup.create_input_file(mf_model, cp if used_endstate else None, endstate = used_endstate)

do_check = False
if mf_model.X_mf[-1].shape[0] >= 3:
    if do_check and not reload_endstate and not check_linearity(mf_model, pp):
        print("WARNING Linearity check: NOT LINEAR enough, but continueing for now.")
    else:
        print("Linearity check: LINEAR enough!")
else:
    print("Too little hifi samples for reliable linearity check!")


" sample from the predicted distribution in EGO fashion"
if isinstance(mf_model, MultiFidelityEGO):
    mf_model.optimize(pp,cp)
    # mf_model.optimize()   

setup.create_input_file(mf_model, cp, endstate = True)
cp.plot_convergence() 
pp.draw_current_levels(mf_model)
pp.render_video()

print(" Simulation finished ")

plt.show()
# plt.draw()

