import numpy as np
from core.sampling.solvers.internal import TestFunction

from core.sampling.solvers.solver import get_solver
np.set_printoptions(precision=3,linewidth = 150,sign=' ')
import sys
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11})

np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)  # type: ignore
# pyright: reportGeneralTypeIssues=false, reportOptionalCall=false

from utils.linearity_utils import check_linearity

from preprocessing.input import Input

from core.kriging.mf_kriging import ProposedMultiFidelityKriging, MultiFidelityEGO, MFK_smt, MFK_org

from postprocessing.plotting import Plotting
from postprocessing.plot_convergence import ConvergencePlotting

# TODO list
# 1) animate the plotting
# 2) set_state get_state for new classes DONE
# 3) RMSE and other analysis tools for base MFK too!! (important for comparison)
# 4) repeated experiments for testfunctions
# 5) 
#    a) bij levelbepaling slechts 1 keer de MFK tunen! -> gebruik get/set_state method
#    b) voor de lager gelegen methodes: als punten dicht bij elkaar liggen (en er noise rond punten wordt aangenomen) 
#       dan is een punt als waarheid aannemen tegenproductief in de zin dat noise moet vermeerderen aldaar. 
#       Ofwel, je krijgt een negatieve hoeveelheid noise vermindering -> FOUT (maar kan goed zijn als je alleen naar het top-level kijkt!)
#       OPLOSSING: Le Gratiet / Meliani beschrijft hoe je de sigma_red vind voor het model van Le Gratiet.
# 6) in proposed method voorkeur geven aan de correlation function van het proposed level als die bestaat! -> meer stabiliteit?
# 7) for the R_diagonal: compare and choose to use either sigma_hat of the OK class or optimal_par.sigma2 of the MFK

# optional TODO list
# 1) use validation dataset, seperate from K_truth! (K_truth can be unreliable too, i.e. based on model)

# schrijven TODO list:
# 1) level selection procedure
# 2) integratie van MFK met eigen methode
# 3) branin aanpassing beschrijven

# inits based on input settings
setup = Input(0)
MFK_kwargs = {'print_global' : True,
                'print_training' : True,
                'print_prediction' : False,
                'eval_noise' : False,
                # 'eval_noise' : setup.noise_regression
                'n_start': 20, # 10 = default, but I want it a bit more robust ( does not always tune to the same -> major influence to own result!)
                }
# mf_model = MFK_smt(setup, max_cost = 150000, initial_nr_samples = 1, **MFK_kwargs)# NOTE cant use one (1) because of GLS in smt!
mf_model = MultiFidelityEGO(setup, initial_nr_samples = 2, max_cost = 150000, MFK_kwargs = MFK_kwargs)
# mf_model = ProposedMultiFidelityKriging(setup, max_cost = 150000, initial_nr_samples = 1, MFK_kwargs = MFK_kwargs)

mf_model.set_L([2, 3, None])
if isinstance(get_solver(setup),TestFunction):
    mf_model.set_L_costs([1,9,10000])   

# init plotting etc
pp = Plotting(setup, plotting_pause = 0.001, plot_once_every=4, fast_plot=True)
pp.set_zoom_inset([0,3], x_rel_range = [0.1,0.2])
cp = ConvergencePlotting(setup)


" level 0 and 1 : setting 'DoE' and 'solve' "
reuse_values = False
reload_endstate = False

used_endstate = hasattr(setup,'model_end') and reload_endstate
if used_endstate:
    mf_model.set_state(setup.model_end)
    cp.set_state(setup)
elif hasattr(setup,'model') and reuse_values:
    mf_model.set_state(setup.model)
else:
    mf_model.prepare_proposed(setup)

setup.create_input_file(mf_model, cp, endstate = False)

do_check = False
if do_check and not check_linearity(mf_model, pp):
    print("Linearity check: NOT LINEAR enough, but continueing for now.")
else:
    print("Linearity check: LINEAR enough!!")


" sample from the predicted distribution in EGO fashion"
if isinstance(mf_model, MultiFidelityEGO):
    mf_model.optimize(pp,cp)

setup.create_input_file(mf_model, cp, endstate = True)
cp.plot_convergence(mf_model)
pp.draw_current_levels(mf_model)

print("Simulation finished")

plt.show()

