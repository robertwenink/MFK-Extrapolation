# pyright: reportGeneralTypeIssues=false, reportOptionalCall=false
import numpy as np
from matplotlib import pyplot as plt
import os

from core.routines.mfk_EGO import MultiFidelityEGO
from core.sampling.solvers.solver import get_solver
from utils.correlation_utils import print_pearson_correlations
from utils.error_utils import RMSE_norm_MF
plt.rcParams['figure.figsize'] = [8, 5]

from smt.applications import MFK
MFK_kwargs = {
            # 'theta0' : theta0, 
            # 'theta_bounds' : [1e-1, 20],
            'print_global' : False,
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

def HF_function(x): # high-fidelity function
    return ((6*x-2)**2)*np.sin((12 * x-4))  

x = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)
x_min = x[np.argmin(HF_function(x))]
y_min = HF_function(x_min)

#Expensive DOE with 7 points
Xt_e = np.array([0.00, 0.65, 1, 0.15, 0.3, 0.45, 0.85, 0.92]).reshape(-1, 1)

# Evaluate the HF and LF functions
yt_e = HF_function(Xt_e)

for i in range(0,4):
    fig, ax = plt.subplots(1, figsize = (15,5))
    ax.scatter(Xt_e[:i], yt_e[:i], color='C0', label='HF DoE')
    if i > 0:
        ax.scatter(Xt_e[0], -10, clip_on=False, zorder =10, marker = "o", s = 2000, color='C0')
        # ax.scatter(Xt_e[0], yt_e[0] + 5, marker = "o", s = 2000, color='C0')
    if i > 1:
        ax.scatter(Xt_e[1], -10, clip_on=False, zorder =10, marker = "v", s = 2000, color='C0')
        # ax.scatter(Xt_e[1], yt_e[1] + 5, marker = "v", s = 2000, color='C0')
    if i > 2:
        ax.scatter(Xt_e[2], -10, clip_on=False, zorder =10, marker = "s", s = 2000, color='C0')
        # ax.scatter(Xt_e[2], yt_e[2] + 5, marker = "s", s = 2000, color='C0')
    
    ax.set_ylim(-10, 17); ax.set_xlim(-0.1, 1.1)
    ax.set_xlabel('Design x'); ax.set_ylabel('Simulation result: Maximum acceleration')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # plt.legend()

fig, ax = plt.subplots(1, figsize = (15,5))
ax.scatter(Xt_e, yt_e, color='C0', label='HF DoE')
ax.set_ylim(-10, 17); ax.set_xlim(-0.1, 1.1)
ax.set_xlabel('Design x'); ax.set_ylabel('Simulation result: Maximum acceleration')
ax.scatter(Xt_e[0], -10, clip_on=False, zorder =10, marker = "o", s = 2000, color='C0')
ax.scatter(Xt_e[1], -10, clip_on=False, zorder =10, marker = "v", s = 2000, color='C0')
ax.scatter(Xt_e[2], -10, clip_on=False, zorder =10, marker = "s", s = 2000, color='C0')
# ax.scatter(Xt_e[0], yt_e[0] + 5, marker = "o", s = 2000, color='C0')
# ax.scatter(Xt_e[1], yt_e[1] + 5, marker = "v", s = 2000, color='C0')
# ax.scatter(Xt_e[2], yt_e[2] + 5, marker = "s", s = 2000, color='C0')
ax.set_xticklabels([])
ax.set_yticklabels([])

fig, ax = plt.subplots(1, figsize = (15,5))
ax.scatter(Xt_e, yt_e, color='C0', label='HF DoE')
ax.plot(x, HF_function(x), '--C0', label='High Fidelity (HF)')
ax.scatter(Xt_e[0], -10, clip_on=False, zorder =10, marker = "o", s = 2000, color='C0')
ax.scatter(Xt_e[1], -10, clip_on=False, zorder =10, marker = "v", s = 2000, color='C0')
ax.scatter(Xt_e[2], -10, clip_on=False, zorder =10, marker = "s", s = 2000, color='C0')
# ax.scatter(Xt_e[0], yt_e[0] + 5, marker = "o", s = 2000, color='C0')
# ax.scatter(Xt_e[1], yt_e[1] + 5, marker = "v", s = 2000, color='C0')
# ax.scatter(Xt_e[2], yt_e[2] + 5, marker = "s", s = 2000, color='C0')
ax.set_ylim(-10, 17); ax.set_xlim(-0.1, 1.1)
ax.set_xlabel('Design x'); ax.set_ylabel('Simulation result: Maximum acceleration')
ax.set_xticklabels([])
ax.set_yticklabels([])

fig, ax = plt.subplots(1, figsize = (15,5))
ax.scatter(Xt_e, yt_e, color='C0', label='HF DoE')
ax.scatter(x_min, y_min, color='C2', s = 100, zorder = 10, label='Next best option')
ax.scatter(Xt_e[0], -10, clip_on=False, zorder =10, marker = "o", s = 2000, color='C0')
ax.scatter(Xt_e[1], -10, clip_on=False, zorder =10, marker = "v", s = 2000, color='C0')
ax.scatter(Xt_e[2], -10, clip_on=False, zorder =10, marker = "s", s = 2000, color='C0')
# ax.scatter(Xt_e[0], yt_e[0] + 5, marker = "o", s = 2000, color='C0')
# ax.scatter(Xt_e[1], yt_e[1] + 5, marker = "v", s = 2000, color='C0')
# ax.scatter(Xt_e[2], yt_e[2] + 5, marker = "s", s = 2000, color='C0')
ax.plot(x, HF_function(x), '--C0', label='High Fidelity (HF)')
ax.set_ylim(-10, 17); ax.set_xlim(-0.1, 1.1)
ax.set_xlabel('Design x'); ax.set_ylabel('Simulation result: Maximum acceleration')
ax.set_xticklabels([])
ax.set_yticklabels([])
# plt.legend()

# save all
figs = [plt.figure(n) for n in plt.get_fignums()]
p = os.path.normpath(r"C:\Users\RobertWenink\OneDrive - Delft University of Technology\Documents\TUDelft\Master\Afstuderen\Presentations\eindpresentatie\forrester only surrogate optimalisatie showcase")
for i, fig in enumerate(figs):
    fig.savefig(os.path.join(p,f"Figure {i+1}"), dpi = 500)

plt.show()