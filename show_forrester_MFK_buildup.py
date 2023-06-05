# pyright: reportGeneralTypeIssues=false, reportOptionalCall=false
import numpy as np
from matplotlib import pyplot as plt

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

def LF_function(x): # low-fidelity function
    return 0.5 * HF_function(x) + (x-0.5) * 10. - 5

x = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)

plt.figure(figsize = (15, 5))
plt.subplot(1,2,1)
plt.plot(x, HF_function(x), '--C0', label='High Fidelity (HF)')
plt.plot(x, LF_function(x), '--C1', label='Low Fidelity (LF)')
plt.ylim(-10, 17); plt.xlim(-0.1, 1.1)
plt.xlabel('x'); plt.ylabel('y')
plt.legend()

#Expensive DOE with 4 points
Xt_e = np.array([0.0, 0.3, 0.6, 1.]).reshape(-1, 1)
#Cheap DOE with 12 points
Xt_c = np.linspace(1/9, 8/9, 8, endpoint=True).reshape(-1, 1)
# The two DOEs must be nest: Xt_e must be included within Xt_c
Xt_c = np.concatenate((Xt_c,Xt_e),axis=0)
Xt_m = Xt_c


# Evaluate the HF and LF functions
yt_e = HF_function(Xt_e)
yt_c = LF_function(Xt_c)



plt.figure(figsize = (15, 5))
plt.subplot(1,2,1)
plt.plot(x, HF_function(x), '--C0', label='High Fidelity (HF)')
plt.plot(x, LF_function(x), '--C1', label='Low Fidelity (LF)')
plt.scatter(Xt_e, yt_e, color='C0', label='HF DoE')
plt.scatter(Xt_c, yt_c, color='C1', label='LF DoE')
plt.ylim(-10, 17); plt.xlim(-0.1, 1.1)
plt.xlabel('x'); plt.ylabel('y')
plt.legend()



# %% Build the MFK object
np.random.seed(7)
sm = MFK(**MFK_kwargs)

# low-fidelity dataset names being integers from 0 to level-1
sm.set_training_values(Xt_c, yt_c, name=0)
# high-fidelity dataset without name
sm.set_training_values(Xt_e, yt_e)
# train the model
sm.train()

# test training
ntest = 101
nlvl = len(sm.X)
x = np.linspace(0, 1, ntest, endpoint=True).reshape(-1, 1)
# HF
y = sm.predict_values(x)
var = sm.predict_variances(x)
# LF
y0 = sm._predict_intermediate_values(x, 1)
var0, _ =  sm.predict_variances_all_levels(x)
var0 = var0[:,0].reshape(-1,1)


# creating a single-fidelity Kriging model to compare with
theta0 = sm.options["theta0"]
noise0 = sm.options["noise0"]

# %% Kriging on monofidelity HF doe
sm_monoHF =  MFK(theta0=theta0[1].tolist(), theta_bounds = [1e-1, 20],
                 noise0=[noise0[1]], use_het_noise=False, n_start=1)

# %% Kriging on monofidelity LF Doe
sm_monoLF =  MFK(theta0=theta0[0].tolist(), theta_bounds = [1e-1, 20],
                 noise0=[noise0[0]], use_het_noise=False, n_start=1)

# train the model
sm_monoHF.set_training_values(Xt_e, yt_e)
sm_monoHF.train()

sm_monoLF.set_training_values(Xt_c, yt_c)
sm_monoLF.train()

# test training
y_monoHF = sm_monoHF.predict_values(x)
var_monoHF = sm_monoHF.predict_variances(x)

y_monoLF = sm_monoLF.predict_values(x)
var_monoLF = sm_monoLF.predict_variances(x)

" sf modeled fidelities "
plt.figure(figsize = (15, 5))
plt.subplot(1,2,1)
plt.plot(x, HF_function(x), '--C0', label='High Fidelity (HF)')
plt.plot(x, LF_function(x), '--C1', label='Low Fidelity (LF)')
plt.plot(x, y_monoHF, 'C2', label='Kriging - HF')
plt.plot(x, y_monoLF, 'C3', label='Kriging - LF')
plt.scatter(np.squeeze(Xt_e), np.squeeze(yt_e), label = "HF DoE")
plt.scatter(np.squeeze(Xt_c), np.squeeze(yt_c), label = "LF DoE")
plt.ylim(-10, 17); plt.xlim(-0.1, 1.1)
plt.xlabel('x'); plt.ylabel('y')
plt.title('Single-fidelity Kriging')
plt.legend()

plt.figure(figsize = (15, 5))
plt.subplot(1,2,1)
plt.plot(x, HF_function(x), '--C0', label='High Fidelity (HF)')
plt.plot(x, LF_function(x), '--C1', label='Low Fidelity (LF)')
plt.plot(x, y_monoHF, 'C2', label='Kriging - HF')
plt.fill_between(np.ravel(x), np.ravel(y_monoHF-3*np.sqrt(var_monoHF)),
                  np.ravel(y_monoHF+3*np.sqrt(var_monoHF)),
                  color='C2',alpha=0.2, label ='Confidence Interval 99% HF')
plt.plot(x, y_monoLF, 'C3', label='Kriging - LF')
plt.fill_between(np.ravel(x), np.ravel(y_monoLF-3*np.sqrt(var_monoLF)),
                  np.ravel(y_monoLF+3*np.sqrt(var_monoLF)),
                  color='C3',alpha=0.2, label ='Confidence Interval 99% LF')
plt.scatter(np.squeeze(Xt_e), np.squeeze(yt_e), label = "HF DoE")
plt.scatter(np.squeeze(Xt_c), np.squeeze(yt_c), label = "LF DoE")
plt.ylim(-10, 17); plt.xlim(-0.1, 1.1)
plt.xlabel('x'); plt.ylabel('y')
plt.title('Single-fidelity Kriging')
plt.legend()



plt.figure(figsize = (15, 5))
plt.subplot(1,2,1)
plt.plot(x, HF_function(x), '--C0', label='High Fidelity (HF)')
plt.plot(x, LF_function(x), '--C1', label='Low Fidelity (LF)')
plt.plot(x, y_monoHF, 'C2', label='Kriging - HF')
plt.fill_between(np.ravel(x), np.ravel(y_monoHF-3*np.sqrt(var_monoHF)),
                  np.ravel(y_monoHF+3*np.sqrt(var_monoHF)),
                  color='C2',alpha=0.2, label ='Confidence Interval 99% HF')
plt.plot(x, y_monoLF, 'C3', label='Kriging - LF')
plt.fill_between(np.ravel(x), np.ravel(y_monoLF-3*np.sqrt(var_monoLF)),
                  np.ravel(y_monoLF+3*np.sqrt(var_monoLF)),
                  color='C3',alpha=0.2, label ='Confidence Interval 99% LF')
plt.scatter(np.squeeze(Xt_e), np.squeeze(yt_e), label = "HF DoE")
plt.scatter(np.squeeze(Xt_c), np.squeeze(yt_c), label = "LF DoE")
plt.ylim(-10, 17); plt.xlim(-0.1, 1.1)
plt.xlabel('x'); plt.ylabel('y')
plt.title('Single-fidelity Kriging')
plt.legend()


plt.subplot(1,2,2)
plt.plot(x, HF_function(x), '--C0', label='High Fidelity (HF)')
plt.plot(x, LF_function(x), '--C1', label='Low Fidelity (LF)')
plt.plot(x, y, 'C0', label='MFK - HF')
plt.fill_between(np.ravel(x), np.ravel(y-3*np.sqrt(var)),
                  np.ravel(y+3*np.sqrt(var)),
                  color='C0',alpha=0.2, label ='Confidence Interval 99% HF')
plt.plot(x, y0, 'C1', label='MFK - LF')
plt.fill_between(np.ravel(x), np.ravel(y0-3*np.sqrt(var0)),
                  np.ravel(y0+3*np.sqrt(var0)),
                  color='C1',alpha=0.2, label ='Confidence Interval 99% LF')
plt.scatter(np.squeeze(Xt_e), np.squeeze(yt_e), label = "HF DoE")
plt.scatter(np.squeeze(Xt_c), np.squeeze(yt_c), label = "LF DoE")
plt.ylim(-10, 17); plt.xlim(-0.1, 1.1)
plt.xlabel('x'); plt.ylabel('y')
plt.title('Multi-fidelity Kriging (MFK)')
plt.legend()




############################################
########### Own method
############################################
# 3 steps: 
# 1) first show better results while using proposed with equal amount of data
# 2) show this is achievable with less data
# 3) show the MFK solution does not improve while using the MF

def LF_function_adapt(x):
    return 0.5 * ((6*x-2)**2)*np.sin((12 * x-4.9)) + (x-0.5) * 10. - 5

def MF_function(x):
    return LF_function_adapt(x) + (HF_function(x) - LF_function_adapt(x))/2
    # return LF_function(x) - HF_function(x)

yt_e = HF_function(Xt_e)
yt_m = MF_function(Xt_m)
yt_c = LF_function_adapt(Xt_c)

X_mf = [Xt_c, Xt_m, Xt_e]
Z_mf = [yt_c, yt_m, yt_e]

X_truth = Xt_c
Z_truth = HF_function(X_truth)

" Creating setup object"
if True:
    # fake setup as a function object
    setup = lambda x: 0

    # standard setup settings
    setup.kernel_name = "kriging"
    setup.noise_regression = True # Always true, but not always add noise!
    setup.mf = True

    setup.solver_str = "Forrester2008"
    setup.d = 1
    ss = get_solver(name = setup.solver_str).get_preferred_search_space(setup.d)
    ss[1], ss[2] = np.array(ss[1]), np.array(ss[2])
    setup.search_space = ss

    # set the variable settings
    setup.conv_mod = 0
    setup.conv_type = "Stable up"
    setup.solver_noise = 0.0

mf_model = MultiFidelityEGO(setup, proposed = True, optim_var= False, initial_nr_samples = 1, max_cost = np.inf, MFK_kwargs = MFK_kwargs)
mf_model.X_mf, mf_model.X_truth, mf_model.Z_mf, mf_model.Z_truth, mf_model.X_unique = X_mf, X_truth, Z_mf, Z_truth, X_mf[0]

mf_model.number_of_levels = 3
mf_model.set_L_costs([1,10,1000])   

mf_model.use_single_fidelities = True

mf_model.train()
mf_model.weighted_prediction()
mf_model.create_update_K_pred()

y_2p, mse_2p = mf_model.K_mf[2].predict(x)
y_1p, mse_1p = mf_model.K_mf[1].predict(x)
y_0p, mse_0p = mf_model.K_mf[0].predict(x)
var_2p, var_1p, var_0p = np.sqrt(mse_2p), np.sqrt(mse_1p), np.sqrt(mse_0p)


mfk_model = MultiFidelityEGO(setup, proposed = False, optim_var= False, initial_nr_samples = 1, max_cost = np.inf, MFK_kwargs = MFK_kwargs)
mfk_model.X_mf, mfk_model.X_truth, mfk_model.Z_mf, mfk_model.Z_truth, mfk_model.X_unique = [X_mf[0],X_mf[2]], X_truth, [Z_mf[0], Z_mf[2]], Z_truth, X_mf[0]
mfk_model.train()

y_2, mse_2 = mfk_model.K_mf[1].predict(x)
# y_1, mse_1 = mfk_model.K_mf[1].predict(x)
y_0, mse_0 = mfk_model.K_mf[0].predict(x)
var_2, var_0 = np.sqrt(mse_2), np.sqrt(mse_0)


y_min_lim = -15
RMSE = RMSE_norm_MF(mf_model, no_samples=True)
mf_model.print_stats(RMSE)
print_pearson_correlations(mf_model)


" plot just truth again for clear shift "
plt.figure(figsize = (15, 5))
plt.subplot(1,2,1)
plt.plot(x, HF_function(x), '--C0', label='High Fidelity (HF)')
plt.plot(x, LF_function(x), '--C1', label='Low Fidelity (LF)')
plt.scatter(Xt_e, yt_e, color='C0', label='HF DoE')
plt.scatter(Xt_c, LF_function(Xt_c), color='C1', label='LF DoE')
plt.ylim(y_min_lim, 17); plt.xlim(-0.1, 1.1)
plt.xlabel('x'); plt.ylabel('y')
plt.legend()

plt.figure(figsize = (15, 5))
plt.subplot(1,2,1)
plt.plot(x, HF_function(x), '--C0', label='High Fidelity (HF)')
plt.plot(x, LF_function_adapt(x), '--C1', label='Low Fidelity (LF)')
plt.scatter(Xt_e, yt_e, color='C0', label='HF DoE')
plt.scatter(Xt_c, yt_c, color='C1', label='LF DoE')
plt.ylim(y_min_lim, 17); plt.xlim(-0.1, 1.1)
plt.xlabel('x'); plt.ylabel('y')
plt.legend()

" Plot MFK solution on top, only high and low fidelity "
plt.figure(figsize = (15, 5))
ax = plt.subplot(1,2,1)
plt.plot(x, HF_function(x), '--C0', label='High Fidelity (HF)')
plt.plot(x, LF_function_adapt(x), '--C1', label='Low Fidelity (LF)')

plt.plot(x, y_2, 'C0', label='MFK - High Fidelity')
plt.plot(x, y_0, 'C1', label='MFK - Low Fidelity')

plt.fill_between(np.ravel(x), np.ravel(y_2-3*np.sqrt(var_2)),
                  np.ravel(y_2+3*np.sqrt(var_2)),
                  color='C0',alpha=0.2, label ='Confidence Interval 99% HF')
plt.fill_between(np.ravel(x), np.ravel(y_0-3*np.sqrt(var_0)),
                  np.ravel(y_0+3*np.sqrt(var_0)),
                  color='C1',alpha=0.2, label ='Confidence Interval 99% LF')

plt.scatter(np.squeeze(Xt_e), np.squeeze(yt_e), c = 'C0', label = "HF DoE")
plt.scatter(np.squeeze(Xt_c), np.squeeze(yt_c), c = 'C1', label = "LF DoE")

plt.ylim(y_min_lim, 17); plt.xlim(-0.1, 1.1)
plt.xlabel('x'); plt.ylabel('y')
plt.title('Multi-fidelity Kriging (MFK)')
handels, labels = ax.get_legend_handles_labels()
inds = [2,3]
plt.legend([handels[i] for i in inds], [labels[i] for i in inds])


" Plot MFK solution that uses medium fidelity "
mfk_model = MultiFidelityEGO(setup, proposed = False, optim_var= False, initial_nr_samples = 1, max_cost = np.inf, MFK_kwargs = MFK_kwargs)
mfk_model.X_mf, mfk_model.X_truth, mfk_model.Z_mf, mfk_model.Z_truth, mfk_model.X_unique = X_mf, X_truth, Z_mf, Z_truth, X_mf[0]
mfk_model.train()

y_2, mse_2 = mfk_model.K_mf[2].predict(x)
y_1, mse_1 = mfk_model.K_mf[1].predict(x)
y_0, mse_0 = mfk_model.K_mf[0].predict(x)
var_2, var_1, var_0 = np.sqrt(mse_2), np.sqrt(mse_1), np.sqrt(mse_0)


plt.figure(figsize = (15, 5))
ax = plt.subplot(1,2,1)
plt.plot(x, HF_function(x), '--C0', label='High Fidelity (HF)')
plt.plot(x, MF_function(x), '--C2', label='Medium Fidelity (MF)')
plt.plot(x, LF_function_adapt(x), '--C1', label='Low Fidelity (LF)')

plt.plot(x, y_2, 'C0', label='MFK - High Fidelity')
plt.plot(x, y_1, 'C2', label='MFK - Medium Fidelity')
plt.plot(x, y_0, 'C1', label='MFK - Low Fidelity')

plt.fill_between(np.ravel(x), np.ravel(y_2-3*np.sqrt(var_2)),
                  np.ravel(y_2+3*np.sqrt(var_2)),
                  color='C0',alpha=0.2, label ='Confidence Interval 99% HF')
plt.fill_between(np.ravel(x), np.ravel(y_1-3*np.sqrt(var_1)),
                  np.ravel(y_1+3*np.sqrt(var_1)),
                  color='C2',alpha=0.2, label ='Confidence Interval 99% MF')
plt.fill_between(np.ravel(x), np.ravel(y_0-3*np.sqrt(var_0)),
                  np.ravel(y_0+3*np.sqrt(var_0)),
                  color='C1',alpha=0.2, label ='Confidence Interval 99% LF')

plt.scatter(np.squeeze(Xt_e), np.squeeze(yt_e), c = 'C0', label = "HF DoE")
plt.scatter(np.squeeze(Xt_m), np.squeeze(yt_m), c = 'C2', label = "MF DoE")
plt.scatter(np.squeeze(Xt_c), np.squeeze(yt_c), c = 'C1', label = "LF DoE")

plt.ylim(y_min_lim, 17); plt.xlim(-0.1, 1.1)
plt.xlabel('x'); plt.ylabel('y')
plt.title('Multi-fidelity Kriging (MFK)')
handels, labels = ax.get_legend_handles_labels()
inds = [3,4,5,9]
plt.legend([handels[i] for i in inds], [labels[i] for i in inds])



" Plot same but with proposed "
plt.figure(figsize = (15, 5))
ax = plt.subplot(1,2,1)
plt.plot(x, HF_function(x), '--C0', label='High Fidelity (HF)')
plt.plot(x, MF_function(x), '--C2', label='Medium Fidelity (MF)')
plt.plot(x, LF_function_adapt(x), '--C1', label='Low Fidelity (LF)')

plt.plot(x, y_2, 'C0', label='MFK - High Fidelity')
plt.plot(x, y_1, 'C2', label='MFK - Medium Fidelity')
plt.plot(x, y_0, 'C1', label='MFK - Low Fidelity')

plt.fill_between(np.ravel(x), np.ravel(y_2-3*np.sqrt(var_2)),
                  np.ravel(y_2+3*np.sqrt(var_2)),
                  color='C0',alpha=0.2, label ='Confidence Interval 99% HF')
plt.fill_between(np.ravel(x), np.ravel(y_1-3*np.sqrt(var_1)),
                  np.ravel(y_1+3*np.sqrt(var_1)),
                  color='C2',alpha=0.2, label ='Confidence Interval 99% MF')
plt.fill_between(np.ravel(x), np.ravel(y_0-3*np.sqrt(var_0)),
                  np.ravel(y_0+3*np.sqrt(var_0)),
                  color='C1',alpha=0.2, label ='Confidence Interval 99% LF')

plt.scatter(np.squeeze(Xt_e), np.squeeze(yt_e), c = 'C0', label = "HF DoE")
plt.scatter(np.squeeze(Xt_m), np.squeeze(yt_m), c = 'C2', label = "MF DoE")
plt.scatter(np.squeeze(Xt_c), np.squeeze(yt_c), c = 'C1', label = "LF DoE")

plt.ylim(y_min_lim, 17); plt.xlim(-0.1, 1.1)
plt.xlabel('x'); plt.ylabel('y')
plt.title('Multi-fidelity Kriging (MFK)')
handels, labels = ax.get_legend_handles_labels()
inds = [3,4,5,9]
plt.legend([handels[i] for i in inds], [labels[i] for i in inds])


ax = plt.subplot(1,2,2)
plt.plot(x, HF_function(x), '--C0', label='High Fidelity (HF)')
plt.plot(x, MF_function(x), '--C2', label='Medium Fidelity (MF)')
plt.plot(x, LF_function_adapt(x), '--C1', label='Low Fidelity (LF)')

plt.plot(x, y_2p, 'C0', label='Proposed - High Fidelity')
plt.plot(x, y_1p, 'C2', label='Kriging - Medium Fidelity')
plt.plot(x, y_0p, 'C1', label='Kriging - Low Fidelity')

plt.fill_between(np.ravel(x), np.ravel(y_2p-3*np.sqrt(var_2p)),
                  np.ravel(y_2p+3*np.sqrt(var_2p)),
                  color='C0',alpha=0.2, label ='Confidence Interval 99% HF')
plt.fill_between(np.ravel(x), np.ravel(y_1p-3*np.sqrt(var_1p)),
                  np.ravel(y_1p+3*np.sqrt(var_1p)),
                  color='C2',alpha=0.2, label ='Confidence Interval 99% MF')
plt.fill_between(np.ravel(x), np.ravel(y_0p-3*np.sqrt(var_0p)),
                  np.ravel(y_0p+3*np.sqrt(var_0p)),
                  color='C1',alpha=0.2, label ='Confidence Interval 99% LF')

plt.scatter(np.squeeze(Xt_e), np.squeeze(yt_e), c = 'C0', label = "HF DoE")
plt.scatter(np.squeeze(Xt_m), np.squeeze(yt_m), c = 'C2', label = "MF DoE")
plt.scatter(np.squeeze(Xt_c), np.squeeze(yt_c), c = 'C1', label = "LF DoE")

plt.ylim(y_min_lim, 17); plt.xlim(-0.1, 1.1)
plt.xlabel('x'); plt.ylabel('y')
plt.title('Kriging Extrapolation (Proposed)')
handels, labels = ax.get_legend_handles_labels()
inds = [3,4,5,9]
plt.legend([handels[i] for i in inds], [labels[i] for i in inds])

" en nu met proposed met 1 sample!"
mf_model = MultiFidelityEGO(setup, proposed = True, optim_var= False, initial_nr_samples = 1, max_cost = np.inf, MFK_kwargs = MFK_kwargs)
mf_model.X_mf, mf_model.X_truth, mf_model.Z_mf, mf_model.Z_truth, mf_model.X_unique = X_mf, X_truth, Z_mf, Z_truth, X_mf[0]
mf_model.X_mf[-1], mf_model.Z_mf[-1] = X_mf[-1][1], Z_mf[-1][1]

mf_model.number_of_levels = 3
mf_model.set_L_costs([1,10,1000])   

mf_model.use_single_fidelities = True

mf_model.train()
mf_model.weighted_prediction()
mf_model.create_update_K_pred()

y_2p, mse_2p = mf_model.K_mf[2].predict(x)
y_1p, mse_1p = mf_model.K_mf[1].predict(x)
y_0p, mse_0p = mf_model.K_mf[0].predict(x)
var_2p, var_1p, var_0p = np.sqrt(mse_2p), np.sqrt(mse_1p), np.sqrt(mse_0p)

plt.figure(figsize = (15, 5))
ax = plt.subplot(1,2,1)
plt.plot(x, HF_function(x), '--C0', label='High Fidelity (HF)')
plt.plot(x, MF_function(x), '--C2', label='Medium Fidelity (MF)')
plt.plot(x, LF_function_adapt(x), '--C1', label='Low Fidelity (LF)')

plt.plot(x, y_2, 'C0', label='MFK - High Fidelity')
plt.plot(x, y_1, 'C2', label='MFK - Medium Fidelity')
plt.plot(x, y_0, 'C1', label='MFK - Low Fidelity')

plt.fill_between(np.ravel(x), np.ravel(y_2-3*np.sqrt(var_2)),
                  np.ravel(y_2+3*np.sqrt(var_2)),
                  color='C0',alpha=0.2, label ='Confidence Interval 99% HF')
plt.fill_between(np.ravel(x), np.ravel(y_1-3*np.sqrt(var_1)),
                  np.ravel(y_1+3*np.sqrt(var_1)),
                  color='C2',alpha=0.2, label ='Confidence Interval 99% MF')
plt.fill_between(np.ravel(x), np.ravel(y_0-3*np.sqrt(var_0)),
                  np.ravel(y_0+3*np.sqrt(var_0)),
                  color='C1',alpha=0.2, label ='Confidence Interval 99% LF')

plt.scatter(np.squeeze(Xt_e), np.squeeze(yt_e), c = 'C0', label = "HF DoE")
plt.scatter(np.squeeze(Xt_m), np.squeeze(yt_m), c = 'C2', label = "MF DoE")
plt.scatter(np.squeeze(Xt_c), np.squeeze(yt_c), c = 'C1', label = "LF DoE")

plt.ylim(y_min_lim, 17); plt.xlim(-0.1, 1.1)
plt.xlabel('x'); plt.ylabel('y')
plt.title('Multi-fidelity Kriging (MFK)')
handels, labels = ax.get_legend_handles_labels()
inds = [3,4,5,9]
plt.legend([handels[i] for i in inds], [labels[i] for i in inds])


ax = plt.subplot(1,2,2)
plt.plot(x, HF_function(x), '--C0', label='High Fidelity (HF)')
plt.plot(x, MF_function(x), '--C2', label='Medium Fidelity (MF)')
plt.plot(x, LF_function_adapt(x), '--C1', label='Low Fidelity (LF)')

plt.plot(x, y_2p, 'C0', label='Proposed - High Fidelity')
plt.plot(x, y_1p, 'C2', label='Kriging - Medium Fidelity')
plt.plot(x, y_0p, 'C1', label='Kriging - Low Fidelity')

plt.fill_between(np.ravel(x), np.ravel(y_2p-3*np.sqrt(var_2p)),
                  np.ravel(y_2p+3*np.sqrt(var_2p)),
                  color='C0',alpha=0.2, label ='Confidence Interval 99% HF')
plt.fill_between(np.ravel(x), np.ravel(y_1p-3*np.sqrt(var_1p)),
                  np.ravel(y_1p+3*np.sqrt(var_1p)),
                  color='C2',alpha=0.2, label ='Confidence Interval 99% MF')
plt.fill_between(np.ravel(x), np.ravel(y_0p-3*np.sqrt(var_0p)),
                  np.ravel(y_0p+3*np.sqrt(var_0p)),
                  color='C1',alpha=0.2, label ='Confidence Interval 99% LF')

plt.scatter(np.squeeze(Xt_e[1]), np.squeeze(yt_e[1]), c = 'C0', label = "HF DoE")
plt.scatter(np.squeeze(Xt_m), np.squeeze(yt_m), c = 'C2', label = "MF DoE")
plt.scatter(np.squeeze(Xt_c), np.squeeze(yt_c), c = 'C1', label = "LF DoE")

plt.ylim(y_min_lim, 17); plt.xlim(-0.1, 1.1)
plt.xlabel('x'); plt.ylabel('y')
plt.title('Kriging Extrapolation (Proposed)')
handels, labels = ax.get_legend_handles_labels()
inds = [3,4,5,9]
plt.legend([handels[i] for i in inds], [labels[i] for i in inds])

plt.show()