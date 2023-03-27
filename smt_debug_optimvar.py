#%%
# # https://colab.research.google.com/github/SMTorg/smt/blob/master/tutorial/SMT_MFK_Noise.ipynb
# pyright: reportGeneralTypeIssues=false
import numpy as np
from copy import deepcopy
np.random.seed(7)

from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 5]

from smt.applications import MFK

# %matplotlib inline
def HF_function(x): # high-fidelity function
    return ((x*6-2)**2)*np.sin((x*6-2)*2)

def LF_function(x):
    return 0.5 * HF_function(x) + (x - 0.5) * 10. - 5


x = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)

#Expensive DOE 
Xt_e = np.array([0.0, 0.3, 0.75, 1.]).reshape(-1, 1)
Xt_e_extra = np.array([0.73, 0.74, 0.755, 0.76, 0.61]).reshape(-1, 1)
Xt_e = np.concatenate((Xt_e, Xt_e_extra),axis=0)

noise0_org = [np.array([2e-01, 7e-02, 2e-03, 6e-01,
                    5e-02, 5e-02, 4e-02, 3e-02,
                    2e-01, 1e-02, 5e-06, 1e-01, 3e-06, 3e-06, 3e-06, 3e-06, 3e-06])/2,
          np.array([1,1,0.1,1,1,1,1,1,1])/5]

#Cheap DOE
Xt_c = np.linspace(1/9, 8/9, 8, endpoint=True).reshape(-1, 1)
# The two DOEs must be nest: Xt_e must be included within Xt_c
Xt_c = np.concatenate((Xt_c,Xt_e),axis=0)
# Xt_c = np.concatenate((Xt_c,np.array([0.301]).reshape(-1, 1)),axis=0)

# Evaluate the HF and LF functions
yt_e = HF_function(Xt_e)
yt_c = LF_function(Xt_c)
print(np.random.random(size=yt_c.shape[0]))
yt_c += (np.random.random(size=yt_c.shape)-0.5)/2
yt_e += (np.random.random(size=yt_e.shape)-0.5)/4
yt_e[4:] -= -1
# yt_c[-1] *= 1.03

# test training
ntest = 101
nlvl = 2
x = np.linspace(0, 1, ntest, endpoint=True).reshape(-1, 1)
theta0 = np.array([[0.5],[0.1]])



# %% Build the MFK object
yt_joint = np.concatenate((yt_c, yt_e))

sm = MFK(use_het_noise = False, eval_noise = True,
         propagate_uncertainty=False, n_start=40, optim_var = False)

# low-fidelity dataset names being integers from 0 to level-1
sm.set_training_values(Xt_c, yt_c, name=0)
sm.set_training_values(Xt_e, yt_e)
sm.train()


plt.figure(figsize = (15, 5))

y = sm.predict_values(x)
var = sm.predict_variances(x)
y0 = sm._predict_intermediate_values(x, 1)
var0, _ =  sm.predict_variances_all_levels(x)
var0 = var0[:,0].reshape(-1,1)

plt.subplot(1,4,1)
plt.plot(x, HF_function(x), '--C0', label='High Fidelity (HF)')
plt.plot(x, LF_function(x), '--C1', label='Low Fidelity (LF)')
plt.plot(x, y, 'C0', label='MFGP - HF')
plt.fill_between(np.ravel(x), np.ravel(y-3*np.sqrt(var)),
                  np.ravel(y+3*np.sqrt(var)),
                  color='C0',alpha=0.2, label ='Confidence Interval 99% HF')
plt.plot(x, y0, 'C1', label='MFGP - LF')
plt.scatter(Xt_e, yt_e)
plt.scatter(Xt_c, yt_c)
plt.fill_between(np.ravel(x), np.ravel(y0-3*np.sqrt(var0)),
                  np.ravel(y0+3*np.sqrt(var0)),
                  color='C1',alpha=0.2, label ='Confidence Interval 99% LF')
plt.ylim(-10, 17); plt.xlim(-0.1, 1.1)
plt.xlabel('x'); plt.ylabel('y')
plt.legend()
plt.title('Only noise evaluation')


noise0 = deepcopy(noise0_org)
noise0[0] = sm.predict_variances_all_levels(Xt_c)[0][:, 0]
# noise0[0] = [0]

# NOTE beste settings: False, False, True
sm.options['eval_noise'] = False
sm.options['optim_var'] = False # NOTE Does (programatically) nothing when eval_noise = False
sm.options['use_het_noise'] = True
sm.options['noise0'] = noise0
sm.train()


y = sm.predict_values(x)
var = sm.predict_variances(x)
y0 = sm._predict_intermediate_values(x, 1)
var0, _ =  sm.predict_variances_all_levels(x)
var0 = abs(var0[:,0].reshape(-1,1))

plt.subplot(1,4,2)
plt.plot(x, HF_function(x), '--C0', label='High Fidelity (HF)')
plt.plot(x, LF_function(x), '--C1', label='Low Fidelity (LF)')
plt.plot(x, y, 'C0', label='MFGP - HF')
plt.fill_between(np.ravel(x), np.ravel(y-3*np.sqrt(var)),
                  np.ravel(y+3*np.sqrt(var)),
                  color='C0',alpha=0.2, label ='Confidence Interval 99% HF')
plt.plot(x, y0, 'C1', label='MFGP - LF')
plt.fill_between(np.ravel(x), np.ravel(y0-3*np.sqrt(var0)),
                  np.ravel(y0+3*np.sqrt(var0)),
                  color='C1',alpha=0.2, label ='Confidence Interval 99% LF')
plt.errorbar(Xt_e, yt_e, yerr=3*np.sqrt(noise0[-1]),
             fmt="o", color="C0", label='HF doe')
plt.errorbar(Xt_c, yt_c, yerr=3*np.sqrt(noise0[0]),
             fmt="o", color="C1", label='LF doe')
plt.ylim(-10, 17); plt.xlim(-0.1, 1.1)
plt.xlabel('x'); plt.ylabel('y')
plt.legend()
plt.title('Evaluated noise, then hetero-scedastic noise.')




sm.set_training_values(Xt_c, sm._predict_intermediate_values(Xt_c, 1), name=0)
sm.set_training_values(Xt_e, sm.predict_values(Xt_e))

sm.options['eval_noise'] = False
sm.options['optim_var'] = False # NOTE Does (programatically) nothing when eval_noise = False
sm.options['use_het_noise'] = False
sm.options['noise0'] = [0]
sm.train()


y = sm.predict_values(x)
var = sm.predict_variances(x)
y0 = sm._predict_intermediate_values(x, 1)
var0, _ =  sm.predict_variances_all_levels(x)
var0 = abs(var0[:,0].reshape(-1,1))

# dus: als de samples in een het_noise environment 0 noise hebben, worden de extrapolations eromheen basically volledig genegeerd
# dit is onwenselijk, er is ook noise bij de samples, alleen het is in principe minder!!
# We hebben hier 3 estimates voor:
# 1) een virtuele weighted prediction op de sample locatie (geeft wss een hoger antwoord)
# 2) het noise level van de medium fidelity (dit is de absolute max)
# 3) noise level van highest fidelity (samples + extrapolations!). 
#   Het kan zijn dat dit iets hoger is dan de werkelijkheid, 
#   maar tegelijkertijd als de extrapolations consistent zijn kan de noise heel laag zijn juist.

plt.subplot(1,4,3)
plt.plot(x, HF_function(x), '--C0', label='High Fidelity (HF)')
plt.plot(x, LF_function(x), '--C1', label='Low Fidelity (LF)')
plt.plot(x, y, 'C0', label='MFGP - HF')
plt.fill_between(np.ravel(x), np.ravel(y-3*np.sqrt(var)),
                  np.ravel(y+3*np.sqrt(var)),
                  color='C0',alpha=0.2, label ='Confidence Interval 99% HF')
plt.plot(x, y0, 'C1', label='MFGP - LF')
plt.fill_between(np.ravel(x), np.ravel(y0-3*np.sqrt(var0)),
                  np.ravel(y0+3*np.sqrt(var0)),
                  color='C1',alpha=0.2, label ='Confidence Interval 99% LF')
plt.errorbar(Xt_e, yt_e, yerr=3*np.sqrt(noise0[-1]),
             fmt="o", color="C0", label='HF doe')
plt.errorbar(Xt_c, yt_c, yerr=3*np.sqrt(noise0[0]),
             fmt="o", color="C1", label='LF doe')
plt.ylim(-10, 17); plt.xlim(-0.1, 1.1)
plt.xlabel('x'); plt.ylabel('y')
plt.legend()
plt.title('Eval_noise + hetero-scedastic + reinterpolate')

#######################################################################
# # %% Kriging on monofidelity HF doe
# sm_org = MFK(noise0 = noise0_org, theta0 = theta0, use_het_noise=True, eval_noise = True, optim_var = False, n_start=10)
# sm_org.name = "MFK ORG"

# # train the model
# sm_org.set_training_values(Xt_c, yt_c,name = 0)
# sm_org.set_training_values(Xt_e, yt_e)
# sm_org.train()

# # test training
# y_monoLF = sm_org._predict_intermediate_values(x, 1)
# y_monoHF = sm_org.predict_values(x)
# MSE = sm_org.predict_variances_all_levels(x)[0]
# var_monoLF = abs(MSE[:,0].reshape(-1,1))
# var_monoHF = abs(MSE[:,1].reshape(-1,1))



# plt.subplot(1,4,3)
# plt.plot(x, HF_function(x), '--C0', label='High Fidelity (HF)')
# plt.plot(x, LF_function(x), '--C1', label='Low Fidelity (LF)')
# plt.plot(x, y_monoHF, 'C2', label='MFGP - HF')
# plt.fill_between(np.ravel(x), np.ravel(y_monoHF-3*np.sqrt(var_monoHF)),
#                   np.ravel(y_monoHF+3*np.sqrt(var_monoHF)),
#                   color='C2',alpha=0.2, label ='Confidence Interval 99% HF')
# plt.plot(x, y_monoLF, 'C3', label='MFGP - LF')
# plt.fill_between(np.ravel(x), np.ravel(y_monoLF-3*np.sqrt(var_monoLF)),
#                   np.ravel(y_monoLF+3*np.sqrt(var_monoLF)),
#                   color='C3',alpha=0.2, label ='Confidence Interval 99% LF')
# plt.errorbar(Xt_e, yt_e, yerr=3*np.sqrt(noise0_org[-1]),
#              fmt="o", color="C0")
# plt.errorbar(Xt_c, yt_c, yerr=3*np.sqrt(noise0_org[0]),
#              fmt="o", color="C1")
# plt.ylim(-10, 17); plt.xlim(-0.1, 1.1)
# plt.xlabel('x'); plt.ylabel('y')
# plt.title('Original Multi-fidelity GPs')
# plt.legend()



# reference model
sm_ref = MFK(theta0 = theta0, eval_noise = True, optim_var = True, n_start=10)

# train the model
sm_ref.set_training_values(Xt_c, yt_c,name = 0)
sm_ref.set_training_values(Xt_e, yt_e)
sm_ref.train()

# test training
y_monoLF = sm_ref._predict_intermediate_values(x, 1)
y_monoHF = sm_ref.predict_values(x)
MSE = sm_ref.predict_variances_all_levels(x)[0]
var_monoLF = abs(MSE[:,0].reshape(-1,1))
var_monoHF = abs(MSE[:,1].reshape(-1,1))

plt.subplot(1,4,4)
plt.plot(x, HF_function(x), '--C0', label='High Fidelity (HF)')
plt.plot(x, LF_function(x), '--C1', label='Low Fidelity (LF)')
plt.plot(x, y_monoHF, 'C2', label='MFGP - HF')
plt.fill_between(np.ravel(x), np.ravel(y_monoHF-3*np.sqrt(var_monoHF)),
                  np.ravel(y_monoHF+3*np.sqrt(var_monoHF)),
                  color='C2',alpha=0.2, label ='Confidence Interval 99% HF')
plt.plot(x, y_monoLF, 'C3', label='MFGP - LF')
plt.fill_between(np.ravel(x), np.ravel(y_monoLF-3*np.sqrt(var_monoLF)),
                  np.ravel(y_monoLF+3*np.sqrt(var_monoLF)),
                  color='C3',alpha=0.2, label ='Confidence Interval 99% LF')
plt.scatter(Xt_e, yt_e)
plt.scatter(Xt_c, yt_c)
plt.ylim(-10, 17); plt.xlim(-0.1, 1.1)
plt.xlabel('x'); plt.ylabel('y')
plt.title('Reference MFK (Noise + reinterpolate)')
plt.legend()




plt.tight_layout()
plt.show()
