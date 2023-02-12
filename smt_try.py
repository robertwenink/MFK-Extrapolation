#%%
# # https://colab.research.google.com/github/SMTorg/smt/blob/master/tutorial/SMT_MFK_Noise.ipynb

import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [8, 5]

from smt.applications import MFK

# %matplotlib inline

def LF_function(x): # low-fidelity function
    return 0.5*((x*6-2)**2)*np.sin((x*6-2)*2)+(x-0.5)*10. - 5

def HF_function(x): # high-fidelity function
    return ((x*6-2)**2)*np.sin((x*6-2)*2)

x = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)

plt.figure()
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

# Evaluate the HF and LF functions
yt_e = HF_function(Xt_e)
yt_c = LF_function(Xt_c)

plt.figure()
plt.plot(x, HF_function(x), '--C0', label='High Fidelity (HF)')
plt.plot(x, LF_function(x), '--C1', label='Low Fidelity (LF)')
plt.scatter(Xt_e, yt_e, color='C0', label='HF doe')
plt.scatter(Xt_c, yt_c, color='C1', label='LF doe')
plt.ylim(-10, 17); plt.xlim(-0.1, 1.1)
plt.xlabel('x'); plt.ylabel('y')
plt.legend()

# %% Build the MFK object
yt_joint = np.concatenate((yt_c, yt_e))
noise0 = [np.array([2e-01, 7e-02, 2e-03, 6e-01,
                    5e-02, 5e-02, 4e-02, 3e-02,
                    2e-01, 1e-02, 5e-06, 1e-01])/2,
          np.array([1.5, 0.4, 0.01, 0.1])/5]
theta0 = np.array([[0.5],[0.1]])
np.random.seed(7)
sm = MFK(theta0=theta0, theta_bounds = [1e-1, 20],
         noise0=noise0, use_het_noise = True, eval_noise = False,
         propagate_uncertainty=False, n_start=1)

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

# %%
# creating a single-fidelity GP model to compare with
theta0 = sm.options["theta0"]
noise0 = sm.options["noise0"]

# %% Kriging on monofidelity HF doe
sm_monoHF =  MFK(theta0=theta0[1].tolist(), theta_bounds = [1e-1, 20],
                 noise0=[noise0[1]], use_het_noise=True, n_start=1)

# %% Kriging on monofidelity LF Doe
sm_monoLF =  MFK(theta0=theta0[0].tolist(), theta_bounds = [1e-1, 20],
                 noise0=[noise0[0]], use_het_noise=True, n_start=1)

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

plt.figure(figsize = (15, 5))
plt.subplot(1,2,1)
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
plt.title('Multi-fidelity GPs (MFGP)')

plt.subplot(1,2,2)
plt.plot(x, HF_function(x), '--C0', label='High Fidelity (HF)')
plt.plot(x, LF_function(x), '--C1', label='Low Fidelity (LF)')
plt.plot(x, y_monoHF, 'C2', label='GP - HF')
plt.fill_between(np.ravel(x), np.ravel(y_monoHF-3*np.sqrt(var_monoHF)),
                  np.ravel(y_monoHF+3*np.sqrt(var_monoHF)),
                  color='C2',alpha=0.2, label ='Confidence Interval 99% HF')
plt.plot(x, y_monoLF, 'C3', label='GP - LF')
plt.fill_between(np.ravel(x), np.ravel(y_monoLF-3*np.sqrt(var_monoLF)),
                  np.ravel(y_monoLF+3*np.sqrt(var_monoLF)),
                  color='C3',alpha=0.2, label ='Confidence Interval 99% LF')
plt.errorbar(Xt_e, yt_e, yerr=3*np.sqrt(noise0[-1]),
             fmt="o", color="C0")
plt.errorbar(Xt_c, yt_c, yerr=3*np.sqrt(noise0[0]),
             fmt="o", color="C1")
plt.ylim(-10, 17); plt.xlim(-0.1, 1.1)
plt.xlabel('x'); plt.ylabel('y')
plt.title('Single-fidelity GPs')
plt.legend()


# adding noisy repetitions
Xt_c_reps = Xt_c.copy()
yt_c_reps = yt_c.copy()
Xt_e_reps = Xt_e.copy()
yt_e_reps = yt_e.copy()

for i in range(4):
    Xt_c_reps = np.concatenate((Xt_c_reps, Xt_c))
    Xt_e_reps = np.concatenate((Xt_e_reps, Xt_e))
    np.random.seed(i)
    yt_c_reps = np.concatenate((yt_c_reps,
                                yt_c + 0.1*np.std(yt_c)*np.random.normal(size=yt_c.shape)))
    yt_e_reps = np.concatenate((yt_e_reps,
                                yt_e + 0.1*np.std(yt_e)*np.random.normal(size=yt_e.shape)))

plt.figure()
plt.plot(x, HF_function(x), '--C0', label='High Fidelity (HF)')
plt.plot(x, LF_function(x), '--C1', label='Low Fidelity (LF)')
plt.scatter(Xt_e_reps, yt_e_reps, color='C0', label='HF doe')
plt.scatter(Xt_c_reps, yt_c_reps, color='C1', label='LF doe')
plt.ylim(-10, 17); plt.xlim(-0.1, 1.1)
plt.xlabel('x'); plt.ylabel('y')
plt.legend()

# %% Build the MFK object
yt_joint = np.concatenate((yt_c_reps, yt_e_reps))
theta0 = np.array([[0.5],[0.1]])
np.random.seed(7)
sm = MFK(theta0=theta0,
         noise0=noise0, eval_noise=True,
         use_het_noise = True,
         propagate_uncertainty = False, n_start=1)

# low-fidelity dataset names being integers from 0 to level-1
sm.set_training_values(Xt_c_reps, yt_c_reps, name=0)
# high-fidelity dataset without name
sm.set_training_values(Xt_e_reps, yt_e_reps)
# train the model
sm.train()

# test training
# HF
y = sm.predict_values(x)
var = sm.predict_variances(x)
# LF
y0 = sm._predict_intermediate_values(x, 1)
var0, _ =  sm.predict_variances_all_levels(x)
var0 = var0[:,0].reshape(-1,1)


plt.figure(figsize=(7,5))
plt.plot(x, HF_function(x), '--C0', label='High Fidelity (HF)')
plt.plot(x, LF_function(x), '--C1', label='Low Fidelity (LF)')
plt.plot(x, y, 'C0', label='MFGP - HF')
plt.fill_between(np.ravel(x), np.ravel(y-3*np.sqrt(var)),
                 np.ravel(y+3*np.sqrt(var)),
                 color='C0',alpha=0.2 ,label ='Confidence Interval 99% HF')
plt.plot(x, y0, 'C1', label='MFGP - LF')
plt.fill_between(np.ravel(x), np.ravel(y0-3*np.sqrt(var0)),
                  np.ravel(y0+3*np.sqrt(var0)),
                  color='C1',alpha=0.2,label ='Confidence Interval 99% LF')
plt.scatter(Xt_e_reps, yt_e_reps, color='C0', label='HF doe with repetitions')
plt.scatter(Xt_c_reps, yt_c_reps, color='C1', label='LF doe with repetitions')
plt.xlabel('x'); plt.ylabel('y')
plt.ylim(-10, 17); plt.xlim(-0.1, 1.1)
plt.legend()



plt.show()
