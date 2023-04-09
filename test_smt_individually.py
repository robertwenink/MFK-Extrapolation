import numpy as np
from smt.applications import MFK
from core.sampling.solvers.solver import get_solver
from utils.error_utils import RMSE_norm, RMSE_norm_MF
from utils.selection_utils import isin_indices


X_mf = [np.array([[ 0.809,  1.82 ],
       [-0.99 ,  1.209],
       [ 1.916,  1.229],
       [-1.518,  0.433],
       [ 0.537, -0.15 ],
       [ 1.56 ,  0.088],
       [ 1.346, -0.665],
       [ 0.938, -1.453],
       [-0.282,  2.029],
       [-0.748, -1.405],
       [ 0.402, -1.912],
       [-1.063, -0.433],
       [-1.738,  1.5  ],
       [ 1.073,  0.903],
       [-1.405, -1.65 ],
       [-0.133,  0.341],
       [ 1.746,  0.707],
       [-0.466, -0.256],
       [-1.938, -0.834],
       [ 0.183, -1.147]]), np.array([[ 0.809,  1.82 ],
       [-0.99 ,  1.209],
       [ 1.916,  1.229],
       [-1.518,  0.433],
       [ 0.537, -0.15 ],
       [ 1.56 ,  0.088],
       [ 1.346, -0.665],
       [ 0.938, -1.453],
       [-0.282,  2.029],
       [-0.748, -1.405],
       [ 0.402, -1.912],
       [-1.063, -0.433],
       [-1.738,  1.5  ],
       [ 1.073,  0.903],
       [-1.405, -1.65 ],
       [-0.133,  0.341],
       [ 1.746,  0.707],
       [-0.466, -0.256],
       [-1.938, -0.834],
       [ 0.183, -1.147]]), np.array([[ 1.073,  0.903],
       [-1.063, -0.433],
       [-0.99 ,  1.209]])]

X_truth = np.array([[-1.938, -0.834],
       [-1.738,  1.5  ],
       [-1.518,  0.433],
       [-1.405, -1.65 ],
       [-1.063, -0.433],
       [-0.99 ,  1.209],
       [-0.748, -1.405],
       [-0.466, -0.256],
       [-0.282,  2.029],
       [-0.133,  0.341],
       [ 0.183, -1.147],
       [ 0.402, -1.912],
       [ 0.537, -0.15 ],
       [ 0.809,  1.82 ],
       [ 0.938, -1.453],
       [ 1.073,  0.903],
       [ 1.346, -0.665],
       [ 1.56 ,  0.088],
       [ 1.746,  0.707],
       [ 1.916,  1.229]])

Z_mf = [np.array([  474.676,   759.719,   780.988,  1223.22 ,   727.654,   968.445,  1152.791,  1310.05 ,   830.06 ,  1391.329,  1397.509,  1220.772,   992.236,
         483.999,  2308.978,   749.823,   865.89 ,   883.622,  2954.624,  1053.244]), np.array([  326.461,   407.658,   708.516,   829.201,   400.95 ,   773.208,   920.535,   960.336,   621.651,   944.244,   932.632,   765.596,   655.824,
         263.663,  1868.112,   407.121,   719.042,   490.24 ,  2563.841,   616.82 ]), np.array([   6.134,  248.998,    9.188])]

Z_truth = np.array([ 2116.227,   238.317,   356.801,  1319.865,   248.998,     9.188,   389.   ,    24.493,   381.822,    11.71 ,   140.147,   430.496,    19.382,
         135.897,   544.096,     6.134,   614.133,   550.638,   549.6  ,   597.705])

MFK_kwargs = {'print_global' : True,
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

model = MFK(**MFK_kwargs)

model.set_training_values(X_mf[0], Z_mf[0], name = 0)
model.set_training_values(X_mf[1], Z_mf[1], name = 1)
model.set_training_values(X_mf[2], Z_mf[2])
model.train()
model.pre

inds = isin_indices(X_truth,X_mf[-1],inversed=True)
print(RMSE_norm(Z_truth[inds], model.predict_values(X_truth[inds]).reshape(Z_truth[inds].shape)))