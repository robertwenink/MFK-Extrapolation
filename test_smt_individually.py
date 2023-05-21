import numpy as np
from smt.applications import MFK
from core.sampling.solvers.solver import get_solver
from utils.error_utils import RMSE_norm, RMSE_norm_MF
from utils.selection_utils import isin_indices

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
# def twee():
#     return
# twee.solver_str = 'EVA'
# solver = get_solver(setup = twee)
# x = np.array([[0.5165, 0.077]])
# z0 = solver.solve(x,0.5)[0].item()
# z1 = solver.solve(x,1.0)[0].item()
# z2 = solver.solve(x,2.0)[0].item()

# Branin
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

# EVA high_mass end 5 hifi samples
# X_mf = [np.array([[ 0.6975,  0.9443],
#        [ 0.2583,  0.7952],
#        [ 0.9679,  0.8001],
#        [ 0.1293,  0.6057],
#        [ 0.6311,  0.4635],
#        [ 0.8809,  0.5214],
#        [ 0.8287,  0.3375],
#        [ 0.7289,  0.1452],
#        [ 0.4312,  0.9954],
#        [ 0.3174,  0.157 ],
#        [ 0.5982,  0.0332],
#        [ 0.2404,  0.3942],
#        [ 0.0757,  0.8663],
#        [ 0.7619,  0.7204],
#        [ 0.1569,  0.0972],
#        [ 0.4674,  0.5832],
#        [ 0.9264,  0.6725],
#        [ 0.3863,  0.4375],
#        [ 0.0268,  0.2964],
#        [ 0.5448,  0.2199]]), np.array([[ 0.6975,  0.9443],
#        [ 0.2583,  0.7952],
#        [ 0.9679,  0.8001],
#        [ 0.1293,  0.6057],
#        [ 0.6311,  0.4635],
#        [ 0.8809,  0.5214],
#        [ 0.8287,  0.3375],
#        [ 0.7289,  0.1452],
#        [ 0.4312,  0.9954],
#        [ 0.3174,  0.157 ],
#        [ 0.5982,  0.0332],
#        [ 0.2404,  0.3942],
#        [ 0.0757,  0.8663],
#        [ 0.7619,  0.7204],
#        [ 0.1569,  0.0972],
#        [ 0.4674,  0.5832],
#        [ 0.9264,  0.6725],
#        [ 0.3863,  0.4375],
#        [ 0.0268,  0.2964],
#        [ 0.5448,  0.2199],
#        [ 0.4744,  0.0142],
#        [ 0.574 ,  0.1568]]), np.array([[ 0.5982,  0.0332],
#        [ 0.4674,  0.5832],
#        [ 0.8287,  0.3375],
#        [ 0.4744,  0.0142],
#        [ 0.574 ,  0.1568]])]

# X_truth = np.array([[ 0.0268,  0.2964],
#        [ 0.0757,  0.8663],
#        [ 0.1293,  0.6057],
#        [ 0.1569,  0.0972],
#        [ 0.2404,  0.3942],
#        [ 0.2583,  0.7952],
#        [ 0.3174,  0.157 ],
#        [ 0.3863,  0.4375],
#        [ 0.4312,  0.9954],
#        [ 0.4674,  0.5832],
#        [ 0.4744,  0.0142],
#        [ 0.5448,  0.2199],
#        [ 0.574 ,  0.1568],
#        [ 0.5982,  0.0332],
#        [ 0.6311,  0.4635],
#        [ 0.6975,  0.9443],
#        [ 0.7289,  0.1452],
#        [ 0.7619,  0.7204],
#        [ 0.8287,  0.3375],
#        [ 0.8809,  0.5214],
#        [ 0.9264,  0.6725],
#        [ 0.9679,  0.8001]])

# Z_mf = [np.array([ 2172471.1953,  1665047.0597,  2661429.1916,  1584742.4702,  1477530.2178,  1938048.5111,  1651871.8135,  1305282.7092,  1876632.5837,
#         1123906.0523,  1077449.6483,  1309294.2259,  1873978.2157,  1987308.0248,  1405986.0396,  1395136.8028,  2253500.2133,  1185518.0118,
#         1602129.3346,  1140284.3704]), np.array([ 2219332.7909,  1751851.9999,  2823942.0166,  1693030.6755,  1440295.4407,  2014371.4768,  1614242.6743,  1257661.8969,  1878143.6081,
#         1168561.4922,   995037.3714,  1383796.9434,  2068458.7303,  2031680.961 ,  1449751.8761,  1322786.9417,  2395734.5131,  1192715.5733,
#         1663457.9788,  1067326.3848,   978603.1417,  1060328.2911]), np.array([ 1005697.7991,  1318602.7464,  1742551.0694,  1086628.9953,  1059745.1279])]

# Z_truth = np.array([ 1715981.5139,  2125465.5666,  1743109.4673,  1546714.2174,  1363037.7234,  1769670.7358,  1186287.5764,  1185532.3446,  1883995.7601,
#         1318602.7464,  1086628.9953,  1081830.1758,  1059745.1279,  1005697.7991,  1433605.6394,  2278856.5303,  1322706.4235,  2106209.8262,
#         1742551.0694,  2162208.3242,  2646152.2757,  3123579.0532])



model.set_training_values(X_mf[0], Z_mf[0], name = 0)
model.set_training_values(X_mf[1], Z_mf[1], name = 1)
model.set_training_values(X_mf[2], Z_mf[2])
model.train()

inds = isin_indices(X_truth,X_mf[-1],inversed=True)
print(RMSE_norm(Z_truth[inds], model.predict_values(X_truth[inds]).reshape(Z_truth[inds].shape)))