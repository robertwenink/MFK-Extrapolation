from copy import deepcopy
import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, linewidth=1000)

mask = np.array(
      [[False, False, False, False, False, False, False, False, False, False,  True, False, False, False],
       [False, False,  True, False, False, False, False, False, False, False, False, False, False, False],
       [False, False, False, False, False, False,  True, False, False, False, False, False, False, False]])
idx = np.array(
       [False  , False  ,  True  , False  , False  , False  ,  True  , False  , False  , False  ,  True  , False  , False  , False])
c_dist_org = np.array(
      [[ 0.117 ,  0.2186,  0.3155,  0.3157,  0.4406,  0.6377,  0.7884,  0.7885,  0.9819,  1.    ,  1.    ,  1.    ,  0.9823,  0.798 ],
       [ 0.8585,  0.975 ,  1.    ,  1.    ,  0.9719,  0.8499,  0.709 ,  0.7088,  0.4141,  0.3156,  0.3155,  0.3154,  0.2327,  0.0908],
       [ 0.385 ,  0.5737,  0.709 ,  0.7092,  0.8399,  0.967 ,  1.    ,  1.    ,  0.8832,  0.7885,  0.7884,  0.7882,  0.6799,  0.3959]])

selection = c_dist_org[:, idx]
c_dist_inter = deepcopy(c_dist_org)
# only use linearly consistent row operations! so, rowwise +- and constant multiply
# for each sample, make the other rows 0 (then other sample row operations will keep this zero, multiplications too)
for i, col in enumerate(selection.T): # NOTE col selects the row otherwise!!!!
    col = c_dist_inter[:, idx][:,i] # NOTE col selects the row!!!!
    j = np.where(col == 1.0)[0]
    aftrek = col[:,np.newaxis] * c_dist_inter[j,:]
    aftrek[j,:] = 0

    c_dist_inter -= aftrek

    # rescale such that rows at samples are one again
    for k in range(c_dist_org.shape[0]):
        # scale all samples back to 1, scale rest of row too
        z = np.where(mask[k,:] == True)[0] # get column of the sample
        c_dist_inter[k,:] /= c_dist_inter[k,z]


# clip to max min (i.e. negative values should not have influence anymore)
c_dist_inter = c_dist_inter.clip(min = 0, max=1)

# rescale: contributions of all samples should add up to 1
c_dist_inter /= np.sum(c_dist_inter, axis=0)

c_dist = c_dist_inter

col_indices = np.where(np.all(c_dist == 0, axis=0))[0]
if col_indices.any():
    print("WARNING: THERE ARE COLUMNS WITH 0 ONLY")
    print(col_indices)

sums = np.sum(c_dist[:, idx], axis=0)
if not np.all(np.isclose(sums, 1)):
    print("WARNING: SAMPLE COLUMNS DONT SUM TO 1")
    print(sums)

c_dist_samples = c_dist[mask]
if not np.all(np.isclose(c_dist_samples,1)):
    print("WARNING: NOT ONLY SAMPLES ARE 1 / SAMPLES NOT 1!")
    print(c_dist_samples)

# c_dist_inter /= np.sum(c_dist_inter, axis = 0)
print(idx)
print(c_dist)