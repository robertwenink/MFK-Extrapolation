from copy import deepcopy
import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, linewidth=1000)

mask = np.array([
                [False, False, False, False, False, False, False,  True, False, False, False, False, False],
                [False, False, False, False, False, False, False, False, False, False, False,  True, False],
                [False, False, False, False, False, False, False, False, False, False, False, False,  True]])
idx = np.array([ False, False, False, False, False, False, False,  True, False, False, False,  True,  True])
c_dist_org = np.array([
       [ 0.9819,  0.7442,  0.9812,  0.5556,  0.8734,  0.9415,  0.8851,  1.    ,  0.7782,  0.8852,  0.885 ,  0.9076,  0.812 ],
       [ 0.9694,  0.4815,  0.8174,  0.3128,  0.6303,  0.9957,  0.9986,  0.9076,  0.9648,  0.9986,  0.9986,  1.    ,  0.5546],
       [ 0.7048,  0.9924,  0.9034,  0.9082,  0.9922,  0.611 ,  0.5225,  0.812 ,  0.4001,  0.5226,  0.5224,  0.5546,  1.    ]])

method_weighing = True
print(f"Method weighing = {method_weighing}")
if method_weighing:
    # method weighing
    D_w = np.array([False,  True,  True])
    D_mse = np.array([  [ 0.0014,  0.0014,  0.0014,  0.0014,  0.0014,  0.0014,  0.0005,  0.    ,  0.0014,  0.0005,  0.0005,  0.0013,  0.0014],
                        [ 0.0101,  0.0201,  0.0111,  0.016 ,  0.0203,  0.0113,  0.0058,  0.0063,  0.009 ,  0.0058,  0.0058,  0.    ,  0.0221],
                        [ 0.0092,  0.0166,  0.0102,  0.0138,  0.0166,  0.0098,  0.0049,  0.0065,  0.0088,  0.0048,  0.0049,  0.0069,  0.    ]])
else:
    # no method weighing
    D_w = np.array([ True,  True,  True])
    D_mse = np.array([
        [  332282.4083,  1967402.5342,   304790.6725,  1003822.6831,  2148355.2597,   666565.3546,   496304.8003,        0.    ,    73907.7125,     497032.0699,   495576.5455,   647740.0141,  2839398.6567],
        [       0.0101,        0.0201,        0.0111,        0.016 ,        0.0203,        0.0113,        0.0058,        0.0063,        0.009 ,          0.0058,        0.0058,        0.    ,        0.0221],
        [       0.0092,        0.0166,        0.0102,        0.0138,        0.0166,        0.0098,        0.0049,        0.0065,        0.0088,          0.0048,        0.0049,        0.0069,        0.    ]])

# sort to order
X_unique = np.array([[ 0.3198],       [ 0.8034],       [ 0.5142],       [ 0.9624],       [ 0.6783],       [ 0.2412],       [ 0.1672],       [ 0.4161],       [ 0.0593],       [ 0.1673],       [ 0.1671],       [ 0.1942],       [ 0.7413]]).flatten()
order = np.argsort(X_unique)
mask, idx, c_dist_org, D_mse = mask[:,order], idx[order], c_dist_org[:,order], D_mse[:,order]

if True: # old part
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
    
" First variance weighing "
# relative weighing of distance scaling wrt variance scaling
c_dist_rel_weight = 1
if np.all(~D_w):
    # if there is nothing
    D_w = ~D_w

mse_mean = np.mean(D_mse[D_w, :], axis = 1) # select using boolean weight (only use samples not completely w = 0)
c_var = np.exp(-(D_mse)/(np.mean(mse_mean))*c_dist_rel_weight) # min omdat bij uitschieters alleen de meest bizarre uitschieter gefilterd wordt terwijl de rest miss ook wel kak is.

# We basically do not want to include the other method in variance weighing. If all are other method neither.
# so we take the mean over the c_var`s of those of the extrapolation, in effect distance weighing becomes the prevalent
c_var[~D_w,:] = np.min(c_var[D_w], axis = 0)
# c_var /= np.sum(c_var, axis=0)
c_var[mask] = 1
print(f"c_var means = {np.mean(c_var,axis=1)}")

# samples giving no contribution should not instigate 0 distance weight for others!
c_dist_var = c_dist_org * c_var
print(c_dist_var)
" Then distance weighing "
# er is geen interpolerende oplossing momenteel.
# daarom: sequentially huidige locatie eraf halen, column opnieuw schalen naar 1, herhalen voor volgend punt
selection = c_dist_org[:, idx]
c_dist_inter = deepcopy(c_dist_org)
# c_dist_inter = deepcopy(c_dist_var)
# c_dist_inter /= np.sum(c_dist_inter, axis=0)

# only use linearly consistent row operations! so, rowwise +- and constant multiply
# for each sample, make the other rows 0 (then other sample row operations will keep this zero, multiplications too)
for i, col_org in enumerate(selection.T): 
    col_alt = c_dist_inter[:, idx][:,i] 
    j = np.where(col_org == 1.0)[0]
    aftrek = col_alt[:,np.newaxis] * c_dist_inter[j,:]
    aftrek[j,:] = 0

    c_dist_inter -= aftrek

    # rescale such that rows at samples are one again
    for k in range(c_dist_org.shape[0]):
        # scale all samples back to 1, scale rest of row too
        z = np.where(mask[k,:] == True)[0] # get column of the sample

        # scale back to original value
        c_dist_inter[k,:] *= c_dist_var[k,z] / c_dist_inter[k,z]

# clip to max min (i.e. negative values should not have influence anymore)
# NOTE max destroys some of the correlation information. The balancing between samples is still done by dividing with the sum
c_dist_inter = c_dist_inter.clip(min = 0)# , max=1) 

# rescale: contributions of all samples should add up to 1
print(c_dist_inter)
c_dist_inter = c_dist_inter #* c_var
# c_dist_inter /= np.sum(c_dist_inter, axis=0)
c_dist = c_dist_inter * c_var
c_dist /= np.sum(c_dist, axis=0)
print(c_dist)

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