import numpy as np


def pearson_correlation_squared(Z0, Z1):
    assert Z0.shape[0] == Z1.shape[0], "\nZ0 and Z1 are not of equal length during correlation check."

    r2 = ( np.sum((Z0-np.mean(Z0)) - (Z1-np.mean(Z1))) /  \
        (np.sqrt(np.sum((Z0-np.mean(Z0)**2))) * np.sqrt(np.sum((Z0-np.mean(Z0)**2))))) ** 2

    return r2

def pearson_correlation_differences(Z0, Z1, Z2):
    assert Z0.shape[0] == Z1.shape[0] == Z2.shape[0], "\nZ0, Z1, and Z2 are not of equal length during correlation check."

    d0 = Z1 - Z0
    d1 = Z2 - Z1

    return pearson_correlation_squared(d0, d1)

def check_correlations(Z):
    r2_low = pearson_correlation_squared(Z[0], Z[1])
    r2_high = pearson_correlation_squared(Z[1], Z[2])
    r2_diff = pearson_correlation_differences(Z[0], Z[1], Z[2])

    print("r2_low = {} \n r2_high = {} \n r2_diff = {}".format(r2_low,r2_high,r2_diff))
    if r2_low < 0.9:
        print("WARNING: r2_low < 0.9!")
    if r2_high < 0.9:
        print("WARNING: r2_high < 0.9!")
    if r2_diff < 0.9:
        print("WARNING: r2_diff < 0.9!")

