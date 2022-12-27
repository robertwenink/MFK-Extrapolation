import numpy as np
from beautifultable import BeautifulTable

def pearson_correlation_squared(Z0, Z1):
    assert (
        Z0.shape[0] == Z1.shape[0]
    ), "\nZ0 and Z1 are not of equal length during correlation check."
    m0 = Z0 - np.mean(Z0)
    m1 = Z1 - np.mean(Z1)
    m = m0 * m1
    top = np.sum(m)
    bottom = np.sqrt(np.sum((Z0 - np.mean(Z0)) ** 2)) * \
             np.sqrt(np.sum((Z1 - np.mean(Z1)) ** 2))
    r2 = (top / bottom) ** 2

    return r2


def pearson_correlation_differences(Z0, Z1, Z2):
    assert (
        Z0.shape[0] == Z1.shape[0] == Z2.shape[0]
    ), "\nZ0, Z1, and Z2 are not of equal length during correlation check."

    d0 = Z1 - Z0
    d1 = Z2 - Z1

    return pearson_correlation_squared(d0, d1)


def check_correlations(Z0, Z1, Z2):
    r2_low = pearson_correlation_squared(Z0, Z1)
    r2_high = pearson_correlation_squared(Z1, Z2)
    r2_diff = pearson_correlation_differences(Z0, Z1, Z2)


    table = BeautifulTable()
    table.columns.header = ['r2_low', 'r2_high', 'r2_diff']
    table.rows.append([r2_low, r2_high, r2_diff])
    warnings = [""] * 3 
    if r2_low < 0.9:
        warnings[0] = "r2_low < 0.9!"
    if r2_high < 0.9:
        warnings[1] = "r2_high < 0.9!"
    if r2_diff < 0.9:
        warnings[2] = "r2_diff < 0.9!"
    if any(warnings):
        table.rows.append(warnings)
    