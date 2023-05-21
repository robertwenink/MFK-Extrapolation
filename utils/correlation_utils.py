import numpy as np
from beautifultable import BeautifulTable

from utils.selection_utils import isin_indices

def pearson_correlation_squared(Z0, Z1):
    assert (
        Z0.shape[0] == Z1.shape[0]
    ), "\nZ0 and Z1 are not of equal length during correlation check."
    m0 = Z0 - np.mean(Z0)
    m1 = Z1 - np.mean(Z1)
    m = m0 * m1
    top = np.sum(m)
    bottom = np.sqrt(np.sum(m0 ** 2)) * np.sqrt(np.sum(m1 ** 2))
    r2 = (top / bottom) ** 2

    return r2


def pearson_correlation_differences(Z0, Z1, Z2):
    assert (
        Z0.shape[0] == Z1.shape[0] == Z2.shape[0]
    ), "\nZ0, Z1, and Z2 are not of equal length during correlation check."

    d0 = Z1 - Z0
    d1 = Z2 - Z1

    return pearson_correlation_squared(d0, d1)

def print_pearson_correlations(mf_model, print_raw = False):
    Z_top = mf_model.Z_truth
    inds = isin_indices(mf_model.X_mf[0],mf_model.X_truth,inversed=False)
    _, inds_sort = np.unique(mf_model.X_mf[0][inds], axis = 0, return_index = True)
    check_correlations(mf_model.Z_mf[0][inds][inds_sort], mf_model.Z_mf[1][inds][inds_sort], Z_top, print_raw)

def check_correlations(Z0, Z1, Z2, print_raw = False):
    r2_low = pearson_correlation_squared(Z0, Z1)
    r2_high = pearson_correlation_squared(Z1, Z2)
    r2_diff = pearson_correlation_differences(Z0, Z1, Z2)

    if print_raw:
        print(f"{r2_low:.3f}\t{r2_high:.3f}\t{r2_diff:.3f}")
    else:
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
        print(table)
    