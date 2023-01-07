
def isin(x,X):
    """Test if any x is in X"""    
    return isin_indices(x,X).any()

def isin_indices(X_test, X, inversed = False):
    """
    return an array of True for indices of X_test where X_test is not in X.
    X_test can be 1 or multiple locations.
    """
    assert X_test.shape[1:] == X.shape[1:], "x and X not of same format"
    mask = (X_test == X[:, None]).all(axis=-1).any(axis=0)
    if inversed:
        return ~mask
    return mask
