import numpy as np # type: ignore

def ensure_intercept(X):
    """
    Checks if a column of ones exists. If not, adds one to the start of X.
    """
    # Check if any column is all ones
    has_intercept = np.any(np.all(X == 1, axis=0))
    
    if not has_intercept:
        return np.c_[np.ones((X.shape[0], 1)), X]
    return X

