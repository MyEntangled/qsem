import numpy as np
import scipy

def regularization_matrix(d: int, p: float, sparse: bool = False):
    ## Construct matrix \sum_k k^p |k><k|
    if p < 0:
        raise ValueError("p must be non-negative.")
    k = np.arange(d + 1)
    values = k ** p
    if sparse:
        R = scipy.sparse.diags(values)
    else:
        R = np.diag(values)
    return R