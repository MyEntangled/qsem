import numpy as np

def regularization_matrix(d: int, p: float):
    ## Construct matrix \sum_k k^p |k><k|
    if p < 0:
        raise ValueError("p must be non-negative.")
    k = np.arange(d + 1)
    R = np.diag(k ** p)
    return R