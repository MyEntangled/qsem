import numpy as np

def chebyshev_nodes(deg):
    xj = [np.cos(np.pi * (j+0.5) / (deg + 1)) for j in range(deg + 1)]
    return np.array(xj)

def chebyshev_encoding(deg, x):
    """
    Encodes the input x into Chebyshev polynomial basis up to degree deg.
    Returns a vector of size (deg + 1) with coefficients for T_0, T_1, ..., T_deg.
    """
    tau = np.zeros(deg + 1)
    scale = 1.0 / np.sqrt(deg + 1)
    for k in range(deg + 1):
        if k == 0:
            tau[k] = 1.0
        elif k == 1:
            tau[k] = x
        else:
            tau[k] = 2 * x * tau[k - 1] - tau[k - 2]

    tau[0] *= scale
    tau[1:] *= scale * np.sqrt(2)
    return tau