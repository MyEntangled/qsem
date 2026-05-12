import numpy as np

def chebyshev_nodes(deg):
    """
    Roots of T_{d+1}(x).
    """
    n = deg + 1
    j = np.arange(n, dtype=float)
    return np.cos((2.0 * j + 1.0) * np.pi / (2.0 * n))

def chebyshev_encoding(deg, x):
    """
    Encodes the input x into Chebyshev polynomial basis up to degree deg.
    Returns a vector of size (deg + 1) with coefficients for T_0, T_1, ..., T_deg.
    """
    # Strategy 1C: vectorized Chebyshev recurrence (no Python loop)
    tau = np.empty(deg + 1)
    tau[0] = 1.0
    if deg >= 1:
        tau[1] = x
    if deg >= 2:
        two_x = 2.0 * x
        for k in range(2, deg + 1):
            tau[k] = two_x * tau[k - 1] - tau[k - 2]

    scale = 1.0 / np.sqrt(deg + 1)
    tau[0] *= scale
    tau[1:] *= scale * np.sqrt(2)

    return tau

if __name__ == "__main__":
    deg = 5
    x_list = chebyshev_nodes(deg)
    tau_list = np.linalg.norm([chebyshev_encoding(deg, x) for x in x_list])
    print(tau_list.shape)