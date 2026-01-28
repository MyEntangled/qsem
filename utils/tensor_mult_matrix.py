import numpy as np
import warnings
from src.utils import multiply_matrix

def get_weight(k: int, deg: int) -> float:
    """
    Return the coefficient of T_k in the generalized Chebyshev encoding
    tau_deg(x) = (T_0(x), sqrt(2) T_1(x), ..., sqrt(2) T_deg(x))^T / sqrt(deg+1).

    Parameters
    ----------
    k : int
        Chebyshev degree index (0 <= k <= deg).
    deg : int
        Maximum Chebyshev degree of the encoding.

    Returns
    -------
    w : float
        The scalar weight such that the k-th component of tau_deg(x) is
        w * T_k(x).
    """
    if k < 0 or k > deg:
        raise ValueError("k must satisfy 0 <= k <= deg.")

    norm = np.sqrt(deg + 1.0)
    if k == 0:
        return 1.0 / norm
    else:
        return np.sqrt(2.0) / norm

def N1_matrix(deg: int, deg_out: int | None = None) -> np.ndarray:
    """
    Construct the self-multiplication operator N_1 such that
        vec( tau_d(x) ⊗ tau_d(x) ) = N_1 @ tau_{deg_out}(x),

    where
        tau_d(x)      = (T_0(x), sqrt(2) T_1(x), ..., sqrt(2) T_d(x))^T / sqrt(d+1),
        tau_{deg_out}(x) = same encoding up with d=deg_out,

    and the outer product is flattened in row-major order:
        vec(tau_d ⊗ tau_d)[i*(d+1) + j] = tau_d,i * tau_d,j.

    By default, deg_out = 2*deg, since the product of two degree-d
    Chebyshev polynomials has maximum degree 2d.

    Shape:
        N_1 ∈ R^{(d+1)^2 × (deg_out+1)}.

    Parameters
    ----------
    deg : int
        Input degree d (for tau_d).
    deg_out : int, optional
        Output degree for the encoding tau_{deg_out}.
        If None, uses deg_out = 2 * deg.

    Returns
    -------
    N1^T : np.ndarray
        Real matrix of shape (deg_out+1, (deg+1)**2): the transpose of N1 such that
        vec(tau_d ⊗ tau_d) = N1 @ tau_{deg_out}.
    """
    if deg < 0:
        raise ValueError("deg must be a non-negative integer.")

    d = deg
    if deg_out is None:
        d_out = 2 * d
    else:
        if deg_out < 0:
            raise ValueError("deg_out must be a non-negative integer.")
        d_out = deg_out

    natural_deg_out = 2 * d
    if d_out < natural_deg_out:
        warnings.warn(
            f"Truncation in N1_matrix: output degree d_out={d_out} "
            f"is smaller than natural degree 2*deg={natural_deg_out}. "
            "Higher Chebyshev modes in tau_d(x) ⊗ tau_d(x) are truncated.",
            RuntimeWarning
        )


    N1 = np.zeros(((d + 1) * (d + 1), d_out + 1), dtype=float)

    # Precompute weights
    w_d = [get_weight(i, d) for i in range(d + 1)]
    w_out = [get_weight(k, d_out) for k in range(d_out + 1)]

    # Use T_n T_m = 1/2 (T_{n+m} + T_{|n-m|})
    for i in range(d + 1):
        for j in range(d + 1):
            idx = i * (d + 1) + j         # row index in vec(tau_d ⊗ tau_d)
            alpha = 0.5 * w_d[i] * w_d[j] # prefactor in front of T_i T_j

            # Contribution to T_{i+j}
            k1 = i + j
            if k1 <= d_out:
                N1[idx, k1] += alpha / w_out[k1]

            # Contribution to T_{|i-j|}
            k2 = abs(i - j)
            if k2 <= d_out:
                N1[idx, k2] += alpha / w_out[k2]

    return N1.T

def Nx_matrix(deg: int, deg_out: int | None = None) -> np.ndarray:
    """
    Construct the operator N_x such that
        x * vec( tau_d(x) ⊗ tau_d(x) ) = N_x @ tau_{deg_out}(x),

    with the same encoding convention as in N1_matrix.

    Internally, this is built via
        vec(tau_d ⊗ tau_d) = N_1 @ tau_{d_mid},
        x * tau_{d_mid}(x) = M_x_power(d_mid, 1, deg_out) @ tau_{deg_out}(x),

    so that
        x * vec(tau_d ⊗ tau_d)
            = N_1 @ [ x tau_{d_mid}(x) ]
            = N_1 @ M_x_power(d_mid, 1, deg_out) @ tau_{deg_out}(x),

        hence
            N_x = N_1 @ M_x_power(d_mid, 1, deg_out).

    We choose d_mid = 2 * deg to capture all degrees up to 2d
    in the self-product, and by default deg_out = 2*deg + 1
    to capture the full degree of x * (degree-2d polynomial).

    Shape:
        N_x ∈ R^{(d+1)^2 × (deg_out+1)}.

    Parameters
    ----------
    deg : int
        Input degree d (for tau_d).
    deg_out : int, optional
        Output degree for the encoding tau_{deg_out}.
        If None, uses deg_out = 2 * deg + 1.

    Returns
    -------
    Nx^T : np.ndarray
        Real matrix of shape (deg_out+1, (deg+1)**2): the transpose of Nx such that
        x * vec(tau_d ⊗ tau_d) = Nx @ tau_{deg_out}.
    """
    if deg < 0:
        raise ValueError("deg must be a non-negative integer.")

    d = deg
    if deg_out is None:
        d_out = 2 * d + 1
    else:
        if deg_out < 0:
            raise ValueError("deg_out must be a non-negative integer.")
        d_out = deg_out

    natural_deg_out = 2 * d + 1
    if d_out < natural_deg_out:
        warnings.warn(
            f"Truncation in Nx_matrix: output degree d_out={d_out} "
            f"is smaller than natural degree 2*deg+1={natural_deg_out}. "
            "Higher Chebyshev modes in x*(tau_d(x) ⊗ tau_d(x)) are truncated.",
            RuntimeWarning
        )

    # Intermediate degree for the pure self-product
    d_mid = 2 * d

    # N_1: vec(tau_d ⊗ tau_d) = N1 @ tau_{d_mid}
    N1 = N1_matrix(d, d_mid).T

    # x * tau_{d_mid}(x) = M_x_power(d_mid, 1, d_out) @ tau_{d_out}(x)
    Mx = multiply_matrix.M_x_power(d_mid, 1, d_out).T

    # x * vec(tau_d ⊗ tau_d) = N1 @ Mx @ tau_{d_out}(x)
    Nx = N1 @ Mx

    return Nx.T

if __name__ == "__main__":
    deg = 3
    N1 = N1_matrix(deg, deg_out=7)
    Nx = Nx_matrix(deg, deg_out=7)
    print("N1 matrix:", N1.shape)

    print("Nx matrix:", Nx.shape)

    def set_N1():
        s2 = np.sqrt(2)
        s2_inv = 1. / s2
        N1_expected = np.zeros((8, 16))
        N1_expected[0, [0, 5, 10, 15]] = s2_inv
        N1_expected[1, [1, 4]] = s2_inv
        N1_expected[1, [6, 9, 11, 14]] = 0.5
        N1_expected[2, [2, 8]] = s2_inv
        N1_expected[2, [5, 7, 13]] = 0.5
        N1_expected[3, [3, 12]] = s2_inv
        N1_expected[3, [6, 9]] = 0.5
        N1_expected[4, [7, 10, 13]] = 0.5
        N1_expected[5, [11, 14]] = 0.5
        N1_expected[6, [15]] = 0.5
        return N1_expected

    def set_Nx():
        s2 = np.sqrt(2)
        s2_inv = 1. / s2
        Nx_expected = np.zeros((8, 16))
        Nx_expected[0, [1, 4]] = 0.5
        Nx_expected[0, [6, 9, 11, 14]] = s2_inv / 2
        Nx_expected[1, [0, 10, 15]] = 0.5
        Nx_expected[1, [2, 8]] = s2_inv / 2
        Nx_expected[1, [5]] = 0.75
        Nx_expected[1, [7, 13]] = 0.25
        Nx_expected[2, [1, 3, 4, 12]] = s2_inv / 2
        Nx_expected[2, [6, 9]] = 0.5
        Nx_expected[2, [11, 14]] = 0.25
        Nx_expected[3, [2, 8]] = s2_inv / 2
        Nx_expected[3, [5, 10]] = 0.25
        Nx_expected[3, [7, 13]] = 0.5
        Nx_expected[4, [3, 12]] = s2_inv / 2
        Nx_expected[4, [6, 9, 11, 14]] = 0.25
        Nx_expected[5, [7, 10, 13, 15]] = 0.25
        Nx_expected[6, [11, 14]] = 0.25
        Nx_expected[7, [15]] = 0.25
        return Nx_expected

    N1_expected = set_N1()
    Nx_expected = set_Nx()

    for i in range(8):
        for j in range(16):
            if not np.isclose(N1[i, j], N1_expected[i, j]):
                print(f"Mismatch at ({i},{j}): got {N1[i, j]}, expected {N1_expected[i, j]}")



    for i in range(8):
        for j in range(16):
            if not np.isclose(Nx[i, j], Nx_expected[i, j]):
                print(f"Mismatch at ({i},{j}): got {Nx[i, j]}, expected {Nx_expected[i, j]}")

    print(Nx)