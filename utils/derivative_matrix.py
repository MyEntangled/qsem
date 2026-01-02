import numpy as np

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

def chebyshev_diff_matrix(deg: int, deg_out: int | None = None) -> np.ndarray:
    """
    Construct the differentiation matrix G for the generalized Chebyshev encoding
    tau_deg(x) = (T_0(x), sqrt(2) T_1(x), ..., sqrt(2) T_deg(x))^T / sqrt(deg+1),

    such that
        d/dx tau_deg(x) = G @ tau_deg_out(x),

    where tau_deg_out uses the same encoding with maximum degree deg_out.

    Requirements
    ------------
    - deg_out must satisfy deg_out >= deg - 1.
      By default, deg_out = deg.

    Rationale
    ---------
    The derivative of a degree-d Chebyshev polynomial has degree at most d-1.
    As long as deg_out >= d-1, all derivative components can be represented
    exactly in the tau_deg_out basis, regardless of whether deg_out < deg
    (truncation of input encoding) or deg_out > deg (extra unused modes).

    Implementation
    --------------
    For each component j of tau_deg:

        tau_deg,j(x) = w_in[j] * T_j(x),

    where w_in[j] = get_weight(j, deg).

    Then
        d/dx tau_deg,j(x) = w_in[j] * d/dx T_j(x)
                          = w_in[j] * j * U_{j-1}(x),

    and we expand U_{j-1}(x) in Chebyshev T_k(x) basis:

      Let m = j - 1.
      If m is even:
          U_m(x) = T_0(x) + 2 * sum_{k=2,4,...,m} T_k(x),
      If m is odd:
          U_m(x) = 2 * sum_{k=1,3,...,m} T_k(x).

    So
        d/dx tau_deg,j(x) = sum_k [ w_in[j] * j * u_k ] T_k(x),

    where u_k are the coefficients from the U_m expansion.

    Expressing this in tau_deg_out:

        tau_deg_out,k(x) = w_out[k] * T_k(x),
        w_out[k] = get_weight(k, deg_out),

    gives
        w_in[j] * j * u_k = G[j,k] * w_out[k]
        => G[j,k] = w_in[j] * j * u_k / w_out[k].

    This holds for any deg_out >= deg-1, no extra conversion step needed.

    Parameters
    ----------
    deg : int
        Maximum degree of the input Chebyshev encoding tau_deg.
    deg_out : int, optional
        Maximum degree of the output encoding tau_deg_out.
        Must satisfy deg_out >= deg - 1.
        If None, uses deg_out = deg.

    Returns
    -------
    G^T : np.ndarray
        Real matrix of shape (deg_out+1, deg+1): the transpose of G such that
        d/dx tau_deg(x) = G @ tau_deg_out(x).
    """
    if deg < 0:
        raise ValueError("deg must be a non-negative integer.")

    if deg_out is None:
        d_out = deg
    else:
        if deg_out < 0:
            raise ValueError("deg_out must be a non-negative integer.")
        if deg_out < deg - 1:
            raise ValueError("deg_out must satisfy deg_out >= deg-1.")
        d_out = deg_out

    d = deg
    G = np.zeros((d + 1, d_out + 1), dtype=float)

    # Precompute weights
    w_in = [get_weight(j, d) for j in range(d + 1)]
    w_out = [get_weight(k, d_out) for k in range(d_out + 1)]

    # j = 0: derivative of T_0 is zero -> row stays zero
    for j in range(1, d + 1):
        m = j - 1  # index for U_m
        # U_m expansion coefficients in T_k basis
        u = np.zeros(d_out + 1, dtype=float)

        if m % 2 == 0:
            # m even: U_m(x) = T_0 + 2 * sum_{even k=2..m} T_k(x)
            u[0] = 1.0
            for k in range(2, m + 1, 2):
                if k <= d_out:
                    u[k] = 2.0
        else:
            # m odd: U_m(x) = 2 * sum_{odd k=1..m} T_k(x)
            for k in range(1, m + 1, 2):
                if k <= d_out:
                    u[k] = 2.0

        # Now d/dx tau_deg,j = w_in[j] * j * sum_k u_k T_k(x)
        # Express in tau_deg_out:
        #   G[j,k] = w_in[j] * j * u_k / w_out[k]
        for k in range(0, min(m, d_out) + 1):
            if w_out[k] != 0.0 and u[k] != 0.0:
                G[j, k] = w_in[j] * j * u[k] / w_out[k]

    return G.T


if __name__ == "__main__":
    G_T = chebyshev_diff_matrix(deg=3, deg_out=None)
    print(G_T)
