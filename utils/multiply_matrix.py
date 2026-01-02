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


def climbing_matrix(deg: int) -> np.ndarray:
    """
    Construct the multiplication matrix M_x for the generalized Chebyshev encoding
    such that
        x * tau_d(x) = M_x @ tau_{d+1}(x),

    where
        tau_d(x)   = (T_0(x), sqrt(2) T_1(x), ..., sqrt(2) T_d(x))^T / sqrt(d+1),
        tau_{d+1}(x) = (T_0(x), sqrt(2) T_1(x), ..., sqrt(2) T_{d+1}(x))^T / sqrt(d+2).

    Here M_x has shape (d+1, d+2), mapping from degree d+1 encoding to degree d encoding
    under multiplication by x.

    Parameters
    ----------
    deg : int
        Maximum degree d of the input Chebyshev encoding tau_d.

    Returns
    -------
    M_x : np.ndarray
        Real matrix of shape (deg+1, deg+2) such that
        x * tau_d(x) = M_x @ tau_{d+1}(x).
    """
    if deg < 0:
        raise ValueError("deg must be a non-negative integer.")

    d = deg
    d_prime = d + 1  # target encoding degree
    M = np.zeros((d + 1, d_prime + 1), dtype=float)

    for j in range(d + 1):
        w_j = get_weight(j, d)

        if j == 0:
            # x T_0(x) = T_1(x)
            k = 1
            a_kj = 1.0
            w_k = get_weight(k, d_prime)
            M[j, k] += w_j * a_kj / w_k
        else:
            # For n >= 1: x T_n(x) = 0.5 (T_{n+1}(x) + T_{n-1}(x))
            ## Lower index term T_{j-1}
            k_low = j - 1
            a_low = 0.5
            w_low = get_weight(k_low, d_prime)
            M[j, k_low] += w_j * a_low / w_low
            ## Upper index term T_{j+1}
            k_up = j + 1
            a_up = 0.5
            w_up = get_weight(k_up, d_prime)
            M[j, k_up] += w_j * a_up / w_up

    return M

def M_x_power(deg: int, p: int, deg_out: int | None = None) -> np.ndarray:
    """
    Construct the multiplication matrix M_{x^p} for the generalized Chebyshev encoding
    such that
        x^p * tau_d(x) = M_{x^p} @ tau_{d'}(x),

    where
        tau_d(x)    = (T_0(x), sqrt(2) T_1(x), ..., sqrt(2) T_d(x))^T / sqrt(d+1),
        tau_{d'}(x) = (T_0(x), sqrt(2) T_1(x), ..., sqrt(2) T_{d'}(x))^T / sqrt(d'+1),

    and p is a non-negative integer. By default, d' = d + p.

    The matrix M_{x^p} has shape (d+1, d'+1).

    Implementation details
    ----------------------
    - For p >= 1, we build M_{x^p} as the product of the climbing matrices:
          M_{x^p} = M_x(d) @ M_x(d+1) @ ... @ M_x(d+p-1),
      where each M_x(k) satisfies:
          x * tau_k(x) = M_x(k) @ tau_{k+1}(x).

      This gives
          x^p * tau_d(x) = M @ tau_{d+p}(x).

    - If d' != d + p, we convert between encodings:
        tau_{d+p,k}(x) = [get_weight(k, d+p) / get_weight(k, d')] * tau_{d',k}(x)
      for all k <= min(d+p, d'). This yields a conversion matrix C of shape
      (d+p+1, d'+1) such that
        tau_{d+p}(x) = C @ tau_{d'}(x),
      and we set
        M_{x^p} = M @ C.

      This handles both:
        * d' > d+p: proper rescaling plus zero-padding for higher degrees.
        * d' < d+p: proper rescaling plus truncation of higher degrees.

    - For p = 0, x^0 = 1, so we only need the conversion matrix from tau_d to tau_{d'}.

    Parameters
    ----------
    deg : int
        Maximum degree d of the input Chebyshev encoding tau_d.
    p : int
        Non-negative integer power in x^p.
    deg_out : int, optional
        Output degree d'. If None, uses d' = d + p.

    Returns
    -------
    M^T : np.ndarray
        Real matrix of shape (deg_out+1, deg+1): the transpose of M such that
        x^p * tau_d(x) = M @ tau_{deg_out}(x).
    """
    if deg < 0:
        raise ValueError("deg must be a non-negative integer.")
    if p < 0 or int(p) != p:
        raise ValueError("p must be a non-negative integer.")
    p = int(p)

    d = deg
    if deg_out is None:
        d_out = d + p
    else:
        if deg_out < 0:
            raise ValueError("deg_out must be a non-negative integer.")
        if deg_out < d:
            raise ValueError("deg_out must satisfy deg_out >= deg.")
        d_out = deg_out

    # Special case: x^0 = 1 -> just convert tau_d to tau_{d_out}
    if p == 0:
        M = np.zeros((d + 1, d_out + 1), dtype=float)
        for j in range(d + 1):
            w_d     = get_weight(j, d)
            w_d_out = get_weight(j, d_out)
            M[j, j] = w_d / w_d_out
        return M.T

    # 1) Build M_{x^p} as product of climbing matrices up to degree d+p
    M = climbing_matrix(d)     # shape: (d+1, d+2)
    current_deg = d + 1          # we now map to tau_{current_deg}

    for k in range(1, p):
        M_next = climbing_matrix(current_deg)  # (current_deg+1, current_deg+2)
        M = M @ M_next
        current_deg += 1

    # At this point: current_deg = d + p, M has shape (d+1, d+p+1),
    # and we have: x^p * tau_d(x) = M @ tau_{d+p}(x).

    # 2) Convert from tau_{d+p} to tau_{d_out} by rescaling.
    # tau_{d+p,k} = (get_weight(k, d+p) / get_weight(k, d_out)) * tau_{d_out,k}
    max_shared = min(current_deg, d_out)
    C = np.zeros((current_deg + 1, d_out + 1), dtype=float)
    for k in range(max_shared + 1):
        w_cur = get_weight(k, current_deg)
        w_out = get_weight(k, d_out)
        C[k, k] = w_cur / w_out
    # rows k > max_shared stay zero (no contribution to any tau_{d_out} component)

    M = M @ C  # final shape: (d+1, d_out+1)

    return M.T


if __name__ == "__main__":
    N = 3
    p_max = 4

    print(M_x_power(deg=3, p=0, deg_out=7))
    print(M_x_power(deg=3, p=1, deg_out=7))
    print(M_x_power(deg=3, p=2, deg_out=7))
    print(M_x_power(deg=3, p=3, deg_out=7))
    print(M_x_power(deg=3, p=4, deg_out=7))
