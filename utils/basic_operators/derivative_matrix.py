import numpy as np

def diff_matrix(deg: int, deg_out: int | None = None) -> np.ndarray:
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
    in_norm, out_norm = np.sqrt(d + 1), np.sqrt(d_out + 1)
    w_in = [1.0 / in_norm] + [np.sqrt(2.0) / in_norm] * d   # == [get_weight(j, d) for j in range(d + 1)]
    w_out = [1.0 / out_norm] + [np.sqrt(2.0) / out_norm] * d_out    # == [get_weight(k, d_out) for k in range(d_out + 1)]

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

def intg_matrix(a: float, deg: int, deg_out: int | None = None) -> np.ndarray:
    """
    Construct the integration matrix S for the generalized Chebyshev encoding
    tau_deg(x) = (T_0(x), sqrt(2) T_1(x), ..., sqrt(2) T_deg(x))^T / sqrt(deg+1),

    such that
        int_a^x tau_deg(t) dt = S @ tau_deg_out(x),

    where tau_deg_out uses the same encoding with maximum degree deg_out.

    Requirements
    ------------
    - deg_out must be a non-negative integer.
    - By default, deg_out = deg + 1.

    Rationale
    ---------
    The antiderivative of a degree-d Chebyshev polynomial has degree d+1.
    To avoid truncation and represent the integral exactly, deg_out should
    ideally be at least deg + 1. However, the matrix gracefully handles
    truncation if a smaller deg_out is provided.

    Implementation
    --------------
    For each component j of tau_deg:

        tau_deg,j(x) = w_in[j] * T_j(x),

    where w_in[j] = get_weight(j, deg).

    Then
        int_a^x tau_deg,j(t) dt = w_in[j] * int_a^x T_j(t) dt.

    We expand the integral of T_j(t) in the Chebyshev T_k(x) basis:
        For j = 0:
            int_a^x T_0(t) dt = T_1(x) - a*T_0(x)
        For j = 1:
            int_a^x T_1(t) dt = 1/4 T_2(x) + (1/4 - a^2/2) T_0(x)
        For j >= 2:
            int_a^x T_j(t) dt = 1/(2(j+1)) T_{j+1}(x) - 1/(2(j-1)) T_{j-1}(x)
                                - [ T_{j+1}(a)/(2(j+1)) - T_{j-1}(a)/(2(j-1)) ] T_0(x)

    So
        int_a^x tau_deg,j(t) dt = sum_k [ w_in[j] * c_k ] T_k(x),

    where c_k are the coefficients from the integral expansions above.

    Expressing this in tau_deg_out:

        tau_deg_out,k(x) = w_out[k] * T_k(x),
        w_out[k] = get_weight(k, deg_out),

    gives
        w_in[j] * c_k = S[j,k] * w_out[k]
        => S[j,k] = w_in[j] * c_k / w_out[k].

    Parameters
    ----------
    a : float
        The lower limit of integration.
    deg : int
        Maximum degree of the input Chebyshev encoding tau_deg.
    deg_out : int, optional
        Maximum degree of the output encoding tau_deg_out.
        If None, uses deg_out = deg + 1 to prevent truncation.

    Returns
    -------
    S^T : np.ndarray
        Real matrix of shape (deg_out+1, deg+1): the transpose of S such that
        int_a^x tau_deg(t) dt = S @ tau_deg_out(x).
    """
    if deg < 0:
        raise ValueError("deg must be a non-negative integer.")

    if deg_out is None:
        d_out = deg + 1
    else:
        if deg_out < 0:
            raise ValueError("deg_out must be a non-negative integer.")
        d_out = deg_out

    d = deg
    S = np.zeros((d + 1, d_out + 1), dtype=float)

    # Precompute normalization weights
    in_norm, out_norm = np.sqrt(d + 1), np.sqrt(d_out + 1)
    w_in = [1.0 / in_norm] + [np.sqrt(2.0) / in_norm] * d
    w_out = [1.0 / out_norm] + [np.sqrt(2.0) / out_norm] * d_out
    # w_in = [1] * (d+1)
    # w_out = [1] * (d_out+1)

    # Precompute evaluations of T_k(a) up to d+1 for the lower bound subtraction
    T_a = np.zeros(d + 2, dtype=float)
    T_a[0] = 1.0
    if d + 1 > 0:
        T_a[1] = float(a)
    for k in range(2, d + 2):
        T_a[k] = 2.0 * a * T_a[k - 1] - T_a[k - 2]

    for j in range(d + 1):
        # c holds the coefficients such that int_a^x T_j(t) dt = sum_k c[k] T_k(x)
        c = np.zeros(d_out + 1, dtype=float)

        if j == 0:
            c[0] = -a
            if 1 <= d_out:
                c[1] = 1.0
        elif j == 1:
            c[0] = 0.25 - 0.5 * (a ** 2)
            if 2 <= d_out:
                c[2] = 0.25
        else:
            # Constant term evaluates the antiderivative at t=a and negates it
            const_term = (T_a[j - 1] / (2.0 * (j - 1))) - (T_a[j + 1] / (2.0 * (j + 1)))
            c[0] = const_term
            if j - 1 <= d_out:
                c[j - 1] = -1.0 / (2.0 * (j - 1))
            if j + 1 <= d_out:
                c[j + 1] = 1.0 / (2.0 * (j + 1))

        # Express in tau_deg_out
        for k in range(d_out + 1):
            if w_out[k] != 0.0 and c[k] != 0.0:
                S[j, k] = w_in[j] * c[k] / w_out[k]

    return S.T


import numpy as np


# (Assume diff_matrix and intg_matrix are defined here as previously discussed)

def verify_matrices():
    d = 4  # Input degree
    d_int = d + 1  # Degree after integration
    a = -1.0  # Lower limit of integration

    # Generate the transposed matrices
    # S_T maps coeffs of degree d -> degree d+1
    S_T = intg_matrix(a=a, deg=d, deg_out=d_int)

    # D_T maps coeffs of degree d+1 -> degree d
    D_T = diff_matrix(deg=d_int, deg_out=d)

    print("--- Test 1: Differentiating the Integral (D * S = I) ---")
    # If we integrate a function and then differentiate it, we should get the exact
    # same coefficients back. Therefore, D_T @ S_T should be the Identity matrix.
    identity_check = D_T @ S_T
    is_identity = np.allclose(identity_check, np.eye(d + 1))

    print(f"Matrix D_T @ S_T is Identity? {is_identity}")
    if not is_identity:
        print(np.round(identity_check, 4))

    print("\n--- Test 2: Integrating a Derivative (S * D * f = f - f(a)) ---")
    # Create a random polynomial f(x) defined by its Chebyshev tau encoding coeffs
    np.random.seed(42)
    c_f = np.random.rand(d_int + 1)

    # Differentiate f(x) to get f'(x)
    c_df = D_T @ c_f

    # Integrate f'(x) from a to x
    c_int_df = S_T @ c_df

    # To check f(x) - f(a), we need to evaluate f(a).
    # We construct the tau basis evaluated at x = a
    out_norm = np.sqrt(d_int + 1)
    w_out = [1.0 / out_norm] + [np.sqrt(2.0) / out_norm] * d_int

    T_a = np.zeros(d_int + 1)
    T_a[0] = 1.0
    T_a[1] = a
    for k in range(2, d_int + 1):
        T_a[k] = 2.0 * a * T_a[k - 1] - T_a[k - 2]

    tau_a = np.array([w_out[k] * T_a[k] for k in range(d_int + 1)])

    # f(a) is the dot product of the coefficients and the basis evaluated at 'a'
    f_a_val = np.dot(c_f, tau_a)

    # The constant term in tau_deg_out is the 0-th index.
    # Since tau_deg_out_0(x) = T_0(x) / sqrt(d_int + 1) = 1 / sqrt(d_int + 1),
    # a constant shift of -f(a) changes the 0-th coefficient by -f(a) * sqrt(d_int + 1).
    c_expected = c_f.copy()
    c_expected[0] -= f_a_val * np.sqrt(d_int + 1)

    is_shifted_correctly = np.allclose(c_int_df, c_expected)
    print(f"S_T @ (D_T @ c_f) == c_f - f(a)? {is_shifted_correctly}")


if __name__ == "__main__":
    #G_T = diff_matrix(deg=3, deg_out=None)
    #print(G_T)

    verify_matrices()

    S_T = intg_matrix(a=-1.0, deg=3, deg_out=3)
    print(S_T)

    nonzero = np.count_nonzero(S_T)
    sparsity = 1 - nonzero / S_T.size
    print(f"S_T has {nonzero} non-zero entries out of {S_T.size} total entries. Sparsity: {sparsity:.4f}")
    print(np.linalg.norm(S_T, ord=2))

