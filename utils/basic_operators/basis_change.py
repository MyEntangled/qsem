from src.utils.encoding import chebyshev_encoding, chebyshev_nodes
import numpy as np

# ---------- boundary functionals in tau coordinates ----------

def tau_endpoint_rows(d: int):
    """
    Return the four boundary functional rows in tau-coordinates:

      value at -1
      value at +1
      derivative at -1
      derivative at +1

    If f(x) = c^T tau(x), then
      f(-1)   = Lm_val @ c
      f(+1)   = Lp_val @ c
      f'(-1)  = Lm_der @ c
      f'(+1)  = Lp_der @ c
    """
    n = d + 1
    s = np.sqrt(2.0 / n)

    Lm_val = np.zeros(n, dtype=float)
    Lp_val = np.zeros(n, dtype=float)
    Lm_der = np.zeros(n, dtype=float)
    Lp_der = np.zeros(n, dtype=float)

    # k = 0
    Lm_val[0] = 1.0 / np.sqrt(n)
    Lp_val[0] = 1.0 / np.sqrt(n)

    # k >= 1
    ks = np.arange(1, n, dtype=float)
    signs = (-1.0) ** ks

    Lp_val[1:] = s
    Lm_val[1:] = s * signs

    # T_k'(1) = k^2,  T_k'(-1) = (-1)^(k-1) k^2
    Lp_der[1:] = s * ks**2
    Lm_der[1:] = s * ((-1.0) ** (ks - 1.0)) * ks**2

    return Lm_val, Lp_val, Lm_der, Lp_der


# ---------- linear algebra helpers ----------

def _svd_rank(s, tol=None):
    if tol is None:
        tol = s.max() * max(1, len(s)) * np.finfo(float).eps
    return int(np.sum(s > tol))


def nullspace(A: np.ndarray, tol=None) -> np.ndarray:
    """
    Orthonormal basis of null(A), as columns.
    """
    U, s, Vh = np.linalg.svd(A, full_matrices=True)
    r = _svd_rank(s, tol)
    return Vh[r:].T.copy()


def rowspace_transpose(A: np.ndarray, tol=None) -> np.ndarray:
    """
    Orthonormal basis of row(A)^T = col(A^T), as columns.
    """
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    r = _svd_rank(s, tol)
    return Vh[:r].T.copy()


# ---------- build the optimal basis in tau coordinates ----------

def optimal_basis_from_tau(d: int, kind: str, order: str = "lr") -> np.ndarray:
    """
    Return C such that

        |basis(x)> = C |tau(x)>.

    Rows of C are the coefficient vectors of the basis functions
    expressed in the tau-basis.

    kind = "mu":
        2-constraint basis
    kind = "nu":
        4-constraint Hermite basis
    """
    if kind not in {"C0", "C1"}:
        raise ValueError("kind must be 'C0' or 'C1'")

    if kind == "C0" and d < 1:
        raise ValueError("Type-C0 bubble basis requires d >= 1")
    if kind == "C1" and d < 3:
        raise ValueError("Type-C1 bubble basis requires d >= 3")

    Lm_val, Lp_val, Lm_der, Lp_der = tau_endpoint_rows(d)

    if kind == "C0":
        if order == "lr":
            # mu_0: nonzero at -1, zero at +1
            # mu_1: nonzero at +1, zero at -1
            L = np.vstack([Lm_val, Lp_val])
        elif order == "rl":
            L = np.vstack([Lp_val, Lm_val])
        else:
            raise ValueError("order must be 'lr' or 'rl'")
    else:
        # nu ordering:
        #   nu_0: left value
        #   nu_1: right value
        #   nu_2: left derivative
        #   nu_3: right derivative
        L = np.vstack([Lm_val, Lp_val, Lm_der, Lp_der])

    # Interior constraint subspace W = null(L)
    W = nullspace(L)                # columns: orthonormal basis of W

    # W^\perp = col(L^T)
    Uc = rowspace_transpose(L)      # columns: orthonormal basis of W^\perp
    Cb = L @ Uc                     # invertible small matrix

    # Canonical boundary modes in W^\perp:
    # columns b_a satisfy L b_a = e_a
    Bcols = Uc @ np.linalg.inv(Cb)

    # Normalize boundary modes to unit norm
    Bcols /= np.linalg.norm(Bcols, axis=0, keepdims=True)

    # Stack boundary rows first, interior rows second
    C = np.vstack([Bcols.T, W.T])
    return C


def optimal_tau_from_basis(d: int, kind: str, order: str = "lr") -> np.ndarray:
    """
    Return A such that

        A |basis(x)> = |tau(x)>.

    Since |basis(x)> = C |tau(x)>, we return A = C^{-1}.
    """
    C = optimal_basis_from_tau(d, kind=kind, order=order)
    A = np.linalg.inv(C)
    return A


# ---------- evaluation helpers ----------

def basis_vector(x: float, d: int, kind: str, order: str = "lr") -> np.ndarray:
    """
    Evaluate |basis(x)> = C |tau(x)>.
    """
    C = optimal_basis_from_tau(d, kind=kind, order=order)
    return C @ chebyshev_encoding(d, x)


def tau_from_basis_vector(basis_vec: np.ndarray, d: int, kind: str, order: str = "lr") -> np.ndarray:
    """
    Apply A so that A |basis> = |tau>.
    """
    A = optimal_tau_from_basis(d, kind=kind, order=order)
    return A @ basis_vec


if __name__ == "__main__":

    # ---------- diagnostics ----------
    def tau_nodes_matrix(d: int) -> np.ndarray:
        """
        U = [tau(x_0) ... tau(x_d)], shape (d+1, d+1).
        This matrix is orthogonal.
        """
        xs = chebyshev_nodes(d)
        return np.column_stack([chebyshev_encoding(d, x) for x in xs])


    def node_gram_from_basis_matrix(C: np.ndarray, d: int) -> np.ndarray:
        """
        Gram matrix G_ij = <basis(x_i) | basis(x_j)>.
        If C were orthogonal, this would be I.
        """
        U = tau_nodes_matrix(d)
        M = C @ U
        return M.T @ M


    def orthonormality_violation(C: np.ndarray, d: int):
        """
        Return Frobenius and spectral norm of G - I on the nodes.
        """
        G = node_gram_from_basis_matrix(C, d)
        E = G - np.eye(d + 1)
        fro = np.linalg.norm(E, ord="fro")
        spec = np.linalg.norm(E, ord=2)
        return fro, spec


    d = 40

    # C maps tau -> basis
    C_mu = optimal_basis_from_tau(d, kind="C0", order="lr")
    C_nu = optimal_basis_from_tau(d, kind="C1")

    # A maps basis -> tau
    A_mu = optimal_tau_from_basis(d, kind="C0", order="lr")
    A_nu = optimal_tau_from_basis(d, kind="C1")

    print("||A_mu C_mu - I|| =", np.linalg.norm(A_mu @ C_mu - np.eye(d + 1)))
    print("||A_nu C_nu - I|| =", np.linalg.norm(A_nu @ C_nu - np.eye(d + 1)))

    x = 0.37

    tau_x = chebyshev_encoding(d, x)
    mu_x = basis_vector(x, d, kind="C0")
    nu_x = basis_vector(x, d, kind="C1")

    assert np.isclose(np.linalg.norm(A_mu @ mu_x - tau_x), 0), "Something went wrong with mu recovery!"
    assert np.isclose(np.linalg.norm(A_nu @ nu_x - tau_x), 0), "Something went wrong with nu recovery!"

    fro_mu, spec_mu = orthonormality_violation(C_mu, d)
    fro_nu, spec_nu = orthonormality_violation(C_nu, d)

    print("Orthogonality violation:", fro_mu, spec_mu, fro_nu, spec_nu)

    print("||A_nu A^t_nu - I|| =", np.linalg.norm(A_nu.T @ A_nu - np.eye(d + 1)))

    # ---------- boundary checks ----------
    def boundary_data(C, d):
        Lm_val, Lp_val, Lm_der, Lp_der = tau_endpoint_rows(d)
        vals_m = C @ Lm_val   # basis values at x = -1
        vals_p = C @ Lp_val   # basis values at x = +1
        ders_m = C @ Lm_der   # basis derivatives at x = -1
        ders_p = C @ Lp_der   # basis derivatives at x = +1
        return vals_m, vals_p, ders_m, ders_p

    vals_m_mu, vals_p_mu, ders_m_mu, ders_p_mu = boundary_data(C_mu, d)
    vals_m_nu, vals_p_nu, ders_m_nu, ders_p_nu = boundary_data(C_nu, d)

    print("\nmu boundary check:")
    print("mu(-1) =", vals_m_mu)
    print("mu(+1) =", vals_p_mu)
    print("mu_0(+1), mu_1(-1) =", vals_p_mu[0], vals_m_mu[1])
    if d >= 2:
        print("mu_k(-1) for k>=2 =", vals_m_mu[2:])
        print("mu_k(+1) for k>=2 =", vals_p_mu[2:])

    print("\nnu boundary check:")
    print("nu(-1) =", vals_m_nu)
    print("nu(+1) =", vals_p_nu)
    print("nu'(-1) =", ders_m_nu)
    print("nu'(+1) =", ders_p_nu)
    print("expected nonzero entries:")
    print("  nu_0(-1) =", vals_m_nu[0])
    print("  nu_1(+1) =", vals_p_nu[1])
    print("  nu_2'(-1) =", ders_m_nu[2])
    print("  nu_3'(+1) =", ders_p_nu[3])
    if d >= 4:
        print("nu_k(-1) for k>=4 =", vals_m_nu[4:])
        print("nu_k(+1) for k>=4 =", vals_p_nu[4:])
        print("nu_k'(-1) for k>=4 =", ders_m_nu[4:])
        print("nu_k'(+1) for k>=4 =", ders_p_nu[4:])