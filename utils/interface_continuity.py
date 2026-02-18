import numpy as np

from src.utils import boundary_matrix, derivative_matrix, multiply_matrix

def boundary_continuity_matrice(type: str, M: int, deg: int, deg_out: int = None):
    """Assemble the 1D continuity penalty matrix across element interfaces.

    This function constructs a global block matrix enforcing continuity conditions
    between adjacent 1D spectral (Chebyshev) elements. Continuity is imposed using
    boundary matrices at the left (x = -1) and right (x = 1) endpoints of each element.

    The choice of boundary operator depends on `type`:

    - `type = 'value'` enforces C^0 continuity (function values match).
    - `type = 'derivative'` enforces C^1 continuity (first derivatives match).

    Parameters
    ----------
    type:
        Boundary condition type. Must be `'value'` or `'derivative'`, forwarded to
        `boundary_matrix.build_boundary_matrix`.
    M:
        Number of 1D elements.
    deg:
        Polynomial degree per element. Each element has (deg+1) Chebyshev modes.
    deg_out:
        Optional output degree for the boundary matrices. If provided, boundary
        operators are lifted accordingly before assembly.

    Returns
    -------
    numpy.ndarray
        Global continuity matrix of shape ((deg+1)*M, (deg+1)*M).
        The matrix is symmetric positive semidefinite; its nullspace corresponds
        to globally continuous functions (or derivatives).

    Notes
    -----
    - For `M = 1`, the matrix is identically zero.
    - The lowest eigenvalue is expected to be zero, with multiplicity corresponding
      to the dimension of the globally continuous subspace.
    """
    if deg_out is None:
        deg_out = deg

    B_left = boundary_matrix.build_boundary_matrix(type, deg, x=-1, deg_out=deg_out)
    B_right = boundary_matrix.build_boundary_matrix(type, deg, x=1, deg_out=deg_out)

    B_ll = B_left.T @ B_left
    B_rr = B_right.T @ B_right
    B_lr = B_left.T @ B_right
    B_rl = B_right.T @ B_left

    C = np.zeros(((deg+1)*M, (deg+1)*M))

    for i in range(M):
        for j in range(M):
            if i == j - 1:
                C_ij = - B_rl
            elif i == j:
                C_ij = B_rr * (i < M-1) + B_ll * (i > 0)
            elif i == j + 1:
                C_ij = - B_lr
            else:
                C_ij = np.zeros((deg+1, deg+1))

            C[i*(deg+1) : (i+1)*(deg+1), j*(deg+1) : (j+1)*(deg+1)] = C_ij
    return C

def boundary_continuity_matrice_alternative(type: str, M: int, deg: int, deg_out: int = None):
    """Assemble the 1D continuity penalty matrix across element interfaces.

    This function constructs a global block matrix enforcing continuity conditions
    between adjacent 1D spectral (Chebyshev) elements. Continuity is imposed using
    boundary matrices at the left (x = -1) and right (x = 1) endpoints of each element.

    The choice of boundary operator depends on `type`:

    - `type = 'value'` enforces C^0 continuity (function values match).
    - `type = 'derivative'` enforces C^1 continuity (first derivatives match).

    Parameters
    ----------
    type:
        Boundary condition type. Must be `'value'` or `'derivative'`, forwarded to
        `boundary_matrix.build_boundary_matrix`.
    M:
        Number of 1D elements.
    deg:
        Polynomial degree per element. Each element has (deg+1) Chebyshev modes.
    deg_out:
        Optional output degree for the boundary matrices. If provided, boundary
        operators are lifted accordingly before assembly.

    Returns
    -------
    numpy.ndarray
        Global continuity matrix of shape ((deg+1)*M, (deg+1)*M).
        The matrix is symmetric positive semidefinite; its nullspace corresponds
        to globally continuous functions (or derivatives).

    Notes
    -----
    - For `M = 1`, the matrix is identically zero.
    - The lowest eigenvalue is expected to be zero, with multiplicity corresponding
      to the dimension of the globally continuous subspace.
    """
    if deg_out is None:
        deg_out = deg

    B_left = boundary_matrix.build_boundary_matrix(type, deg, x=-1, deg_out=deg_out)
    B_right = boundary_matrix.build_boundary_matrix(type, deg, x=1, deg_out=deg_out)

    #  H = sum_e C_{e, e+1}^T C_{e, e+1}, where
    #  C_{e, e+1} = |e\rangle\langle e| \otimes B_n(1) - |e\rangle\langle e+1| \otimes B_n(-1)
    C = np.zeros(((deg+1)*M, (deg+1)*M))

    for e in range(M-1):
        C_e = np.zeros(((deg_out+1)*M, (deg+1)*M))
        C_e[e*(deg_out+1) : (e+1)*(deg_out+1), e*(deg+1) : (e+1)*(deg+1)] = B_right
        C_e[e*(deg_out+1) : (e+1)*(deg_out+1), (e+1)*(deg+1) : (e+2)*(deg+1)] = -B_left

        C += C_e.T @ C_e

    return C


# --- Multivariate nD continuity penalty matrix ---

def multivar_boundary_continuity_matrix(
    type: str,
    M_list: list,
    deg: int,
    deg_out: int = None,
):
    """Assemble the nD continuity penalty matrix across element interfaces.

    This is the nD analogue of `boundary_continuity_matrice()` /
    `boundary_continuity_matrice_alternative()` for an axis-aligned tensor-product
    rectangular mesh.

    Mesh representation (nD):
      - `M_list[d]`: number of elements along dimension d
      - Elements are indexed by a multi-index (i0, i1, ..., i_{n-1}) with
        0 <= i_d < M_list[d]. A flat element index is obtained using row-major strides.

    Continuity is enforced across every interior face between adjacent elements:
      - For each dimension v and each interface between element i_v and i_v+1,
        we penalize the jump of the boundary operator:
            B_v( +1 ) u_{left}  -  B_v( -1 ) u_{right}
        where B_v acts as a 1D boundary operator in dimension v and as an
        identity-lift in all other dimensions.

    Parameters
    ----------
    type:
        Boundary condition type forwarded to `boundary_matrix.build_boundary_matrix`.
        Must be `'value'` or `'derivative'`.
    M_list:
        List of number of elements per dimension.
    deg:
        Polynomial degree per element per dimension. Each element has (deg+1)^n modes.
    deg_out:
        Optional output degree for the boundary matrices. If provided, boundary
        operators are lifted accordingly before assembly.

    Returns
    -------
    numpy.ndarray
        Global continuity penalty matrix of shape (M*(deg+1)^n, M*(deg+1)^n),
        where M = prod_d N[d]. The matrix is symmetric positive semidefinite.

    Notes
    -----
    - This construction matches the 1D formula:
        C = sum_interfaces C_face^T C_face,
      but is implemented by assembling the equivalent block contributions.
    - For large meshes / degrees, this dense assembly is expensive. If needed,
      this can be rewritten using sparse matrices.
    """
    if deg_out is None:
        deg_out = deg

    num_variables = len(M_list)
    if num_variables == 0:
        raise ValueError("M_list must contain at least one dimension")

    N = list(M_list)
    if any(n_d <= 0 for n_d in N):
        raise ValueError(f"Invalid M_list: each entry must be >= 1, got M_list={M_list}")

    # Row-major strides: last dimension varies fastest.
    strides = [1] * num_variables
    for d in range(num_variables - 2, -1, -1):
        strides[d] = strides[d + 1] * N[d + 1]

    def flat_index(idx):
        return int(sum(idx[d] * strides[d] for d in range(num_variables)))

    M = int(np.prod(N))
    in_block = (deg + 1) ** num_variables

    # 1D boundary operators on reference interval endpoints.
    B_left_1d = boundary_matrix.build_boundary_matrix(type, deg, x=-1, deg_out=deg_out)
    B_right_1d = boundary_matrix.build_boundary_matrix(type, deg, x=1, deg_out=deg_out)

    # Identity-like lift for unconstrained dimensions
    M1 = multiply_matrix.M_x_power(deg, 0, deg_out)

    # Precompute face operators for each dimension v:
    #   B_face_right[v] corresponds to xi_v = +1 on the left element
    #   B_face_left[v]  corresponds to xi_v = -1 on the right element
    out_block = (deg_out + 1) ** num_variables
    B_face_right = []
    B_face_left = []

    for v in range(num_variables):
        B_r = None
        B_l = None
        for d in range(num_variables):
            if d == v:
                Br_d = B_right_1d
                Bl_d = B_left_1d
            else:
                Br_d = M1
                Bl_d = M1

            B_r = Br_d if B_r is None else np.kron(B_r, Br_d)
            B_l = Bl_d if B_l is None else np.kron(B_l, Bl_d)

        if B_r.shape != (out_block, in_block) or B_l.shape != (out_block, in_block):
            raise ValueError(
                f"Internal shape error for v={v}: got B_r {B_r.shape}, B_l {B_l.shape}, "
                f"expected ({out_block}, {in_block})."
            )

        B_face_right.append(B_r)
        B_face_left.append(B_l)

    # Precompute per-dimension block contributions.
    Q_rr = [B_face_right[v].T @ B_face_right[v] for v in range(num_variables)]
    Q_ll = [B_face_left[v].T @ B_face_left[v] for v in range(num_variables)]
    Q_rl = [B_face_right[v].T @ B_face_left[v] for v in range(num_variables)]
    Q_lr = [B_face_left[v].T @ B_face_right[v] for v in range(num_variables)]

    C = np.zeros((in_block * M, in_block * M))

    # Loop over all interior interfaces along each dimension.
    # For each interface (left element e, right element f):
    #   add Q_rr to (e,e), Q_ll to (f,f), -Q_rl to (e,f), -Q_lr to (f,e).
    for v in range(num_variables):
        # Iterate over all element multi-indices, but only those with idx[v] < N[v]-1
        # so that the neighbor idx[v]+1 exists.
        # We generate indices via nested loops using np.ndindex.
        for idx in np.ndindex(*N):
            if idx[v] >= N[v] - 1:
                continue

            idx_nb = list(idx)
            idx_nb[v] += 1
            idx_nb = tuple(idx_nb)

            e = flat_index(idx)
            f = flat_index(idx_nb)

            e0 = e * in_block
            e1 = (e + 1) * in_block
            f0 = f * in_block
            f1 = (f + 1) * in_block

            C[e0:e1, e0:e1] += Q_rr[v]
            C[f0:f1, f0:f1] += Q_ll[v]
            C[e0:e1, f0:f1] += -Q_rl[v]
            C[f0:f1, e0:e1] += -Q_lr[v]

    return C


if __name__ == '__main__':
    M = 4
    n = 2
    deg = 2**n - 1
    deg_out = 2**(n) - 1
    C0 = boundary_continuity_matrice('value', M, deg, deg_out=deg_out)
    C1 = boundary_continuity_matrice('derivative', M, deg, deg_out=deg_out)

    C0_new = boundary_continuity_matrice_alternative('value', M, deg, deg_out=deg_out)
    C1_new = boundary_continuity_matrice_alternative('derivative', M, deg, deg_out=deg_out)
    print("Difference in C0:", np.linalg.norm(C0 - C0_new))
    print("Difference in C1:", np.linalg.norm(C1 - C1_new))

    print("C0 shape:", C0.shape)
    print("C1 shape:", C1.shape)
    print("Commutator [C0,C1] norm:", np.linalg.norm(C0 @ C1 - C1 @ C0))
    print("Commutator [C0_new,C1_new] norm:", np.linalg.norm(C0_new @ C1_new - C1_new @ C0_new))

    ## Check lowest eigenvalue and the rank of the eigenspace corresponding to the lowest eigenvalue.
    # We expect the lowest eigenvalue to be 0.
    eigvals_0, eigvecs_0 = np.linalg.eigh(C0)
    eigvals_1, eigvecs_1 = np.linalg.eigh(C1)
    print(eigvals_0[eigvals_0 > 1e-10])
    print(eigvals_1[eigvals_1 > 1e-10])

    tol = 1e-10
    zero_eigvals_0 = np.sum(eigvals_0 < tol)
    zero_eigvals_1 = np.sum(eigvals_1 < tol)
    print(f"Number of zero eigenvalues in C0: {zero_eigvals_0}")
    print(f"Number of zero eigenvalues in C1: {zero_eigvals_1}")

    print("Norm of C0 and C1", np.linalg.norm(C0), np.linalg.norm(C1))

    eigvecs_0_zero = eigvecs_0[:, :zero_eigvals_0]
    eigvecs_1_zero = eigvecs_1[:, :zero_eigvals_1]



    ## Plotting the piecewise polynomial functions corresponding to the lowest eigenspace of C0 and C1.
    import matplotlib.pyplot as plt
    from src.utils import encoding

    C = C0_new + C1_new
    eigvals, eigvecs = np.linalg.eigh(C)
    print(eigvals)
    tol = 1e-10
    zero_eigvals = np.sum(eigvals < tol)
    print(f"Number of zero eigenvalues in C0 + C1: {zero_eigvals}")
    eigvecs_zero = eigvecs[:, :zero_eigvals]

    x_plot = np.linspace(-1, 1, 1000)

    for i in range(zero_eigvals_0):
        f_i = []
        for x in x_plot:
            e = min(int((x + 1) / 2 * M), M-1)
            xi = (2 * x - (2*e/M - 1 + 2*(e+1)/M - 1)) / (2/M)  # Map to [-1, 1]
            f_i.append(np.dot(eigvecs_0_zero[:, i][e*(deg+1):(e+1)*(deg+1)], encoding.chebyshev_encoding(deg, xi)))
        plt.plot(x_plot, f_i, label=f"C0 eig {i}")

    # for i in range(zero_eigvals_1):
    #     f_i = []
    #     for x in x_plot:
    #         e = min(int((x + 1) / 2 * M), M-1)
    #         xi = (2 * x - (2*e/M - 1 + 2*(e+1)/M - 1)) / (2/M)  # Map to [-1, 1]
    #         f_i.append(np.dot(eigvecs_1_zero[:, i][e*(deg+1):(e+1)*(deg+1)], encoding.chebyshev_encoding(deg, xi)))
    #     plt.plot(x_plot, f_i, label=f"C1 eig {i}")

    # for i in range(zero_eigvals):
    #     f_i = []
    #     for x in x_plot:
    #         e = min(int((x + 1) / 2 * M), M-1)
    #         xi = (2 * x - (2*e/M - 1 + 2*(e+1)/M - 1)) / (2/M)  # Map to [-1, 1]
    #         f_i.append(np.dot(eigvecs_zero[:, i][e*(deg+1):(e+1)*(deg+1)], encoding.chebyshev_encoding(deg, xi)))
    #     plt.plot(x_plot, f_i, label=f"C eig {i}", alpha=0.75)

    plt.legend()
    plt.title("Functions in the lowest eigenspace of C0 and C1")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.show()






