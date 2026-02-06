import numpy as np

import boundary_matrix, derivative_matrix

def boundary_continuity_matrice_1D(type: int, M: int, deg: int, deg_out: int = None):
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
    B_left = boundary_matrix.build_boundary_matrix(type, deg, x=-1, deg_out=deg_out)
    B_right = boundary_matrix.build_boundary_matrix(type, deg, x=1, deg_out=deg_out)

    C = np.zeros(((deg+1)*M, (deg+1)*M))
    for i in range(M):
        for j in range(M):
            if i == j - 1:
                C_ij = -B_right.T @ B_left
            elif i == j:
                C_ij = B_right.T @ B_right * (i <= M-2) + B_left.T @ B_left * (i >= 1)
            elif i == j + 1:
                C_ij = -B_left.T @ B_right
            else:
                C_ij = np.zeros((deg+1, deg+1))
            C[i*(deg+1) : (i+1)*(deg+1), j*(deg+1) : (j+1)*(deg+1)] = C_ij
    return C

if __name__ == '__main__':
    M = 2
    n = 1
    deg = 2**n - 1
    C0 = boundary_continuity_matrice_1D('value', M, deg)
    C1 = boundary_continuity_matrice_1D('derivative', M, deg)
    print("C0 shape:", C0.shape)
    print("C1 shape:", C1.shape)
    print("Commutator [C0,C1] norm:", np.linalg.norm(C0 @ C1 - C1 @ C0))

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

    ## Check if the lowest eigenspace of C0 and C1 commute
    eigvecs_0_zero = eigvecs_0[:, :zero_eigvals_0]
    eigvecs_1_zero = eigvecs_1[:, :zero_eigvals_1]

    print(eigvecs_0_zero)
    print(eigvecs_1_zero)

    A = eigvecs_0_zero @ eigvecs_0_zero.T
    B = eigvecs_1_zero @ eigvecs_1_zero.T
    print("Commutator [P_C0, P_C1] norm:", np.linalg.norm(A@B - B@A))







