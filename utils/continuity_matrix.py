import numpy as np
from utils import boundary_matrix, multiply_matrix, derivative_matrix

def generate_C0(N: int, deg: int, deg_out: int | None = None):
    """
    Generate C0 matrix which enforces C0 continuity between elements.

    Parameters
    ----------
    N : int
        The number of elements in the domain.
    deg : int
        Degree of Chebyshev polynomials. Defines the shape of C0 matrix, which is (N*(deg+1),N*(deg+1))
    deg_out : int
        If deg_out exists, it changes the way the matrix is constructed. It will be used to show that multiply matrices are used in the problem.

    Returns
    -------
    C0 : np.array(dtype=float)
        Matrix that enforces C0 continuity
    """
    if deg_out is None:
        Bn = boundary_matrix.zero_value_boundary_matrix(deg, -1)
        Bp = boundary_matrix.zero_value_boundary_matrix(deg, 1)
    else:
        M1 = multiply_matrix.M_x_power(deg, 0, deg_out)
        Bn = boundary_matrix.zero_value_boundary_matrix(deg_out, -1) @ M1
        Bp = boundary_matrix.zero_value_boundary_matrix(deg_out, 1) @ M1

    BnTBn = Bn.T @ Bn
    BpTBp = Bp.T @ Bp
    BpTBn = Bp.T @ Bn
    BnTBp = Bn.T @ Bp

    C0 = np.zeros((N*(deg+1),N*(deg+1)))

    for i in range(N):
        C0[i*(deg+1):(i+1)*(deg+1),
          i*(deg+1):(i+1)*(deg+1)] = BnTBn + BpTBp
        if i < N-1:
            C0[(i+1)*(deg+1):(i+2)*(deg+1),
              i*(deg+1):(i+1)*(deg+1)] = -BnTBp
            C0[i*(deg+1):(i+1)*(deg+1),
              (i+1)*(deg+1):(i+2)*(deg+1)] = -BpTBn

    C0[0:1*(deg+1),
       0:1*(deg+1)] -= BnTBn
    C0[(N-1)*(deg+1):N*(deg+1),
       (N-1)*(deg+1):N*(deg+1)] -= BpTBp

    return C0

def generate_C1(N: int, deg: int, deg_out: int | None = None):
    """
    Generate C1 matrix which enforces C1 continuity between elements.

    Parameters
    ----------
    N : int
        The number of elements in the domain.
    deg : int
        Degree of Chebyshev polynomials. Defines the shape of C1 matrix, which is (N*(deg+1),N*(deg+1))
    deg_out : int
        If deg_out exists, it changes the way the matrix is constructed. It will be used to show that multiply matrices are used in the problem.

    Returns
    -------
    C1 : np.array(dtype=float)
        Matrix that enforces C1 continuity
    """
    GT = derivative_matrix.chebyshev_diff_matrix(deg=deg)

    if deg_out is None:
        Bn = boundary_matrix.zero_value_boundary_matrix(deg, -1) @ GT
        Bp = boundary_matrix.zero_value_boundary_matrix(deg, 1) @ GT
    else:
        M1 = multiply_matrix.M_x_power(deg, 0, deg_out)
        Bn = boundary_matrix.zero_value_boundary_matrix(deg_out, -1) @ M1 @ GT
        Bp = boundary_matrix.zero_value_boundary_matrix(deg_out, 1) @ M1 @ GT

    BnTBn = Bn.T @ Bn
    BpTBp = Bp.T @ Bp
    BpTBn = Bp.T @ Bn
    BnTBp = Bn.T @ Bp

    C1 = np.zeros((N*(deg+1),N*(deg+1)))

    for i in range(N):
        C1[i*(deg+1):(i+1)*(deg+1),
          i*(deg+1):(i+1)*(deg+1)] = BnTBn + BpTBp
        if i < N-1:
            C1[(i+1)*(deg+1):(i+2)*(deg+1),
              i*(deg+1):(i+1)*(deg+1)] = -BnTBp
            C1[i*(deg+1):(i+1)*(deg+1),
              (i+1)*(deg+1):(i+2)*(deg+1)] = -BpTBn

    C1[0:1*(deg+1),
       0:1*(deg+1)] -= BnTBn
    C1[(N-1)*(deg+1):N*(deg+1),
       (N-1)*(deg+1):N*(deg+1)] -= BpTBp

    return C1
