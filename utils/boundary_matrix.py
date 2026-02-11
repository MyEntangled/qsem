import numpy as np
from src.utils import multiply_matrix, encoding, derivative_matrix

def zero_value_boundary_matrix(deg: int, x_z: float, deg_out: int = None):
    """Construct the boundary matrix enforcing a homogeneous value condition f(x_z) = 0.

        In the project's notation, the condition
            f(x_z) = 0
        is represented as
            sqrt(eta) * <tau(x)|_deg  B(x_z) |psi> = 0,
        and the matrix is chosen as
            B(x_z) = sqrt(deg+1) |0><tau(x_z)|_deg.

        Parameters
        ----------
        deg:
            Input Chebyshev degree d. The input space has dimension (d+1).
        x_z:
            Point in [-1, 1] where the value constraint is imposed.
        deg_out:
            Optional output degree. If provided, the matrix is lifted to shape
            (deg_out+1, deg+1) via `multiply_matrix.M_x_power(deg, p=0, deg_out)`.

        Returns
        -------
        numpy.ndarray
            Boundary matrix B. Shape is (deg+1, deg+1) if `deg_out is None`, otherwise
            (deg_out+1, deg+1).

        Raises
        ------
        ValueError
            If `deg_out < deg`.
        """
    d = deg
    B = np.zeros((d + 1, d + 1))

    tau = encoding.chebyshev_encoding(deg, x_z)

    B[0, :] = tau
    B *= np.sqrt(d + 1)

    if deg_out is None:
        return B
    else:
        if deg_out < deg:
            raise ValueError("deg_out must be greater than or equal to deg.")
        M1 = multiply_matrix.M_x_power(deg=deg, p=0, deg_out=deg_out)
        return M1 @ B

def zero_derivative_boundary_matrix(deg: int, x_m: float, deg_out: int = None):
    """Construct the boundary matrix enforcing a homogeneous derivative condition f'(x_m) = 0.

        This builds the matrix \\hat{B}(x_m) that represents evaluation of the derivative at `x_m`.
        Internally it composes the value boundary matrix with the Chebyshev differentiation matrix:

            \\hat{B}(x_m) = B(x_m) @ G^T,

        where `G^T` is returned by `derivative_matrix.chebyshev_diff_matrix(deg)`.

        Parameters
        ----------
        deg:
            Input Chebyshev degree d.
        x_m:
            Point in [-1, 1] where the derivative constraint is imposed.
        deg_out:
            Optional output degree. If provided, the matrix is lifted to shape
            (deg_out+1, deg+1) via `multiply_matrix.M_x_power(deg, p=0, deg_out)`.

        Returns
        -------
        numpy.ndarray
            Derivative boundary matrix \\hat{B}. Shape is (deg+1, deg+1) if `deg_out is None`,
            otherwise (deg_out+1, deg+1).

        Raises
        ------
        ValueError
            If `deg_out < deg`.
        """

    B = zero_value_boundary_matrix(deg, x_m)
    B_hat = B @ derivative_matrix.chebyshev_diff_matrix(deg, deg_out=None)

    if deg_out is None:
        return B_hat
    else:
        if deg_out < deg:
            raise ValueError("deg_out must be greater than or equal to deg.")
        M1 = multiply_matrix.M_x_power(deg=deg, p=0, deg_out=deg_out)
        return M1 @ B_hat

def regular_value_boundary_matrix(deg: int, x_s: float, y_s: float, deg_out: int = None):
    """Construct a boundary matrix enforcing an inhomogeneous value condition f(x_s) = y_s.

        The matrix is the normalized version of the homogeneous value matrix:

            D^{(0)}(x_s) = B(x_s) / y_s,

        so that the constraint can be written (in the project's convention) as
            sqrt(eta) * <tau(x)|_deg  D^{(0)}(x_s) |psi> = 1.

        Parameters
        ----------
        deg:
            Input Chebyshev degree d.
        x_s:
            Point in [-1, 1] where the value is prescribed.
        y_s:
            Prescribed value f(x_s). Must be nonzero.
        deg_out:
            Optional output degree. If provided, the matrix is lifted to shape
            (deg_out+1, deg+1) via `multiply_matrix.M_x_power(deg, p=0, deg_out)`.

        Returns
        -------
        numpy.ndarray
            Boundary matrix D^{(0)}. Shape is (deg+1, deg+1) if `deg_out is None`, otherwise
            (deg_out+1, deg+1).

        Raises
        ------
        ValueError
            If `deg_out < deg`.

        Notes
        -----
        No explicit check is performed for `y_s == 0`; passing zero will raise a NumPy warning and
        produce infinities.
        """

    B = zero_value_boundary_matrix(deg, x_s)
    D0 = B / y_s

    if deg_out is None:
        return D0
    else:
        if deg_out < deg:
            raise ValueError("deg_out must be greater than or equal to deg.")
        M1 = multiply_matrix.M_x_power(deg=deg, p=0, deg_out=deg_out)
        return M1 @ D0

def regular_derivative_boundary_matrix(deg: int, x_s: float, t_s: float, deg_out: int = None):
    """Construct a boundary matrix enforcing an inhomogeneous derivative condition f'(x_s) = t_s.

        This is the normalized version of the homogeneous derivative matrix:

            D^{(1)}(x_s) = \\hat{B}(x_s) / t_s,

        so that the constraint can be written (in the project's convention) as
            sqrt(eta) * <tau(x)|_deg  D^{(1)}(x_s) |psi> = 1.

        Parameters
        ----------
        deg:
            Input Chebyshev degree d.
        x_s:
            Point in [-1, 1] where the derivative is prescribed.
        t_s:
            Prescribed derivative value f'(x_s). Must be nonzero.
        deg_out:
            Optional output degree. If provided, the matrix is lifted to shape
            (deg_out+1, deg+1) via `multiply_matrix.M_x_power(deg, p=0, deg_out)`.

        Returns
        -------
        numpy.ndarray
            Boundary matrix D^{(1)}. Shape is (deg+1, deg+1) if `deg_out is None`, otherwise
            (deg_out+1, deg+1).

        Raises
        ------
        ValueError
            If `deg_out < deg`.

        Notes
        -----
        No explicit check is performed for `t_s == 0`; passing zero will raise a NumPy warning and
        produce infinities.
        """

    B_hat = zero_derivative_boundary_matrix(deg, x_s)
    D1 = B_hat / t_s
    if deg_out is None:
        return D1
    else:
        if deg_out < deg:
            raise ValueError("deg_out must be greater than or equal to deg.")
        M1 = multiply_matrix.M_x_power(deg=deg, p=0, deg_out=deg_out)
        return M1 @ D1

def build_boundary_matrix(type: str, deg: int, x: float, y: float = None, deg_out: int = None):
    """Factory for boundary matrices (value or derivative; homogeneous or inhomogeneous).

    Parameters
    ----------
    type:
        Either `'value'` or `'derivative'`.
    deg:
        Input Chebyshev degree d.
    x:
        Point in [-1, 1] where the boundary constraint is imposed.
    y:
        If `None`, construct a homogeneous constraint (equal to 0).
        If provided, construct an inhomogeneous constraint (equal to `y`).
        For `type='value'`, `y` corresponds to f(x). For `type='derivative'`, `y` corresponds
        to f'(x).
    deg_out:
        Optional output degree for lifting the operator.

    Returns
    -------
    numpy.ndarray
        The requested boundary matrix.

    Raises
    ------
    ValueError
        If `type` is not one of `'value'` or `'derivative'`, or if `deg_out < deg`.
    """

    if type == 'value':
        if y is None:
            B = zero_value_boundary_matrix(deg, x, deg_out)
        else:
            B = regular_value_boundary_matrix(deg, x, y, deg_out)
    elif type == 'derivative':
        if y is None:
            B = zero_derivative_boundary_matrix(deg, x, deg_out)
        else:
            B = regular_derivative_boundary_matrix(deg, x, y, deg_out)
    else:
        raise ValueError("type must be either 'value' or 'derivative'.")
    return B

def sem_boundary_hamiltonian(deg: int, deg_out: int, endpoints: np.ndarray, type: str, x: float):
    """Construct the boundary Hamiltonian for a single boundary constraint.

    The boundary Hamiltonian is defined as B^T B, where B is the boundary matrix corresponding
    to the specified constraint. This can be used in the semigroup formulation to enforce the
    boundary condition.

    Parameters
    ----------
    deg:
        Input Chebyshev degree d.
    deg_out:
        Output Chebyshev degree for lifting the operator.
    type:
        Either `'value'` or `'derivative'`.
    x:
        Point where the boundary constraint is imposed.

    Returns
    -------
    numpy.ndarray
        The boundary Hamiltonian matrix B^T B.
    """
    assert endpoints.ndim == 2
    num_elements = endpoints.shape[0]
    lo = endpoints[:, 0]
    hi = endpoints[:, 1]

    # Check that x is within the domain
    e_s = None
    for e in range(num_elements):
        if lo[e] <= x <= hi[e]:
            e_s = e
            xi_s_in_e_s = (2 * x - (lo[e] + hi[e])) / (hi[e] - lo[e])  # Map to [-1, 1]
            break
    if e_s is None:
        raise ValueError(f"Regularization point x={x} is not found in the mesh.")

    B = build_boundary_matrix(type=type, deg=deg, x=xi_s_in_e_s, deg_out=deg_out)
    B_total = np.zeros(((deg_out + 1)*num_elements, (deg + 1)*num_elements))

    B_total[e_s*(deg_out+1):(e_s+1)*(deg_out+1), e_s*(deg+1):(e_s+1)*(deg+1)] = B

    return B_total.T @ B_total


if __name__ == '__main__':
    d_in = 33
    d_out = 135

    M1 = multiply_matrix.M_x_power(deg=d_in, p=0, deg_out=d_out)
    GT = derivative_matrix.chebyshev_diff_matrix(deg=d_in)
    B_in = zero_value_boundary_matrix(deg=d_in, x_z=-1)
    B_out = zero_value_boundary_matrix(deg=d_out, x_z=-1)

    type_1 = M1 @ B_in @ GT
    type_2 = B_out @ M1 @ GT
    type_3 = zero_derivative_boundary_matrix(d_in, -1, d_out)
    print(np.linalg.norm(type_1 - type_2), np.linalg.norm(type_2 - type_3))

    type_1 = M1 @ B_in
    type_2 = B_out @ M1
    type_3 = zero_value_boundary_matrix(d_in, -1, d_out)
    print(np.linalg.norm(type_1 - type_2), np.linalg.norm(type_2 - type_3))