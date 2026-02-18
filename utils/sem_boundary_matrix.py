import numpy as np

from src.utils import boundary_matrix, multiply_matrix


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

    B = boundary_matrix.build_boundary_matrix(type=type, deg=deg, x=xi_s_in_e_s, deg_out=deg_out)
    B_total = np.zeros(((deg_out + 1)*num_elements, (deg + 1)*num_elements))

    B_total[e_s*(deg_out+1):(e_s+1)*(deg_out+1), e_s*(deg+1):(e_s+1)*(deg+1)] = B

    return B_total.T @ B_total

def sem_multivar_boundary_hamiltonian(deg: int, deg_out: int, endpoints: np.ndarray, type: str, coords: tuple):
    """Construct the boundary Hamiltonian for multiple boundary constraints in multiple dimensions.

    Parameters
    ----------
    deg:
        Input Chebyshev degree d.
    deg_out:
        Output Chebyshev degree for lifting the operator.
    endpoints:
        Array of shape (num_elements, 2, num_variables) containing the endpoints of each element in the mesh.
    type:
        Either `'value'` or `'derivative'`.
    coords:
        Tuple of shape (num_variables,) containing the coordinates where the boundary constraint is imposed.
        An entry with value None indicates that the constraint is applied for every value of that variable (i.e., no constraint in that variable).

    Returns
    -------
    numpy.ndarray
        The boundary Hamiltonian matrix B^T B for the constraint.
    """

    # Basic shape / argument checks
    if endpoints.ndim != 3 or endpoints.shape[1] != 2:
        raise ValueError(
            "endpoints must have shape (num_elements, 2, num_variables) for the multivariate case."
        )

    num_elements, _, num_variables = endpoints.shape

    if len(coords) != num_variables:
        raise ValueError(
            f"coords must have length {num_variables} (num_variables), got {len(coords)}."
        )

    # Endpoints per element per variable
    lo = endpoints[:, 0, :]  # (E, d)
    hi = endpoints[:, 1, :]  # (E, d)

    # Identify which elements contain the constrained coordinate(s).
    # A None entry in coords means "apply throughout" that variable, so it does not
    # constrain element selection.
    selected_elements = []
    xi_per_element = {}  # e -> list of xi (length d), with None for unconstrained vars

    for e in range(num_elements):
        ok = True
        xi_list = [None] * num_variables
        for v in range(num_variables):
            c = coords[v]
            if c is None:
                continue
            if not (lo[e, v] <= c <= hi[e, v]):
                ok = False
                break
            # Map coordinate to reference interval [-1, 1]
            xi_list[v] = (2.0 * c - (lo[e, v] + hi[e, v])) / (hi[e, v] - lo[e, v])
        if ok:
            selected_elements.append(e)
            xi_per_element[e] = xi_list

    if len(selected_elements) == 0:
        raise ValueError(
            f"Boundary coords={coords} are not found in the mesh (for the constrained variables)."
        )

    # Build the d-dimensional boundary operator as a Kronecker product of 1D factors.
    # - For constrained variables: boundary_matrix.build_boundary_matrix(...)
    # - For unconstrained variables (coords[v] is None): identity lift deg -> deg_out

    M1 = multiply_matrix.M_x_power(deg, 0, deg_out) # Identity-like lift for unconstrained variables

    def _build_B_d(xi_list: list) -> np.ndarray:
        B_d = None
        for v in range(num_variables):
            if xi_list[v] is None:
                B_v = M1
            else:
                B_v = boundary_matrix.build_boundary_matrix(
                    type=type, deg=deg, x=xi_list[v], deg_out=deg_out
                )

            B_d = B_v if B_d is None else np.kron(B_d, B_v)
        return B_d

    in_block = (deg + 1) ** num_variables
    out_block = (deg_out + 1) ** num_variables

    B_total = np.zeros((out_block * num_elements, in_block * num_elements))

    for e in selected_elements:
        B = _build_B_d(xi_per_element[e])
        r0 = e * out_block
        r1 = (e + 1) * out_block
        c0 = e * in_block
        c1 = (e + 1) * in_block
        B_total[r0:r1, c0:c1] = B

    return B_total.T @ B_total
