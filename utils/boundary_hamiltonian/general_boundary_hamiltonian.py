import numpy as np

from src.utils.basic_operators import multiply_matrix
from src.utils.boundary_hamiltonian import simple_boundary

def sem_boundary_hamiltonian(type: str, x: float, deg: int, deg_out: int, endpoints: np.ndarray):
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

    B = simple_boundary.build_boundary_matrix(type=type, x=xi_s_in_e_s, y=None, deg=deg, deg_out=deg_out)
    B_total = np.zeros(((deg_out + 1)*num_elements, (deg + 1)*num_elements))

    B_total[e_s*(deg_out+1):(e_s+1)*(deg_out+1), e_s*(deg+1):(e_s+1)*(deg+1)] = B

    return B_total.T @ B_total


def sem_multivar_boundary_hamiltonian(type: str, coords: tuple, deg: int, deg_out: int, endpoints: np.ndarray):
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
    # - For constrained variables: simple_boundary.build_boundary_matrix(...)
    # - For unconstrained variables (coords[v] is None): identity lift deg -> deg_out

    M1 = multiply_matrix.M_x_power(0, deg, deg_out)  # Identity-like lift for unconstrained variables

    def _build_B_d(xi_list: list) -> np.ndarray:
        B_d = None
        for v in range(num_variables):
            if xi_list[v] is None:
                B_v = M1
            else:
                B_v = simple_boundary.build_boundary_matrix(type=type, x=xi_list[v], y=None, deg=deg, deg_out=deg_out)

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


def sem_vector_multivar_boundary_hamiltonian(type: str, coords: tuple, deg: int, deg_out: int, endpoints: np.ndarray,
                                             num_components: int, component_idx: int):
    """Construct the boundary Hamiltonian for a specific component of a vector-valued function.

    Memory Layout: |element> (x) |component> (x) |spatial_basis>

    Parameters
    ----------
    deg:
        Input Chebyshev degree d.
    deg_out:
        Output Chebyshev degree for lifting the operator.
    endpoints:
        Array of shape (num_elements, 2, num_variables) containing the mesh endpoints.
    type:
        Either `'value'` or `'derivative'`.
    coords:
        Tuple of shape (num_variables,) containing the constrained coordinates.
        `None` indicates no constraint along that axis.
    num_components:
        Total number of components in the vector-valued system (e.g., 2 for [u, v]).
    component_idx:
        The specific component index (0 to num_components-1) that this constraint applies to.

    Returns
    -------
    numpy.ndarray
        The boundary Hamiltonian matrix B^T B for the constraint.
    """
    if endpoints.ndim != 3 or endpoints.shape[1] != 2:
        raise ValueError(
            "endpoints must have shape (num_elements, 2, num_variables) for the multivariate case."
        )

    num_elements, _, num_variables = endpoints.shape

    if len(coords) != num_variables:
        raise ValueError(f"coords must have length {num_variables}, got {len(coords)}.")

    if not (0 <= component_idx < num_components):
        raise ValueError(f"component_idx {component_idx} out of bounds for {num_components} components.")

    lo = endpoints[:, 0, :]
    hi = endpoints[:, 1, :]

    # Identify which elements contain the constrained coordinate(s).
    selected_elements = []
    xi_per_element = {}

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
        raise ValueError(f"Boundary coords={coords} are not found in the mesh.")

    M1 = multiply_matrix.M_x_power(0, deg, deg_out)

    def _build_B_d(xi_list: list) -> np.ndarray:
        B_d = None
        for v in range(num_variables):
            if xi_list[v] is None:
                B_v = M1
            else:
                B_v = simple_boundary.build_boundary_matrix(type=type, x=xi_list[v], y=None, deg=deg, deg_out=deg_out)
            B_d = B_v if B_d is None else np.kron(B_d, B_v)
        return B_d

    # Calculate Block Sizes
    spatial_in = (deg + 1) ** num_variables
    spatial_out = (deg_out + 1) ** num_variables

    elem_in = num_components * spatial_in
    elem_out = num_components * spatial_out

    # Global Block Matrix initialization
    B_total = np.zeros((num_elements * elem_out, num_elements * elem_in))

    for e in selected_elements:
        B_scalar = _build_B_d(xi_per_element[e])

        # Determine the exact row/column indices inside the global matrix
        # Structure: Element Block -> Component Sub-block -> Spatial Operator
        row_start = e * elem_out + component_idx * spatial_out
        row_end = row_start + spatial_out

        col_start = e * elem_in + component_idx * spatial_in
        col_end = col_start + spatial_in

        B_total[row_start:row_end, col_start:col_end] = B_scalar

    return B_total.T @ B_total

def build_general_boundary_matrix(type: str, coords: float | tuple, deg: int, deg_out: int, endpoints: np.ndarray, num_components: int = None, component_idx: int = None):
    """Factory for building boundary Hamiltonians for various types of constraints.

    Parameters
    ----------
    type:
        Either 'value' or 'derivative'.
    coords:
        For 1D: a single float coordinate where the constraint is applied.
        For multivariate: a tuple of coordinates, where None indicates no constraint along that axis.
    deg:
        Input Chebyshev degree d.
    deg_out:
        Output Chebyshev degree for lifting the operator.
    endpoints:
        For multivariate cases, an array of shape (num_elements, 2, num_variables) containing the mesh endpoints.
    num_components:
        For vector-valued cases, the total number of components in the system.
    component_idx:
        For vector-valued cases, the specific component index that this constraint applies to.

    Returns
    -------
    numpy.ndarray
        The boundary matrix for the specified constraint.
    """
    if (num_components is None) ^ (component_idx is None):
        raise ValueError("Both num_components and component_idx must be provided together or not at all.")

    if not isinstance(coords, tuple) and not isinstance(coords, float):
        raise ValueError("coords must be either a float (for 1D) or a tuple of floats (for multivariate).")


    if num_components is None or num_components == 1:
        if isinstance(coords, float):
            return sem_boundary_hamiltonian(type=type, x=coords, deg=deg, deg_out=deg_out, endpoints=endpoints)
        else:
            return sem_multivar_boundary_hamiltonian(type=type, coords=coords, deg=deg, deg_out=deg_out, endpoints=endpoints)

    elif num_components >= 2:
        if isinstance(coords, float):
            return sem_vector_multivar_boundary_hamiltonian(type=type, coords=(coords,), deg=deg, deg_out=deg_out,
                                                          endpoints=endpoints, num_components=num_components,
                                                          component_idx=component_idx)
        else:
            return sem_vector_multivar_boundary_hamiltonian(type=type, coords=coords, deg=deg, deg_out=deg_out,
                                                      endpoints=endpoints, num_components=num_components,
                                                      component_idx=component_idx)

    else:
        raise ValueError("num_components must be a positive integer.")
