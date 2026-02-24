import functools

from src.utils import encoding
from src.utils.basic_operators import multiply_matrix
import numpy as np

def evaluate_sem_encoding(psi_sol: np.ndarray,
                          deg: int,
                          deg_out: int,
                          endpoints: np.ndarray,
                          x_eval_list: np.ndarray,
                          scaling_factor: float = 1.0) -> np.ndarray:

    assert endpoints.ndim == 2
    num_elements = endpoints.shape[0]
    assert len(psi_sol) == num_elements * (deg + 1)

    lo = endpoints[:, 0]
    hi = endpoints[:, 1]

    # Reshape psi_sol to (num_elements, deg+1)
    psi_sol = psi_sol.reshape((num_elements, deg + 1))

    # Evaluate at the provided x_list
    f_eval_list = []
    for x_eval in x_eval_list:
        # Find the element containing x
        e_s = None
        for e in range(num_elements):
            if lo[e] <= x_eval <= hi[e]:
                e_s = e
                break
        if e_s is None:
            raise ValueError(f"Point x={x_eval} is not found in the mesh.")
        xi_eval = (2 * x_eval - (lo[e_s] + hi[e_s])) / (hi[e_s] - lo[e_s])  # Map to [-1, 1]

        # Evaluate the solution at x using the Chebyshev expansion for element e_s
        lo, hi = endpoints[e_s]
        psi_e = psi_sol[e_s]

        tau_e = encoding.chebyshev_encoding(deg=deg_out, x=xi_eval)

        M1 = multiply_matrix.M_x_power(p=0, deg=deg, deg_out=deg_out)

        if deg_out != deg:
            f_e = np.dot(tau_e, M1 @ psi_e)
        else:
            f_e = np.dot(tau_e, psi_e)

        f_eval_list.append(f_e)

    f_eval_list = np.array(f_eval_list) * scaling_factor
    return f_eval_list


def evaluate_multivar_sem_encoding(psi_sol: np.ndarray,
                          deg: int,
                          deg_out: int,
                          endpoints: np.ndarray,
                          coords_eval_list: np.ndarray,
                          scaling_factor: float = 1.0) -> np.ndarray:
    assert endpoints.ndim == 3
    num_elements = endpoints.shape[0]
    num_vars = endpoints.shape[2]
    assert len(psi_sol) == num_elements * (deg + 1)** num_vars

    lo = endpoints[:, 0, :]
    hi = endpoints[:, 1, :]

    # Reshape psi_sol to (num_elements, (deg+1)^num_vars)
    psi_sol = psi_sol.reshape((num_elements, (deg + 1) ** num_vars))

    # Build the full tensor-product lift matrix only if needed
    lift = None
    if deg_out != deg:
        # Precompute 1D lift matrix deg -> deg_out
        M1_1d = multiply_matrix.M_x_power(p=0, deg=deg, deg_out=deg_out)
        lift = functools.reduce(np.kron, [M1_1d] * num_vars)

    # Evaluate at the provided coordinate list
    f_eval_list = []
    for coords_eval in coords_eval_list:
        coords_eval = np.asarray(coords_eval, dtype=float)
        if coords_eval.shape != (num_vars,):
            raise ValueError(
                f"Each evaluation point must have shape ({num_vars},), got {coords_eval.shape}."
            )

        # Find the element containing coords_eval
        e_s = None
        for e in range(num_elements):
            if np.all(lo[e, :] <= coords_eval) and np.all(coords_eval <= hi[e, :]):
                e_s = e
                break
        if e_s is None:
            raise ValueError(f"Point coords={tuple(coords_eval)} is not found in the mesh.")

        # Map to reference coordinates xi in [-1, 1]^d
        xi_eval = (2.0 * coords_eval - (lo[e_s, :] + hi[e_s, :])) / (hi[e_s, :] - lo[e_s, :])

        # Tensor-product Chebyshev encoding tau(xi) = kron_v tau_v(xi_v)
        tau = None
        for v in range(num_vars):
            tau_v = encoding.chebyshev_encoding(deg=deg_out, x=xi_eval[v])
            tau = tau_v if tau is None else np.kron(tau, tau_v)

        psi_e = psi_sol[e_s]

        if deg_out != deg:
            f_e = np.dot(tau, lift @ psi_e)
        else:
            f_e = np.dot(tau, psi_e)

        f_eval_list.append(f_e)

    f_eval_list = np.array(f_eval_list) * scaling_factor
    return f_eval_list


def evaluate_vector_multivar_sem_encoding(psi_sol: np.ndarray,
                                          deg: int,
                                          deg_out: int,
                                          num_components: int,
                                          endpoints: np.ndarray,
                                          coords_eval_list: np.ndarray,
                                          scaling_factor: float = 1.0) -> np.ndarray:
    """
    Evaluate a vector-valued SEM solution at a list of physical coordinates.

    Parameters
    ----------
    psi_sol : np.ndarray
        The solution vector (flat).
        Expected ordering: |element> |component> |spatial>
    deg : int
        Polynomial degree of the solution basis.
    deg_out : int
        Degree to use for the evaluation basis (tau).
    num_components : int
        Number of vector components (e.g., 2 for [u, v]).
    endpoints : np.ndarray
        Geometry definition. Shape (num_elements, 2, num_vars).
    coords_eval_list : np.ndarray
        List of coordinates [x, y, ...] to evaluate.
    scaling_factor : float
        Optional scaling factor for the output.

    Returns
    -------
    np.ndarray
        Array of shape (num_points, num_components) containing the evaluated vector
        at each input point.
    """
    assert endpoints.ndim == 3
    num_elements = endpoints.shape[0]
    num_vars = endpoints.shape[2]

    spatial_dof = (deg + 1) ** num_vars
    expected_len = num_elements * num_components * spatial_dof

    if len(psi_sol) != expected_len:
        raise ValueError(f"psi_sol length {len(psi_sol)} does not match expected length {expected_len} "
                         f"for {num_elements} elements, {num_components} components, "
                         f"and {spatial_dof} spatial DOFs.")

    lo = endpoints[:, 0, :]
    hi = endpoints[:, 1, :]

    # Reshape psi_sol to (num_elements, num_components, spatial_dof)
    # This matches the |element>|component>|spatial> structure
    psi_reshaped = psi_sol.reshape((num_elements, num_components, spatial_dof))

    # Precompute Lift Matrix if deg_out != deg
    # This maps coefficients from degree 'deg' to the evaluation basis 'deg_out'
    lift_op = None
    if deg_out != deg:
        # 1D Identity-like lift
        M1_1d = multiply_matrix.M_x_power(p=0, deg=deg, deg_out=deg_out)
        # Tensor product for nD
        lift_op = functools.reduce(np.kron, [M1_1d] * num_vars)

    f_eval_list = []

    for coords_eval in coords_eval_list:
        coords_eval = np.asarray(coords_eval, dtype=float)

        # 1. Locate the Element
        e_s = None
        for e in range(num_elements):
            if np.all(lo[e, :] <= coords_eval) and np.all(coords_eval <= hi[e, :]):
                e_s = e
                break

        if e_s is None:
            raise ValueError(f"Point coords={tuple(coords_eval)} is not found in the mesh.")

        # 2. Map to Reference Coordinates [-1, 1]
        xi_eval = (2.0 * coords_eval - (lo[e_s, :] + hi[e_s, :])) / (hi[e_s, :] - lo[e_s, :])

        # 3. Construct Tensor-Product Basis Vector tau(xi)
        # tau shape: ((deg_out+1)**num_vars, )
        tau = None
        for v in range(num_vars):
            tau_v = encoding.chebyshev_encoding(deg=deg_out, x=xi_eval[v])
            tau = tau_v if tau is None else np.kron(tau, tau_v)

        # 4. Extract Solution Coefficients for this Element
        # psi_e shape: (num_components, spatial_dof)
        psi_e = psi_reshaped[e_s]

        # 5. Compute Dot Product for each Component
        # Result vector shape: (num_components,)
        if deg_out != deg:
            # Apply lift: psi_lifted = psi_e @ lift_op.T
            # Then dot: psi_lifted @ tau
            # Equivalent to: psi_e @ (lift_op.T @ tau)
            # Or simply: psi_e @ (lifted basis)
            # lift_op shape is (spatial_out, spatial_in)

            # Efficient calc: transform the coefficients first or the basis?
            # Usually basis is smaller vector, but here we have multiple components.
            # psi_e (C, N_in) . lift (N_out, N_in)^T  -> (C, N_out)
            # (C, N_out) . tau (N_out) -> (C,)

            # Using matrix multiplication:
            # psi_lifted = np.dot(psi_e, lift_op.T)
            # val = np.dot(psi_lifted, tau)

            # Optimized: np.dot(psi_e, np.dot(lift_op.T, tau))
            basis_at_point = np.dot(lift_op.T, tau)
            f_val = np.dot(psi_e, basis_at_point)
        else:
            f_val = np.dot(psi_e, tau)

        f_eval_list.append(f_val)

    return np.array(f_eval_list) * scaling_factor