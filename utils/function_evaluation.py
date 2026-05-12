import functools

from src.utils import encoding
from src.utils.basic_operators import multiply_matrix
from src.utils.meshing import RectMesh
import numpy as np

def evaluate_sem_encoding(psi_sol: np.ndarray,
                          deg: int,
                          deg_out: int,
                          mesh: RectMesh,
                          x_eval_list: list | np.ndarray,
                          scaling_factor: float = 1.0,
                          local_basis_transform: np.ndarray = None) -> np.ndarray:

    endpoints = mesh.endpoints
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
        e_s = mesh.find_elements((x_eval,))
        if isinstance(e_s, int):
            pass
        elif len(e_s) == 0:
            raise ValueError(f"Point x={x_eval} is not found in the mesh.")
        else: # len(e_s) > 1
            raise ValueError(f"Point x={x_eval} is found in multiple elements: {e_s}. For function evaluation, a single element should be specified.")

        xi_eval = (2 * x_eval - (lo[e_s] + hi[e_s])) / (hi[e_s] - lo[e_s])  # Map to [-1, 1]

        # Evaluate the solution at x using the Chebyshev expansion for element e_s
        lo, hi = endpoints[e_s]
        psi_e = psi_sol[e_s]
        psi_e = np.ravel(psi_e)

        tau_e = encoding.chebyshev_encoding(deg=deg_out, x=xi_eval)

        M1 = multiply_matrix.M_x_power(p=0, deg=deg, deg_out=deg_out)

        #print(tau_e.shape, C_recomb_inv.shape)

        if local_basis_transform is not None:
            psi_e = local_basis_transform.T @ psi_e

        if deg_out != deg:
            f_e = np.dot(tau_e, M1 @ psi_e)
        else:
            f_e = np.dot(tau_e, psi_e)

        f_eval_list.append(f_e)

    f_eval_list = np.array(f_eval_list) * scaling_factor
    return f_eval_list


def _evaluate_multivar_sem_encoding_inefficient_(psi_sol: np.ndarray,
                                                 deg: int,
                                                 deg_out: int,
                                                 mesh: RectMesh,
                                                 coords_eval_list: np.ndarray,
                                                 scaling_factor: float = 1,
                                                 local_basis_transform: np.ndarray = None,
                                                 local_basis_transform_joint: np.ndarray = None) -> np.ndarray:

    endpoints = mesh.endpoints
    assert endpoints.ndim == 3

    num_elements = endpoints.shape[0]
    num_vars = endpoints.shape[2]

    assert len(psi_sol) == num_elements * (deg + 1)** num_vars

    lo = endpoints[:, 0, :]
    hi = endpoints[:, 1, :]

    spatial_in = (deg + 1) ** num_vars
    if local_basis_transform_joint is not None:
        psi_sol = (psi_sol.reshape(-1, spatial_in) @ local_basis_transform_joint).flatten()
    elif local_basis_transform is not None:
        C = local_basis_transform
        C_joint = C
        for _ in range(1, num_vars):
            C_joint = np.kron(C_joint, C)
        psi_sol = (psi_sol.reshape(-1, spatial_in) @ C_joint).flatten()

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
        e_s = mesh.find_elements(coords_eval)
        if isinstance(e_s, int):
            pass
        elif len(e_s) == 0:
            raise ValueError(f"Point coords={tuple(coords_eval)} is not found in the mesh.")
        else: # len(e_s) > 1
            raise ValueError(f"Point coords={tuple(coords_eval)} is found in multiple elements: {e_s}. For function evaluation, a single element should be specified.")


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

def evaluate_multivar_sem_encoding(psi_sol: np.ndarray,
                                   deg: int,
                                   deg_out: int,
                                   mesh: RectMesh,
                                   coords_eval_list: np.ndarray,
                                   scaling_factor: float = 1,
                                   local_basis_transform: np.ndarray = None,
                                   local_basis_transform_joint: np.ndarray = None) -> np.ndarray:

    endpoints = mesh.endpoints
    assert endpoints.ndim == 3
    num_elements = endpoints.shape[0]
    num_vars = endpoints.shape[2]
    assert len(psi_sol) == num_elements * (deg + 1) ** num_vars

    lo = endpoints[:, 0, :]
    hi = endpoints[:, 1, :]

    spatial_in = (deg + 1) ** num_vars
    if local_basis_transform_joint is not None:
        psi_sol = (psi_sol.reshape(-1, spatial_in) @ local_basis_transform_joint).flatten()
    elif local_basis_transform is not None:
        C = local_basis_transform
        C_joint = C
        for _ in range(1, num_vars):
            C_joint = np.kron(C_joint, C)
        psi_sol = (psi_sol.reshape(-1, spatial_in) @ C_joint).flatten()

    # Reshape psi_sol to (num_elements, (deg+1)^num_vars)
    psi_sol = psi_sol.reshape((num_elements, (deg + 1) ** num_vars))

    # Precompute 1D lift matrix only if needed
    M1_1d = None
    if deg_out != deg:
        # Assuming M_x_power returns shape (deg_out + 1, deg + 1)
        M1_1d = multiply_matrix.M_x_power(p=0, deg=deg, deg_out=deg_out)
        # Note: We completely skip building the full 'lift' tensor product matrix

    # Evaluate at the provided coordinate list
    f_eval_list = []
    for coords_eval in coords_eval_list:
        coords_eval = np.asarray(coords_eval, dtype=float)
        if coords_eval.shape != (num_vars,):
            raise ValueError(
                f"Each evaluation point must have shape ({num_vars},), got {coords_eval.shape}."
            )

        # Find the element containing coords_eval
        e_s = mesh.find_elements(coords_eval)
        if isinstance(e_s, int):
            pass
        elif len(e_s) == 0:
            raise ValueError(f"Point coords={tuple(coords_eval)} is not found in the mesh.")
        else: # len(e_s) > 1
            raise ValueError(f"Point coords={tuple(coords_eval)} is found in multiple elements: {e_s}. For function evaluation, a single element should be specified.")

        # Map to reference coordinates xi in [-1, 1]^d
        xi_eval = (2.0 * coords_eval - (lo[e_s, :] + hi[e_s, :])) / (hi[e_s, :] - lo[e_s, :])

        # 1. Compute the effective 1D evaluations
        effective_taus = []
        for v in range(num_vars):
            tau_v = encoding.chebyshev_encoding(deg=deg_out, x=xi_eval[v])
            if M1_1d is not None:
                # Apply 1D lift directly to the encoding: shape (deg_out+1,) @ (deg_out+1, deg+1) -> (deg+1,)
                tau_v = tau_v @ M1_1d
            effective_taus.append(tau_v)

        # 2. Reshape psi_e into a d-dimensional grid
        psi_e = psi_sol[e_s]
        psi_grid = psi_e.reshape((deg + 1,) * num_vars)

        # 3. Sequential Tensor Contraction
        # We iterate in reverse to properly contract the C-contiguous flattened array
        f_e = psi_grid
        for v in reversed(range(num_vars)):
            # Contract the last dimension of the grid with the 1D tau
            f_e = np.tensordot(f_e, effective_taus[v], axes=([-1], [0]))

        f_eval_list.append(float(f_e))

    f_eval_list = np.array(f_eval_list) * scaling_factor
    return f_eval_list


def evaluate_vector_multivar_sem_encoding(psi_sol: np.ndarray,
                                          deg: int,
                                          deg_out: int,
                                          num_components: int,
                                          mesh: RectMesh,
                                          coords_eval_list: np.ndarray,
                                          scaling_factor: list | np.ndarray = None,
                                          local_basis_transform: np.ndarray = None,
                                          local_basis_transform_joint: np.ndarray = None) -> np.ndarray:
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
    mesh : RectMesh
        The mesh object containing element endpoints.
    coords_eval_list : np.ndarray
        List of coordinates [x, y, ...] to evaluate.
    scaling_factor : list | np.ndarray
        Optional scaling factor for the output.
    local_basis_transform : np.ndarray
        Optional transformation matrix for the local basis.
    local_basis_transform_joint : np.ndarray
        Optional transformation matrix for the local basis, pre-computed as a tensor product.

    Returns
    -------
    np.ndarray
        Array of shape (num_points, num_components) containing the evaluated vector
        at each input point.
    """
    endpoints = mesh.endpoints
    assert endpoints.ndim in [2,3]
    if endpoints.ndim == 2:
        # 1D case: reshape to (num_elements, 2, 1) for uniform handling
        endpoints = endpoints[:, :, np.newaxis]

    num_elements = endpoints.shape[0]
    num_vars = endpoints.shape[2]

    if scaling_factor is None:
        scaling_factor = [1.0]*num_components
    else:
        assert len(scaling_factor) == num_components, f"scaling_factor length {len(scaling_factor)} must match num_components {num_components}"

    spatial_dof = (deg + 1) ** num_vars
    expected_len = num_elements * num_components * spatial_dof

    if len(psi_sol) != expected_len:
        raise ValueError(f"psi_sol length {len(psi_sol)} does not match expected length {expected_len} "
                         f"for {num_elements} elements, {num_components} components, "
                         f"and {spatial_dof} spatial DOFs.")

    lo = endpoints[:, 0, :]
    hi = endpoints[:, 1, :]

    if local_basis_transform_joint is not None:
        psi_sol = (psi_sol.reshape(-1, spatial_dof) @ local_basis_transform_joint).flatten()
    elif local_basis_transform is not None:
        C = local_basis_transform
        C_joint = C
        for _ in range(1, num_vars):
            C_joint = np.kron(C_joint, C)
        psi_sol = (psi_sol.reshape(-1, spatial_dof) @ C_joint).flatten()

    # Reshape psi_sol to (num_elements, num_components, (deg+1), ..., (deg+1))
    psi_reshaped = psi_sol.reshape((num_elements, num_components) + (deg + 1,) * num_vars)

    M1_1d = None
    if deg_out != deg:
        M1_1d = multiply_matrix.M_x_power(p=0, deg=deg, deg_out=deg_out)

    f_eval_list = []

    for coords_eval in coords_eval_list:
        coords_eval = np.asarray(coords_eval, dtype=float)

        # 1. Locate the Element
        e_s = mesh.find_elements(coords_eval)
        if isinstance(e_s, int):
            pass
        elif len(e_s) == 0:
            raise ValueError(f"Point coords={tuple(coords_eval)} is not found in the mesh.")
        else: # len(e_s) > 1
            raise ValueError(f"Point coords={tuple(coords_eval)} is found in multiple elements: {e_s}. For function evaluation, a single element should be specified.")

        # 2. Map to Reference Coordinates [-1, 1]
        xi_eval = (2.0 * coords_eval - (lo[e_s, :] + hi[e_s, :])) / (hi[e_s, :] - lo[e_s, :])

        # 3. Compute effective 1D evaluations
        effective_taus = []
        for v in range(num_vars):
            tau_v = encoding.chebyshev_encoding(deg=deg_out, x=xi_eval[v])
            if M1_1d is not None:
                tau_v = tau_v @ M1_1d
            effective_taus.append(tau_v)

        # 4. Extract and Contract Tensors Sequentially
        # psi_e shape: (num_components, deg+1, ..., deg+1)
        f_e = psi_reshaped[e_s]

        # Contract spatial dimensions (from last to first)
        for v in reversed(range(num_vars)):
            # Contract the last dimension of f_e with the 1D tau
            f_e = np.tensordot(f_e, effective_taus[v], axes=([-1], [0]))

        f_eval_list.append(f_e)

    return np.array(f_eval_list) * np.array(scaling_factor)

if __name__ == "__main__":
    import time
    num_nodes = 10+1
    nodes_x = np.linspace(-1, 1, num_nodes)
    nodes_y = np.linspace(-1, 1, num_nodes)
    nodes_z = np.linspace(-1, 1, num_nodes)
    mesh = RectMesh([nodes_x, nodes_y, nodes_z])
    print(mesh.num_elems)

    deg = 5
    num_vars = 3

    psi_sol = np.random.rand(mesh.num_elems * (deg + 1) ** num_vars)
    psi_sol = psi_sol / np.linalg.norm(psi_sol)
    x_eval_list = np.random.uniform(-1, 1, size=(100000, num_vars))

    start = time.time()
    f_eval = _evaluate_multivar_sem_encoding_inefficient_(psi_sol, deg, deg_out=deg, mesh=mesh, coords_eval_list=x_eval_list)
    end = time.time()
    print(f"Original evaluation time: {end - start:.4f} seconds")

    start = time.time()
    f_eval_new = evaluate_multivar_sem_encoding(psi_sol, deg, deg_out=deg, mesh=mesh, coords_eval_list=x_eval_list)
    end = time.time()
    print(f"New evaluation time: {end - start:.4f} seconds")

    assert np.allclose(f_eval, f_eval_new)
