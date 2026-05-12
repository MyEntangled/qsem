from src.utils.basic_operators import multiply_matrix
from src.utils.boundary_hamiltonian import simple_boundary
from src.utils.meshing import RectMesh
from src.utils.boundary_hamiltonian.cheb_approx_boundary import cheb_coeffs_projector

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix, kron, eye
from typing import Callable, Optional

def sem_boundary(type: str, x: float, d: int, d_out: int, mesh: RectMesh,
                 local_basis_transform: Optional[np.ndarray] = None,
                 sparse: bool = False, get_hamiltonian: bool = True,
                 ops_cache: Optional[dict] = None):
    """Construct the boundary Hamiltonian for a single boundary constraint.

    The boundary Hamiltonian is defined as B^T B, where B is the boundary matrix corresponding
    to the specified constraint. This can be used in the semigroup formulation to enforce the
    boundary condition.

    Parameters
    ----------
    d:
        Input Chebyshev degree d.
    d_out:
        Output Chebyshev degree for lifting the operator.
    type:
        Either `'value'` or `'derivative'`.
    x:
        Point where the boundary constraint is imposed.
    mesh:
        The RectMesh object defining the spatial discretization and element endpoints.
    get_hamiltonian:
        If True, return the Hamiltonian B^T B. If False, return the boundary matrix B itself.

    Returns
    -------
    numpy.ndarray
        The boundary matrix B or the Hamiltonian B^T B
    """
    if local_basis_transform is not None:
        if local_basis_transform.shape != (d + 1, d + 1):
            raise ValueError("local_basis_transform must have shape (d+1, d+1).")

    assert mesh.endpoints.ndim == 2
    num_elements = mesh.num_elems
    lo = mesh.endpoints[:, 0]
    hi = mesh.endpoints[:, 1]

    e_s = mesh.find_elements((x,))

    if isinstance(e_s, int):
        pass
    elif len(e_s) == 0:
        raise ValueError(f"Boundary point x={x} is not found in the mesh.")
    else:
        raise ValueError(f"For 1D mesh, constructing boundary expects at most one element containing x={x}, but found {len(e_s)}.")

    xi_s_in_e_s = (2 * x - (lo[e_s] + hi[e_s])) / (hi[e_s] - lo[e_s])  # Map to [-1, 1]

    B = simple_boundary.build_boundary_matrix(type=type, x=xi_s_in_e_s, y=None, deg=d, deg_out=d_out)
    
    # Apply local basis transform if provided
    if local_basis_transform is not None:
        C = local_basis_transform
        B = B @ C.T

    if get_hamiltonian:
        # Strategy 2A: Direct Hamiltonian assembly
        H_local = B.T @ B
        if sparse:
            # Place the local Hamiltonian block in the global matrix
            rows, cols = np.indices(H_local.shape)
            r_off = e_s * (d + 1)
            c_off = e_s * (d + 1)
            H_total = coo_matrix((H_local.flatten(), (rows.flatten() + r_off, cols.flatten() + c_off)),
                                 shape=((d + 1) * num_elements, (d + 1) * num_elements)).tocsr()
            return H_total
        else:
            H_total = np.zeros(((d + 1) * num_elements, (d + 1) * num_elements))
            H_total[e_s * (d + 1):(e_s + 1) * (d + 1), e_s * (d + 1):(e_s + 1) * (d + 1)] = H_local
            return H_total
    else:
        if sparse:
            rows, cols = np.indices(B.shape)
            r_off = e_s * (d_out + 1)
            c_off = e_s * (d + 1)
            B_total = coo_matrix((B.flatten(), (rows.flatten() + r_off, cols.flatten() + c_off)),
                                 shape=((d_out + 1) * num_elements, (d + 1) * num_elements)).tocsr()
            return B_total
        else:
            B_total = np.zeros(((d_out + 1) * num_elements, (d + 1) * num_elements))
            B_total[e_s * (d_out + 1):(e_s + 1) * (d_out + 1), e_s * (d + 1):(e_s + 1) * (d + 1)] = B
            return B_total


def sem_multivar_boundary(type: str, coords: tuple, d: int, d_out: int, mesh: RectMesh, func: Callable = None,
                          local_basis_transform: Optional[np.ndarray] = None,
                          sparse: bool = False, get_hamiltonian: bool = True,
                          ops_cache: Optional[dict] = None):
    """Construct the boundary Hamiltonian for multiple boundary constraints in multiple dimensions.

    Supports both standard boundary conditions (where unconstrained dimensions are lifted via identity)
    and functional-form boundary conditions (e.g., wave initial conditions) applied across the 
    unconstrained dimensions via a Chebyshev orthogonal projector.
    """
    if local_basis_transform is not None:
        if local_basis_transform.shape != (d + 1, d + 1):
            raise ValueError("local_basis_transform must have shape (d+1, d+1).")

    endpoints = mesh.endpoints

    if endpoints.ndim != 3 or endpoints.shape[1] != 2:
        raise ValueError("endpoints must have shape (num_elements, 2, num_variables).")

    num_elements, _, num_variables = endpoints.shape

    if len(coords) != num_variables:
        raise ValueError(f"coords must have length {num_variables}, got {len(coords)}.")

    lo = endpoints[:, 0, :]
    hi = endpoints[:, 1, :]

    matches = mesh.find_elements(tuple(coords))
    if isinstance(matches, int):
        matches = [matches]

    selected_elements = []
    xi_per_element = {}

    for e in matches:
        lo_e, hi_e = lo[e], hi[e]
        xi_list = [None] * num_variables
        for var, coord in enumerate(coords):
            if coord is not None:
                xi_list[var] = (2.0 * coord - (lo_e[var] + hi_e[var])) / (hi_e[var] - lo_e[var])
        selected_elements.append(e)
        xi_per_element[e] = xi_list

    if len(selected_elements) == 0:
        raise ValueError(f"Boundary coords={coords} are not found in the mesh.")

    # Identify variable types
    I_constr = [v for v in range(num_variables) if coords[v] is not None]
    I_unconstr = [v for v in range(num_variables) if coords[v] is None]

    M1 = multiply_matrix.M_x_power(0, d, d_out)
    in_block = (d + 1) ** num_variables
    out_block = (d_out + 1) ** num_variables

    # Strategy 2B/4A: Pre-compute joint operators
    C_joint = None
    if ops_cache is not None and 'C_joint' in ops_cache:
        C_joint = ops_cache['C_joint']
    elif local_basis_transform is not None:
        C = local_basis_transform
        C_joint = C
        for _ in range(1, num_variables):
            C_joint = np.kron(C_joint, C)
        if ops_cache is not None:
            ops_cache['C_joint'] = C_joint
    
    M_unconstr_joint = None
    if ops_cache is not None and 'M_unconstr_joint' in ops_cache:
        M_unconstr_joint = ops_cache['M_unconstr_joint']
    elif len(I_unconstr) > 0:
        M_unconstr_joint = M1
        for _ in range(1, len(I_unconstr)):
            M_unconstr_joint = np.kron(M_unconstr_joint, M1)
        if ops_cache is not None:
            ops_cache['M_unconstr_joint'] = M_unconstr_joint
    
    M_constr_joint = None
    if func is None and len(I_constr) > 0:
        # We don't pre-compute M_constr_joint here because B_v depends on xi_list
        pass
    
    if sparse:
        if get_hamiltonian:
            H_total = lil_matrix((in_block * num_elements, in_block * num_elements))
        else:
            B_total = lil_matrix((out_block * num_elements, in_block * num_elements))
    else:
        if get_hamiltonian:
            H_total = np.zeros((in_block * num_elements, in_block * num_elements))
        else:
            B_total = np.zeros((out_block * num_elements, in_block * num_elements))

    for e in selected_elements:
        xi_list = xi_per_element[e]
        H_local = None
        B_d = None

        # -------------------------------------------------------------
        # Fallback: No function provided OR fully constrained point
        # -------------------------------------------------------------
        if func is None or len(I_unconstr) == 0:
            if get_hamiltonian:
                # Strategy 2C: Factored Kronecker B^T B
                for var in range(num_variables):
                    if xi_list[var] is None:
                        B_v = M1
                    else:
                        B_v = simple_boundary.build_boundary_matrix(type=type, x=xi_list[var], y=None, deg=d, deg_out=d_out)
                    H_v = B_v.T @ B_v
                    H_local = H_v if H_local is None else np.kron(H_local, H_v)
            else:
                for var in range(num_variables):
                    if xi_list[var] is None:
                        B_v = M1
                    else:
                        B_v = simple_boundary.build_boundary_matrix(type=type, x=xi_list[var], y=None, deg=d, deg_out=d_out)
                    B_d = B_v if B_d is None else np.kron(B_d, B_v)

        # -------------------------------------------------------------
        # Projector-based functional boundary
        # -------------------------------------------------------------
        else:
            # ... (Projector logic) ...
            # (Keeping the projector logic as is for now, it sets B_d)
            # 1. Build the Kronecker product for CONSTRAINED variables
            B_constr = None
            for var in I_constr:
                B_v = simple_boundary.build_boundary_matrix(type=type, x=xi_list[var], y=None, deg=d, deg_out=d_out)
                B_constr = B_v if B_constr is None else np.kron(B_constr, B_v)

            if B_constr is None:
                B_constr = np.array(1.0)  # Handle edge case of 0 constraints

            # 2. Build the Projector for UNCONSTRAINED variables
            lo_e, hi_e = lo[e], hi[e]

            def local_func(*xi_args):
                """Translates local Chebyshev grid [-1, 1] to global physical coordinates."""
                global_args = []
                for i, v in enumerate(I_unconstr):
                    x_val = ((hi_e[v] - lo_e[v]) / 2.0) * xi_args[i] + ((hi_e[v] + lo_e[v]) / 2.0)
                    global_args.append(x_val)
                return func(*global_args)

            # Extract the orthogonal projector for this local element piece
            tau_coeffs, _, _ = cheb_coeffs_projector(local_func, (d,) * len(I_unconstr), include_matrices=False)

            # Strategy 3A: Implicit projector application
            # B_unconstr = M_joint @ (I - |tau><tau| / ||tau||^2)
            #            = M_joint - (M_joint @ tau) @ tau^T / ||tau||^2
            norm_sq = np.sum(tau_coeffs**2)
            M_tau = M_unconstr_joint @ tau_coeffs
            # B_unconstr is still materialized here for tensor interleave, 
            # but we construct it efficiently.
            B_unconstr = M_unconstr_joint - np.outer(M_tau, tau_coeffs) / norm_sq

            # 3. Interleave variables using tensor permutations
            N_out = d_out + 1
            N_in = d + 1

            # Reshape block matrices into separated input/output dimension tensors
            shape_constr = [N_out] * len(I_constr) + [N_in] * len(I_constr)
            shape_unconstr = [N_out] * len(I_unconstr) + [N_in] * len(I_unconstr)

            B_c_tensor = B_constr.reshape(shape_constr)
            B_u_tensor = B_unconstr.reshape(shape_unconstr)

            # Combine them: Tensor has axes (c_out..., c_in..., u_out..., u_in...)
            outer_tensor = np.tensordot(B_c_tensor, B_u_tensor, axes=0)

            # Map the grouped axes back into their absolute global variable index
            perm = []
            # Map Output Axes
            for var in range(num_variables):
                if var in I_constr:
                    perm.append(I_constr.index(var))
                else:
                    perm.append(2 * len(I_constr) + I_unconstr.index(var))
            # Map Input Axes
            for var in range(num_variables):
                if var in I_constr:
                    perm.append(len(I_constr) + I_constr.index(var))
                else:
                    perm.append(2 * len(I_constr) + len(I_unconstr) + I_unconstr.index(var))

            # Apply permutation and flatten back to standard block matrix
            B_e_tensor = np.transpose(outer_tensor, perm)
            B_d = B_e_tensor.reshape((out_block, in_block))

        # Apply pre-computed local basis transform
        if C_joint is not None:
            if H_local is not None:
                H_local = C_joint @ H_local @ C_joint.T
            if B_d is not None:
                B_d = B_d @ C_joint.T

        # Place the local boundary operator in the global matrix
        if get_hamiltonian:
            # Strategy 2A: Direct Hamiltonian assembly
            if H_local is None:
                if B_d is None:
                    raise ValueError("Failed to compute H_local or B_d")
                H_local = B_d.T @ B_d
            
            r0, r1 = e * in_block, (e + 1) * in_block
            if sparse:
                H_total[r0:r1, r0:r1] = lil_matrix(H_local)
            else:
                H_total[r0:r1, r0:r1] = H_local
        else:
            r0, r1 = e * out_block, (e + 1) * out_block
            c0, c1 = e * in_block, (e + 1) * in_block
            if sparse:
                B_total[r0:r1, c0:c1] = lil_matrix(B_d)
            else:
                B_total[r0:r1, c0:c1] = B_d

    if get_hamiltonian:
        return H_total.tocsr() if sparse else H_total
    else:
        return B_total.tocsr() if sparse else B_total


def sem_vector_multivar_boundary(type: str, coords: tuple, d: int, d_out: int, mesh: RectMesh,
                                 num_components: int, component_idx: int,
                                 local_basis_transform: Optional[np.ndarray] = None,
                                 sparse: bool = False, get_hamiltonian: bool = True,
                                 ops_cache: Optional[dict] = None):
    """Construct the boundary Hamiltonian for a specific component of a vector-valued function.

    Memory Layout: |element> |component> |spatial_basis>

    Parameters
    ----------
    d:
        Input Chebyshev degree d.
    d_out:
        Output Chebyshev degree for lifting the operator.
    mesh:
        The RectMesh object defining the spatial discretization and element endpoints.
    type:
        Either `'value'` or `'derivative'`.
    coords:
        Tuple of shape (num_variables,) containing the constrained coordinates.
        `None` indicates no constraint along that axis.
    num_components:
        Total number of components in the vector-valued system (e.g., 2 for [u, v]).
    component_idx:
        The specific component index (0 to num_components-1) that this constraint applies to.
    get_hamiltonian:
        If True, return the Hamiltonian B^T B. If False, return the boundary matrix B itself.

    Returns
    -------
    numpy.ndarray
        The boundary Hamiltonian matrix B^T B for the constraint.
        :param get_hamiltonian:
    """

    endpoints = mesh.endpoints
    assert endpoints.ndim in [2,3]
    if endpoints.ndim == 2:
        # 1D mesh case: reshape to (num_elements, 2, 1) for uniform handling
        endpoints = endpoints[:, :, np.newaxis]


    num_elements, _, num_variables = endpoints.shape

    if len(coords) != num_variables:
        raise ValueError(f"coords must have length {num_variables}, got {len(coords)}.")

    if not (0 <= component_idx < num_components):
        raise ValueError(f"component_idx {component_idx} out of bounds for {num_components} components.")

    if local_basis_transform is not None:
        if local_basis_transform.shape != (d + 1, d + 1):
            raise ValueError("local_basis_transform must have shape (d+1, d+1).")

    lo = endpoints[:, 0, :]
    hi = endpoints[:, 1, :]

    # Identify which elements contain the constrained coordinate(s).
    selected_elements = []
    xi_per_element = {}

    matches = mesh.find_elements(tuple(coords))
    if isinstance(matches, int):
        matches = [matches]

    for e in matches:
        lo_e, hi_e = lo[e], hi[e]

        xi_list = [None] * num_variables
        for var, coord in enumerate(coords):
            if coord is not None:
                xi_list[var] = (2.0 * coord - (lo_e[var] + hi_e[var])) / (hi_e[var] - lo_e[var])

        selected_elements.append(e)
        xi_per_element[e] = xi_list

    if len(selected_elements) == 0:
        raise ValueError(f"Boundary coords={coords} are not found in the mesh.")

    M1 = multiply_matrix.M_x_power(0, d, d_out)

    # Strategy 2B/4A: Pre-compute local basis transform
    C_joint = None
    if ops_cache is not None and 'C_joint' in ops_cache:
        C_joint = ops_cache['C_joint']
    elif local_basis_transform is not None:
        C = local_basis_transform
        C_joint = C
        for _ in range(1, num_variables):
            C_joint = np.kron(C_joint, C)
        if ops_cache is not None:
            ops_cache['C_joint'] = C_joint

    def _build_H_d(xi_list: list) -> np.ndarray:
        # Strategy 2C: Factored Kronecker B^T B
        H_d = None
        for v in range(num_variables):
            if xi_list[v] is None:
                B_v = M1
            else:
                B_v = simple_boundary.build_boundary_matrix(type=type, x=xi_list[v], y=None, deg=d, deg_out=d_out)
            H_v = B_v.T @ B_v
            H_d = H_v if H_d is None else np.kron(H_d, H_v)
        return H_d

    def _build_B_d(xi_list: list) -> np.ndarray:
        B_d = None
        for v in range(num_variables):
            if xi_list[v] is None:
                B_v = M1
            else:
                B_v = simple_boundary.build_boundary_matrix(type=type, x=xi_list[v], y=None, deg=d, deg_out=d_out)
            B_d = B_v if B_d is None else np.kron(B_d, B_v)
        return B_d

    # Calculate Block Sizes
    spatial_in = (d + 1) ** num_variables
    spatial_out = (d_out + 1) ** num_variables

    elem_in = num_components * spatial_in
    elem_out = num_components * spatial_out

    # Global Block Matrix initialization
    if sparse:
        if get_hamiltonian:
            H_total = lil_matrix((num_elements * elem_in, num_elements * elem_in))
        else:
            B_total = lil_matrix((num_elements * elem_out, num_elements * elem_in))
    else:
        if get_hamiltonian:
            H_total = np.zeros((num_elements * elem_in, num_elements * elem_in))
        else:
            B_total = np.zeros((num_elements * elem_out, num_elements * elem_in))

    for e in selected_elements:
        if get_hamiltonian:
            H_scalar = _build_H_d(xi_per_element[e])
            if C_joint is not None:
                H_scalar = C_joint @ H_scalar @ C_joint.T
            
            # Strategy 2A: Direct Hamiltonian assembly
            row_start = e * elem_in + component_idx * spatial_in
            row_end = row_start + spatial_in
            col_start = row_start
            col_end = row_end
            
            if sparse:
                H_total[row_start:row_end, col_start:col_end] = lil_matrix(H_scalar)
            else:
                H_total[row_start:row_end, col_start:col_end] = H_scalar
        else:
            B_scalar = _build_B_d(xi_per_element[e])

            # Apply pre-computed local basis transform
            if C_joint is not None:
                B_scalar = B_scalar @ C_joint.T
            # Determine the exact row/column indices inside the global matrix
            # Structure: Element Block -> Component Sub-block -> Spatial Operator
            row_start = e * elem_out + component_idx * spatial_out
            row_end = row_start + spatial_out

            col_start = e * elem_in + component_idx * spatial_in
            col_end = col_start + spatial_in

            if sparse:
                B_total[row_start:row_end, col_start:col_end] = lil_matrix(B_scalar)
            else:
                B_total[row_start:row_end, col_start:col_end] = B_scalar

    if get_hamiltonian:
        return H_total.tocsr() if sparse else H_total
    else:
        return B_total.tocsr() if sparse else B_total


def build_general_boundary(type: str, coords: float | tuple, d: int, d_out: int, mesh: RectMesh,
                           num_components: int = None, component_idx: int = None,
                           local_basis_transform: Optional[np.ndarray] = None,
                           sparse: bool = False, get_hamiltonian: bool = True,
                           ops_cache: Optional[dict] = None):
    """Factory for building boundary Hamiltonians for various types of constraints.

    Parameters
    ----------
    type:
        Either 'value' or 'derivative'.
    coords:
        For 1D: a single float coordinate where the constraint is applied.
        For multivariate: a tuple of coordinates, where None indicates no constraint along that axis.
    d:
        Input Chebyshev degree d.
    d_out:
        Output Chebyshev degree for lifting the operator.
    mesh:
        The RectMesh object defining the spatial discretization and element endpoints.
    num_components:
        For vector-valued cases, the total number of components in the system.
    component_idx:
        For vector-valued cases, the specific component index that this constraint applies to.
    get_hamiltonian:
        If True, return the Hamiltonian B^T B. If False, return the boundary matrix B itself.

    Returns
    -------
    numpy.ndarray
        The boundary matrix for the specified constraint.
        :param get_hamiltonian:
    """

    if (num_components is None) ^ (component_idx is None):
        raise ValueError("Both num_components and component_idx must be provided together or not at all.")

    if not isinstance(coords, tuple) and not isinstance(coords, float):
        raise ValueError("coords must be either a float (for 1D) or a tuple of floats (for multivariate).")


    if num_components is None or num_components == 1:
        if isinstance(coords, float):
            return sem_boundary(type=type, x=coords, d=d, d_out=d_out, mesh=mesh,
                                local_basis_transform=local_basis_transform,
                                sparse=sparse, get_hamiltonian=get_hamiltonian,
                                ops_cache=ops_cache)
        else:
            return sem_multivar_boundary(type=type, coords=coords, d=d, d_out=d_out, mesh=mesh,
                                         local_basis_transform=local_basis_transform,
                                         sparse=sparse, get_hamiltonian=get_hamiltonian,
                                         ops_cache=ops_cache)

    elif num_components >= 2:
        if isinstance(coords, float):
            coords = (coords,)

        return sem_vector_multivar_boundary(type=type, coords=coords, d=d, d_out=d_out, mesh=mesh,
                                            num_components=num_components, component_idx=component_idx,
                                            sparse=sparse, get_hamiltonian=get_hamiltonian,
                                            ops_cache=ops_cache)

    else:
        raise ValueError("num_components must be a positive integer.")



if __name__ == "__main__":
    # Example usage for a 2D vector-valued function with a functional boundary condition on component 0
    from src.utils.meshing import RectMesh
    from src.utils.eigensolvers import lanczos_solver
    from src.utils.function_evaluation import evaluate_multivar_sem_encoding
    from src.utils import encoding
    from src.utils.interface_continuity import multivar_boundary_continuity_matrix

    # Define a simple 2D mesh (4x4 elements = 16 elements total)
    x_nodes = np.linspace(-1, 1, 5)
    y_nodes = np.linspace(-1, 1, 5)
    mesh = RectMesh([x_nodes, y_nodes])

    # Define a functional boundary condition: u(-1, y) = sin(pi * y)
    func_bc = lambda y: np.sin(np.pi * y)

    # Build the boundary Hamiltonian for this functional constraint at x=-1
    deg = 3
    B_hamiltonian = sem_multivar_boundary(type='value', coords=(-1.0, None), d=deg, d_out=deg, mesh=mesh, func=func_bc)

    Cv_hamiltonian = multivar_boundary_continuity_matrix('value', mesh.N, deg, deg)
    Cd_hamiltonian = multivar_boundary_continuity_matrix('derivative', mesh.N, deg, deg)
    C_hamiltonian = Cd_hamiltonian + 10*Cv_hamiltonian

    print(C_hamiltonian.shape)

    H = B_hamiltonian + C_hamiltonian

    print("Boundary Hamiltonian shape:", B_hamiltonian.shape)
    print("Continuity Hamiltonian shape:", C_hamiltonian.shape)

    # Solve for the ground state (which minimizes the boundary penalty)
    eigvals, eigvecs = lanczos_solver.pylanczos_solve(H, 2, find_max=False)
    psi_sol = eigvecs[:, 0]
    print("Smallest eigenvalue:", eigvals[0])
    print("Spectral gap:", eigvals[1] - eigvals[0])

    # ==========================================
    # 1. Find the physical scaling factor (Coefficient Projection)
    # ==========================================
    # Find one element that lives on the x = -1.0 boundary
    eval_pt = (-1.0, 1.0)
    e_s = mesh.find_elements(eval_pt)
    print(f"Elements containing the boundary point {eval_pt}: {e_s}")
    if isinstance(e_s, list):
        e_s = e_s[0]  # Pick the first matching element if multiple are found

    lo_e, hi_e = mesh.endpoints[e_s, 0, :], mesh.endpoints[e_s, 1, :]

    # Define the local function for the unconstrained variable (y is index 1)
    def local_func(xi_y):
        y_global = ((hi_e[1] - lo_e[1]) / 2.0) * xi_y + ((hi_e[1] + lo_e[1]) / 2.0)
        return func_bc(y_global)


    # Get the Chebyshev coefficients for this local boundary piece
    f_coeffs_local, _, _ = cheb_coeffs_projector(local_func, deg, include_matrices=False)

    # Extract the local quantum state vector for element e_s
    psi_sol_reshaped = psi_sol.reshape((mesh.num_elems, (deg + 1) ** 2))
    psi_e = psi_sol_reshaped[e_s]
    print(e_s, psi_e.shape)

    # Project the 2D local state onto the boundary x = -1.0 (local xi_x = -1.0)
    tau_x = encoding.chebyshev_encoding(deg, -1.0)

    # Reshape psi_e into a 2D matrix (rows=x, cols=y) because tau = kron(tau_x, tau_y)
    psi_e_matrix = psi_e.reshape((deg + 1, deg + 1))

    # Evaluate the x-dimension at the boundary, leaving just the y-dimension state
    psi_y_boundary = np.dot(tau_x, psi_e_matrix)
    print(np.linalg.norm(tau_x), np.linalg.norm(psi_e), np.linalg.norm(psi_y_boundary))

    # Now calculate the scaling factor exactly as in wave_equation.py
    projection = np.dot(psi_y_boundary, f_coeffs_local)
    norm_psi_t = np.dot(f_coeffs_local, f_coeffs_local)

    if abs(projection) > 1e-12:
        scaling_factor = norm_psi_t / projection
    else:
        scaling_factor = 1.0  # Fallback safeguard

    print(f"Scaling factor determined via coefficient projection: {scaling_factor:.4f}")


    # ==========================================
    # 2. Evaluate the solution across the boundary
    # ==========================================
    # Generate test points along the x = -1.0 edge
    y_test_vals = np.linspace(-1., 1., 100)
    coords_test = np.column_stack((np.full_like(y_test_vals, -1.), y_test_vals))

    # Evaluate using the provided function and our new scaling factor
    computed_vals = evaluate_multivar_sem_encoding(
        psi_sol=psi_sol,
        deg=deg,
        deg_out=deg,
        mesh=mesh,
        coords_eval_list=coords_test,
        scaling_factor=scaling_factor
    )

    exact_vals = func_bc(y_test_vals)

    # ==========================================
    # 3. Compare Results
    # ==========================================

    # print("\n--- Evaluation Results along x = -1.0 ---")
    # for i, y_val in enumerate(y_test_vals):
    #     exact = exact_vals[i]
    #     computed = computed_vals[i]
    #     error = abs(exact - computed) / (abs(exact) + 1e-12)  # Relative error with safeguard
    #     print(f"y = {y_val:5.2f} | Exact: {exact:8.5f} | Computed: {computed:8.5f} | Error: {error:.2e}")
    #

    ## Plotting the results for visual comparison
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(y_test_vals, exact_vals, 'k-', label='Exact Boundary Function')
    plt.plot(y_test_vals, computed_vals, 'r-', label='Computed from SEM Solution')
    plt.title("Boundary Evaluation at x = -1.0")
    plt.xlabel("y")
    plt.ylabel("u(-1, y)")
    plt.legend()
    plt.grid()
    plt.show()
