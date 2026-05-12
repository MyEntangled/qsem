import numpy as np
import scipy.sparse as sps
from src.utils.meshing import RectMesh
from typing import Union

def basis_index(e: int, k: int, d: int) -> int:
    """
    Flatten |e>|k> into a single index.
    """
    return e * (d + 1) + k


def _get_shared_indices(e_array: np.ndarray, var_index: int, k_val: int, 
                        num_funcs: int, d: int, num_vars: int) -> np.ndarray:
    """
    Returns the flattened indices corresponding to the given elements, 
    for all functions and all remaining variables, with k set to k_val along var_index.
    """
    local_dim = (d + 1) ** num_vars
    # All local indices inside one function block
    all_local = np.arange(local_dim)
    stride_v = (d + 1) ** (num_vars - 1 - var_index)
    
    # Filter for the specific k_val
    mask = (all_local // stride_v) % (d + 1) == k_val
    valid_local = all_local[mask]
    
    # Function offsets
    f_offsets = np.arange(num_funcs) * local_dim
    
    # Base for elements
    e_base = e_array * (num_funcs * local_dim)
    
    indices = (e_base[:, None, None] + 
               f_offsets[None, :, None] + 
               valid_local[None, None, :])
               
    return indices.flatten()

def _get_adjacent_elements(mesh: Union[RectMesh, int], v: int, is_periodic: bool):
    if isinstance(mesh, int):
        N = [mesh]
        n_vars = 1
        strides = np.array([1])
    else:
        N = mesh.N
        n_vars = mesh.n
        strides = mesh.strides

    ranges = [np.arange(N_d) for N_d in N]
    if not is_periodic:
        ranges[v] = np.arange(N[v] - 1)
        if len(ranges[v]) == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
            
    grid = np.meshgrid(*ranges, indexing='ij')
    multi_left = np.stack(grid, axis=-1).reshape(-1, n_vars)
    
    multi_right = multi_left.copy()
    multi_right[:, v] = (multi_right[:, v] + 1) % N[v]
    
    e_left = multi_left.dot(strides)
    e_right = multi_right.dot(strides)
    return e_left, e_right

def construct_interface_sharing(mesh: Union[RectMesh, int], d: int, num_funcs: int = 1, num_vars: int = 1, share_mode: str = "C0", is_periodic: bool = False, sparse: bool = False):
    """
    Construct the matrix S acting on the basis {|e>|k>} with
    e = 0, ..., M-1 and k = 0, ..., d.

    Always enforces
        c_{e,1} = c_{e+1,0} = (a_{e,1} + a_{e+1,0}) / 2

    If share_mode == "C1", also enforces
        c_{e,3} = c_{e+1,2} = (a_{e,3} + a_{e+1,2}) / 2

    Here e+1 means:
      - (e+1) % M if is_periodic=True
      - ordinary e+1, with the last pair omitted if is_periodic=False
      
    If sparse == True, returns a scipy.sparse.csr_matrix.
    """
    if isinstance(mesh, int):
        n_vars = 1
        num_elements = mesh
    else:
        n_vars = mesh.n
        num_elements = mesh.num_elems
    assert num_vars == n_vars, f"num_vars ({num_vars}) must equal mesh dimension ({n_vars})"
    N = num_elements * num_funcs * (d + 1) ** n_vars
    
    if sparse:
        S = sps.eye(N, dtype=float, format="csr")
    else:
        S = np.eye(N, dtype=float)

    for v in range(n_vars):
        if sparse:
            S_v = sps.eye(N, dtype=float, format="lil")
        else:
            S_v = np.eye(N, dtype=float)
            
        e_left, e_right = _get_adjacent_elements(mesh, v, is_periodic)
        if len(e_left) > 0:
            if d >= 1:
                i1 = _get_shared_indices(e_left, v, 1, num_funcs, d, n_vars)
                i2 = _get_shared_indices(e_right, v, 0, num_funcs, d, n_vars)
                S_v[i1, i1] = 0.5
                S_v[i1, i2] = 0.5
                S_v[i2, i1] = 0.5
                S_v[i2, i2] = 0.5
                
            if share_mode == "C1":
                if d < 3:
                    raise ValueError("share_mode='C1' requires d >= 3.")
                i3 = _get_shared_indices(e_left, v, 3, num_funcs, d, n_vars)
                i4 = _get_shared_indices(e_right, v, 2, num_funcs, d, n_vars)
                S_v[i3, i3] = 0.5
                S_v[i3, i4] = 0.5
                S_v[i4, i3] = 0.5
                S_v[i4, i4] = 0.5

        if sparse:
            S_v = S_v.tocsr()
        
        S = S_v @ S

    return S

def apply_interface_sharing(psi: np.ndarray, mesh: Union[RectMesh, int], d: int,
                            num_funcs: int = 1, num_vars: int = 1,
                            share_mode: str = "C0", is_periodic: bool = False) -> np.ndarray:
    """
    Applies the interface sharing projector S to a vector psi directly,
    averaging the specified boundary coefficients.
    """
    # Create a copy so we don't modify the input vector in-place
    out = np.array(psi, copy=True, dtype=float)
    if isinstance(mesh, int):
        n_vars = 1
    else:
        n_vars = mesh.n
    assert num_vars == n_vars, f"num_vars ({num_vars}) must equal mesh dimension ({n_vars})"

    for v in range(n_vars):
        e_left, e_right = _get_adjacent_elements(mesh, v, is_periodic)
        if len(e_left) == 0:
            continue
            
        # C0 Sharing
        if d >= 1:
            i1 = _get_shared_indices(e_left, v, 1, num_funcs, d, n_vars)
            i2 = _get_shared_indices(e_right, v, 0, num_funcs, d, n_vars)

            avg_C0 = (out[i1] + out[i2]) / 2.0
            out[i1] = avg_C0
            out[i2] = avg_C0

        # C1 Sharing
        if share_mode == "C1":
            if d < 3:
                raise ValueError("share_mode='C1' requires d >= 3.")

            i3 = _get_shared_indices(e_left, v, 3, num_funcs, d, n_vars)
            i4 = _get_shared_indices(e_right, v, 2, num_funcs, d, n_vars)

            avg_C1 = (out[i3] + out[i4]) / 2.0
            out[i3] = avg_C1
            out[i4] = avg_C1

    return out


def apply_SHS(H: Union[np.ndarray, sps.spmatrix], mesh: Union[RectMesh, int], d: int,
              num_funcs: int = 1, num_vars: int = 1,
              share_mode: str = "C0", is_periodic: bool = False) -> Union[np.ndarray, sps.spmatrix]:
    """
    Computes S @ A @ S by averaging the shared columns, then the shared rows.
    """
    if sps.issparse(H):
        S = construct_interface_sharing(mesh, d, num_funcs=num_funcs, num_vars=num_vars,
                                        share_mode=share_mode, is_periodic=is_periodic, sparse=True)
        return S @ H @ S

    out = np.array(H, copy=True, dtype=float)
    if isinstance(mesh, int):
        n_vars = 1
    else:
        n_vars = mesh.n
    assert num_vars == n_vars, f"num_vars ({num_vars}) must equal mesh dimension ({n_vars})"

    for v in range(n_vars):
        e_left, e_right = _get_adjacent_elements(mesh, v, is_periodic)
        if len(e_left) == 0:
            continue
            
        if d >= 1:
            i1 = _get_shared_indices(e_left, v, 1, num_funcs, d, n_vars)
            i2 = _get_shared_indices(e_right, v, 0, num_funcs, d, n_vars)

            # 1. Right Multiplication (A @ S_v): Average the columns
            avg_cols = (out[:, i1] + out[:, i2]) / 2.0
            out[:, i1] = avg_cols
            out[:, i2] = avg_cols

            # 2. Left Multiplication (S_v @ (A @ S_v)): Average the rows
            avg_rows = (out[i1, :] + out[i2, :]) / 2.0
            out[i1, :] = avg_rows
            out[i2, :] = avg_rows

        if share_mode == "C1" and d >= 3:
            i3 = _get_shared_indices(e_left, v, 3, num_funcs, d, n_vars)
            i4 = _get_shared_indices(e_right, v, 2, num_funcs, d, n_vars)

            # 1. Right Multiplication: Average the columns
            avg_cols = (out[:, i3] + out[:, i4]) / 2.0
            out[:, i3] = avg_cols
            out[:, i4] = avg_cols

            # 2. Left Multiplication: Average the rows
            avg_rows = (out[i3, :] + out[i4, :]) / 2.0
            out[i3, :] = avg_rows
            out[i4, :] = avg_rows

    return out

if __name__ == "__main__":
    def multiply_interface_sharing(psi: np.ndarray, share_mode: str = "C0", is_periodic: bool = False) -> np.ndarray:
        """
        Apply S to a coefficient array psi[e,k] of shape (M, d+1).
        """
        M, local_dim = psi.shape
        d = local_dim - 1
        
        # Mock mesh for 1D structure 
        nodes = np.linspace(0, 1, M + 1)
        mesh = RectMesh(nodes)
        S = construct_interface_sharing(mesh, d, share_mode=share_mode, is_periodic=is_periodic)

        vec_a = psi.reshape(-1)
        vec_c = S @ vec_a
        return vec_c.reshape(M, d + 1)

    M = 4
    d = 7
    a = np.arange(M * (d + 1), dtype=float).reshape(M, d + 1)
    nodes = np.linspace(0, 1, M + 1)
    mesh = RectMesh(nodes)

    ## Verify S is a projector
    S = construct_interface_sharing(mesh, d, share_mode="C1", is_periodic=False)
    # Use array multiplication for verification with dense matrix
    res = S @ S - S
    if np.linalg.norm(res) < 1e-6:
        print("S is projector")
    else:
        print("S is not projector")

    ## Compute the shared coeffs.
    c_nonperiodic = multiply_interface_sharing(a, share_mode="C1", is_periodic=False)
    c_periodic = multiply_interface_sharing(a, share_mode="C1", is_periodic=True)

    print("Original a:")
    print(a)

    print("\nNon-periodic:")
    print(c_nonperiodic)

    print("\nPeriodic:")
    print(c_periodic)

    H = np.random.rand(M * (d + 1), M * (d + 1))
    res = apply_SHS(H, mesh, d, share_mode="C1", is_periodic=False) - S @ H @ S
    if np.linalg.norm(res) < 1e-6:
        print("SHS matches S@H@S")
    else:
        print("SHS does not match S@H@S")

    print("\n--- Testing Vector-Valued Multivariate Case ---")
    M_x, M_y = 5, 6
    d_multi = 3
    num_funcs_multi = 2
    num_vars_multi = 2

    nodes_x = np.linspace(0, 1, M_x + 1)
    nodes_y = np.linspace(0, 1, M_y + 1)
    mesh_2d = RectMesh([nodes_x, nodes_y])
    N_multi = mesh_2d.num_elems * num_funcs_multi * (d_multi + 1)**num_vars_multi

    S_multi = construct_interface_sharing(mesh_2d, d_multi, num_funcs=num_funcs_multi, num_vars=num_vars_multi, share_mode="C1", is_periodic=False)
    if np.linalg.norm(S_multi @ S_multi - S_multi) < 1e-6:
        print("S_multi is projector")
    else:
        print("S_multi is not projector")

    psi_multi = np.random.rand(N_multi)
    res_vec = apply_interface_sharing(psi_multi, mesh_2d, d_multi, num_funcs=num_funcs_multi, num_vars=num_vars_multi, share_mode="C1", is_periodic=False) - S_multi @ psi_multi
    if np.linalg.norm(res_vec) < 1e-6:
        print("apply_interface_sharing matches S @ psi")
    else:
        print("apply_interface_sharing does not match S @ psi")

    H_multi = np.random.rand(N_multi, N_multi)
    res_shs = apply_SHS(H_multi, mesh_2d, d_multi, num_funcs=num_funcs_multi, num_vars=num_vars_multi, share_mode="C1", is_periodic=False) - S_multi @ H_multi @ S_multi
    if np.linalg.norm(res_shs) < 1e-6:
        print("apply_SHS matches S @ H @ S for multivariate vector-valued")
    else:
        print("apply_SHS does not match S @ H @ S for multivariate vector-valued")

    print("\n--- Verifying C0 and C1 equalities for Multivariate Vector-Valued ---")
    c_multi = S_multi @ psi_multi
    max_diff_c0 = 0.0
    max_diff_c1 = 0.0

    for v in range(num_vars_multi):
        e_left, e_right = _get_adjacent_elements(mesh_2d, v, is_periodic=False)
        print("v=", v)
        print(e_left, e_right)
        if len(e_left) > 0:
            # Check C0 equality: |e>|1> vs |e+1>|0>
            i1 = _get_shared_indices(e_left, v, 1, num_funcs_multi, d_multi, num_vars_multi)
            i2 = _get_shared_indices(e_right, v, 0, num_funcs_multi, d_multi, num_vars_multi)
            diff_c0 = np.abs(c_multi[i1] - c_multi[i2]).max()
            max_diff_c0 = max(max_diff_c0, diff_c0)

            # Check C1 equality: |e>|3> vs |e+1>|2>
            i3 = _get_shared_indices(e_left, v, 3, num_funcs_multi, d_multi, num_vars_multi)
            i4 = _get_shared_indices(e_right, v, 2, num_funcs_multi, d_multi, num_vars_multi)
            diff_c1 = np.abs(c_multi[i3] - c_multi[i4]).max()
            max_diff_c1 = max(max_diff_c1, diff_c1)

    print(f"Max C0 difference (|e>|1> vs |e+1>|0>): {max_diff_c0:.2e}")
    if max_diff_c0 < 1e-12:
        print("C0 equality holds strictly.")
    else:
        print("C0 equality failed!")

    print(f"Max C1 difference (|e>|3> vs |e+1>|2>): {max_diff_c1:.2e}")
    if max_diff_c1 < 1e-12:
        print("C1 equality holds strictly.")
    else:
        print("C1 equality failed!")