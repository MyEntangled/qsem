import sympy as sp
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, kron, eye

from src.utils import function_evaluation, interface_continuity
from src.utils.basic_operators import derivative_matrix, multiply_matrix, tensor_mult_matrix, boundary_share
from src.utils.diffeq_hamiltonian import equation_parsing
from src.utils.boundary_hamiltonian.simple_boundary import build_boundary_matrix
from src.utils.meshing import RectMesh
import functools
import warnings
from typing import Dict, Tuple, Optional, Any, Literal

def _free_coeff_op(d:int, d_out:int, c:sp.Expr, var:sp.Symbol, truncated_order:int=None):
    c = sp.sympify(c)

    # Special case: scalar coefficient c = c0
    if c.is_Number:
        M_1 = multiply_matrix.M_x_power(p=0, deg=d, deg_out=d_out)
        return float(c) * M_1

    if not c.is_polynomial(var):
        ## If not polynomial, return truncated Maclaurin series
        if truncated_order is None:
            raise ValueError("truncated_order must be specified for non-polynomial coefficients.")
        coeff = sp.series(c, x=var, x0=0, n=truncated_order).removeO()
    else:
        coeff = c

    ## Coeff is a polynomial (c0 + c1 * x + c2 * x**2 + ...)
    ## Construct the corresponding H = c0 * M_1 + c1 * M_x + c2 * M_x2 + ...
    c_terms = coeff.as_ordered_terms()
    A = 0
    for term in c_terms:
        power = sp.degree(term, gen=var)
        scalar = term.coeff(var, power)
        M_xp = multiply_matrix.M_x_power(p=power, deg=d, deg_out=d_out)
        A += scalar * M_xp
    return A

def build_equation_operator(
        d: int,
        d_out: int,
        terms_dict: Dict[Tuple[int, int, int], Any],
        var: Any,
        truncated_order: Optional[int] = None,
        regular_data: tuple = (None,None),
        regular_data_type: Literal['value', 'derivative'] = None
) -> np.ndarray:
    """
    Constructs the effective Hamiltonian for a DE.

    This version assumes tensor_mult_matrix.N1_matrix returns the reduction matrix
    (Tensor -> Single) directly, with shape (deg_out+1, (deg+1)**2).

    Args:
        d (int): Input degree of the Chebyshev basis (N = d+1).
        d_out (int): Output dimension for the final coefficient projection.
        terms_dict (dict): Mapping (n, m, k) -> coefficient function c(x).
        var (Any): Symbolic variable.
        truncated_order (int, optional): Truncation order for coefficients.
        regular_data (tuple, optional): (x_s, y_s) for regular constraint. Required if padding or pure source term is needed.
        regular_data_type (str): 'value' (f) or 'derivative' (f').

    Returns:
        np.ndarray: The effective Hamiltonian H = A.T @ A.
    """

    # 1. Determine Tensor Rank (P)
    max_total_pow = max((m + k) for (n, m, k) in terms_dict.keys())
    tensor_rank = max(1, max_total_pow)

    if tensor_rank > 2:
        raise NotImplementedError("N1_matrix strictly supports Rank 2 reduction.")

    # 2. Pre-compute Base Matrices
    GT = derivative_matrix.diff_matrix(deg=d)
    Id = np.eye(d + 1)

    # 3. Regular constraint operator (only build if actually needed)
    # Regular matrices are only required when we need padding (to match tensor rank)
    # or when we have a pure source term (no derivative/function operators).
    need_regular = False
    regular_x, regular_y = regular_data
    for (n, m, k) in terms_dict.keys():
        term_power = m + k
        # Padding needed when term lives in a lower tensor-rank space than the maximum
        if term_power < tensor_rank:
            need_regular = True
            break
        # Pure source term: no derivative/function operators
        if term_power == 0:
            need_regular = True
            break

    # If regular data are provided but not needed, warn the caller.
    if not need_regular:
        if (regular_x is not None) or (regular_y is not None) or (regular_data_type is not None):
            warnings.warn(
                "Regular constraint data were provided but are not needed: "
                "no padding and no pure source term requires them.",
                UserWarning
            )
        D_reg = None
    else:
        # Regular data are required; validate inputs.
        if regular_data_type not in ('value', 'derivative'):
            raise ValueError("regular_constraint_type must be 'value' or 'derivative' when regular constraints are needed.")
        if regular_x is None or regular_y is None:
            raise ValueError("regular_data must be provided when regular constraints are needed.")
        if abs(float(regular_y)) < 1e-12:
            if regular_data_type == 'value':
                raise ValueError("Regular value y_s should be non-zero.")

        D_reg = build_boundary_matrix(type=regular_data_type, x=regular_x, y=regular_y, deg=d)

    A = 0.0

    # 4. Iterate over terms to build Operator A
    for (n, m, k), coeff in terms_dict.items():
        term_power = m + k
        n_pads = tensor_rank - term_power

        # --- A. Construct the Raw Tensor Operator ---
        # Order: [Derivatives] (kron) [Functions] (kron) [Padding]
        ops_list = []

        # Derivatives: (d^n/dx^n)^m
        if m > 0:
            GT_n = np.linalg.matrix_power(GT, n)
            ops_list.extend([GT_n] * m)

        # Function: f^k
        if k > 0:
            ops_list.extend([Id] * k)

        # Padding: Regular Constraints to match Hilbert space P
        if n_pads > 0:
            if D_reg is None:
                raise RuntimeError("Internal error: padding requested but D_reg was not constructed.")
            ops_list.extend([D_reg] * n_pads)

        # Pure source terms
        if not ops_list:
            if D_reg is None:
                raise RuntimeError("Internal error: source term requested but D_reg was not constructed.")
            ops_list = [D_reg] * tensor_rank

        combined_op = functools.reduce(np.kron, ops_list)

        # --- B. Apply N1 Reduction (Tensor -> Single Register) ---
        # The N1_matrix is already the transpose (mapping Tensor -> Single).
        # Shape: (reduced_deg+1, (d+1)**2)

        if tensor_rank == 2:
            # Use N1 to reduce quadratic terms
            Reducer = tensor_mult_matrix.N1_matrix(deg=d, deg_out=None)

            # Apply Reducer to the LEFT of the tensor operator
            reduced_op = Reducer @ combined_op

            # Output degree determines input size for the next step
            reduced_deg = Reducer.shape[0] - 1

        else:  # tensor_rank == 1
            reduced_op = combined_op
            reduced_deg = d

        # --- C. Apply Coefficient ---
        # Map the reduced register (degree reduced_deg) to the final output dimension d_out
        coeff_op = _free_coeff_op(
            d=reduced_deg,
            d_out=d_out,
            c=coeff,
            var=var,
            truncated_order=truncated_order
        )

        A += coeff_op @ reduced_op

    return A.astype(float)


def get_hamiltonian_from_operator(A):
    return A.T @ A


def sem_equation_hamiltonian(d: int, d_out: int, diff_eq, func, var, mesh: RectMesh, truncated_order: int = None,
                             regular_data: tuple = None, regular_data_type: str = None, intg_cond_order: int = 0,
                             local_basis_transform: Optional[np.ndarray] = None,
                             sparse: bool = False, get_hamiltonian: bool = True):

    """
    Transform the differential Hamiltonian to the reference domain [-1, 1] using the affine transformation x = (b-a)/2 * xi + (a+b)/2, where a and b are the endpoints of the original domain.
    """
    assert mesh.endpoints.ndim == 2
    if not (isinstance(intg_cond_order, int) and intg_cond_order >= 0):
        raise ValueError("intg_cond_order must be a non-negative integer.")
    if local_basis_transform is not None:
        if local_basis_transform.shape != (d+1, d+1):
            raise ValueError("local_basis_transform must have shape (d+1, d+1).")
    
    num_elements = mesh.num_elems
    lo = mesh.endpoints[:, 0]
    hi = mesh.endpoints[:, 1]

    N = d + 1
    N_out = d_out + 1

    parser = equation_parsing.ElementalEquationParser()
    terms = parser.parse_equation(diff_eq, func)
    ## Check if tensor rank = 1, otherwise raise NotImplementedError (need to handle padding and regularization for higher tensor ranks)
    max_total_pow = max((m + k) for (n, m, k) in terms.keys())
    if max_total_pow > 1:
        raise NotImplementedError("sem_differential_hamiltonian currently only supports tensor rank 1 (no products of derivatives/functions).")

    ## Find which element x_s belongs to and the corresponding local variable for regularization
    e_s = None
    xi_s_in_e_s = None

    if regular_data is not None:
        (x_s, y_s) = regular_data
        e_s = mesh.find_elements((x_s,))

        if isinstance(e_s, int):
            pass
        elif len(e_s) == 0:
            raise ValueError(f"Regularization point x_s={x_s} is not found in the mesh.")
        else:
            raise ValueError(f"Regularization point x_s={x_s} is found in multiple elements: {e_s}. A regularization point should be uniquely assigned to one element.")
        xi_s_in_e_s = (2 * x_s - (lo[e_s] + hi[e_s])) / (hi[e_s] - lo[e_s])  # Map to [-1, 1]
    else:
        x_s, y_s = None, None

    if sparse:
        A_total = lil_matrix((N_out * num_elements, N * num_elements))
    else:
        A_total = np.zeros((N_out * num_elements, N * num_elements))

    for e in range(num_elements):
        local_var_terms, local_var = parser.transform(terms_dict=terms, var_sym=var, a=lo[e], b=hi[e])
        #print(f"Transformed Terms for variable {var} to {local_var}", local_var_terms)

        ## Separate the inhomogeneous term (if exists) for regularization
        hom_terms = local_var_terms
        inhom_terms = {(0,0,0): hom_terms.pop((0,0,0))} if (0,0,0) in hom_terms.keys() else None
        A_hom = build_equation_operator(d=d,
                                        d_out=d_out,
                                        terms_dict=hom_terms,
                                        var=local_var,
                                        truncated_order=truncated_order)
        
        if sparse:
            A_total[e*N_out : (e+1)*N_out, e*N : (e+1)*N] += csr_matrix(A_hom)
        else:
            A_total[e*N_out : (e+1)*N_out, e*N : (e+1)*N] += A_hom

        if inhom_terms is not None:
            if e_s is None or xi_s_in_e_s is None:
                raise ValueError("Inhomogeneous terms require regular_data to be provided.")
            A_inhom = build_equation_operator(d=d,
                                              d_out=d_out,
                                              terms_dict=inhom_terms,
                                              var=local_var,
                                              truncated_order=truncated_order,
                                              regular_data=(xi_s_in_e_s, y_s),
                                              regular_data_type=regular_data_type)
            if sparse:
                A_total[e*N_out : (e+1)*N_out, e_s*N : (e_s+1)*N] += csr_matrix(A_inhom)
            else:
                A_total[e*N_out : (e+1)*N_out, e_s*N : (e_s+1)*N] += A_inhom

    # --- Integral Conditioning ---
    if intg_cond_order > 0:
        # Construct integration matrix for one element.
        # Enforce deg_out=d to maintain square matrix size (N x N) for preconditioning.
        S_block = derivative_matrix.intg_matrix(a=-1.0, deg=d_out, deg_out=d_out)
        
        if intg_cond_order > 1:
            S_block = np.linalg.matrix_power(S_block, intg_cond_order)
            
        if sparse:
            # Create global block-diagonal S using Kronecker product
            # S_total = I_elements (x) S_block
            S_total = kron(eye(num_elements), csr_matrix(S_block), format='csr')
            
            # Apply preconditioning: A_total = S_total @ A_total
            if isinstance(A_total, lil_matrix):
                A_total = A_total.tocsr()
            
            A_total = S_total @ A_total
            
        else: # Dense case: Apply S_block to each element block column-wise
            for e in range(num_elements):
                row_slice = slice(e*N_out, (e+1)*N_out)
                A_total[row_slice, :] = S_block @ A_total[row_slice, :]


    # --- Local basis transformation ---
    if local_basis_transform is not None:
        C = local_basis_transform

        if sparse:
            C_total = kron(eye(num_elements), csr_matrix(C), format='csr')
            if isinstance(A_total, lil_matrix):
                A_total = A_total.tocsr()

            A_total = A_total @ C_total.T

        else: # Dense case: Apply C to each element block row-wise
            for e in range(num_elements):
                col_slice = slice(e*N, (e+1)*N)
                A_total[:, col_slice] = A_total[:, col_slice] @ C.T

    # --- Convert to csr if needed ---
    if sparse and isinstance(A_total, lil_matrix):
        A_total = A_total.tocsr()

    if get_hamiltonian is False:
        return A_total
    else:
        return get_hamiltonian_from_operator(A_total)