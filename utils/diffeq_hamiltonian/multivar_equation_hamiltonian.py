from src.utils.basic_operators import derivative_matrix, multiply_matrix
from src.utils.boundary_hamiltonian.simple_boundary import build_boundary_matrix
from src.utils.diffeq_hamiltonian import multivar_equation_parsing
from src.utils.diffeq_hamiltonian.symbolic_utils import factored_series_poly_dict
from src.utils.basic_operators.derivative_matrix import intg_matrix

import numpy as np
import sympy as sp
import functools
import warnings
from typing import Dict, Tuple, Any, Optional, List, Literal

from src.utils.meshing import RectMesh
from scipy.sparse import lil_matrix, csr_matrix, kron, eye


def _free_coeff_op_multivariable(d: int, d_out: int, c: sp.Expr, vars: list, truncated_order: int = None, computed_Mxp: dict = None):
    c = sp.sympify(c)
    vars = list(vars)
    if computed_Mxp is None or computed_Mxp == {}:
        computed_Mxp = {}


    # 1. Handle Scalar Case
    if c.is_Number:
        # p=0 represents the identity-like operator for every variable.
        computed_Mxp[0] = multiply_matrix.M_x_power(p=0, deg=d, deg_out=d_out)
        return float(c) * functools.reduce(np.kron, [computed_Mxp[0]] * len(vars)), computed_Mxp

    # 2+3. Handle Non-Polynomials via Factored Taylor Expansion + Extract Monomials
    poly_dict = factored_series_poly_dict(c, tuple(vars), truncated_order)

    A = 0
    for powers, scalar in poly_dict.items():
        # 'powers' is a tuple corresponding to the degrees of each variable in 'vars'
        # e.g., if vars=[x, y] and term is 3*x**1*y**2, powers=(1, 2)

        term_operators = []

        for i, p in enumerate(powers):
            # Generate the operator for variable vars[i] raised to power p
            if p not in computed_Mxp:
                M_p = multiply_matrix.M_x_power(p=p, deg=d, deg_out=d_out)
                computed_Mxp[p] = M_p
            else:
                M_p = computed_Mxp[p]

            term_operators.append(M_p)

        A += float(scalar) * functools.reduce(np.kron, term_operators)

    return A.astype(float), computed_Mxp


def _prepare_fast_evaluators(terms_dict, vars, d, d_out, n_vars, truncated_order, precomputed):
    """Perform all SymPy symbolic work once; return lambdified evaluators for per-element use.

    For each non-source term, produces a list of (scalar_func, product_matrix) pairs where:
      - scalar_func(J_0,...,J_{n-1}, c_0,...,c_{n-1}) -> float  (fast numeric)
      - product_matrix = Mxp_kron @ combined_op                 (precomputed matrix, sparse or dense)

    The per-element operator is then: A_hom = sum_i scalar_func_i(J, center) * product_matrix_i
    """
    J_syms = sp.symbols(f'J_0:{n_vars}', positive=True)
    c_syms = sp.symbols(f'c_0:{n_vars}')
    xi_syms = sp.symbols(f'xi_1:{n_vars + 1}')

    var_maps = [J_syms[i] * xi_syms[i] + c_syms[i] for i in range(n_vars)]
    subs_map = list(zip(vars, var_maps))
    params = list(J_syms) + list(c_syms)

    computed_Mxp = precomputed['computed_Mxp']
    combined_ops = precomputed['combined_ops']
    use_sparse = precomputed.get('use_sparse', False)

    # Cache for Mxp Kronecker products: powers_tuple -> kron(Mxp[p0], ..., Mxp[pn])
    kron_cache = {}

    hom_evaluators = []   # list of (scalar_func, product_matrix)
    inhom_evaluators = [] # list of (scalar_func, kron_op) for source terms
    c_syms_set = set(c_syms)
    center_independent = True  # True if no scalar depends on center symbols

    for (deriv_signature, k_pow), coeff in terms_dict.items():
        # Detect source terms
        term_has_derivs = any(
            alpha_tuple and any(order > 0 for order in alpha_tuple)
            for alpha_tuple, power_int in deriv_signature
        ) if deriv_signature else False
        is_source = (k_pow == 0 and not term_has_derivs)

        # 1. Symbolic affine substitution (with special handling for Piecewise)
        # We scalarize Piecewise conditions by substituting centers instead of the full affine map.
        # This assumes jumps align with element boundaries for spectral accuracy,
        # but prevents PolynomialError in any case.
        def _affine_map_with_piecewise(e):
            if isinstance(e, sp.Piecewise):
                return sp.Piecewise(*[(_affine_map_with_piecewise(val), cond.subs(list(zip(vars, c_syms))))
                                      for val, cond in e.args])
            if not e.args:
                return e.subs(subs_map)
            return e.func(*[_affine_map_with_piecewise(arg) for arg in e.args])

        new_coeff = _affine_map_with_piecewise(coeff)

        # 2. Chain rule scaling (symbolic)
        for item in deriv_signature:
            if not item or not isinstance(item, tuple) or len(item) < 2:
                continue
            alpha_tuple, power = item
            for i, order in enumerate(alpha_tuple):
                if order > 0:
                    new_coeff *= (1 / J_syms[i]) ** (order * power)

        transformed = sp.expand(new_coeff)

        # 3+4. Extract monomials in xi, using factored Taylor expansion for non-polynomial terms
        poly_dict = factored_series_poly_dict(transformed, xi_syms, truncated_order)

        term_key = (deriv_signature, k_pow)
        combined_op = combined_ops.get(term_key)

        for powers, scalar_expr in poly_dict.items():
            if center_independent and scalar_expr.free_symbols & c_syms_set:
                center_independent = False
            scalar_func = sp.lambdify(params, scalar_expr, modules='numpy')

            # Build/cache Mxp Kronecker product for this monomial
            if powers not in kron_cache:
                mono_ops = []
                for i, p in enumerate(powers):
                    if p not in computed_Mxp:
                        computed_Mxp[p] = multiply_matrix.M_x_power(p=p, deg=d, deg_out=d_out)
                    mono_ops.append(computed_Mxp[p])
                if use_sparse:
                    kron_cache[powers] = functools.reduce(
                        lambda a, b: kron(csr_matrix(a), csr_matrix(b), format='csr'), mono_ops)
                else:
                    kron_cache[powers] = functools.reduce(np.kron, mono_ops)

            kron_op = kron_cache[powers]

            if is_source:
                inhom_evaluators.append((scalar_func, kron_op))
            else:
                product_matrix = kron_op @ combined_op
                hom_evaluators.append((scalar_func, product_matrix))

    return hom_evaluators, inhom_evaluators, center_independent


def _evaluate_operator_fast(evaluators, J, center):
    """Evaluate the operator for a single element using precomputed evaluators.

    Args:
        evaluators: list of (scalar_func, product_matrix) pairs
        J: array of Jacobian values per dimension (hi - lo) / 2
        center: array of center values per dimension (hi + lo) / 2

    Returns:
        A: the operator matrix for this element
    """
    params = list(J) + list(center)
    A = None
    for scalar_func, product_matrix in evaluators:
        s = float(scalar_func(*params))
        if s != 0.0:
            term = s * product_matrix
            A = term if A is None else A + term
    return A if A is not None else 0.0


def _precompute_operators(d, d_out, terms_keys, n_vars, use_sparse=False):
    """Precompute element-independent operators shared across all mesh elements.

    Returns a dict with GT, Id, combined_ops (per term key), and computed_Mxp cache.
    Source-term keys (k_pow=0, no derivatives) are skipped since they depend on
    element-specific boundary operators.
    """
    GT = derivative_matrix.diff_matrix(deg=d)
    Id = np.eye(d + 1)

    GT_powers = {}
    combined_ops = {}
    computed_Mxp = {0: multiply_matrix.M_x_power(p=0, deg=d, deg_out=d_out)}

    def _kron_reduce(ops):
        if use_sparse:
            return functools.reduce(lambda a, b: kron(csr_matrix(a), csr_matrix(b), format='csr'), ops)
        return functools.reduce(np.kron, ops)

    for (deriv_signature, k_pow) in terms_keys:
        # Detect source terms: k_pow=0 and no actual derivatives
        term_has_derivs = any(
            alpha_tuple and any(order > 0 for order in alpha_tuple)
            for alpha_tuple, power_int in deriv_signature
        ) if deriv_signature else False

        if k_pow == 0 and not term_has_derivs:
            continue  # source terms use element-specific D_regs

        var_ops = [None] * n_vars

        for (alpha_tuple, power_int) in deriv_signature:
            if not alpha_tuple:
                continue
            for i, deriv_order in enumerate(alpha_tuple):
                if deriv_order > 0:
                    if deriv_order not in GT_powers:
                        GT_powers[deriv_order] = np.linalg.matrix_power(GT, deriv_order)
                    var_ops[i] = GT_powers[deriv_order]

        if k_pow == 1:
            for i in range(n_vars):
                if var_ops[i] is None:
                    var_ops[i] = Id

        for i in range(n_vars):
            if var_ops[i] is None:
                var_ops[i] = Id

        combined_ops[(deriv_signature, k_pow)] = _kron_reduce(var_ops)

    return {
        'GT': GT,
        'Id': Id,
        'GT_powers': GT_powers,
        'combined_ops': combined_ops,
        'computed_Mxp': computed_Mxp,
        'use_sparse': use_sparse,
    }


def build_multivar_equation_operator(
        d: int,
        d_out: int,
        terms_dict: Dict[Tuple[Tuple[Tuple[int, ...], int], int], Any],
        vars: List[sp.Symbol],
        truncated_order: Optional[int] = None,
        regular_data: Optional[Tuple[Tuple, Any]] = (None, None),
        regular_data_type: Literal['value', 'derivative'] = None,
        precomputed: Optional[dict] = None):
    """
    Constructs the effective Hamiltonian for a Multivariate DE with Rank 1 restriction.
    Validates necessity of regular_data for source terms.
    """
    n_vars = len(vars)
    regular_coords, regular_value = regular_data if regular_data is not None else (None, None)

    # 1. Determine Tensor Rank and Necessity of Regular Data
    need_regular = False

    for (deriv_signature, k_pow), _ in terms_dict.items():
        var_rank_counts = [0] * n_vars

        # --- Check for Pure Source Term ---
        term_has_derivs = False
        for alpha_tuple, power_int in deriv_signature:
            if alpha_tuple and any(order > 0 for order in alpha_tuple):
                term_has_derivs = True
                break

        if k_pow == 0 and not term_has_derivs:
            need_regular = True

        # --- Check Tensor Rank 1 Constraint ---
        # Check k_pow
        if k_pow > 1:
            raise NotImplementedError(f"k_pow={k_pow} exceeds supported Rank 1.")
        if k_pow == 1:
            for i in range(n_vars):
                var_rank_counts[i] += 1

        # Check deriv_signature
        for alpha_tuple, power_int in deriv_signature:
            if not alpha_tuple: continue
            if power_int > 1:
                raise NotImplementedError(f"Derivative power {power_int} exceeds supported Rank 1.")

            for i, deriv_order in enumerate(alpha_tuple):
                if deriv_order > 0:
                    var_rank_counts[i] += 1

        # Final Rank Check per variable
        if any(r > 1 for r in var_rank_counts):
            raise NotImplementedError(
                "Combined derivative and function operators for a single variable detected. "
                "Multivariate reduction strictly supports Rank 1."
            )

    # 2. Validate Regular Data Input
    has_regular_data = (regular_coords is not None) and (regular_value is not None)

    if need_regular and not has_regular_data:
        raise ValueError(
            "Regular data (constraints) are required for pure source terms but were not provided."
        )

    D_regs = [None] * n_vars
    if has_regular_data:
        if not need_regular:
            warnings.warn(
                "Regular data provided but not required (no pure source terms found).",
                UserWarning
            )
            # We do not build D_regs if not needed to save compute/avoid potential errors
        else:
            # Build D_regs only if needed and provided
            if regular_data_type not in ['value', 'derivative']:
                raise ValueError("regular_data_type must be 'value' or 'derivative'.")

            # Ensure reg_coords matches n_vars
            if len(regular_coords) != n_vars:
                raise ValueError(
                    f"Length of regular_coords ({len(regular_coords)}) does not match number of variables ({n_vars}).")

            for i in range(n_vars):
                # Note: Assuming build_boundary_matrix handles scalar y correctly or ignores it for the operator structure
                D_regs[i] = build_boundary_matrix(type=regular_data_type, x=regular_coords[i], y=None,
                                                                  deg=d, deg_out=d)

    # 3. Pre-compute Base Matrices (or use precomputed cache)
    if precomputed is not None:
        GT = precomputed['GT']
        Id = precomputed['Id']
        GT_powers = precomputed['GT_powers']
        combined_ops_cache = precomputed['combined_ops']
        computed_Mxp = precomputed['computed_Mxp']
    else:
        GT = derivative_matrix.diff_matrix(deg=d)
        Id = np.eye(d + 1)
        GT_powers = {}
        combined_ops_cache = {}
        computed_Mxp = {}

    A_total = 0.0

    # 4. Iterate over terms to build Operator A
    for (deriv_signature, k_pow), coeff in terms_dict.items():

        term_key = (deriv_signature, k_pow)

        # --- A. Construct the Raw Tensor Operator ---
        if term_key in combined_ops_cache:
            # Use precomputed combined_op (element-independent)
            combined_op = combined_ops_cache[term_key]
            is_source_term = False
        else:
            # Build combined_op from scratch (source terms, or no precomputation)
            var_ops = [None] * n_vars

            # Assign Derivatives
            for (alpha_tuple, power_int) in deriv_signature:
                if not alpha_tuple:
                    continue
                for i, deriv_order in enumerate(alpha_tuple):
                    if deriv_order > 0:
                        if deriv_order in GT_powers:
                            var_ops[i] = GT_powers[deriv_order]
                        else:
                            var_ops[i] = np.linalg.matrix_power(GT, deriv_order)

            # Assign Functions
            if k_pow == 1:
                for i in range(n_vars):
                    if var_ops[i] is None:
                        var_ops[i] = Id

            # Check for Pure Source Term (this specific term)
            is_source_term = False
            if k_pow == 0 and all(op is None for op in var_ops):
                is_source_term = True
                # For source terms, we strictly use the Boundary Operators
                # D_regs is guaranteed to be populated here due to Step 2 validation
                var_ops = D_regs

            # Fill remaining Nones with Identity (for non-source terms)
            if not is_source_term:
                for i in range(n_vars):
                    if var_ops[i] is None:
                        var_ops[i] = Id

            # Combined = M_var0 ⊗ M_var1 ⊗ ... ⊗ M_varN
            combined_op = functools.reduce(np.kron, var_ops)

        if is_source_term:
            # Apply scaling by the constraint value y_s (reg_val)
            if regular_value is None:
                raise ValueError("Internal Error: reg_val is None for source term.")
            combined_op = combined_op / float(regular_value)

        # --- B. Apply Coefficient ---
        coeff_op, computed_Mxp = _free_coeff_op_multivariable(
            d=d,
            d_out=d_out,
            c=coeff,
            vars=vars,
            truncated_order=truncated_order,
            computed_Mxp = computed_Mxp
        )

        A_total += coeff_op @ combined_op

    return A_total


def get_hamiltonian_from_operator(A):
    """Computes A.T @ A for both dense and sparse matrices."""
    return A.T @ A


def sem_multivar_equation_hamiltonian(d: int, d_out: int, diff_eq, func, vars: list | tuple, mesh: RectMesh,
                                      truncated_order: Optional[int] = None,
                                      regular_data: Optional[Tuple[Tuple, Any]] = (None, None),
                                      regular_data_type: Literal['value', 'derivative'] = None,
                                      intg_cond_order: int = 0,
                                      local_basis_transform: Optional[np.ndarray] = None,
                                      sparse: bool = False, get_hamiltonian: bool = True):

    if not (isinstance(intg_cond_order, int) and intg_cond_order >= 0):
        raise ValueError("intg_cond_order must be a non-negative integer.")
    if local_basis_transform is not None:
        if local_basis_transform.shape != (d+1, d+1):
            raise ValueError("local_basis_transform must have shape (d+1, d+1).")

    endpoints = mesh.endpoints
    assert endpoints.ndim == 3
    num_elements = mesh.num_elems
    lo = mesh.endpoints[:, 0]
    hi = mesh.endpoints[:, 1]

    N, N_out = d + 1, d_out + 1
    num_vars = len(vars)
    
    in_block_size = N ** num_vars
    out_block_size = N_out ** num_vars

    parser = multivar_equation_parsing.ElementalMultiVariateEquationParser()
    terms = parser.parse_equation(diff_eq, func)

    ## Find which element x_s belongs to and the corresponding local variable for regularization
    e_s = None
    xi_s_in_e_s = None

    if regular_data is not None:
        (coord_s, val_s) = regular_data
        coord_s = np.array(coord_s)
        e_s = mesh.find_elements(coord_s)

        if isinstance(e_s, int):
            pass
        elif len(e_s) == 0:
            raise ValueError(f"Regularization point coord_s={coord_s} is not found in the mesh.")
        else:
            raise ValueError(f"Regularization point coord_s={coord_s} is found in multiple elements: {e_s}. A regularization point should be uniquely assigned to one element.")
        xi_s_in_e_s = (2 * coord_s - (lo[e_s] + hi[e_s])) / (hi[e_s] - lo[e_s])  # Map to [-1, 1]

    else:
        coord_s, val_s = None, None


    # Precompute element-independent operators once (Strategy B optimization)
    precomputed = _precompute_operators(d=d, d_out=d_out, terms_keys=terms.keys(), n_vars=num_vars, use_sparse=sparse)

    # Precompute symbolic evaluators once — eliminates SymPy from element loop (Strategy A)
    hom_evaluators, inhom_evaluators, center_independent = _prepare_fast_evaluators(
        terms_dict=terms, vars=vars, d=d, d_out=d_out,
        n_vars=num_vars, truncated_order=truncated_order, precomputed=precomputed
    )

    # Precompute inhomogeneous combined_op if source terms exist
    inhom_combined_op = None
    if inhom_evaluators and e_s is not None and xi_s_in_e_s is not None:
        D_regs = []
        for i in range(num_vars):
            D_regs.append(build_boundary_matrix(
                type=regular_data_type, x=xi_s_in_e_s[i], y=None, deg=d, deg_out=d))
        inhom_combined_op = functools.reduce(np.kron, D_regs) / float(val_s)

    has_inhom = (inhom_evaluators and inhom_combined_op is not None)

    # --- Strategy C: Uniform mesh fast path ---
    # When mesh is uniform AND coefficients don't depend on element center,
    # all homogeneous blocks are identical → use kron(I, block) directly.
    use_uniform_fast_path = mesh.is_uniform and center_independent

    if use_uniform_fast_path:
        # Compute a single representative block (element 0)
        J_0 = (hi[0] - lo[0]) / 2.0
        center_0 = (hi[0] + lo[0]) / 2.0
        A_hom_block = _evaluate_operator_fast(hom_evaluators, J_0, center_0)

        if sparse:
            A_total = kron(eye(num_elements), csr_matrix(A_hom_block), format='csr')
        else:
            A_total = np.kron(np.eye(num_elements), A_hom_block)

        # Inhomogeneous terms still need per-element handling (off-diagonal blocks)
        if has_inhom:
            if sparse:
                A_inhom_total = lil_matrix((out_block_size * num_elements, in_block_size * num_elements))
            for e in range(num_elements):
                J_e = (hi[e] - lo[e]) / 2.0
                center_e = (hi[e] + lo[e]) / 2.0
                A_inhom_coeff = _evaluate_operator_fast(inhom_evaluators, J_e, center_e)
                A_inhom = A_inhom_coeff @ inhom_combined_op

                r0_inhom, c0_inhom = e * out_block_size, e_s * in_block_size
                r1_inhom, c1_inhom = (e + 1) * out_block_size, (e_s + 1) * in_block_size
                if sparse:
                    A_inhom_total[r0_inhom:r1_inhom, c0_inhom:c1_inhom] += csr_matrix(A_inhom)
                else:
                    A_total[r0_inhom:r1_inhom, c0_inhom:c1_inhom] += A_inhom
            if sparse:
                A_total = A_total + A_inhom_total.tocsr()

    else:
        # --- Normal per-element loop (Strategy A + E) ---
        hom_blocks = []
        if not sparse:
            A_total = np.zeros((out_block_size * num_elements, in_block_size * num_elements))
        if has_inhom and sparse:
            A_inhom_total = lil_matrix((out_block_size * num_elements, in_block_size * num_elements))

        for e in range(num_elements):
            J_e = (hi[e] - lo[e]) / 2.0
            center_e = (hi[e] + lo[e]) / 2.0

            A_hom = _evaluate_operator_fast(hom_evaluators, J_e, center_e)

            if sparse:
                hom_blocks.append(csr_matrix(A_hom))
            else:
                r0, c0 = e * out_block_size, e * in_block_size
                r1, c1 = (e + 1) * out_block_size, (e + 1) * in_block_size
                A_total[r0:r1, c0:c1] = A_hom

            if has_inhom:
                A_inhom_coeff = _evaluate_operator_fast(inhom_evaluators, J_e, center_e)
                A_inhom = A_inhom_coeff @ inhom_combined_op

                r0_inhom, c0_inhom = e * out_block_size, e_s * in_block_size
                r1_inhom, c1_inhom = (e + 1) * out_block_size, (e_s + 1) * in_block_size
                if sparse:
                    A_inhom_total[r0_inhom:r1_inhom, c0_inhom:c1_inhom] += csr_matrix(A_inhom)
                else:
                    A_total[r0_inhom:r1_inhom, c0_inhom:c1_inhom] += A_inhom

        # Assemble sparse block-diagonal from collected blocks (Strategy E)
        if sparse:
            from scipy.sparse import block_diag as sp_block_diag
            A_total = sp_block_diag(hom_blocks, format='csr')
            if has_inhom:
                A_total = A_total + A_inhom_total.tocsr()

    # --- Integral Conditioning ---
    if intg_cond_order > 0:
        # Construct 1D integration matrix
        # Since we are left-multiplying A (which has output dimension N_out),
        # we need S to act on coefficients of degree d_out.
        # We enforce deg_out=d_out to maintain square matrix size (N_out x N_out).
        S_1D = intg_matrix(a=-1.0, deg=d_out, deg_out=d_out)
        
        if intg_cond_order > 1:
            S_1D = np.linalg.matrix_power(S_1D, intg_cond_order)
            
        # Create tensor product S_block = S_1D (x) ... (x) S_1D
        S_block = S_1D
        for _ in range(num_vars - 1):
            S_block = np.kron(S_block, S_1D)
            
        if sparse:
            S_block_sparse = csr_matrix(S_block)
            # Create global block-diagonal S
            S_total = kron(eye(num_elements), S_block_sparse, format='csr')
            
            # Apply preconditioning: A_total = S_total @ A_total (Left Multiplication)
            if isinstance(A_total, lil_matrix):
                A_total = A_total.tocsr()
            
            A_total = S_total @ A_total
            
        else:
            # Dense case: Apply S_block to each element block row-wise
            # A_total is (N_out**num_vars * M, ...)
            # We iterate over row blocks
            for e in range(num_elements):
                row_slice = slice(e * out_block_size, (e + 1) * out_block_size)
                A_total[row_slice, :] = S_block @ A_total[row_slice, :]

    if local_basis_transform is not None:
        # local_basis_transform C is (N, N). We need to apply it to each variable in the tensor product.
        C = local_basis_transform
        C_joint = C
        for _ in range(num_vars - 1):
            C_joint = np.kron(C_joint, C)

        if sparse:
            C_total = kron(eye(num_elements), csr_matrix(C_joint), format='csr')
            A_total = A_total @ C_total.T
        else:
            # Apply C_block to each element's column block to avoid full N_total x N_total multiplication
            for e in range(num_elements):
                col_slice = slice(e * in_block_size, (e + 1) * in_block_size)
                A_total[:, col_slice] = A_total[:, col_slice] @ C_joint.T


    if sparse and isinstance(A_total, lil_matrix):
        A_total = A_total.tocsr()

    if get_hamiltonian is False:
        return A_total
    else:
        return get_hamiltonian_from_operator(A_total)


if __name__ == "__main__":
    # Example usage
    x, y = sp.symbols('x y')
    f = sp.Function('f')(x, y)
    fx = sp.Derivative(f, x)
    fxx = sp.Derivative(f, x, 2)
    fy = sp.Derivative(f, y)
    fyy = sp.Derivative(f, y, 2)
    diff_eq = fxx + fyy + sp.sin(x) * fx + sp.cos(y) * f + (x*y-1)
    parser = multivar_equation_parsing.ElementalMultiVariateEquationParser()
    terms_dict = parser.parse_equation(diff_eq, f)
    print(terms_dict)

    data_s = ((0.5, 0.5), 1.0)  # Example regular data: (reg_coords, reg_val)

    A_op = build_multivar_equation_operator(d=5, d_out=7,
                                            terms_dict=terms_dict,
                                            vars=[x, y],
                                            truncated_order=3,
                                            regular_data=data_s,
                                            regular_data_type='value')
    print("Constructed multivariate equation operator shape:", A_op.shape)


    H = get_hamiltonian_from_operator(A_op)
    print("Constructed Hamiltonian shape:", H.shape)

    # Example usage of sem_multivar_equation_hamiltonian
    x_nodes = np.linspace(-1, 1, 2)
    y_nodes = np.linspace(-1, 1, 2)
    mesh = RectMesh(nodes = [x_nodes, y_nodes])

    H_sem = sem_multivar_equation_hamiltonian(d=5, d_out=7, diff_eq=diff_eq, func=f, vars=[x, y], mesh=mesh,
                                              truncated_order=3, regular_data=data_s, regular_data_type='value')
    print("Constructed SEM Hamiltonian shape:", H_sem.shape)

    assert np.linalg.norm(H - H_sem) < 1e-10, "Hamiltonians from both methods do not match!"