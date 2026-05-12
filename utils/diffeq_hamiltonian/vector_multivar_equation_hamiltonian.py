import numpy as np
import sympy as sp
import functools
from typing import Dict, Tuple, Any, Optional, List, Literal

from src.utils.basic_operators import derivative_matrix, multiply_matrix
from src.utils.basic_operators.derivative_matrix import intg_matrix
from src.utils.boundary_hamiltonian import simple_boundary
from src.utils.diffeq_hamiltonian import vector_multivar_equation_parsing
from src.utils.diffeq_hamiltonian.symbolic_utils import factored_series_poly_dict
from src.utils.meshing import RectMesh
from scipy.sparse import lil_matrix, csr_matrix, kron, eye


def _free_coeff_op_multivariable(d: int, d_out: int, c: sp.Expr, vars: list, truncated_order: int = None,
                                 computed_Mxp: dict = None):
    """
    Constructs the matrix operator for multiplying by a scalar coefficient function c(x).
    """
    c = sp.sympify(c)
    vars = list(vars)
    if computed_Mxp is None or computed_Mxp == {}:
        computed_Mxp = {}

    # 1. Handle Scalar Constants
    if c.is_Number:
        # p=0 represents Identity in M_x_power
        if 0 not in computed_Mxp:
            computed_Mxp[0] = multiply_matrix.M_x_power(p=0, deg=d, deg_out=d_out)

        # Tensor product of Identity matrices scaled by c
        return float(c) * functools.reduce(np.kron, [computed_Mxp[0]] * len(vars)), computed_Mxp

    # 2+3. Handle Non-Polynomials via Factored Taylor Expansion + Extract Monomials
    poly_dict = factored_series_poly_dict(c, tuple(vars), truncated_order)
    A = 0
    for powers, scalar in poly_dict.items():
        term_operators = []

        for i, p in enumerate(powers):
            if p not in computed_Mxp:
                M_p = multiply_matrix.M_x_power(p=p, deg=d, deg_out=d_out)
                computed_Mxp[p] = M_p
            else:
                M_p = computed_Mxp[p]
            term_operators.append(M_p)

        A += float(scalar) * functools.reduce(np.kron, term_operators)

    return A.astype(float), computed_Mxp


def _precompute_operators_vector(d, d_out, all_parsed_eqs, n_vars, use_sparse=False):
    """Precompute element-independent operators shared across all mesh elements (vector case).

    Strategy B: Caches GT, GT_powers, combined_ops, and Mxp matrices.
    Source-term keys (no active component) are skipped.
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

    for parsed_eq in all_parsed_eqs:
        for (deriv_sig, f_sig) in parsed_eq.keys():
            if (deriv_sig, f_sig) in combined_ops:
                continue
            active_comp_idx = None
            if f_sig:
                active_comp_idx = f_sig[0][0]
            elif deriv_sig:
                active_comp_idx = deriv_sig[0][0]
            if active_comp_idx is None:
                continue  # source terms use element-specific D_regs

            var_ops = [None] * n_vars
            for (c_idx, alpha_tuple, power_int) in deriv_sig:
                if c_idx != active_comp_idx:
                    continue
                for i, deriv_order in enumerate(alpha_tuple):
                    if deriv_order > 0:
                        if deriv_order not in GT_powers:
                            GT_powers[deriv_order] = np.linalg.matrix_power(GT, deriv_order)
                        var_ops[i] = GT_powers[deriv_order]
            for i in range(n_vars):
                if var_ops[i] is None:
                    var_ops[i] = Id
            combined_ops[(deriv_sig, f_sig)] = _kron_reduce(var_ops)

    return {
        'GT': GT, 'Id': Id, 'GT_powers': GT_powers,
        'combined_ops': combined_ops, 'computed_Mxp': computed_Mxp,
        'use_sparse': use_sparse,
    }


def _prepare_fast_evaluators_vector(terms_dict, vars, d, d_out, n_vars,
                                     num_components, target_eq_idx,
                                     truncated_order, precomputed):
    """Perform all SymPy symbolic work once for one equation; return lambdified evaluators.

    Strategy A adapted for vector-valued PDEs. Each evaluator entry includes
    target_block_idx to route contributions to the correct component column.

    Returns
    -------
    hom_evaluators : list of (scalar_func, product_matrix, target_block_idx)
    inhom_evaluators : list of (scalar_func, kron_op, target_block_idx)
    center_independent : bool
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

    kron_cache = {}
    hom_evaluators = []
    inhom_evaluators = []
    c_syms_set = set(c_syms)
    center_independent = True

    for (deriv_sig, f_sig), coeff in terms_dict.items():
        active_comp_idx = None
        if f_sig:
            active_comp_idx = f_sig[0][0]
        elif deriv_sig:
            active_comp_idx = deriv_sig[0][0]
        is_source = (active_comp_idx is None)
        target_block_idx = active_comp_idx if not is_source else target_eq_idx

        # 1. Symbolic affine substitution
        new_coeff = coeff.subs(subs_map)

        # 2. Chain rule scaling
        for (c_idx, alpha_tuple, power_int) in deriv_sig:
            for i, order in enumerate(alpha_tuple):
                if order > 0:
                    new_coeff *= (1 / J_syms[i]) ** (order * power_int)
        transformed = sp.expand(new_coeff)

        # 3+4. Extract monomials in xi, using factored Taylor expansion for non-polynomial terms
        poly_dict = factored_series_poly_dict(transformed, xi_syms, truncated_order)
        term_key = (deriv_sig, f_sig)
        combined_op = combined_ops.get(term_key)

        for powers, scalar_expr in poly_dict.items():
            if center_independent and scalar_expr.free_symbols & c_syms_set:
                center_independent = False
            scalar_func = sp.lambdify(params, scalar_expr, modules='numpy')

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
                inhom_evaluators.append((scalar_func, kron_op, target_block_idx))
            else:
                product_matrix = kron_op @ combined_op
                hom_evaluators.append((scalar_func, product_matrix, target_block_idx))

    return hom_evaluators, inhom_evaluators, center_independent


def _evaluate_operator_row_fast(evaluators, J, center, num_components):
    """Evaluate a row of component blocks for one equation using precomputed evaluators.

    Pure numeric: scalar function calls + matrix-scalar multiplications. No SymPy.
    """
    params = list(J) + list(center)
    blocks = [None] * num_components
    for scalar_func, product_matrix, target_idx in evaluators:
        s = float(scalar_func(*params))
        if s != 0.0:
            term = s * product_matrix
            if blocks[target_idx] is None:
                blocks[target_idx] = term
            else:
                blocks[target_idx] = blocks[target_idx] + term
    for i in range(num_components):
        if blocks[i] is None:
            blocks[i] = 0.0
    return blocks


def build_vector_equation_operator_row(
        d: int,
        d_out: int,
        terms_dict: Dict,
        vars: List[sp.Symbol],
        num_components: int,
        target_eq_idx: int,
        truncated_order: Optional[int] = None,
        regular_data: Optional[Tuple[Any, Any]] = None,
        regular_data_type: Literal['value', 'derivative'] = None):
    """
    Constructs a ROW of blocks for the global system matrix corresponding to ONE equation.

    Returns
    -------
    list[np.ndarray]
        A list of length `num_components`. The j-th element is the operator block
        representing how component `j` affects this specific equation.
    """
    n_vars = len(vars)
    regular_coords, regular_value = regular_data if regular_data is not None else (None, None)

    # --- 1. Validate Linearity and Necessity of Regular Data ---
    need_regular = False

    # keys are: ( (deriv_signature), (f_signature) )
    # deriv_sig: tuple of (comp_idx, alpha_tuple, power)
    # f_sig: tuple of (comp_idx, power)

    for (deriv_sig, f_sig), _ in terms_dict.items():
        # Calculate total degree of unknown functions
        deg = sum(m for _, _, m in deriv_sig) + sum(k for _, k in f_sig)

        if deg == 0:
            need_regular = True  # Pure source term
        elif deg > 1:
            raise NotImplementedError(
                f"Nonlinear term detected (Degree {deg}). "
                "This solver currently only supports Linear (Rank 1) coupled systems."
            )

    # --- 2. Build Regularization (Boundary) Matrices ---
    has_regular_data = (regular_coords is not None) and (regular_value is not None)

    if need_regular and not has_regular_data:
        raise ValueError("Regular data (constraints) required for source terms but not provided.")

    D_regs = [None] * n_vars
    if has_regular_data:
        if regular_data_type not in ['value', 'derivative']:
            raise ValueError("regular_data_type must be 'value' or 'derivative'.")

        for i in range(n_vars):
            # Note: deg_out=d keeps the matrix square (d+1 x d+1) for the evaluation operator
            # This is important so that it can be composed with coeff_op (d_out x d)
            D_regs[i] = simple_boundary.build_boundary_matrix(type=regular_data_type, x=regular_coords[i], y=None,
                                                              deg=d, deg_out=d)

    # --- 3. Pre-compute Base Matrices ---
    GT = derivative_matrix.diff_matrix(deg=d)
    Id = np.eye(d + 1)

    # We need one operator block per unknown component
    row_blocks = [0.0] * num_components
    computed_Mxp = {}

    # --- 4. Process Terms ---
    for (deriv_sig, f_sig), coeff in terms_dict.items():

        # Determine the active component for this term
        active_comp_idx = None

        if f_sig:
            active_comp_idx = f_sig[0][0]  # (idx, power)
        elif deriv_sig:
            active_comp_idx = deriv_sig[0][0]  # (idx, alpha, power)

        is_source_term = (active_comp_idx is None)

        # Source terms are added to the block corresponding to the target equation's primary variable
        target_block_idx = active_comp_idx if not is_source_term else target_eq_idx

        # Build Spatial Operators for this term
        var_ops = [None] * n_vars

        if is_source_term:
            # Source terms use the regularization operators (point evaluation)
            var_ops = D_regs
        else:
            # Linear term: Find derivatives
            for (c_idx, alpha_tuple, power_int) in deriv_sig:
                if c_idx != active_comp_idx: continue

                for i, deriv_order in enumerate(alpha_tuple):
                    if deriv_order > 0:
                        var_ops[i] = np.linalg.matrix_power(GT, deriv_order)

            # Fill missing slots with Identity
            for i in range(n_vars):
                if var_ops[i] is None:
                    var_ops[i] = Id

        # Construct Tensor Product
        combined_op = functools.reduce(np.kron, var_ops)

        if is_source_term:
            # Apply source scaling (1 / constraint_value)
            if regular_value is None:
                raise ValueError("Internal Error: regular_value missing for source term.")
            combined_op = combined_op / float(regular_value)

        # Apply Coefficient Function c(x)
        # c(x) is defined in the local coordinate system of the current element 'e'
        coeff_op, computed_Mxp = _free_coeff_op_multivariable(
            d=d, d_out=d_out, c=coeff, vars=vars,
            truncated_order=truncated_order, computed_Mxp=computed_Mxp
        )

        # Add to the appropriate block
        term_op = coeff_op @ combined_op

        row_blocks[target_block_idx] += term_op

    return row_blocks


def get_hamiltonian_from_operator(A: np.ndarray):
    return A.T @ A


def sem_vector_multivar_equation_hamiltonian(d: int, d_out: int, diff_eqs: List[Any], funcs: List[sp.Expr],
                                             vars: List[sp.Symbol], mesh: RectMesh,
                                             truncated_order: Optional[int] = None,
                                             regular_data_list: Optional[List[Tuple[Tuple, Any]]] = None,
                                             regular_data_type: Literal['value', 'derivative'] = None,
                                             intg_cond_order: int = 0,
                                             local_basis_transform: Optional[np.ndarray] = None,
                                             sparse: bool = False, get_hamiltonian: bool = True):
    """
    Main entry point for Vector PDEs.

    Parameters
    ----------
    diff_eqs : List[Expr]
        List of sympy expressions, one for each equation.
    funcs : List[Function]
        List of unknown functions [u, v, ...].
    regular_data_list : List[Tuple]
        A list matching `diff_eqs` in length. Each item is ((x,y,..), value)
        specifying the constraint for that specific equation.
    """
    if not (isinstance(intg_cond_order, int) and intg_cond_order >= 0):
        raise ValueError("intg_cond_order must be a non-negative integer.")
    if local_basis_transform is not None:
        if local_basis_transform.shape != (d+1, d+1):
            raise ValueError("local_basis_transform must have shape (d+1, d+1).")

    endpoints = mesh.endpoints
    assert endpoints.ndim in [2, 3]
    if endpoints.ndim == 2:
        # 1D case: reshape to (num_elements, 1, 2)
        endpoints = endpoints[:, :, np.newaxis]
    num_elements = endpoints.shape[0]
    lo = endpoints[:, 0, :]
    hi = endpoints[:, 1, :]

    N = d + 1
    N_out = d_out + 1
    num_vars = len(vars)
    num_funcs = len(funcs)
    num_eqs = len(diff_eqs)

    if num_eqs != num_funcs:
        raise ValueError(f"Number of equations ({num_eqs}) must match number of functions ({num_funcs}).")

    # 1. Parse all equations
    parser = vector_multivar_equation_parsing.VectorMultiVariateEquationParser()

    parsed_eqs = []
    for eq in diff_eqs:
        parsed_eqs.append(parser.parse_equation(eq, funcs))

    # 2. Map Regularization Data to Elements
    # reg_infos[eq_idx] = (element_index, local_coords_xi, value)
    reg_infos = [None] * num_eqs

    if regular_data_list:
        if len(regular_data_list) != num_eqs:
            raise ValueError("regular_data_list must have same length as diff_eqs")

        for i, data in enumerate(regular_data_list):
            if data is None: continue

            coord_s, val_s = data
            coord_s = np.array(coord_s)

            # Find which element contains this point
            e_s = mesh.find_elements(coord_s)
            if isinstance(e_s, int):
                pass
            elif len(e_s) == 0:
                raise ValueError(f"Regularization point {coord_s} for Eq {i} is outside all elements.")
            else:  # len(e_s) > 1
                raise ValueError(
                    f"Regularization point {coord_s} for Eq {i} is found in multiple elements: {e_s}. Each regularization point should be uniquely located within a single element.")

            # Map global coord to local xi for the containing element
            xi_s = (2 * coord_s - (lo[e_s] + hi[e_s])) / (hi[e_s] - lo[e_s])
            reg_infos[i] = (e_s, xi_s, val_s)

    # 3. Precompute operators (Strategy B) and fast evaluators (Strategy A)
    block_rows = N_out ** num_vars
    block_cols = N ** num_vars

    # Split each equation's terms into hom/inhom
    hom_parsed_eqs = []
    inhom_parsed_eqs = []
    for eq_idx in range(num_eqs):
        hom_terms = parsed_eqs[eq_idx].copy()
        inhom_terms = {}
        source_keys = []
        for k in hom_terms.keys():
            d_sig, f_sig = k
            deg = sum(p for _, p in f_sig) + sum(p for _, _, p in d_sig)
            if deg == 0:
                source_keys.append(k)
        for k in source_keys:
            inhom_terms[k] = hom_terms.pop(k)
        hom_parsed_eqs.append(hom_terms)
        inhom_parsed_eqs.append(inhom_terms)

    # Strategy B: precompute element-independent operators
    precomputed = _precompute_operators_vector(
        d=d, d_out=d_out, all_parsed_eqs=hom_parsed_eqs + inhom_parsed_eqs,
        n_vars=num_vars, use_sparse=sparse)

    # Strategy A: prepare fast evaluators per equation (all SymPy done here)
    all_hom_evaluators = []
    all_inhom_evaluators = []
    all_center_independent = True
    for eq_idx in range(num_eqs):
        hom_ev, _, ci = _prepare_fast_evaluators_vector(
            terms_dict=hom_parsed_eqs[eq_idx], vars=vars, d=d, d_out=d_out,
            n_vars=num_vars, num_components=num_funcs, target_eq_idx=eq_idx,
            truncated_order=truncated_order, precomputed=precomputed)
        all_hom_evaluators.append(hom_ev)
        # Always prepare inhom evaluators for source terms
        _, inhom_ev_eq, ci_inhom = _prepare_fast_evaluators_vector(
            terms_dict=inhom_parsed_eqs[eq_idx], vars=vars, d=d, d_out=d_out,
            n_vars=num_vars, num_components=num_funcs, target_eq_idx=eq_idx,
            truncated_order=truncated_order, precomputed=precomputed)
        all_inhom_evaluators.append(inhom_ev_eq)
        if not (ci and ci_inhom):
            all_center_independent = False

    # Precompute inhomogeneous combined_ops per equation (D_regs kron / val)
    inhom_combined_ops = [None] * num_eqs
    for eq_idx in range(num_eqs):
        if all_inhom_evaluators[eq_idx] and reg_infos[eq_idx] is not None:
            r_e, r_xi, r_val = reg_infos[eq_idx]
            D_regs = []
            for i in range(num_vars):
                D_regs.append(simple_boundary.build_boundary_matrix(
                    type=regular_data_type, x=r_xi[i], y=None, deg=d, deg_out=d))
            inhom_combined_ops[eq_idx] = functools.reduce(np.kron, D_regs) / float(r_val)

    has_any_inhom = any(inhom_combined_ops[i] is not None for i in range(num_eqs))

    # Helper: build per-element mega-block (num_eqs*block_rows, num_funcs*block_cols)
    def _build_element_block(J_e, center_e):
        mega = None
        for eq_idx in range(num_eqs):
            row_blocks = _evaluate_operator_row_fast(
                all_hom_evaluators[eq_idx], J_e, center_e, num_funcs)
            for func_j in range(num_funcs):
                block_val = row_blocks[func_j]
                if np.isscalar(block_val):
                    continue
                r0 = eq_idx * block_rows
                c0 = func_j * block_cols
                if mega is None:
                    if sparse:
                        from scipy.sparse import lil_matrix as _lil
                        mega = _lil((num_eqs * block_rows, num_funcs * block_cols))
                    else:
                        mega = np.zeros((num_eqs * block_rows, num_funcs * block_cols))
                if sparse:
                    mega[r0:r0+block_rows, c0:c0+block_cols] += csr_matrix(block_val)
                else:
                    mega[r0:r0+block_rows, c0:c0+block_cols] += block_val
        if mega is None:
            mega = np.zeros((num_eqs * block_rows, num_funcs * block_cols))
        if sparse and not isinstance(mega, csr_matrix):
            mega = csr_matrix(mega)
        return mega

    # --- Strategy C: Uniform mesh fast path ---
    use_uniform_fast_path = mesh.is_uniform and all_center_independent

    elem_block_rows = num_eqs * block_rows
    elem_block_cols = num_funcs * block_cols

    if use_uniform_fast_path:
        J_0 = (hi[0] - lo[0]) / 2.0
        center_0 = (hi[0] + lo[0]) / 2.0
        single_block = _build_element_block(J_0, center_0)

        if sparse:
            A_total = kron(eye(num_elements), csr_matrix(single_block), format='csr')
        else:
            A_total = np.kron(np.eye(num_elements), single_block)

        # Inhomogeneous terms still need per-element handling (off-diagonal)
        if has_any_inhom:
            if sparse:
                A_inhom_total = lil_matrix((num_elements * elem_block_rows, num_elements * elem_block_cols))
            for e in range(num_elements):
                J_e = (hi[e] - lo[e]) / 2.0
                center_e = (hi[e] + lo[e]) / 2.0
                for eq_idx in range(num_eqs):
                    if inhom_combined_ops[eq_idx] is None:
                        continue
                    r_e = reg_infos[eq_idx][0]
                    inhom_row_blocks = _evaluate_operator_row_fast(
                        all_inhom_evaluators[eq_idx], J_e, center_e, num_funcs)
                    row_start = e * elem_block_rows + eq_idx * block_rows
                    for func_j in range(num_funcs):
                        if np.isscalar(inhom_row_blocks[func_j]):
                            continue
                        A_inhom = inhom_row_blocks[func_j] @ inhom_combined_ops[eq_idx]
                        col_start = r_e * elem_block_cols + func_j * block_cols
                        if sparse:
                            A_inhom_total[row_start:row_start+block_rows, col_start:col_start+block_cols] += csr_matrix(A_inhom)
                        else:
                            A_total[row_start:row_start+block_rows, col_start:col_start+block_cols] += A_inhom
            if sparse and has_any_inhom:
                A_total = A_total + A_inhom_total.tocsr()
    else:
        # --- Normal per-element loop (Strategy A + E) ---
        hom_blocks = []  # Strategy E: collect for block_diag
        if not sparse:
            A_total = np.zeros((num_elements * elem_block_rows, num_elements * elem_block_cols))
        if has_any_inhom and sparse:
            A_inhom_total = lil_matrix((num_elements * elem_block_rows, num_elements * elem_block_cols))

        for e in range(num_elements):
            J_e = (hi[e] - lo[e]) / 2.0
            center_e = (hi[e] + lo[e]) / 2.0

            mega_block = _build_element_block(J_e, center_e)

            if sparse:
                hom_blocks.append(csr_matrix(mega_block))
            else:
                r0 = e * elem_block_rows
                c0 = e * elem_block_cols
                A_total[r0:r0+elem_block_rows, c0:c0+elem_block_cols] = mega_block

            # Handle inhomogeneous terms
            if has_any_inhom:
                for eq_idx in range(num_eqs):
                    if inhom_combined_ops[eq_idx] is None:
                        continue
                    r_e = reg_infos[eq_idx][0]
                    inhom_row_blocks = _evaluate_operator_row_fast(
                        all_inhom_evaluators[eq_idx], J_e, center_e, num_funcs)
                    row_start = e * elem_block_rows + eq_idx * block_rows
                    for func_j in range(num_funcs):
                        if np.isscalar(inhom_row_blocks[func_j]):
                            continue
                        A_inhom = inhom_row_blocks[func_j] @ inhom_combined_ops[eq_idx]
                        col_start = r_e * elem_block_cols + func_j * block_cols
                        if sparse:
                            A_inhom_total[row_start:row_start+block_rows, col_start:col_start+block_cols] += csr_matrix(A_inhom)
                        else:
                            A_total[row_start:row_start+block_rows, col_start:col_start+block_cols] += A_inhom

        # Strategy E: sparse block-diagonal assembly
        if sparse:
            from scipy.sparse import block_diag as sp_block_diag
            A_total = sp_block_diag(hom_blocks, format='csr')
            if has_any_inhom:
                A_total = A_total + A_inhom_total.tocsr()

    # --- Integral Conditioning ---
    if intg_cond_order > 0:
        S_1D = intg_matrix(a=-1.0, deg=d_out, deg_out=d_out)
        if intg_cond_order > 1:
            S_1D = np.linalg.matrix_power(S_1D, intg_cond_order)
        S_spatial = S_1D
        for _ in range(num_vars - 1):
            S_spatial = np.kron(S_spatial, S_1D)

        if sparse:
            S_spatial_sparse = csr_matrix(S_spatial)
            S_block_sparse = kron(eye(num_eqs), S_spatial_sparse, format='csr')
            S_total = kron(eye(num_elements), S_block_sparse, format='csr')
            if isinstance(A_total, lil_matrix):
                A_total = A_total.tocsr()
            A_total = S_total @ A_total
        else:
            S_block = np.kron(np.eye(num_eqs), S_spatial)
            elem_rows = num_eqs * block_rows
            for e in range(num_elements):
                row_slice = slice(e * elem_rows, (e + 1) * elem_rows)
                A_total[row_slice, :] = S_block @ A_total[row_slice, :]

    if local_basis_transform is not None:
        C = local_basis_transform
        C_joint = C
        for _ in range(num_vars - 1):
            C_joint = np.kron(C_joint, local_basis_transform)

        if sparse:
            C_total = kron(eye(num_elements * num_funcs), csr_matrix(C_joint), format='csr')
            if isinstance(A_total, lil_matrix):
                A_total = A_total.tocsr()
            A_total = A_total @ C_total.T
        else:
            C_block = np.kron(np.eye(num_funcs), C_joint)
            elem_cols = num_funcs * block_cols
            for e in range(num_elements):
                col_slice = slice(e * elem_cols, (e + 1) * elem_cols)
                A_total[:, col_slice] = A_total[:, col_slice] @ C_block.T

    if sparse and isinstance(A_total, lil_matrix):
        A_total = A_total.tocsr()

    if get_hamiltonian is False:
        return A_total
    else:
        return get_hamiltonian_from_operator(A_total)
