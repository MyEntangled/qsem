from src.utils import multivar_equation_parsing, multiply_matrix, derivative_matrix, boundary_matrix

import numpy as np
import sympy as sp
import functools
import warnings
from typing import Dict, Tuple, Any, Optional, List, Literal
#from scipy import sparse

def _free_coeff_op_multivariable(d: int, d_out: int, c: sp.Expr, vars: list, truncated_order: int = None):
    c = sp.sympify(c)
    vars = list(vars)
    computed_Mxp = {}


    # 1. Handle Scalar Case
    if c.is_Number:
        # p=0 represents the identity-like operator for every variable.
        computed_Mxp[0] = multiply_matrix.M_x_power(deg=d, p=0, deg_out=d_out)
        return float(c) * functools.reduce(np.kron, [computed_Mxp[0]] * len(vars))

    # 2. Handle Non-Polynomials via Multivariable Taylor Expansion
    if any(not c.is_polynomial(v) for v in vars):
        if truncated_order is None:
            raise ValueError("truncated_order must be specified for non-polynomial coefficients.")
        # Taylor expansion around origin for all variables
        coeff = sp.series(c, vars[0], 0, truncated_order).removeO()
        for v in vars[1:]:
            coeff = sp.series(coeff, v, 0, truncated_order).removeO()
    else:
        coeff = c

    # 3. Process Monomials
    # as_coefficients_dict() returns {monomial: coefficient} e.g., {x*y**2: 3}
    poly_dict = sp.Poly(coeff, *vars).as_dict()

    A = 0
    for powers, scalar in poly_dict.items():
        # 'powers' is a tuple corresponding to the degrees of each variable in 'vars'
        # e.g., if vars=[x, y] and term is 3*x**1*y**2, powers=(1, 2)

        term_operators = []

        for i, p in enumerate(powers):
            # Generate the operator for variable vars[i] raised to power p
            if p not in computed_Mxp:
                M_p = multiply_matrix.M_x_power(deg=d, p=p, deg_out=d_out)
                computed_Mxp[p] = M_p
            else:
                M_p = computed_Mxp[p]

            term_operators.append(M_p)

        A += float(scalar) * functools.reduce(np.kron, term_operators)

    return A.astype(float)

def build_multivar_equation_operator(
        d: int,
        d_out: int,
        terms_dict: Dict[Tuple[Tuple[Tuple[int, ...], int], int], Any],
        vars: List[sp.Symbol],
        truncated_order: Optional[int] = None,
        regular_data: Optional[Tuple[Tuple, Any]] = (None, None),
        regular_data_type: Literal['value', 'derivative'] = None):
    """
    Constructs the effective Hamiltonian for a Multivariate NDE with Rank 1 restriction.
    Validates necessity of regular_data for source terms.
    """
    n_vars = len(vars)
    regular_coords, regular_value = regular_data if regular_data is not None else (None, None)

    # 1. Determine Tensor Rank and Necessity of Regular Data
    need_regular = False

    for (deriv_signature, k_pow), _ in terms_dict.items():
        var_rank_counts = [0] * n_vars

        # --- Check for Pure Source Term ---
        # defined as: No function power (k=0) AND No derivatives
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
                D_regs[i] = boundary_matrix.build_boundary_matrix(
                    type=regular_data_type,
                    deg=d,
                    deg_out=d,
                    x=regular_coords[i]
                )

    # 3. Pre-compute Base Matrices
    GT = derivative_matrix.chebyshev_diff_matrix(deg=d)
    Id = np.eye(d + 1)

    A_total = 0.0

    # 4. Iterate over terms to build Operator A
    for (deriv_signature, k_pow), coeff in terms_dict.items():

        var_ops = [None] * n_vars

        # Assign Derivatives
        for (alpha_tuple, power_int) in deriv_signature:
            if not alpha_tuple:
                continue
            for i, deriv_order in enumerate(alpha_tuple):
                if deriv_order > 0:
                    var_ops[i] = np.linalg.matrix_power(GT, deriv_order)

        # Assign Functions
        if k_pow == 1:
            for i in range(n_vars):
                if var_ops[i] is None:
                    var_ops[i] = Id
                # Note: The Rank 1 check in Step 1 ensures we don't overwrite existing ops here

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

        # --- A. Construct the Raw Tensor Operator ---
        # Combined = M_var0 ⊗ M_var1 ⊗ ... ⊗ M_varN
        combined_op = functools.reduce(np.kron, var_ops)

        if is_source_term:
            # Apply scaling by the constraint value y_s (reg_val)
            # Warning: checks for division by zero might be prudent depending on physics
            if regular_value is None:
                raise ValueError("Internal Error: reg_val is None for source term.")
            combined_op = combined_op / float(regular_value)

        # --- B. Apply Coefficient ---
        coeff_op = _free_coeff_op_multivariable(
            d=d,
            d_out=d_out,
            c=coeff,
            vars=vars,
            truncated_order=truncated_order
        )

        A_total += coeff_op @ combined_op

    return A_total

def get_hamiltonian_from_operator(A: np.ndarray):
    return A.T @ A


def sem_multivar_equation_hamiltonian(
        d: int,
        d_out: int,
        diff_eq,
        func,
        vars: List[sp.Symbol],
        endpoints: np.ndarray,
        truncated_order: Optional[int] = None,
        regular_data: Optional[Tuple[Tuple, Any]] = (None, None),
        regular_data_type: Literal['value', 'derivative'] = None):

    assert endpoints.ndim == 3
    num_elements = endpoints.shape[0]
    lo = endpoints[:, 0, :]
    hi = endpoints[:, 1, :]

    N = d + 1
    N_out = d_out + 1
    num_vars = len(vars)

    parser = multivar_equation_parsing.ElementalMultiVariateEquationParser()
    terms = parser.parse_equation(diff_eq, func)

    ## Find which element x_s belongs to and the corresponding local variable for regularization
    e_s = None
    xi_s_in_e_s = None

    if regular_data is not None:
        (coord_s, val_s) = regular_data
        coord_s = np.array(coord_s)

        for e in range(num_elements):
            ## Check if every coordinate lies in the low-high
            if (lo[e] <= coord_s).all() and (coord_s <= hi[e]).all():
                e_s = e
                print(lo[e], hi[e], coord_s)
                xi_s_in_e_s = (2 * coord_s - (lo[e] + hi[e])) / (hi[e] - lo[e])  # Map to [-1, 1]
                break
        if e_s is None:
            raise ValueError(f"Regularization point x_s={coord_s} is not found in the mesh.")
    else:
        coord_s, val_s = None, None


    A_total = np.zeros((N_out**num_vars * num_elements, N**num_vars * num_elements))

    for e in range(num_elements):
        local_terms, local_vars = parser.transform(terms_dict=terms, vars=vars, lo=lo[e], hi=hi[e])
        print(f"Transformed Terms for variable {vars} to {local_vars}", local_terms)

        ## Separate the inhomogeneous term (if exists) for regularization
        hom_terms = local_terms
        inhom_terms = {((),0): hom_terms.pop(((),0))} if ((),0) in hom_terms.keys() else None

        A_hom = build_multivar_equation_operator(d=d,
                                                 d_out=d_out,
                                                 terms_dict=hom_terms,
                                                 vars=local_vars,
                                                 truncated_order=truncated_order)

        if inhom_terms is not None:
            if e_s is None or xi_s_in_e_s is None:
                raise ValueError("Inhomogeneous terms require regular_data to be provided.")
            A_inhom = build_multivar_equation_operator(d=d,
                                                       d_out=d_out,
                                                       terms_dict=inhom_terms,
                                                       vars=local_vars,
                                                       truncated_order=truncated_order,
                                                       regular_data=(xi_s_in_e_s, val_s),
                                                       regular_data_type=regular_data_type)

            A_total[e*N_out**num_vars:(e+1)*N_out**num_vars, e*N**num_vars:(e+1)*N**num_vars] = A_hom + A_inhom
        else:
            A_total[e*N_out**num_vars:(e+1)*N_out**num_vars, e*N**num_vars:(e+1)*N**num_vars] = A_hom
    H = get_hamiltonian_from_operator(A_total)
    return H


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
    #endpoints = np.array([[[0, 0], [1, 1]], [[1, 1], [2, 2]]])  # Two elements in 2D
    endpoints = np.array([[[-1, -1], [1, 1]]])  # Single element in 2D
    H_sem = sem_multivar_equation_hamiltonian(d=5,
                                              d_out=7,
                                              diff_eq=diff_eq,
                                              func=f,
                                                vars=[x, y],
                                                endpoints=endpoints,
                                                truncated_order=3,
                                                regular_data=data_s,
                                                regular_data_type='value')
    print("Constructed SEM Hamiltonian shape:", H_sem.shape)

    assert np.linalg.norm(H - H_sem) < 1e-10, "Hamiltonians from both methods do not match!"