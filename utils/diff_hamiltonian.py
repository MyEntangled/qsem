import sympy as sp
import numpy as np
from src.utils import boundary_matrix, derivative_matrix, multiply_matrix, tensor_mult_matrix
import functools
from typing import Dict, Tuple, Optional, Any, Literal

def build_H_diff(
        d: int,
        d_out: int,
        terms_dict: Dict[Tuple[int, int, int], Any],
        var: Any,
        truncated_order: Optional[int] = None,
        regular_constraint_val: float = 1.0,
        regular_constraint_pos: float = 0.0,
        regular_constraint_type: Literal['value', 'derivative'] = 'value'
) -> np.ndarray:
    """
    Constructs the effective Hamiltonian for a NDE.

    This version assumes tensor_mult_matrix.N1_matrix returns the reduction matrix
    (Tensor -> Single) directly, with shape (deg_out+1, (deg+1)**2).

    Args:
        d (int): Input degree of the Chebyshev basis (N = d+1).
        d_out (int): Output dimension for the final coefficient projection.
        terms_dict (dict): Mapping (n, m, k) -> coefficient function c(x).
        var (Any): Symbolic variable.
        truncated_order (int, optional): Truncation order for coefficients.
        regular_constraint_val (float): Target value for regularization (y_s).
        regular_constraint_pos (float): Position x_s for regularization.
        regular_constraint_type (str): 'value' (f) or 'derivative' (f').

    Returns:
        np.ndarray: The effective Hamiltonian H = A.T @ A.
    """

    # 1. Determine Tensor Rank (P)
    max_total_pow = max((m + k) for (n, m, k) in terms_dict.keys())
    tensor_rank = max(1, max_total_pow)

    if tensor_rank > 2:
        raise NotImplementedError("N1_matrix strictly supports Rank 2 reduction.")

    # 2. Pre-compute Base Matrices
    GT = derivative_matrix.chebyshev_diff_matrix(deg=d)
    Id = np.eye(d + 1)

    # 3. Regular Constraint Operator (Padding)
    # [cite_start]Ref: Eq (30) [cite: 533] [cite_start]and Eq (32) [cite: 575]
    B_mat = boundary_matrix.regular_value_boundary_matrix(deg=d, x_s=regular_constraint_pos, y_s=1.0)

    if regular_constraint_type == 'value':
        if abs(regular_constraint_val) < 1e-12:
            raise ValueError("Regular constraint value f(x_s) cannot be zero.")
        D_reg = B_mat / regular_constraint_val

    elif regular_constraint_type == 'derivative':
        # [cite_start]Ref: Eq (12) [cite: 172]
        if abs(regular_constraint_val) < 1e-12:
            raise ValueError("Regular constraint derivative f'(x_s) cannot be zero.")
        D_reg = (B_mat @ GT) / regular_constraint_val

    else:
        raise ValueError("regular_constraint_type must be 'value' or 'derivative'")

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
            ops_list.extend([D_reg] * n_pads)

        # Pure source terms
        if not ops_list:
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

    # 5. Form Effective Hamiltonian
    H = A.T @ A

    return H.astype(float)


def _free_coeff_op(d:int, d_out:int, c:sp.Expr, var:sp.Symbol, truncated_order:int=None):
    c = sp.sympify(c)

    # Special case: scalar coefficient c = c0
    if c.is_Number:
        M_1 = multiply_matrix.M_x_power(deg=d, p=0, deg_out=d_out)
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
        M_xp = multiply_matrix.M_x_power(deg=d, p=power, deg_out=d_out)
        A += scalar * M_xp
    return A


if __name__ == "__main__":
    from src.utils import local_eq_parser, encoding, boundary_matrix

    ## Testing the build_H_diff function with a sample ODE

    n = 2
    d = 2**n - 1
    d_out = 2**(n+1) - 1
    print(f"Using Chebyshev degree d={d}, output degree d_out={d_out}\n")

    x = sp.Symbol('x')
    f = sp.Function('f')(x)

    d1 = f.diff(x)
    d2 = f.diff(x, 2)

    # l,m = 5,1
    #lhs = (1 - x**2)**2 * d2 - 2*x*(1-x**2)*d1 + (l*(l+1)*(1-x**2) - m**2)*f
    #lhs = (1 - x**2) * d2 - 2*x*d1 + (l*(l+1) - m**2/(1-x**2))*f
    #lhs = (x-1) * d2 - x * d1 + f - (x-1)**2
    lhs = d2 - 2*f**2 + x

    # Analytical solution
    from scipy.special import lpmv
    #x_s = 0.5
    #sol = lambda x: lpmv(m,l,x)
    #sol = lambda x: 1.5 * np.exp(x) - 0.125*x*(8*x+13) - 1
    #data_s = (x_s, sol(x_s))
    data_s = (-1, -0.1)

    print(f"Equation LHS: {lhs}\n")

    parser = local_eq_parser.ElementalEquationParser()
    terms = parser.parse_equation(lhs, f, x)
    print("Parsed Terms:\n")
    print(terms)

    local_terms = parser.transform(a=-1, b=1)
    print("\nTransformed Terms (Reference Domain) with variable xi:", local_terms)

    H = build_H_diff(d, d_out, terms,
                           var=parser.x_sym,
                           truncated_order=None,
                           regular_constraint_val=0.1,
                           regular_constraint_pos=1,
                           regular_constraint_type='value')
    #H_diff_xi = diff_hamiltonian.build_H_diff(d, d_out, local_terms, var=parser.xi_sym, truncated_order=300)
    #assert np.allclose(H_diff_xi, H_diff)


    x_z = 0.02615
    B_z = boundary_matrix.zero_value_boundary_matrix(deg=d, x_z=x_z)
    N1 = tensor_mult_matrix.N1_matrix(deg=d, deg_out=d_out)


    # B_s = boundary_matrix.zero_value_boundary_matrix(deg=d, x_z=1)
    # D_s = B_s / 0.1
    # B = N1 @ np.kron(D_s, B_z)
    # H += B.T @ B

    B_s = boundary_matrix.zero_value_boundary_matrix(deg=d, x_z=-1)
    D_s = B_s / (-0.1)
    B = N1 @ np.kron(D_s, B_z)
    H += 1 * B.T @ B


    eigvals, eigvecs = np.linalg.eigh(H)
    psi_sol = eigvecs[:, 0]
    print("Spectral gap:", eigvals[1] - eigvals[0])
    print("Solution coefficients (Chebyshev basis):")
    print(psi_sol)

    #f_s = np.dot(encoding.chebyshev_encoding(deg=d, x=data_s[0]), psi_sol)
    f_s = np.dot(encoding.chebyshev_encoding(deg=d_out, x=data_s[0]), N1 @ np.kron(D_s, np.eye(d+1)) @ psi_sol)
    s_eta = data_s[1] / f_s
    print(s_eta**2)


    # Plot the solution
    import matplotlib.pyplot as plt
    x_plot = np.linspace(-1, 1, 10000)
    #x_plot = encoding.chebyshev_nodes(deg=deg)
    fQ_plot = []
    f_plot = []

    for xj in x_plot:
        tau = encoding.chebyshev_encoding(deg=d_out, x=xj)

        #B_j = boundary_matrix.zero_value_boundary_matrix(deg=d, x_z=xj)
        fQ_plot.append(s_eta * np.dot(tau, N1 @ np.kron(D_s, np.eye(d+1)) @ psi_sol))
        #f_plot.append(sol(xj))
    plt.plot(x_plot, fQ_plot, c='red', label=r'$f^*_{Q}(x)$')
    #plt.plot(x_plot, f_plot, '--', label=rf'$P^{m}_{l}(x)$')
    plt.title(f"Solution to ODE: n={n}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()
