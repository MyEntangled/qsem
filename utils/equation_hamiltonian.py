import sympy as sp
import numpy as np
import scipy
from src.utils import boundary_matrix, derivative_matrix, multiply_matrix, tensor_mult_matrix, equation_parsing, function_evaluation, interface_continuity
import functools
import warnings
from typing import Dict, Tuple, Optional, Any, Literal

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
    Constructs the effective Hamiltonian for a NDE.

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
    GT = derivative_matrix.chebyshev_diff_matrix(deg=d)
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

        # if regular_data_type == 'value':
        #     # Ref: Eq (11)
        #     D_reg = boundary_matrix.regular_value_boundary_matrix(deg=d, x_s=regular_x, y_s=regular_y)
        # else:  # regular_constraint_type == 'derivative'
        #     # Ref: Eq (12)
        #     D_reg = boundary_matrix.regular_derivative_boundary_matrix(deg=d, x_s=regular_x, t_s=regular_y
        #     )

        D_reg = boundary_matrix.build_boundary_matrix(type=regular_data_type, deg=d, x=regular_x, y=regular_y)

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

    # # 5. Form Effective Hamiltonian
    # H = A.T @ A

    return A.astype(float)


def build_equation_hamiltonian(equation_op):
    A = equation_op
    return A.T @ A


def sem_equation_hamiltonian(d: int,
                             d_out: int,
                             diff_eq,
                             func,
                             var,
                             endpoints: np.ndarray,
                             truncated_order: int = None,
                             regular_data: tuple = None,
                             regular_data_type: str = None):

    """
    Transform the differential Hamiltonian to the reference domain [-1, 1] using the affine transformation x = (b-a)/2 * xi + (a+b)/2, where a and b are the endpoints of the original domain.
    """
    assert endpoints.ndim == 2
    num_elements = endpoints.shape[0]
    lo = endpoints[:, 0]
    hi = endpoints[:, 1]

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
        for e in range(num_elements):
            if lo[e] <= x_s <= hi[e]:
                e_s = e
                xi_s_in_e_s = (2 * x_s - (lo[e] + hi[e])) / (hi[e] - lo[e])  # Map to [-1, 1]
                break
        if e_s is None:
            raise ValueError(f"Regularization point x_s={x_s} is not found in the mesh.")
    else:
        x_s, y_s = None, None

    print("(e_s, xi_s, y_s):", e_s, xi_s_in_e_s, y_s)

    H_total = np.zeros(((d+1)*num_elements, (d+1)*num_elements))

    for e in range(num_elements):
        local_var_terms, local_var = parser.transform(terms_dict=terms, var_sym=var, a=lo[e], b=hi[e])
        print(f"Transformed Terms for variable {var} to {local_var}", local_var_terms)

        ## Separate the inhomogeneous term (if exists) for regularization
        hom_terms = local_var_terms
        inhom_terms = {(0,0,0): hom_terms.pop((0,0,0))} if (0,0,0) in hom_terms.keys() else None
        A_hom = build_equation_operator(d=d,
                                        d_out=d_out,
                                        terms_dict=hom_terms,
                                        var=local_var,
                                        truncated_order=truncated_order)
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

        ## A_e = |e><e| \otimes A_hom + |e><e_s| \otimes A_inhom
        A_e = np.zeros(((d_out+1)*num_elements, (d+1)*num_elements))
        A_e[e*(d_out+1):(e+1)*(d_out+1), e*(d+1):(e+1)*(d+1)] = A_hom
        if inhom_terms is not None:
            A_e[e*(d_out+1):(e+1)*(d_out+1), e_s*(d+1):(e_s+1)*(d+1)] = A_inhom

        H_total += build_equation_hamiltonian(A_e)

    return H_total


if __name__ == "__main__":
    from src.utils import equation_parsing, encoding, boundary_matrix

    ## Testing the build_H_diff function with a sample ODE

    n = 5
    d = 2**n - 1
    d_out = 2**(n+1) - 1
    print(f"Using Chebyshev degree d={d}, output degree d_out={d_out}")

    x = sp.Symbol('x')
    f = sp.Function('f')(x)

    d1 = f.diff(x)
    d2 = f.diff(x, 2)

    l,m = 5,1
    lhs = (1 - x**2)**2 * d2 - 2*x*(1-x**2)*d1 + (l*(l+1)*(1-x**2) - m**2)*f
    #lhs = (1 - x**2) * d2 - 2*x*d1 + (l*(l+1) - m**2/(1-x**2))*f
    #lhs = (x-1) * d2 - x * d1 + f - (x-1)**2
    print(f"Equation LHS: {lhs}")


    # Analytical solution
    from scipy.special import lpmv
    x_s = 0.5
    sol = lambda x: lpmv(m,l,x)
    #sol = lambda x: 1.5 * np.exp(x) - 0.125*x*(8*x+13) - 1
    data_s = (x_s, sol(x_s))
    print("Regular data point:", data_s)


    # parser = equation_parsing.ElementalEquationParser()
    # terms = parser.parse_equation(lhs, f)
    # print("Parsed Terms:", terms)
    #
    # local_terms, xi = parser.transform(terms, x, a=-1, b=1)
    # print("Transformed Terms (Reference Domain) with variable xi:", local_terms)
    #
    # A = build_equation_operator(d, d_out, terms,
    #                             var=x,
    #                             truncated_order=None,
    #                             regular_data=data_s,
    #                             regular_data_type='value')
    # A_xi = build_equation_operator(d, d_out, local_terms,
    #                                var=xi,
    #                                truncated_order=None,
    #                                regular_data=data_s,
    #                                regular_data_type='value')

    endpoints = np.array([[-1, 0], [0, 1]])
    #endpoints = np.array([[-1, 1]])
    num_elements = 8
    ## Create endpoints for the mesh
    nodes = np.linspace(-1, 1, num_elements + 1)
    endpoints = np.column_stack((nodes[:-1], nodes[1:]))
    H_sem = sem_equation_hamiltonian(d, d_out, lhs, f, x, endpoints=endpoints, regular_data=None, regular_data_type='value')

    # H_diff = build_equation_hamiltonian(A)
    # H_diff_xi = build_equation_hamiltonian(A_xi)
    # H = A.T @ A

    # assert np.allclose(H_diff_xi, H_diff), "Transformed Hamiltonian does not match original!"
    #assert np.allclose(H_sem, H_diff), "SEM Hamiltonian does not match original!"

    # x_m = 0.0 #-0.195976
    # B = boundary_matrix.build_boundary_matrix('derivative', deg=d, x=x_m, y=None, deg_out=d_out)
    # H += B.T @ B

    x_m = 0.0  # -0.195976
    B_sem = boundary_matrix.sem_boundary_hamiltonian(type='derivative', deg=d, deg_out=d_out, endpoints=endpoints, x=x_m)
    H_sem += B_sem

    num_elements = endpoints.shape[0]
    H_sem += 100000 * interface_continuity.boundary_continuity_matrice_1D('value', num_elements, d)
    H_sem += 100000 * interface_continuity.boundary_continuity_matrice_1D('derivative', num_elements, d)


    eigvals, eigvecs = np.linalg.eigh(H_sem)
    psi_sol = eigvecs[:, 0]
    print("Ground energy", eigvals[0])
    print("Spectral gap:", eigvals[1] - eigvals[0])
    print("Solution coefficients (Chebyshev basis):", np.round(psi_sol,2))

    # M1 = multiply_matrix.M_x_power(d, p=0, deg_out=d_out)
    # f_s = np.dot(encoding.chebyshev_encoding(deg=d_out, x=data_s[0]), M1 @ psi_sol)

    f_s = function_evaluation.evaluate_sem_encoding(psi_sol=psi_sol, deg=d, deg_out=d_out, endpoints=endpoints, x_eval_list=[data_s[0]], scaling_factor=1.0)[0]
    print("f_s at regularization point:", f_s)
    s_eta = data_s[1] / f_s
    print("Scaling^2:", s_eta**2)


    # Plot the solution
    import matplotlib.pyplot as plt
    x_plot = np.linspace(-1, 1, 101)
    #x_plot = encoding.chebyshev_nodes(deg=deg)
    fQ_plot = []
    f_plot = []

    for xj in x_plot:
        # tau = encoding.chebyshev_encoding(deg=d_out, x=xj)
        # fQ_plot.append(s_eta * np.dot(tau, M1 @ psi_sol))
        yj = function_evaluation.evaluate_sem_encoding(psi_sol=psi_sol, deg=d, deg_out=d_out, endpoints=endpoints, x_eval_list=np.array([xj]), scaling_factor=s_eta)[0]
        #print("yj:", yj)
        fQ_plot.append(yj)
        f_plot.append(sol(xj))
    plt.plot(x_plot, fQ_plot, c='red', label=r'$f^*_{Q}(x)$')
    plt.plot(x_plot, f_plot, '--', label=r'f^*(x)')
    plt.title(f"Solution to ODE: n={n}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()



    # ## For Eq (34)
    #lhs = d2 - 2*f**2 + x

    # x_z = 0.02615
    # B_z = boundary_matrix.zero_value_boundary_matrix(deg=d, x_z=x_z)
    # N1 = tensor_mult_matrix.N1_matrix(deg=d, deg_out=d_out)
    #
    # B_s = boundary_matrix.zero_value_boundary_matrix(deg=d, x_z=1)
    # D_s = B_s / 0.1
    # B = N1 @ np.kron(D_s, B_z)
    # H += B.T @ B
    #
    # B_s = boundary_matrix.zero_value_boundary_matrix(deg=d, x_z=-1)
    # D_s = B_s / (-0.1)
    # B = N1 @ np.kron(D_s, B_z)
    # H += 1 * B.T @ B

    # eigvals, eigvecs = np.linalg.eigh(H)
    # psi_sol = eigvecs[:, 0]
    # print("Spectral gap:", eigvals[1] - eigvals[0])
    # print("Solution coefficients (Chebyshev basis):", psi_sol)
    #
    # f_s = np.dot(encoding.chebyshev_encoding(deg=d_out, x=data_s[0]), N1 @ np.kron(D_s, np.eye(d+1)) @ psi_sol)
    # s_eta = data_s[1] / f_s
    # print(s_eta**2)
    #
    #
    # # Plot the solution
    # import matplotlib.pyplot as plt
    # x_plot = np.linspace(-1, 1, 10000)
    # #x_plot = encoding.chebyshev_nodes(deg=deg)
    # fQ_plot = []
    # f_plot = []
    #
    # for xj in x_plot:
    #     tau = encoding.chebyshev_encoding(deg=d_out, x=xj)
    #     fQ_plot.append(s_eta * np.dot(tau, N1 @ np.kron(D_s, np.eye(d+1)) @ psi_sol))
    #     f_plot.append(sol(xj))
    # plt.plot(x_plot, fQ_plot, c='red', label=r'$f^*_{Q}(x)$')
    # #plt.plot(x_plot, f_plot, '--', label=rf'$P^{m}_{l}(x)$')
    # plt.title(f"Solution to ODE: n={n}")
    # plt.xlabel("x")
    # plt.ylabel("f(x)")
    # plt.legend()
    # plt.grid()
    # plt.show()
