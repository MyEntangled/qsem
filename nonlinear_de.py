import numpy as np
from src.utils import tensor_mult_matrix, derivative_matrix, boundary_matrix, encoding

def solve_NDE(deg, deg_out, regBC, xz):
    xs, ys = regBC
    N1 = tensor_mult_matrix.N1_matrix(deg,deg_out)
    GT = derivative_matrix.chebyshev_diff_matrix(deg)
    Dn = boundary_matrix.regular_value_boundary_matrix(deg,xs,ys)
    In = np.eye(deg+1)

#     Bc = boundary_matrix.zero_value_boundary_matrix(deg,xz)
    Bc = boundary_matrix.zero_derivative_boundary_matrix(deg,xz)

    A = N1 @ (np.kron(Dn, 4*GT@GT + In) + 2*np.kron(GT,GT))
    B = N1 @ np.kron(Dn,Bc)

    H = A.T@A + B.T@B

    eigvals, eigvecs = np.linalg.eigh(H)
    psi_sol = eigvecs[:,2]
    print(eigvals)
    print("Spectral gap:", eigvals[1] - eigvals[0])
    return psi_sol

def evaluate_NDE(deg, deg_out, regBC, psi_sol, x):
    xs, ys = regBC
    N1 = tensor_mult_matrix.N1_matrix(deg,deg_out)
    Dn = boundary_matrix.regular_value_boundary_matrix(deg,xs,ys)
    In = np.eye(deg+1)

    A = N1 @ np.kron(Dn,In)
    tau = encoding.chebyshev_encoding(deg_out, x)

    return np.dot(tau,A@psi_sol)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n = 2
    deg = 2**n - 1
    deg_out = 2**(n+1) - 1
    
    true_sol = lambda x: 1 - x*x/8
    regBC = (0,1)
    xz = 0

    psi_sol = solve_NDE(deg,deg_out,regBC,xz)
    appr_sol = lambda x: evaluate_NDE(deg,deg_out,regBC,psi_sol,x)
    s_eta = true_sol(0)/appr_sol(0)
    print(s_eta)
    print(s_eta**2)

    # Plot the solution
    plt.figure(figsize=(8,6))

    true_plot = []
    appr_plot = []
    xs = np.linspace(-1,1,100)
    for x in xs:
        true_plot.append(true_sol(x))
        appr_plot.append(s_eta * appr_sol(x))

    plt.plot(xs, appr_plot, c='red', label=r'$u^*_{Q}(x)$')
    plt.plot(xs, true_plot, '--', label=r'$u_{true}(x)$')
    plt.title(f"Solution to NDE: n={n}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()

    plt.show()
