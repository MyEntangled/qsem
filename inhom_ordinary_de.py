import numpy as np
import scipy as sp
from utils import encoding, derivative_matrix, boundary_matrix, multiply_matrix

def solve_ODE(deg, deg_out, x_s, y_s, x_m):
    GT = derivative_matrix.chebyshev_diff_matrix(deg=deg)
    GT_sq = GT @ GT

    M1 = multiply_matrix.M_x_power(deg, 0, deg_out=deg_out)
    Mx = multiply_matrix.M_x_power(deg, 1, deg_out=deg_out)
    Mx2 = multiply_matrix.M_x_power(deg, 2, deg_out=deg_out)

    D0_s = boundary_matrix.regular_value_boundary_matrix(deg, x_s, y_s)

    A = (Mx - M1) @ GT_sq - Mx @ GT + M1 - (Mx2 - 2*Mx + M1) @ D0_s
    H = A.T @ A

    Bm = boundary_matrix.zero_value_boundary_matrix(deg_out, x_z=x_m)
    Bm_M_GT = Bm @ M1 @ GT
    H += Bm_M_GT.T @ Bm_M_GT


    eigvals, eigvecs = np.linalg.eigh(H)
    psi_sol = eigvecs[:, 0]
    print("Spectral gap:", eigvals[1] - eigvals[0])
    return psi_sol

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n = 3
    deg = 2**n-1
    deg_out = 2**(n+1) - 1
    sol = lambda x: 1.5 * np.exp(x) - 0.125 * x * (8 * x + 13) - 1
    dsol = lambda x: 1.5 * np.exp(x) - 2*x -0.125*13
    djac = lambda x: 1.5 * np.exp(x) - 2
    root = sp.optimize.root(dsol,-0.2,jac=djac)
    print(root.x)

    psi_sol = solve_ODE(deg,deg_out,x_s=0.5,y_s=sol(0.5), x_m=root.x[0])
    print("Solution coefficients (Chebyshev basis):")
    print(psi_sol)

    # Compute scaling factor s_eta
    x = 0.5
    data_s = (x,sol(x))
    print("Data point for scaling:", data_s)
    s_eta = data_s[1] / np.dot(encoding.chebyshev_encoding(deg=deg, x=data_s[0]), psi_sol)
    #s_eta = -np.sqrt(108.51)
    print(s_eta, s_eta**2)

    # Plot the solution
    x_plot = np.linspace(-1, 1, 200)
    f_plot = []
    for x in x_plot:
        tau_x = encoding.chebyshev_encoding(deg=deg, x=x)
        f_x = s_eta * np.dot(tau_x, psi_sol)
        f_plot.append(f_x)
    print(np.linalg.norm(f_plot - sol(x_plot)) / np.linalg.norm(sol(x_plot)))
    f_plot = np.array(f_plot)
    plt.plot(x_plot, f_plot, label='Quantum ODE Solver', color='blue')
    plt.plot(x_plot, sol(x_plot), label='Exact Solution', linestyle='dashed', color='red')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Solution of ODE using Quantum Algorithm')
    plt.legend()
    plt.grid()
    plt.savefig("test.png")
    plt.show()
