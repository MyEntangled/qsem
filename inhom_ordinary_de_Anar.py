import numpy as np
import scipy as sp
import math
from src.utils import encoding, derivative_matrix, boundary_matrix, multiply_matrix

def solve_ODE(deg, deg_out, x_s, y_s, x_m, coeffs):
    GT = derivative_matrix.chebyshev_diff_matrix(deg=deg)
    GT_sq = GT @ GT
    Id = np.identity(deg+1, dtype=float)

    M1 = multiply_matrix.M_x_power(deg, 0, deg_out=deg_out)
    Mx = multiply_matrix.M_x_power(deg, 1, deg_out=deg_out)
    Mx2 = multiply_matrix.M_x_power(deg, 2, deg_out=deg_out)
    D0_s = boundary_matrix.regular_value_boundary_matrix(deg, x_s, y_s)

    A = (Mx - M1) @ GT_sq - Mx @ GT + M1 - (Mx2 - 2*Mx + M1) @ D0_s

#     A = M1 @ (coeffs[0] * GT_sq + coeffs[1] * GT + coeffs[2] * Id)
#     Mr = np.zeros(M1.shape, dtype=float)
# 
#     for i in range(deg+1):
#         cp = (-2)**i / math.factorial(i)
#         Mi = multiply_matrix.M_x_power(deg, i, deg_out=deg_out)
#         Mr += cp * Mi

#     for i in range(1,deg+1):
#         cp = i*(2)**(i-1) / math.factorial(i)
#         Mi = multiply_matrix.M_x_power(deg, i, deg_out=deg_out)
#         Mr += cp * Mi

#     z = complex(-2,4)
#     for i in range(2,deg+1):
#         cp = (z**(i-1)).imag / math.factorial(i-1)
#         print(cp)
#         Mi = multiply_matrix.M_x_power(deg, i, deg_out=deg_out)
#         Mr += cp * Mi

#    A -= Mr @ D0_s
    H = A.T @ A

    Bm = boundary_matrix.zero_value_boundary_matrix(deg_out, x_z=x_m)
    Bm_M_GT = Bm @ M1 @ GT
    H += Bm_M_GT.T @ Bm_M_GT

#     Bm_M = Bm @ M1
#     H += Bm_M.T @ Bm_M


    eigvals, eigvecs = np.linalg.eigh(H)
    psi_sol = eigvecs[:, 0]
    print("Spectral gap:", eigvals[1] - eigvals[0])
    return psi_sol

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n = 7
    deg = 2**n-1
    deg_out = 2**(n+1) - 1

    coeffs = None
    x_s = 0.5
    sol = lambda x: 1.5 * np.exp(x) - 0.125 * x * (8 * x + 13) - 1
    dsol = lambda x: 1.5 * np.exp(x) - 2*x -0.125*13
    djac = lambda x: 1.5 * np.exp(x) - 2
    root = sp.optimize.root(dsol,-0.2,jac=djac)

#     coeffs = [1.0, 4.0, 4.0]
#     x_s = -1
#     sol = lambda x: np.exp(-2*x)*(x*x - 2*x - 2)/2
#     jac = lambda x: np.exp(-2*x)*(-x*x + 3*x + 1)
#     root = sp.optimize.root(sol,-0.73,jac=jac)

#     coeffs = [1.0, -5.0, 6.0]
#     x_s = 1/3
#     sol = lambda x: (4*np.exp(3*x) - np.exp(2*x)*(3*x*x + 6*x + 2))/6
#     jac = lambda x: (6*np.exp(3*x) - np.exp(2*x)*(3*x*x + 9*x + 5))/3
#     root = sp.optimize.root(sol,0.91,jac=jac)

#     coeffs = [1.0, 4.0, 20.0]
#     x_s = -1
#     sol = lambda x: np.exp(-2*x)*(-4*(x*x + 16)*np.cos(4*x) + (x - 48)*np.sin(4*x))/64
#     dsol = lambda x: np.exp(-2*x)*((16*x*x - 2*x + 353)*np.sin(4*x) + (8*x*x - 4*x - 64)*np.cos(4*x))/64
#     djac = lambda x: np.exp(-2*x)*((-16*x*x + 13*x - 113)*np.sin(4*x) + 4*(3*x*x + x + 96)*np.cos(4*x))/16
#     root = sp.optimize.root(dsol,-0.75,jac=djac)

    psi_sol = solve_ODE(deg,deg_out,x_s=x_s,y_s=sol(x_s), x_m=root.x[0], coeffs=coeffs)
    print("Solution coefficients (Chebyshev basis):")
    print(psi_sol)

    # Compute scaling factor s_eta
    x = x_s
    data_s = (x,sol(x))
    print("Data point for scaling:", data_s)
    print(np.dot(encoding.chebyshev_encoding(deg=deg, x=data_s[0]), psi_sol))
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
    f_plot = np.array(f_plot)
    plt.plot(x_plot, f_plot, label='Quantum ODE Solver', color='blue')
    plt.plot(x_plot, sol(x_plot), label='Exact Solution', linestyle='dashed', color='red')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Solution of ODE using Quantum Algorithm')
    plt.legend()
    plt.grid()
    plt.show()
