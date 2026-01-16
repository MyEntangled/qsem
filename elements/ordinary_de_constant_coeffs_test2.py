import numpy as np
from src.utils import derivative_matrix, boundary_matrix, encoding, multiply_matrix


def solve_ODE(deg, deg_out, coeffs, x_z, x_m, x_s, y_s, x_r, y_r, map_coeff):
    G_T = derivative_matrix.chebyshev_diff_matrix(deg=deg)
    D0_s = boundary_matrix.regular_value_boundary_matrix(deg, x_s, y_s)
    M1 = multiply_matrix.M_x_power(deg, 0, deg_out=deg_out)

    a = map_coeff[0]
#     A = a*a*coeffs[0]*G_T@G_T + a*coeffs[1]*G_T + coeffs[2]*np.eye(deg+1)
    A = M1@(a*a*coeffs[0]*G_T@G_T + a*coeffs[1]*G_T +
            coeffs[2]*(np.eye(deg+1)+y_r*D0_s))
    T_A = A.T @ A
    H = T_A

    if x_z != None:
        B_z = boundary_matrix.zero_value_boundary_matrix(deg=deg, x_z=x_z)
        T_Bz = B_z.T @ B_z
        H += T_Bz
    if x_r != None:
        B_r = boundary_matrix.zero_value_boundary_matrix(deg=deg, x_z=x_r)
        T_Br = B_r.T @ B_r
        H += T_Br
    if x_m != None:
        B_m = boundary_matrix.zero_derivative_boundary_matrix(deg=deg, x_m=x_m)
        T_Bm = B_m.T @ B_m
        H += T_Bm

    eigvals, eigvecs = np.linalg.eigh(H)
    psi_sol = eigvecs[:, 0]
    print("Spectral gap:", eigvals[1] - eigvals[0])
    return psi_sol

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n = 3
    deg = 2**n - 1
    deg_out = 2**(n+1) - 1
    coeffs = (1.0,4.0,4.0)

#     true_sol = lambda x: 0.5 * (1+x) * np.exp(-2*x)
#     true_sol = lambda x: 0.25 * (1+x) * np.exp(1-x)
    true_sol = lambda x: 0.25 * (3+x) * np.exp(-1-x)

    x_z, x_m, x_s, x_r, y_r = None, None, -0.5, 0.5, true_sol(0.5)
    y_s = true_sol(x_s) - y_r


    intervals = np.array([[-1,0],[0,1]])
    map_coeffs = np.array([2/(intervals[:,1]-intervals[:,0]),
        -(intervals[:,1]+intervals[:,0])/(intervals[:,1]-intervals[:,0])]).T
    print(map_coeffs)

    psi_sol = solve_ODE(deg,deg_out,coeffs,x_z,x_m,x_s,y_s,x_r,y_r,map_coeffs[1])
    print("Solution coefficients (Chebyshev basis):")
    print(psi_sol)

    s_eta = y_s / np.dot(encoding.chebyshev_encoding(deg=deg, x=x_s), psi_sol)
    print(s_eta**2, "\n")

    # Plot the solution
    x_plot = np.linspace(-1, 1, 100)
    f_plot = []
    f_true = []
    for xj in x_plot:
        tau = encoding.chebyshev_encoding(deg=deg, x=xj)
        fj = s_eta * np.dot(tau, psi_sol) + y_r
        f_plot.append(fj)
        f_true.append(true_sol(xj))
    plt.plot(x_plot, f_plot, c='red', label=r'$f^*_{Q}(x)$')
    plt.plot(x_plot, f_true, '--', label=r'$f_{true}(x)$')
    plt.title(f"Solution to ODE: n={n}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()
