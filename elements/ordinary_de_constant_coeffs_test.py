import numpy as np
from src.utils import derivative_matrix, boundary_matrix, encoding

def solve_ODE(deg, coeffs, x_z, x_m, map_coeffs):
    G_T = derivative_matrix.chebyshev_diff_matrix(deg=deg)
    Cneg = boundary_matrix.zero_value_boundary_matrix(deg=deg, x_z=-1)
    Cpos = boundary_matrix.zero_value_boundary_matrix(deg=deg, x_z=1)
    GT_Cneg = Cneg @ G_T
    GT_Cpos = Cpos @ G_T
    T_Cneg = Cneg.T @ Cneg
    T_Cpos = Cpos.T @ Cpos
    T_Cnegpos = Cneg.T @ Cpos
    T_Cposneg = Cpos.T @ Cneg
    T_GT_Cneg = GT_Cneg.T @ GT_Cneg
    T_GT_Cpos = GT_Cpos.T @ GT_Cpos
    T_GT_Cnegpos = GT_Cneg.T @ GT_Cpos
    T_GT_Cposneg = GT_Cpos.T @ GT_Cneg

    T_As = []
    for map_coeff in map_coeffs:
        a = map_coeff[0]
        A = a*a*coeffs[0]*G_T@G_T + a*coeffs[1]*G_T + coeffs[2]*np.eye(deg+1)
        T_A = A.T @ A
        T_As.append(T_A)

    if x_z != None:
        B_z = boundary_matrix.zero_value_boundary_matrix(deg=deg, x_z=-1)
        T_Bz = B_z.T @ B_z

    H = np.block([[T_As[0]+T_Cpos+T_GT_Cpos+T_Bz, -T_Cposneg-T_GT_Cposneg, np.zeros_like(T_Bz)], 
                  [-T_Cnegpos-T_GT_Cnegpos, T_Cneg+T_GT_Cneg+T_As[1]+T_Cpos+T_GT_Cpos, -T_Cposneg-T_GT_Cposneg],
                  [np.zeros_like(T_Bz), -T_Cnegpos-T_GT_Cnegpos, T_Cneg+T_GT_Cneg+T_As[2]]])

#     if x_m != None:
#         B_m = boundary_matrix.zero_derivative_boundary_matrix(deg=deg, x_m=x_m)
#         T_Bm = B_m.T @ B_m

    eigvals, eigvecs = np.linalg.eigh(H)
    psi_sol = eigvecs[:, 0]
    print("Spectral gap:", eigvals[1] - eigvals[0])
    return psi_sol

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    N = 3
    n = 3
    deg = 2**n - 1
    coeffs = (1.0,4.0,4.0)
    x_z, x_m = -1, None

    true_sol = lambda x: 0.5 * (1+x) * np.exp(-2*x)
#     true_sol = lambda x: 0.25 * (1+x) * np.exp(1-x)
    x_s = -0.5
    f_s = true_sol(x_s)

    nodes = np.linspace(-1,1,N+1)
    intervals = np.column_stack((nodes[:-1],nodes[1:]))
    map_coeffs = np.array([2/(intervals[:,1]-intervals[:,0]),
        -(intervals[:,1]+intervals[:,0])/(intervals[:,1]-intervals[:,0])]).T

    psi_sol = solve_ODE(deg,coeffs,x_z,x_m,map_coeffs)
    psis = psi_sol.reshape(N,deg+1)

    def map(x,map_coeff):
        return map_coeff[0]*x + map_coeff[1]

    def f(x):
        for i, interval in enumerate(intervals):
            if interval[0] <= x <= interval[1]:
                return np.dot(encoding.chebyshev_encoding(deg,
                                            map(x,map_coeffs[i])), psis[i])

    s_eta = f_s / f(x_s)
    print(s_eta**2, "\n")

    # Plot the solution
    x_plot = np.linspace(-1, 1, 100)
    f_plot = []
    f_true = []
    for xj in x_plot:
        tau = encoding.chebyshev_encoding(deg=deg, x=xj)
        f_plot.append(s_eta * f(xj))
        f_true.append(true_sol(xj))
    plt.plot(x_plot, f_plot, c='red', label=r'$f^*_{Q}(x)$')
    plt.plot(x_plot, f_true, '--', label=r'$f_{true}(x)$')
    plt.title(f"Solution to ODE: n={n}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()
