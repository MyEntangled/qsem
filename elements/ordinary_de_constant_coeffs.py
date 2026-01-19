import numpy as np
from src.utils import derivative_matrix, boundary_matrix, encoding

def solve_ODE(deg, N, coeffs, map_coeffs_a, xi_z=None, xi_m=None):
    GT = derivative_matrix.chebyshev_diff_matrix(deg=deg)
    Cneg = boundary_matrix.zero_value_boundary_matrix(deg=deg, x_z=-1)
    Cpos = boundary_matrix.zero_value_boundary_matrix(deg=deg, x_z=1)
    Cneg_GT = Cneg @ GT
    Cpos_GT = Cpos @ GT
    T_Cneg = Cneg.T @ Cneg
    T_Cpos = Cpos.T @ Cpos
    T_Cnegpos = Cneg.T @ Cpos
    T_Cposneg = Cpos.T @ Cneg
    T_Cneg_GT = Cneg_GT.T @ Cneg_GT
    T_Cpos_GT = Cpos_GT.T @ Cpos_GT
    T_Cnegpos_GT = Cneg_GT.T @ Cpos_GT
    T_Cposneg_GT = Cpos_GT.T @ Cneg_GT

    H = np.zeros((N*(deg+1),N*(deg+1)))

    for i,a in enumerate(map_coeffs_a):
        A = a*a*coeffs[0]*GT@GT + a*coeffs[1]*GT + coeffs[2]*np.eye(deg+1)
        T_A = A.T @ A
        H[i*(deg+1):(i+1)*(deg+1),
          i*(deg+1):(i+1)*(deg+1)] = T_Cneg+T_Cneg_GT+T_A+T_Cpos+T_Cpos_GT

    H[0:deg+1,0:deg+1] -= T_Cneg+T_Cneg_GT
    H[(N-1)*(deg+1):N*(deg+1),
      (N-1)*(deg+1):N*(deg+1)] -= T_Cpos+T_Cpos_GT

    if xi_z != None:
        x_z, i = xi_z[0], xi_z[1]
        B_z = boundary_matrix.zero_value_boundary_matrix(deg=deg, x_z=x_z)
        T_Bz = B_z.T @ B_z
        H[i*(deg+1):(i+1)*(deg+1),
          i*(deg+1):(i+1)*(deg+1)] += T_Bz

    if xi_m != None:
        x_m, i = xi_m[0], xi_m[1]
        B_m = boundary_matrix.zero_derivative_boundary_matrix(deg=deg, x_m=x_m)
        T_Bm = B_m.T @ B_m
        H[i*(deg+1):(i+1)*(deg+1),
          i*(deg+1):(i+1)*(deg+1)] += T_Bm

    for i in range(N-1):
        H[i*(deg+1):(i+1)*(deg+1),
          (i+1)*(deg+1):(i+2)*(deg+1)] = -T_Cposneg -T_Cposneg_GT
        H[(i+1)*(deg+1):(i+2)*(deg+1),
          i*(deg+1):(i+1)*(deg+1)] = -T_Cnegpos -T_Cnegpos_GT
 
    eigvals, eigvecs = np.linalg.eigh(H)
    psi_sol = eigvecs[:, 0]
    print("Spectral gap:", eigvals[1] - eigvals[0])
    return psi_sol

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    N = 6
    n = 4
    deg = 2**n - 1

### FIG 3. b)
#     coeffs = (1.0,4.0,4.0)
#     true_sol = lambda x: 0.5 * (1+x) * np.exp(-2*x)
#     x_s = -0.5
#     f_s = true_sol(x_s)
#     x_z, x_m = -1, None

### FIG 3. c)
    coeffs = (1.0,-2.0,-3.0)
    true_sol = lambda x: 0.25 * (np.exp(3*x) - np.exp(-x))
    x_s = 0.5
    f_s = true_sol(x_s)
    x_z, x_m = 0, None

### FIG 3. d)
#     coeffs = (1.0,5.0,400.0)
#     true_sol = lambda x: np.exp(-5*x/2) * (np.cos(15*np.sqrt(7)*x/2) +
#                                 np.sqrt(7)*np.sin(15*np.sqrt(7)*x/2)/21)
#     x_s = 0
#     f_s = true_sol(x_s)
#     x_z, x_m = None, 0

    nodes = np.linspace(-1,1,N+1)
    intervals = np.column_stack((nodes[:-1],nodes[1:]))
    map_coeffs = np.array([2/(intervals[:,1]-intervals[:,0]),
        -(intervals[:,1]+intervals[:,0])/(intervals[:,1]-intervals[:,0])]).T

    def map(x,map_coeff):
        return map_coeff[0]*x + map_coeff[1]

    i_z = np.searchsorted(nodes[:-1],x_z,'right')-1
    x_z = map(x_z,map_coeffs[i_z])
#     i_m = np.searchsorted(nodes[:-1],x_m,'right')-1
#     x_m = map(x_m,map_coeffs[i_m])

    psi_sol = solve_ODE(deg, N, coeffs, map_coeffs[:,0], (x_z,i_z), None)
#     psi_sol = solve_ODE(deg, N, coeffs, map_coeffs[:,0], None, (x_m,i_m))
    psis = psi_sol.reshape(N,deg+1)

    def f(x):
        i = np.searchsorted(nodes[:-1],x,'right')-1
        return np.dot(encoding.chebyshev_encoding(deg,
                      map(x,map_coeffs[i])), psis[i])

    s_eta = f_s / f(x_s)
    print(s_eta**2, "\n")

    # Plot the solution
    x_plot = np.linspace(-1, 1, 1000)
    f_plot = []
    f_true = []
    for xj in x_plot:
        tau = encoding.chebyshev_encoding(deg=deg, x=xj)
        f_plot.append(s_eta * f(xj))
        f_true.append(true_sol(xj))
    L2_err = np.sqrt(2*np.sum(np.power(np.array(f_plot)-np.array(f_true),2))/1000)
    print(f"L2 error: {L2_err}")

    plt.plot(x_plot, f_plot, c='red', label=r'$f^*_{Q}(x)$')
    plt.plot(x_plot, f_true, '--', label=r'$f_{true}(x)$')
    plt.title(f"Solution to ODE: n={n}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()
