import numpy as np
from src.utils import derivative_matrix, boundary_matrix, encoding, multiply_matrix, continuity_matrix
from scipy.special import lpmv

def solve_ODE(deg, deg_out, N, l, m, map_coeffs, xi_z=None, xi_m=None):
    GT = derivative_matrix.chebyshev_diff_matrix(deg=deg)
    GT2 = GT@GT
    M1 = multiply_matrix.M_x_power(deg, 0, deg_out=deg_out)
    Mx = multiply_matrix.M_x_power(deg, 1, deg_out=deg_out)
    Mx2 = multiply_matrix.M_x_power(deg, 2, deg_out=deg_out)
    Mx3 = multiply_matrix.M_x_power(deg, 3, deg_out=deg_out)
    Mx4 = multiply_matrix.M_x_power(deg, 4, deg_out=deg_out)

    Am = np.zeros((N*(deg+1),N*(deg+1)))

    for i,map_coeff in enumerate(map_coeffs):
        a, b = map_coeff[0], map_coeff[1]
        if m == 0:
            A = ((a*a-b*b)*M1 + 2*b*Mx - Mx2)@GT2 + 2*(b*M1-Mx)@GT + l*(l+1)*M1
        else:
            termGT2 = ((a*a-b*b)*(a*a-b*b)*M1 + 4*b*(a*a-b*b)*Mx -
                       2*(a*a-3*b*b)*Mx2 - 4*b*Mx3 + Mx4)
            termGT  = 2*(b*(a*a-b*b)*M1 - (a*a-3*b*b)*Mx - 3*b*Mx2 + Mx3)
            term    = ((l*(l+1)*(a*a-b*b)-m*m*a*a)*M1 +
                       2*b*l*(l+1)*Mx - l*(l+1)*Mx2)
            A = termGT2@GT2 + termGT@GT + term
        T_A = A.T @ A

        Am[i*(deg+1):(i+1)*(deg+1),
          i*(deg+1):(i+1)*(deg+1)] = T_A

    if xi_z != None:
        x_z, i = xi_z[0], xi_z[1]
        B_z = boundary_matrix.zero_value_boundary_matrix(deg=deg, x_z=x_z)
        T_Bz = B_z.T @ B_z

        Am[i*(deg+1):(i+1)*(deg+1),
          i*(deg+1):(i+1)*(deg+1)] += T_Bz

    if xi_m != None:
        x_m, i = xi_m[0], xi_m[1]
        B_m = boundary_matrix.zero_derivative_boundary_matrix(deg=deg, x_m=x_m)
        T_Bm = B_m.T @ B_m

        Am[i*(deg+1):(i+1)*(deg+1),
          i*(deg+1):(i+1)*(deg+1)] += T_Bm

#     print(np.linalg.norm(Am@C0 - C0@Am))
#     print(np.linalg.norm(Am@C1 - C1@Am))
#     print(np.linalg.norm(C0@C1 - C1@C0))

    C0 = continuity_matrix.generate_C0(N,deg)
    C1 = continuity_matrix.generate_C1(N,deg)

    H = Am + 1000*C0@C0 + 1000*C1@C1

    eigvals, eigvecs = np.linalg.eigh(H)
#     print(eigvals[0])
    psi_sol = eigvecs[:, 0]
    print("Spectral gap:", eigvals[1] - eigvals[0])
    return psi_sol

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    N = 40
    n = 2
    deg = 2**n - 1
    deg_out = 2**(n+1) - 1

### FIG 4.
    m, l = 0, 5
    true_sol = lambda x: lpmv(m,l,x)
    x_s = 1/3
    f_s = true_sol(x_s)
    x_z, x_m = 0, None

    nodes = np.linspace(-1,1,N+1)
    intervals = np.column_stack((nodes[:-1],nodes[1:]))
    map_coeffs = np.array([2/(intervals[:,1]-intervals[:,0]),
        -(intervals[:,1]+intervals[:,0])/(intervals[:,1]-intervals[:,0])]).T
    i_z = np.searchsorted(nodes[:-1],x_z,'right')-1
#     i_m = np.searchsorted(nodes[:-1],x_m,'right')-1

    psi_sol = solve_ODE(deg, deg_out, N, l, m, map_coeffs, (x_z,i_z), None)
#     psi_sol = solve_ODE(deg, deg_out, N, l, m, map_coeffs, None, (x_m,i_m))
    psis = psi_sol.reshape(N,deg+1)

    def map(x,map_coeff):
        return map_coeff[0]*x + map_coeff[1]

    def f(x):
        i = np.searchsorted(nodes[:-1],x,'right')-1
        return np.dot(encoding.chebyshev_encoding(deg,
                      map(x,map_coeffs[i])), psis[i])

    s_eta = f_s / f(x_s)
    print(s_eta**2, "\n")

    # Plot the solution
    x_plot = np.linspace(-1, 1, 5000)
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
