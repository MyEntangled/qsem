from typing import Iterable

import numpy as np
from src.utils import encoding, derivative_matrix, boundary_matrix, multiply_matrix
from scipy.special import lpmv

def solve_ODE(deg, deg_out, m, l, x_z=None, x_m=None):
    GT = derivative_matrix.chebyshev_diff_matrix(deg=deg)
    GT_sq = GT @ GT

    M1 = multiply_matrix.M_x_power(deg, 0, deg_out=deg_out)
    Mx = multiply_matrix.M_x_power(deg, 1, deg_out=deg_out)
    Mx2 = multiply_matrix.M_x_power(deg, 2, deg_out=deg_out)
    Mx3 = multiply_matrix.M_x_power(deg, 3, deg_out=deg_out)
    Mx4 = multiply_matrix.M_x_power(deg, 4, deg_out=deg_out)

    if m == 0:
        A = (M1 - Mx2) @ GT_sq - 2 * Mx @ GT + l*(l+1) * M1
    else:
        A = (M1 -2*Mx2 + Mx4) @ GT_sq + 2 * (Mx3 - Mx) @ GT - l*(l+1) * Mx2 + (l*(l+1) - m*m) * M1
    H = A.T @ A

    if x_z is not None:
        if isinstance(x_z, Iterable):
            for xz in x_z:
                Bz = boundary_matrix.zero_value_boundary_matrix(deg_out, x_z=xz)
                Bz_M = Bz @ M1
                H += Bz_M.T @ Bz_M
        else:
            Bz = boundary_matrix.zero_value_boundary_matrix(deg_out, x_z=x_z)
            Bz_M = Bz @ M1
            H += Bz_M.T @ Bz_M
    if x_m is not None:
        if isinstance(x_m, Iterable):
            for xm in x_m:
                Bm = boundary_matrix.zero_value_boundary_matrix(deg_out, x_z=xm)
                Bm_M_GT = Bm @ M1 @ GT
                H += Bm_M_GT.T @ Bm_M_GT
        else:
            Bm = boundary_matrix.zero_value_boundary_matrix(deg_out, x_z=x_m)
            Bm_M_GT = Bm @ M1 @ GT
            H += Bm_M_GT.T @ Bm_M_GT

    eigvals, eigvecs = np.linalg.eigh(H)
    print(eigvals)
    psi_sol = eigvecs[:, 0]
    print("Spectral gap:", eigvals[1] - eigvals[0])
    return psi_sol

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    m, l = 0, 5
    n = 3
    deg = 2**n-1
    deg_out = 2**(n+1) - 1

    psi_sol = solve_ODE(deg,deg_out,m=m,l=l,x_m=None,x_z=[0])
    print("Solution coefficients (Chebyshev basis):")
    print(psi_sol)

    # Compute scaling factor s_eta
    x = 0.5
    data_s = (x,lpmv(m,l,x))
    s_eta = data_s[1] / np.dot(encoding.chebyshev_encoding(deg=deg, x=data_s[0]), psi_sol)
    #s_eta = -np.sqrt(108.51)
    print(s_eta, s_eta**2)

    # Plot the solution
    x_plot = np.linspace(-1, 1, 10000)
    #x_plot = encoding.chebyshev_nodes(deg=deg)
    fQ_plot = []
    f_plot = []

    for xj in x_plot:
        tau = encoding.chebyshev_encoding(deg=deg, x=xj)
        fQ_plot.append(s_eta * np.dot(tau, psi_sol))
        f_plot.append(lpmv(m,l,xj))
    plt.plot(x_plot, fQ_plot, c='red', label=r'$f^*_{Q}(x)$')
    plt.plot(x_plot, f_plot, '--', label=rf'$P^{m}_{l}(x)$')
    plt.title(f"Solution to ODE: n={n}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()




