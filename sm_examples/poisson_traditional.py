import numpy as np
from src.utils import derivative_matrix, boundary_matrix, encoding

def solve_Poisson(deg):
    G_T = derivative_matrix.chebyshev_diff_matrix(deg=deg)
    D = boundary_matrix.regular_value_boundary_matrix(deg,0,0.5)
    A = G_T @ G_T + D

    B = boundary_matrix.zero_value_boundary_matrix(deg=deg, x_z=1)
    T_A = A.T @ A
    T_B = B.T @ B

    H = T_A + T_B
    eigvals, eigvecs = np.linalg.eigh(H)
    psi_sol = eigvecs[:, 0]
    print("Ground State Energy:", eigvals[0])
    print("Spectral gap:", eigvals[1] - eigvals[0])
    return psi_sol

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n = 3
    deg = 2**n - 1

    psi_sol = solve_Poisson(deg)

    print("Solution coefficients (Chebyshev basis):")
    print(psi_sol)

    s_eta = 0.5 / np.dot(encoding.chebyshev_encoding(deg=deg, x=0), psi_sol)
    print(s_eta**2)

    # Plot the solution
    x_plot = np.linspace(-1, 1, 100)
    f_plot = []
    for xj in x_plot:
        tau = encoding.chebyshev_encoding(deg=deg, x=xj)
        fj = s_eta * np.dot(tau, psi_sol)
        f_plot.append(fj)
    plt.plot(x_plot, f_plot)
    plt.title("Solution to ODE")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.show()
