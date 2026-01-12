import numpy as np
from src.utils import derivative_matrix, boundary_matrix, encoding


def solve_ODE(deg, coeffs, x_z):
    GT = derivative_matrix.chebyshev_diff_matrix(deg=deg)

    A = coeffs[0] * GT @ GT + coeffs[1] * GT + coeffs[2] * np.eye(deg+1)
    B = boundary_matrix.zero_value_boundary_matrix(deg=deg, x_z=x_z)
    Cneg = boundary_matrix.zero_value_boundary_matrix(deg=deg, x_z=-1)
    Cpos = boundary_matrix.zero_value_boundary_matrix(deg=deg, x_z=1)
#     B_GT = B @ GT

    T_A = A.T @ A # Made a mistake. Matrix A is different for each element,
                  # and it depends on the mapping function.
    T_B = B.T @ B
    H = np.block([[T_A+T_B+Cpos, -Cneg],[-Cpos, Cneg+T_A]])

    eigvals, eigvecs = np.linalg.eigh(H)
    psi_sol = eigvecs[:, 0]
    print("Spectral gap:", eigvals[1] - eigvals[0])
    return psi_sol

if __name__ == "__main__":
    import matplotlib.pyplot as plt
#     N = 2 # Number of elements

    n = 3 # Number of qubits
    deg = 2**n - 1 # Degree of Chebyshev Polynomials
    coeffs = (1.0, 4.0, 4.0)
    x_z = 0
    x_s = 0.5

    psi_sol = solve_ODE(deg,coeffs,x_z)
    psi_1 = psi_sol[:deg+1]
    psi_2 = psi_sol[deg+1:]
    psis = (psi_1,psi_2)
    intervals = ([-1,0],[0,1])

    sol = lambda x: 0.5 * (1 + x) * np.exp(-2*x)

    def map(x,interval):
        return (2*x - interval[1] - interval[0])/(interval[1]-interval[0])

    def f(x):
        for i, interval in enumerate(intervals):
            if interval[0] <= x <= interval[1]:
                return np.dot(encoding.chebyshev_encoding(deg, map(x,interval)),
                              psis[i])

    s_eta = sol(x_s) / f(x_s)

#     psi_sol = solve_ODE(deg,coeffs,x_z,N)
#     print("Solution coefficients (Chebyshev basis):")
#     print(psi_sol)
# 
#     s_eta = data_s[1] / np.dot(encoding.chebyshev_encoding(deg=deg, x=data_s[0]), psi_sol)
#     print(s_eta**2, "\n")
# 
    # Plot the solution
    x_plot = np.linspace(-1, 1, 100)
    f_plot = []
    f_true = []
    for x in x_plot:
        f_plot.append(s_eta * f(x))
        f_true.append(sol(x))
    plt.plot(x_plot, f_plot)
    plt.plot(x_plot, f_true)
    plt.title("Solution to ODE")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.show()
