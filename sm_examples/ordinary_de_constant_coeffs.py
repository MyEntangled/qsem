import numpy as np
from src.utils import derivative_matrix, boundary_matrix, encoding

def solve_ODE(deg, a,b,c,x_z):
    G_T = derivative_matrix.chebyshev_diff_matrix(deg=deg)

    A = a * G_T @ G_T + b * G_T + c * np.eye(deg+1)
    B = boundary_matrix.zero_value_boundary_matrix(deg=deg, x_z=x_z)
    #B_hat = boundary_matrix.zero_derivative_boundary_matrix(deg=3, x_m=0.0)

    T_A = A.T @ A
    T_B = B.T @ B

    H = T_A + T_B
    eigvals, eigvecs = np.linalg.eigh(H)
    psi_sol = eigvecs[:, 0]
    print(f"t > {eigvals[-1]/(eigvals[1]*(5))}")
    print("Ground State Energy:", eigvals[0])
    print("Spectral gap:", eigvals[1] - eigvals[0])
    return H/eigvals[-1], psi_sol

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path

    deg = 31
    a = 1.0
    b = 4.0
    c = 4.0
    x_z = -1
    data_s = (0,0.5)

    H, _ = solve_ODE(deg,a,b,c,x_z)
    psi_sol = np.load('./src/penny_examples/ground_state.npy')

    print("Solution coefficients (Chebyshev basis):")
    print(psi_sol)

    s_eta = data_s[1] / np.dot(encoding.chebyshev_encoding(deg=deg, x=data_s[0]), psi_sol)
    print(s_eta**2)

    # Save the hamiltonian
    current_path = Path(__file__).resolve()
    target_dir = current_path.parent.parent / "penny_examples"
    file_path = target_dir / "hamiltonian.npy"
    np.save(file_path, H)
    print(f"Hamiltonian saved to: {file_path}")

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
