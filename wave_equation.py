from utils import derivative_matrix, boundary_matrix, encoding
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def get_2d_cheb_coeffs(func, deg):
    nodes = np.cos(np.pi * (np.arange(deg + 1) + 0.5) / (deg + 1))
    X, Y = np.meshgrid(nodes, nodes, indexing='ij')
    F_vals = func(X, Y)

    coeffs_y = np.polynomial.chebyshev.chebfit(nodes, F_vals.T, deg).T
    coeffs = np.polynomial.chebyshev.chebfit(nodes, coeffs_y, deg)
    weights = np.array([derivative_matrix.get_weight(k, deg) for k in range(deg+1)])

    W_x = weights[:, np.newaxis]
    W_y = weights[np.newaxis, :]
    target_coeffs = coeffs / (W_x * W_y)
    return target_coeffs.flatten()

def solve_wave_equation(deg, c_func, x_z, x_m, y_z, y_m, initial_cond_func=None, initial_deriv_func=None):
    """
    Solves the 2D Transient Wave Equation using Spectral methods with Hamiltonian formulation.
    
    Parameters
    ----------
    deg: int
        Degree of the Chebyshev polynomial basis.
    c_func: float
        Wave speed function c(x, y, t).
    x_z: list of float
        Spatial points where the solution is zero.
    x_m: list of float
        Spatial points where the spatial derivative of the solution is zero.
    y_z: list of float
        Spatial points where the solution is zero.
    y_m: list of float
        Spatial points where the spatial derivative of the solution is zero.
    t_z: list of float
        Temporal points where the solution is zero.
    t_m: list of float
        Temporal points where the temporal derivative of the solution is zero.
    """
    G = derivative_matrix.chebyshev_diff_matrix(deg=deg)
    Lx = G @ G
    Ly = G @ G
    Lt = G @ G
    I = np.eye(deg+1)

    A = np.kron(Lt, np.kron(I, I)) - c_func**2 * (np.kron(I, np.kron(Lx, I)) + np.kron(I, np.kron(I, Ly)))
    H = A.T @ A
    if x_z is not None:
        for x in x_z:
            Bx = boundary_matrix.zero_value_boundary_matrix(deg, x)
            H += np.kron(I, np.kron(Bx.T @ Bx, I))
    if x_m is not None:
        for x in x_m:
            Bx = boundary_matrix.zero_value_boundary_matrix(deg, x)
            BxG = Bx @ G
            H += np.kron(I, np.kron(BxG.T @ BxG, I))
    if y_z is not None:
        for y in y_z:
            By = boundary_matrix.zero_value_boundary_matrix(deg, y)
            H += np.kron(I, np.kron(I, By.T @ By))
    if y_m is not None:
        for y in y_m:
            By = boundary_matrix.zero_value_boundary_matrix(deg, y)
            ByG = By @ G
            H += np.kron(I, np.kron(I, ByG.T @ ByG))

    if initial_cond_func is not None:
        f_coeffs = get_2d_cheb_coeffs(initial_cond_func, deg)
        norm_f = np.linalg.norm(f_coeffs)
        if norm_f > 1e-12:
            f_hat = f_coeffs / norm_f
            P_f = np.outer(f_hat, f_hat)
            P_orth = np.eye((deg+1)**2) - P_f
            Bt = boundary_matrix.zero_value_boundary_matrix(deg, -1.0)
            H += np.kron(Bt.T @ Bt, P_orth)

    if initial_deriv_func is not None:
        Bt = boundary_matrix.zero_value_boundary_matrix(deg, -1.0)
        BtG = Bt @ G
        H += np.kron(BtG.T @ BtG, np.kron(I, I))

    # print("Maximum value in H:", np.max(np.abs(H)))
    # print("Minimum value in H:", np.min(np.abs(H)))
    # print("Condition number of H:", np.linalg.cond(H))
    eigvals, eigvecs = np.linalg.eigh(H)
    # np.save("wave_hamiltonian.npy", H)
    psi_sol = eigvecs[:, 0]

    if initial_cond_func is not None:
        tau_start = encoding.chebyshev_encoding(deg, -1.0)
        psi_tensor = psi_sol.reshape((deg+1, (deg+1)**2))
        psi_t_start = tau_start @ psi_tensor
        f_coeffs = get_2d_cheb_coeffs(initial_cond_func, deg)
        projection = np.dot(psi_t_start, f_coeffs)
        norm_psi_t = np.dot(f_coeffs, f_coeffs)
        
        if abs(projection) > 1e-12:
            scale = norm_psi_t / projection
            psi_sol *= scale

    print("Spectral gap:", eigvals[1] - eigvals[0])
    return psi_sol

def initial_conditions(x, y):
    return np.sin(np.pi * (x + 1) / 2) * np.sin(np.pi * (y + 1) / 2)

def derivative_initial_conditions():
    return 0.0

def analytic_solution(x, y, t):
    omega = np.pi / np.sqrt(2)
    return np.sin(np.pi * (x + 1) / 2) * np.sin(np.pi * (y + 1) / 2) * np.cos(omega * t)

def create_animation(x_vals, y_vals, t_vals, wave_solution, name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x_vals, y_vals)
    def update_frame(frame):
        ax.clear()
        ax.set_zlim(-1, 1)
        ax.plot_surface(X, Y, wave_solution[frame], cmap='viridis')
        ax.set_title(f'Time = {t_vals[frame]:.2f}')
    
    ani = animation.FuncAnimation(fig, update_frame, frames=len(t_vals), interval=100)
    ani.save('{}.gif'.format(name), writer='imagemagick')
    plt.show()

def plot_comparison_slice(x_vals, sem_slice, analytic_slice, time_val, y_val):
    sem_slice = np.array(sem_slice)
    analytic_slice = np.array(analytic_slice)
    abs_error = np.abs(sem_slice - analytic_slice)
    max_err = np.max(abs_error)
    mse_err = np.mean(abs_error**2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=100, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot solutions
    ax1.plot(x_vals, sem_slice, label="SEM Solution", color='blue', linewidth=2)
    ax1.plot(x_vals, analytic_slice, label="Analytic Solution", linestyle='--', color='red', linewidth=2)
    ax1.set_ylabel("Wave Amplitude", fontsize=14)
    ax1.set_title(f"Wave Solution Slice at y={y_val:.2f}, t={time_val:.2f}", fontsize=16)
    ax1.legend(fontsize=12, loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # Add text box with error stats
    textstr = '\n'.join((
        r'$\mathrm{Max\ Error}=%.2e$' % (max_err, ),
        r'$\mathrm{MSE}=%.2e$' % (mse_err, )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    # Plot Error
    ax2.plot(x_vals, abs_error, label="Absolute Error", color='black', linewidth=1.5)
    ax2.fill_between(x_vals, abs_error, color='gray', alpha=0.3)
    ax2.set_xlabel("Position x", fontsize=14)
    ax2.set_ylabel("Abs. Error", fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.legend(fontsize=12, loc='upper right')

    plt.tight_layout()
    plt.savefig("Wave_slice_comparison.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    n = 4
    deg = 2**n - 1
    c = 1.0
    T_end = 3.0
    N_gridx = 100
    N_gridy = 100
    N_gridt = 100
    c_eff = c * T_end / 2.0
    plot = False
    
    x_z = [-1, 1]
    y_z = [-1, 1]
    x_m = None
    y_m = None

    psi_sol = solve_wave_equation(deg, c_eff, x_z, x_m, y_z, y_m, 
                                  initial_cond_func=initial_conditions, 
                                  initial_deriv_func=derivative_initial_conditions)
    
    x_vals = np.linspace(-1, 1, N_gridx)
    y_vals = np.linspace(-1, 1, N_gridy)
    t_vals = np.linspace(-1, 1, N_gridt)
    
    wave_solution = np.zeros((len(t_vals), len(x_vals), len(y_vals)))
    for ti, t in enumerate(t_vals):
        for xi, x in enumerate(x_vals):
            for yi, y in enumerate(y_vals):
                tau_x = encoding.chebyshev_encoding(deg, x)
                tau_y = encoding.chebyshev_encoding(deg, y)
                tau_t = encoding.chebyshev_encoding(deg, t)
                tau_txy = np.kron(tau_t, np.kron(tau_x, tau_y))
                w_val = np.dot(tau_txy, psi_sol)
                wave_solution[ti, xi, yi] = w_val
    
    t_vals_original = (t_vals + 1) * (T_end / 2)
    true_solution = np.zeros((len(t_vals), len(x_vals), len(y_vals)))
    for ti, t in enumerate(t_vals_original):
        for xi, x in enumerate(x_vals):
            for yi, y in enumerate(y_vals):
                true_solution[ti, xi, yi] = analytic_solution(x, y, t)
                
    error = np.linalg.norm(wave_solution - true_solution) / np.linalg.norm(true_solution)
    print("Relative L2 Error:", error)

    np.save("wave_solution_sem.npy", wave_solution)
    np.save("wave_solution_analytic.npy", true_solution)

    if plot:
        create_animation(x_vals, y_vals, t_vals_original, wave_solution, "Computed Solution")
        create_animation(x_vals, y_vals, t_vals_original, true_solution, "Analytic Solution")