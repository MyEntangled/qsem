import numpy as np
from utils import derivative_matrix, boundary_matrix, encoding, multiply_matrix
import matplotlib.pyplot as plt


def analytical_deflection(x, q0, EI):
    factor = q0 / (24 * EI)
    return factor * (1 - 2*x**2 + x**4)


def evaluate_deflection(psi, x_vals):
    """
    Evaluate w(x) from Chebyshev coefficients (tau basis).
    x_vals can be scalar or array-like.
    """
    deg = len(psi) - 1
    if np.isscalar(x_vals):
        tau = encoding.chebyshev_encoding(deg, x_vals)
        return np.dot(tau, psi)
    
    res = []
    for x in x_vals:
        tau = encoding.chebyshev_encoding(deg, x)
        res.append(np.dot(tau, psi))
    return np.array(res)


def solve_beam(deg, deg_out, q0, EI, x_z, x_m):
    G = derivative_matrix.chebyshev_diff_matrix(deg=deg)
    L = G @ G @ G @ G
    f = np.zeros((deg + 1, 1))
    f[0,0] = q0[0] / EI
    A = np.hstack([L/np.linalg.norm(L), -f/np.linalg.norm(f)])
    H = A.T @ A

    H_bc = np.zeros_like(H)
    for x in x_z:
        Bx = boundary_matrix.zero_value_boundary_matrix(deg, x)
        Bx = np.hstack([Bx, np.zeros((deg + 1, 1))])
        H_bc += Bx.T @ Bx
    for x in x_m:
        Bx = boundary_matrix.zero_value_boundary_matrix(deg, x)
        BxM = Bx @ G
        BxM = np.hstack([BxM, np.zeros((deg + 1, 1))])
        H_bc += BxM.T @ BxM

    # D0 = boundary_matrix.regular_value_boundary_matrix(deg, 0.0, q0[0]/24/EI)
    # D = np.hstack([D0, np.zeros((deg+1, 1))])
    # H_bc += D.T @ D

    H_eff = H + H_bc / np.linalg.norm(H_bc)
    
    eigs, eigv = np.linalg.eigh(H_eff)
    psi_sol = eigv[:, 0]
    print("Spectral gap:", eigs[1] - eigs[0])
    return psi_sol[:-1] #/ psi_sol[-1]

def euler_beam_fd(EI, L, q0, N):
    """
    Solve EI w'''' = q0 on [0,L] with clamped ends using finite differences.

    Parameters
    ----------
    EI : float
        Flexural rigidity
    L : float
        Beam length
    q0 : float
        Uniform load
    N : int
        Number of intervals (grid points = N+1)

    Returns
    -------
    x : ndarray
        Grid points
    w : ndarray
        Deflection at grid points
    """
    dx = L / N
    x = np.linspace(-L/2, L/2, N+1)
    
    # Initialize matrix
    A = np.zeros((N+1, N+1))
    f = np.full(N+1, q0 / EI)
    
    # Interior points: 2..N-2
    for i in range(2, N-1):
        A[i, i-2] = 1
        A[i, i-1] = -4
        A[i, i]   = 6
        A[i, i+1] = -4
        A[i, i+2] = 1
    # Scale by dx^4
    A[2:N-1, :] /= dx**4
    
    # Boundary conditions
    # w(0) = 0
    A[0,0] = 1
    f[0] = 0
    # w(L) = 0
    A[N,N] = 1
    f[N] = 0
    # w'(0) = 0 -> (-3 w0 +4 w1 - w2)/(2 dx) =0
    A[1,0] = -3/(2*dx)
    A[1,1] = 4/(2*dx)
    A[1,2] = -1/(2*dx)
    f[1] = 0
    # w'(L) =0 -> (3 wN -4 w_{N-1} + w_{N-2})/(2 dx) =0
    A[N-1,N]   = 3/(2*dx)
    A[N-1,N-1] = -4/(2*dx)
    A[N-1,N-2] = 1/(2*dx)
    f[N-1] = 0

    # Solve
    w = np.linalg.solve(A, f)
    
    return w


if __name__ == "__main__":
    n = 5
    deg = 2**n - 1
    deg_out = 2**(n+1) +1
    E = 200e8
    h = 0.25
    b = 0.25
    I = h * b**3 / 12
    EI = E*I
    uniform_load = -500000
    grid_points = 20
    x_z = [-1, 1]
    x_m = [-1, 1]
    
    psi_sol = solve_beam(deg, deg_out, [uniform_load], EI, x_z, x_m)
    print("Solution coefficients (Chebyshev basis):")

    x_plot = np.linspace(-1, 1, 100)
    x_plot_fd = np.linspace(-1, 1, grid_points + 1)
    deflection = evaluate_deflection(psi_sol, x_plot)
    scale = abs(uniform_load) / (24 * EI) / max(np.abs(deflection))
    deflection *= scale

    true_deflection = analytical_deflection(x_plot, uniform_load, EI)
    true_deflection_fd = analytical_deflection(x_plot_fd, uniform_load, EI)

    deflection_fd = euler_beam_fd(EI, L=2, q0=uniform_load, N=grid_points)
    
    error = true_deflection - deflection
    print("Chebyshev Spectral Tau Results:")
    print("Max deflection (numerical): ", np.max(np.abs(deflection)))
    print("Max deflection (analytical): ", np.max(np.abs(true_deflection)))
    print("Max deflection error: ", np.max(np.abs(error)))
    print("L2 Norm of Error: ", np.linalg.norm(error))
    print("Linf Norm of Error: ", np.linalg.norm(error, np.inf))
    print("Relative L2 Error: ", np.linalg.norm(error) / np.linalg.norm(true_deflection))
    error_fd = true_deflection_fd - deflection_fd
    print("\nFinite Difference Results:")
    print("Max deflection (numerical): ", np.max(np.abs(deflection_fd)))
    print("Max deflection (analytical): ", np.max(np.abs(true_deflection)))
    print("Max deflection error: ", np.max(np.abs(error_fd)))
    print("L2 Norm of Error: ", np.linalg.norm(error_fd))
    print("Linf Norm of Error: ", np.linalg.norm(error_fd, np.inf))
    print("Relative L2 Error: ", np.linalg.norm(error_fd) / np.linalg.norm(true_deflection))

    fig, ax = plt.subplots(2, 2, figsize=(8, 10))
    x_plot = np.linspace(-1, 1, 100)
    ax[0,0].plot(x_plot, deflection, label='Numerical (Chebyshev)', linestyle='--', color='blue')
    ax[0,0].plot(x_plot, true_deflection, label='Analytical', linestyle='-', color='red')
    ax[0,0].set_title('Beam Deflection under Uniform Load')
    ax[0,0].set_xlabel('Position along Beam (x)')
    ax[0,0].set_ylabel('Deflection w(x)')
    ax[0,0].legend()
    ax[0,0].grid()
    ax[0,1].plot(x_plot, error, label='Deflection Error', color='green')
    ax[0,1].set_title('Error in Beam Deflection')
    ax[0,1].set_xlabel('Position along Beam (x)')
    ax[0,1].set_ylabel('Error')
    ax[0,1].legend()
    ax[0,1].grid()

    ax[1,0].plot(x_plot_fd, deflection_fd, label="Numerical (Finite Difference)", linestyle='--', color='purple')
    ax[1,0].plot(x_plot_fd, true_deflection_fd, label='Analytical', linestyle='-', color='red')
    ax[1,0].set_title('Beam Deflection (Finite Difference)')
    ax[1,0].set_xlabel('Position along Beam (x)')
    ax[1,0].set_ylabel('Deflection w(x)')
    ax[1,0].legend()
    ax[1,0].grid()
    ax[1,1].plot(x_plot_fd, error_fd, label='Deflection Error (FD)', color='orange')
    ax[1,1].set_title('Error in Beam Deflection (FD)')
    ax[1,1].set_xlabel('Position along Beam (x)')
    ax[1,1].set_ylabel('Error')
    ax[1,1].legend()
    ax[1,1].grid()
    plt.tight_layout()
    plt.savefig('beam_deflection_not_working.png')
    plt.show()
    plt.close()