import numpy as np

from src.utils.boundary_matrix import zero_value_boundary_matrix
from src.utils.derivative_matrix import chebyshev_diff_matrix
from src.utils.encoding import chebyshev_encoding
from src.utils.weak_formulation import chebyshev_int_state, constract_iota_state, construct_boundary_matrix_1D, test_basis_matrix

def solve_DE(deg,xs,boundaries,fs,lam):
    Dx = zero_value_boundary_matrix(deg, x_z=xs)
    GTn = chebyshev_diff_matrix(deg=deg)
    ITn = chebyshev_int_state(deg=deg,dim=1)
    In = np.eye(deg+1)

    ## a calculation
    term1 = [GTn@GTn,GTn@GTn]
    a = constract_iota_state(ITn,term1)

    ## l calculation
    term1 = [Dx,In]
    l = constract_iota_state(ITn,term1)

    v = a-l

    C = v.reshape((deg+1),(deg+1))
    Z_1D = test_basis_matrix(deg)
    A = C @ Z_1D @ Z_1D.T @ C.T

    der, xbs, ybs = zip(*boundaries)
#     B = np.zeros((deg+1,deg+1))
#     for i in range(4):
#         print(der[i])
#         print(xbs[i])
#         print(ybs[i])
#         B += construct_boundary_matrix_1D(deg,xs,fs,der[i],xbs[i],ybs[i],lam)

    tau_xs = chebyshev_encoding(deg=deg, x=xs)
    tau_m1 = chebyshev_encoding(deg=deg, x=-1)
    tau_m1_ = chebyshev_encoding(deg=deg, x=-1) @ GTn @ GTn
    tau_p1 = chebyshev_encoding(deg=deg, x=1)
    tau_p1_ = chebyshev_encoding(deg=deg, x=1) @ GTn @ GTn
    b_m1 = fs*tau_m1 - 0*tau_xs
    b_m1_ = fs*tau_m1_ - 0*tau_xs
    b_p1 = fs*tau_p1 - 0*tau_xs
    b_p1_ = fs*tau_p1_ - 0*tau_xs
    B = lam * (np.outer(b_m1,b_m1) + np.outer(b_m1_,b_m1_) +
               np.outer(b_p1,b_p1) + np.outer(b_p1_,b_p1_))

    H = A + B

    eigvals, eigvecs = np.linalg.eigh(H)
    print("Lowest eigval: ", eigvals[0])
    print("Spectral Gap: ", eigvals[1]-eigvals[0])
    u = eigvecs[:,0]
    return u

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.set_printoptions(linewidth=200)

    # Some helper functions
    def tau_n(x):
        return chebyshev_encoding(deg=deg, x=x)

    # True solution
    def true_f(x):
        return (x*x*x*x - 6*x*x + 5)/24

    # Solver and problem parameters
    n = 2
    deg = 2**n-1
    xs, fs = 0, true_f(0)
    boundaries = ((0,-1.0,0.0),
                  (0,1.0,0.0),
                  (2,-1.0,0.0),
                  (2,1.0,0.0))
    lam = 100000

    # Solving the problem
    u = solve_DE(deg,xs,boundaries,fs,lam)
    s_eta = fs/np.dot(tau_n(xs),u)

    # Approximate solution
    def appr_f(x):
        return s_eta * np.dot(tau_n(x),u)

    # Sample data for plotting
    x_plot = np.linspace(-1, 1, 100)
    appr_f_vec = np.vectorize(appr_f)
    true_f_vec = np.vectorize(true_f)
    y_appr = appr_f_vec(x_plot)
    y_true = true_f_vec(x_plot)
    y_diff = y_appr - y_true

    dx = x_plot[1] - x_plot[0]
    error_sq = (y_true-y_appr)**2
    errorL2 = np.sqrt(np.sum(error_sq)*dx)
    print("L2 Error: ", errorL2)

    # Ploting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Comparison of True and Approximate
    ax1.plot(x_plot, y_true, label='True Solution',
             color='tab:blue', linewidth=2)
    ax1.plot(x_plot, y_appr, label='Approximate Solution',
             color='tab:orange', linestyle='--', linewidth=2)
    ax1.set_title("Approximate vs True Solution")
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Difference (Error)
    ax2.plot(x_plot, y_diff, label='Error (Appr - True)', color='tab:red')
    ax2.axhline(0, color='black', lw=1)
    ax2.set_title("Difference (Residual)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("Error")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
