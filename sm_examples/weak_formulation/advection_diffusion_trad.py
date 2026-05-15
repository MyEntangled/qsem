import numpy as np
from src.utils import derivative_matrix, boundary_matrix, encoding

def solve_Poisson(deg,xs,ys):
    GTn = derivative_matrix.chebyshev_diff_matrix(deg=deg)
    Dn = boundary_matrix.regular_value_boundary_matrix(deg=deg,x_s=xs,y_s=ys)

    A = -0.1*GTn@GTn + GTn - 0.4*Dn
    T_A = A.T@A

    u_m1 = encoding.chebyshev_encoding(deg=deg, x=-1)
    u_p1 = encoding.chebyshev_encoding(deg=deg, x=1)
    B = np.vstack([u_m1,u_p1])
    _,_,Vh = np.linalg.svd(B)
    Z = Vh[2:].T

    H = Z.T @ T_A @ Z
    eigvals, eigvecs = np.linalg.eigh(H)
    psi_sol = eigvecs[:, 0]

    print("Ground State Energy:", eigvals[0])
    print("Spectral gap:", eigvals[1] - eigvals[0])
    return Z @ psi_sol

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.set_printoptions(linewidth=200)

    def true_f(x):
        return 0.4*(x+1-2*(np.exp((x+1)/0.1)-1)/(np.exp(2/0.1)-1))

    n = 3
    deg = 2**n-1
    xs, ys = 0, true_f(0)

    def tau_n(x):
        return encoding.chebyshev_encoding(deg=deg, x=x)

    u = solve_Poisson(deg,xs,ys)
    s_eta = ys/np.dot(tau_n(xs),u)

    def appr_f(x):
        return s_eta * np.dot(tau_n(x),u)

    # Plot the solution
    x_plot = np.linspace(-1, 1, 200)
    appr_f_plot = []
    true_f_plot = []
    for x in x_plot:
        appr_f_plot.append(appr_f(x))
        true_f_plot.append(true_f(x))
    errorL2 = np.sqrt(2*np.sum((np.array(true_f_plot)-
                                np.array(appr_f_plot))**2)/200)
    print("L2 Error: ", errorL2)
    plt.plot(x_plot, appr_f_plot, label='appr_u')
    plt.plot(x_plot, true_f_plot, label='true_u')
    plt.title("Solution to ODE")
    plt.xlabel("x")
    plt.xlim(-1,1)
    plt.ylabel("f(x)")
    plt.ylim(-1,1)
    plt.grid()
    plt.legend()
    plt.show()
