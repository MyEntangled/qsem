import numpy as np
from src.utils import derivative_matrix, boundary_matrix, encoding

def chebyshev_int_matrix_taotao(deg):
    I = np.array([1/(1-(i+j)**2) + 1/(1-(i-j)**2) if (i+j)%2 == 0 else 0
         for i in range(deg+1)
         for j in range(deg+1)])
    tau = encoding.chebyshev_encoding(deg=deg, x=1)
    return np.kron(tau,tau)*I

def chebyshev_test_basis_matrix(deg):
    Z = np.eye(deg+1, deg-1, k=0) - np.eye(deg+1, deg-1, k=-2)
    Z[0,0] = np.sqrt(2)
    return Z

def solve_Poisson(deg,boundaries,xs,ys,lam=1):
    GTn = derivative_matrix.chebyshev_diff_matrix(deg=deg)
    ITn = chebyshev_int_matrix_taotao(deg=deg)
    In = np.eye(deg+1)
    Dn = boundary_matrix.regular_value_boundary_matrix(deg=deg,x_s=xs,y_s=ys)

    v = ITn @ (0.1*np.kron(GTn,GTn) + np.kron(GTn,In) - 0.4*np.kron(Dn,In))
    C = v.reshape(deg+1,deg+1)
    Z = chebyshev_test_basis_matrix(deg)
    A = C @ Z @ Z.T @ C.T

    tau_xs = encoding.chebyshev_encoding(deg=deg, x=xs)
    tau_m1 = encoding.chebyshev_encoding(deg=deg, x=boundaries[0][0])
    tau_p1 = encoding.chebyshev_encoding(deg=deg, x=boundaries[1][0])
    b_m1 = ys*tau_m1 - boundaries[0][1]*tau_xs
    b_p1 = ys*tau_p1 - boundaries[1][1]*tau_xs
    B = lam * (np.outer(b_m1,b_m1) + np.outer(b_p1,b_p1))

    H = A + B

    eigvals, eigvecs = np.linalg.eigh(H)
    print("Lowest eigval: ", eigvals[0])
    print("Spectral Gap: ", eigvals[1]-eigvals[0])
    u = eigvecs[:,0]
    return u

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.set_printoptions(linewidth=200)

    def true_f(x):
        return 0.4*(x+1-2*(np.exp((x+1)/0.1)-1)/(np.exp(2/0.1)-1))

    n = 3
    deg = 2**n-1
    xs, ys = 0, true_f(0)
    boundaries = ((-1,true_f(-1)),(1,true_f(1)))
    lam = 1000

    def tau_n(x):
        return encoding.chebyshev_encoding(deg=deg, x=x)

    u = solve_Poisson(deg,boundaries,xs,ys,lam)
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
