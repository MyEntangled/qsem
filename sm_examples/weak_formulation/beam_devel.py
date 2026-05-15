import numpy as np
from src.utils import derivative_matrix, boundary_matrix, encoding

def chebyshev_int_matrix_taotao(deg):
    I = np.array([1/(1-(i+j)**2) + 1/(1-(i-j)**2) if (i+j)%2 == 0 else 0
         for i in range(deg+1)
         for j in range(deg+1)])
    tau = encoding.chebyshev_encoding(deg=deg, x=1)
    return np.kron(tau,tau)*I

def solve_Poisson(deg,xs,ys):
    GTn = derivative_matrix.chebyshev_diff_matrix(deg=deg)
    GTn2 = GTn@GTn
    ITn = chebyshev_int_matrix_taotao(deg=deg)
    In = np.eye(deg+1)
    Dn = boundary_matrix.regular_value_boundary_matrix(deg=deg,x_s=xs,y_s=ys)

## Case 1: Boundary values u(-1)=ddu(-1)=u(1)=ddu(1)=0
#     u_m1 = encoding.chebyshev_encoding(deg=deg, x=-1)
#     ddu_m1 = encoding.chebyshev_encoding(deg=deg, x=-1) @ GTn @ GTn
#     u_p1 = encoding.chebyshev_encoding(deg=deg, x=1)
#     ddu_p1 = encoding.chebyshev_encoding(deg=deg, x=1) @ GTn @ GTn
#     B = np.vstack([u_m1,ddu_m1,u_p1,ddu_p1])

## Case 2: Boundary values u(-1)=du(-1)=ddu(1)=dddu(1)=0
    u_m1 = encoding.chebyshev_encoding(deg=deg, x=-1)
    du_m1 = encoding.chebyshev_encoding(deg=deg, x=-1) @ GTn
    ddu_p1 = encoding.chebyshev_encoding(deg=deg, x=1) @ GTn @ GTn
    dddu_p1 = encoding.chebyshev_encoding(deg=deg, x=1) @ GTn @ GTn @ GTn
    B = np.vstack([u_m1,du_m1,ddu_p1,dddu_p1])

    _,_,Vh = np.linalg.svd(B)
    Z = Vh[4:].T

    v = ITn @ (np.kron(GTn2,GTn2) - np.kron(Dn,In))
    C = v.reshape(deg+1,deg+1)
    A = C @ Z @ Z.T @ C.T
    W = Z.T @ A @ Z

    eigvals, eigvecs = np.linalg.eigh(W)
    print("Lowest eigval: ", eigvals[0])
    print("Spectral Gap: ", eigvals[1]-eigvals[0])
    w = eigvecs[:,0]
    u = Z @ w
    return u

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.set_printoptions(linewidth=200)

## Case 1: Boundary values u(-1)=ddu(-1)=u(1)=ddu(1)=0
#     def true_f(x):
#         return (x*x*x*x - 6*x*x + 5)/24
## Case 2: Boundary values u(-1)=du(-1)=ddu(1)=dddu(1)=0
    def true_f(x):
        return ((x-1)**4 + 32*x + 16)/60

    n = 5
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
