import numpy as np
from src.utils import derivative_matrix, boundary_matrix, encoding

def chebyshev_int_matrix_taotao(deg):
    I = np.array([1/(1-(i+j)**2) + 1/(1-(i-j)**2) if (i+j)%2 == 0 else 0 
         for i in range(deg+1)
         for j in range(deg+1)])
    Bn = boundary_matrix.zero_value_boundary_matrix(deg=deg, x_z=1)
    res = (np.kron(Bn,Bn) * I)[0]
    return res/np.linalg.norm(res)

def solve_Poisson(deg,xs,ys):
    GTn = derivative_matrix.chebyshev_diff_matrix(deg=deg)
    ITn = chebyshev_int_matrix_taotao(deg=deg)
    In = np.eye(deg+1)
    Dn = boundary_matrix.regular_value_boundary_matrix(deg=deg,x_s=xs,y_s=ys)

    u_m1 = encoding.chebyshev_encoding(deg=deg, x=-1)
    u_p1 = encoding.chebyshev_encoding(deg=deg, x=1)
    B = np.vstack([u_m1,u_p1])
    _,_,Vh = np.linalg.svd(B)
    Z = Vh[2:].T

    v = ITn @ ((np.kron(GTn,GTn)+np.kron(In,In))/2 - np.kron(Dn,In))
    V = v.reshape(deg+1,deg+1)
    V = (V+V.T)/2
    V_proj = Z.T @ V @ Z
    tau_proj = Z.T @ encoding.chebyshev_encoding(deg=deg, x=xs)

#     eigvals,_ = np.linalg.eigh(V_proj)
#     print(eigvals[:3])
    w = np.linalg.solve(V_proj,tau_proj)
    u = Z @ w
    return u

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.set_printoptions(linewidth=200)

    def true_f(x):
        return 1-np.cosh(x)/np.cosh(1)

    n = 3
    deg = 2**n-1
    xs, ys = 0, true_f(0)

    def tau_n(x):
        return encoding.chebyshev_encoding(deg=deg, x=x)

    u = solve_Poisson(deg,xs,ys)
    print(np.linalg.norm(u))
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
