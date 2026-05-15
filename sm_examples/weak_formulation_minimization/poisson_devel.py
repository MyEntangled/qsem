import numpy as np
from scipy.linalg import eig
from src.utils import derivative_matrix, boundary_matrix, encoding

def chebyshev_int_matrix_taotao(deg):
    I = np.array([1/(1-(i+j)**2) + 1/(1-(i-j)**2) if (i+j)%2 == 0 else 0 
         for i in range(deg+1)
         for j in range(deg+1)])
    Bn = boundary_matrix.zero_value_boundary_matrix(deg=deg, x_z=1)
    return (np.kron(Bn,Bn) * I)[0]

def solve_Poisson(deg,xs,ys):
    GTn = derivative_matrix.chebyshev_diff_matrix(deg=deg)
    ITn = chebyshev_int_matrix_taotao(deg=deg)
    In = np.eye(deg+1)
    Dn = boundary_matrix.regular_value_boundary_matrix(deg=deg,x_s=xs,y_s=ys)

    A = ITn @ (np.kron(GTn,GTn)/2 - np.kron(In,Dn))
    V = A.reshape(deg+1,deg+1)
    M = (V+V.T)/2

    tau_m1 = encoding.chebyshev_encoding(deg=deg, x=-1)
    tau_p1 = encoding.chebyshev_encoding(deg=deg, x=1)
    C = np.array([tau_m1,tau_p1])
    _,_,Vh = np.linalg.svd(C)
    Z = Vh[2:].T
    Mproj = Z.T @ M @ Z

    eigvals, eigvecs = np.linalg.eigh(Mproj)
    w = eigvecs[:, 0]
    v = Z @ w
    print(v)
    print("Ground State Energy:", eigvals[0])
    print("Spectral gap:", eigvals[1] - eigvals[0])
    return v

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.set_printoptions(linewidth=200)

    n = 2
    deg = 2**n-1
    xs, ys = 0, 0.5

    def generate_coeffs(deg):
        a = np.random.uniform(-1,1,deg+1)
        a = a/np.sum(np.abs(a))
        return a

    def tau_n(x):
        return encoding.chebyshev_encoding(deg=deg, x=x)
    GTn = derivative_matrix.chebyshev_diff_matrix(deg=deg)

#     coeffs = np.array([0.25,0,-0.25,0,0,0,0,0])
#     true_u = coeffs/tau_n(1)
#     true_u = true_u/np.linalg.norm(true_u)

    coeffs = generate_coeffs(deg)
    u = coeffs/tau_n(1)
    u = u/np.linalg.norm(u)

    s_eta = ys/np.dot(tau_n(xs),u)
    eta = s_eta*s_eta

    def f_u(x):
        return np.dot(tau_n(x),u)
    def df_u(x):
        return np.dot(tau_n(x),GTn@u)
    def f_uu(x):
        return np.dot(np.kron(tau_n(x),tau_n(x)),np.kron(u,u))
    def df_uu(x):
        return np.dot(np.kron(tau_n(x),tau_n(x)),np.kron(GTn,GTn)@np.kron(u,u))

    In = np.eye(deg+1)
    Dn = boundary_matrix.regular_value_boundary_matrix(deg=deg,x_s=xs,y_s=ys)
    ITn = chebyshev_int_matrix_taotao(deg=deg)
    int_f_u_quant = np.dot(ITn,np.kron(Dn,In)@np.kron(u,u))
    int_df_u_quant = np.dot(ITn,np.kron(Dn,GTn)@np.kron(u,u))
    int_f_uu_quant = np.dot(ITn,np.kron(u,u))
    int_df_uu_quant = np.dot(ITn,np.kron(GTn,GTn)@np.kron(u,u))

    B = np.vstack([tau_n(-1),tau_n(1)])
    _,_,Vh = np.linalg.svd(B)
    Z = Vh[2:].T

    v = ITn @ (np.kron(GTn,GTn)/2 - np.kron(In,Dn))
    V = v.reshape(deg+1,deg+1)
    V = 0.5 * (V + V.T)
    V_proj = Z.T @ V @ Z

    tau_proj = Z.T @ tau_n(xs)
    G_proj = np.outer(tau_proj,tau_proj)

    print("Lowest energy: ", eta * (u.T @ V @ u))

#     eigvals, eigvecs = eig(V_proj,G_proj)
#     print(eigvals)
#     w = eigvecs[:, 0]
#     u = Z @ w
    w = np.linalg.solve(V_proj,tau_proj)
    u = Z @ w
    s_eta = ys/np.dot(tau_n(xs),u)
    eta = s_eta*s_eta
    print("Found Solution: ", u)
#     print("True Solution: ", true_u)

    print("Found energy: ", eta * (u.T @ V @ u))

    int_f_u_gauss = 0
    int_df_u_gauss = 0
    int_f_uu_gauss = 0
    int_df_uu_gauss = 0
    nodes, weights = np.polynomial.legendre.leggauss(10)
    for x, w in zip(nodes,weights):
        int_f_u_gauss += w*f_u(x)
        int_df_u_gauss += w*df_u(x)
        int_f_uu_gauss += w*f_uu(x)
        int_df_uu_gauss += w*df_uu(x)

    print("\n\nIntegrals of u")
    print(int_f_u_gauss)
    print(int_f_u_quant)
    print(int_f_u_gauss/int_f_u_quant)
    print(1/2**n)

    print("\n\nIntegrals of du")
    print(int_df_u_gauss)
    print(int_df_u_quant)
    print(int_df_u_gauss/int_df_u_quant)
    print(1/2**n)

    print("\n\nIntegrals of u^2")
    print(int_f_uu_gauss)
    print(int_f_uu_quant)
    print(int_f_uu_gauss/int_f_uu_quant)
    print(1/2**n)

    print("\n\nIntegrals of du^2")
    print(int_df_uu_gauss)
    print(int_df_uu_quant)
    print(int_df_uu_gauss/int_df_uu_quant)
    print(1/2**n)

    print("\n\n")

    # Plot the solution
    x_plot = np.linspace(-1, 1, 200)
    f_plot = []
    df_plot = []
    dfdf_plot = []
    for x in x_plot:
        f_plot.append(s_eta*f_u(x))
        df_plot.append(s_eta*df_u(x))
        dfdf_plot.append(eta*df_uu(x))
    plt.axhline(1)
    plt.axhline(-1)
    plt.plot(x_plot, f_plot, label='u')
    # plt.plot(x_plot, df_plot, label='du')
    plt.plot(x_plot, dfdf_plot, label='du^2')
    plt.title("Solution to ODE")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.legend()
    plt.show()
