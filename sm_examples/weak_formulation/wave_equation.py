import numpy as np
from src.utils import derivative_matrix, boundary_matrix, encoding, multiply_matrix
from functools import reduce
from math import factorial as fac

def contract_iota(iota, mats):
    shape_in = [m.shape[0] for m in mats]
    res = iota.reshape(shape_in)
    for i in range(len(mats)):
        res = np.tensordot(res,mats[i],axes=([0],[0]))
    return res.flatten()

def chebyshev_int_state(deg,dim):
    I = np.array([1/(1-(i+j)**2) + 1/(1-(i-j)**2) if (i+j)%2 == 0 else 0
         for i in range(deg+1)
         for j in range(deg+1)])
    tau = encoding.chebyshev_encoding(deg=deg, x=1)
    iota = np.kron(tau,tau)*I
    if dim == 1:
        return iota
    else:
        M = reduce(np.kron, [iota.reshape(deg+1,deg+1)] * dim)
        return M.flatten()

def solve_Poisson(deg,deg_out,xs,ts,fs):
    M1 = multiply_matrix.M_x_power(deg, 0, deg_out=deg_out)
    GTn = derivative_matrix.chebyshev_diff_matrix(deg=deg)
    ITn = chebyshev_int_state(deg=deg_out,dim=2)

## Uncomment to use assuming no information about the solution
#     Dx = boundary_matrix.zero_value_boundary_matrix(deg, x_z=xs)
#     Dt = boundary_matrix.zero_value_boundary_matrix(deg, x_z=ts)

    ## a calculation
    term1 = [-M1,M1@GTn,M1,M1@GTn]
    term2 = [4*M1@GTn,M1,M1@GTn,M1]
    a = (contract_iota(ITn,term1)+
         contract_iota(ITn,term2))

    ## l calculation
    l = 0

## Uncomment to use assuming no information about the solution
#     c1 = -16*np.pi*np.pi
#     css = [(-1)**i * (2*np.pi)**(2*i+1)/fac(2*i+1) for i in range(deg//2+1)]
#     for i in range(deg//2+1):
#         Mxn = multiply_matrix.M_x_power(deg, 2*i+1, deg_out=deg_out)
#         l += contract_iota(ITn,[(c1*css[i])*Mxn@Dx,M1@Dt,M1,M1])
#     l /= fs

    v = a-l

    ux_m1 = encoding.chebyshev_encoding(deg=deg, x=-1)
    ux_p1 = encoding.chebyshev_encoding(deg=deg, x=1)
    ut_m1 = encoding.chebyshev_encoding(deg=deg, x=-7/8)
    udt_m1 = encoding.chebyshev_encoding(deg=deg, x=-1) @ GTn
    Bx = np.vstack([ux_m1,ux_p1])
    _,_,Vh = np.linalg.svd(Bx)
    Zx = Vh[2:].T
    Bt = np.vstack([ut_m1,udt_m1])
    _,_,Vh = np.linalg.svd(Bt)
    Zt = Vh[2:].T
    Z = np.kron(Zx,Zt)

    C = v.reshape((deg+1)*(deg+1),(deg+1)*(deg+1))
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
    from matplotlib import cm
    np.set_printoptions(linewidth=200)

    def true_f(x,t):
        return np.sin(2*np.pi*x)*np.cos(4*np.pi*(t+1))

    n = 4
    deg = 2**n-1
    deg_out = 2**(n+1)-1
    xs, ts, fs = 0.25, -1.0, true_f(0.25,-1.0)

    def tau_n(x):
        return encoding.chebyshev_encoding(deg=deg, x=x)
    def tau_2D_n(x,t):
        return np.kron(tau_n(x),tau_n(t))

    u = solve_Poisson(deg,deg_out,xs,ts,fs)
    s_eta = fs/np.dot(tau_2D_n(xs,ts),u)

    def appr_f(x,t):
        return s_eta * np.dot(tau_2D_n(x,t),u)

    # Plot the solution
    x_plot = np.linspace(-1, 1, 100)
    t_plot = np.linspace(-1, 1, 100)
    X,T = np.meshgrid(x_plot,t_plot)
    appr_f_vec = np.vectorize(appr_f)
    true_f_vec = np.vectorize(true_f)
    Z_appr = appr_f_vec(X,T)
    Z_true = true_f_vec(X,T)

    dx = x_plot[1] - x_plot[0]
    dt = t_plot[1] - t_plot[0]
    error_sq = (Z_true-Z_appr)**2
    errorL2 = np.sqrt(np.sum(error_sq)*dx*dt)
    print("L2 Error: ", errorL2)

    # 4. Plotting
    fig = plt.figure(figsize=(14, 6))

    # Plot Approximate Solution
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X, T, Z_appr, cmap=cm.viridis, antialiased=True)
    ax1.set_title("Approximate Solution")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    # Plot True Solution
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X, T, Z_true, cmap=cm.plasma, antialiased=True)
    ax2.set_title("True Solution")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

    plt.tight_layout()
    plt.show()
