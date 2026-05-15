import numpy as np

from src.utils.boundary_matrix import zero_value_boundary_matrix
from src.utils.derivative_matrix import chebyshev_diff_matrix
from src.utils.encoding import chebyshev_encoding
from src.utils.multiply_matrix import M_x_power
from src.utils.weak_formulation import chebyshev_int_state, constract_iota_state, construct_boundary_matrix, test_basis_matrix


def solve_DE(deg,deg_out,xs,ys,fs,lam):
    M1 = M_x_power(deg, 0, deg_out=deg_out)
    Mx1 = M_x_power(deg, 1, deg_out=deg_out)
    Mx2 = M_x_power(deg, 2, deg_out=deg_out)
    Dx = zero_value_boundary_matrix(deg, x_z=xs)
    Dy = zero_value_boundary_matrix(deg, x_z=ys)
    GTn = chebyshev_diff_matrix(deg=deg)
    ITn = chebyshev_int_state(deg=deg_out,dim=2)

    ## a calculation
    term1 = [M1@GTn,M1,M1@GTn,M1]
    term2 = [M1,M1@GTn,M1,M1@GTn]
    term3 = [Mx1@GTn,M1,M1,M1]
    term4 = [M1,Mx1@GTn,M1,M1]
    a = (constract_iota_state(ITn,term1)+
         constract_iota_state(ITn,term2)+
         constract_iota_state(ITn,term3)*10+
         constract_iota_state(ITn,term4)*10)

    ## l calculation
    term1 = [Mx1@Dx,M1@Dy,M1,M1]
    term2 = [M1@Dx,M1@Dy,M1,M1]
    term3 = [Mx1@Dx,Mx2@Dy,M1,M1]
    term4 = [M1@Dx,Mx2@Dy,M1,M1]
    l = (constract_iota_state(ITn,term1)*4.8+
         constract_iota_state(ITn,term2)*0.4-
         constract_iota_state(ITn,term3)*12-
         constract_iota_state(ITn,term4)*4)/fs

    v = a-l

    C = v.reshape((deg+1)*(deg+1),(deg+1)*(deg+1))
    Z_1D = test_basis_matrix(deg)
    Z_2D = np.kron(Z_1D,Z_1D)
    A = C @ Z_2D @ Z_2D.T @ C.T

    g = np.zeros(deg+1)
    g[0] = np.sqrt(2)
    g[2] = -1
    c1 = -0.2/np.dot(chebyshev_encoding(deg,0),g)
    c2 = 0.6/np.dot(chebyshev_encoding(deg,0),g)
    g1 = c1*g
    g2 = c2*g

    B1 = construct_boundary_matrix(deg,xs,ys,fs,-1,None,g1,lam)
    B2 = construct_boundary_matrix(deg,xs,ys,fs,1,None,g2,lam)
    B3 = construct_boundary_matrix(deg,xs,ys,fs,None,-1,None,lam)
    B4 = construct_boundary_matrix(deg,xs,ys,fs,None,1,None,lam)

    H = A + B1 + B2 + B3 + B4

    eigvals, eigvecs = np.linalg.eigh(H)
    print("Lowest eigval: ", eigvals[0])
    print("Spectral Gap: ", eigvals[1]-eigvals[0])
    u = eigvecs[:,0]
    return u

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import colormaps as cm
    np.set_printoptions(linewidth=200)

    # Some helper functions
    def tau_n(x):
        return chebyshev_encoding(deg=deg, x=x)
    def tau_2D_n(x,y):
        return np.kron(tau_n(x),tau_n(y))

    # True solution
    def true_f(x,y):
        return 0.4*(x+0.5)*(1-y*y)

    # Solver and problem parameters
    n = 2
    deg = 2**n-1
    deg_out = 2**(n+1)-1
    xs, ys, fs = 0, 0, true_f(0,0)
    lam = 1000

    # Solving the problem
    u = solve_DE(deg,deg_out,xs,ys,fs,lam)
    s_eta = fs/np.dot(tau_2D_n(xs,ys),u)

    # Approximate solution
    def appr_f(x,y):
        return s_eta * np.dot(tau_2D_n(x,y),u)

    # Sample data for plotting
    x_plot = np.linspace(-1, 1, 100)
    y_plot = np.linspace(-1, 1, 100)
    X,Y = np.meshgrid(x_plot,y_plot)
    appr_f_vec = np.vectorize(appr_f)
    true_f_vec = np.vectorize(true_f)
    Z_appr = appr_f_vec(X,Y)
    Z_true = true_f_vec(X,Y)
    Z_diff = Z_appr - Z_true

    dx = x_plot[1] - x_plot[0]
    dy = y_plot[1] - y_plot[0]
    error_sq = (Z_true-Z_appr)**2
    errorL2 = np.sqrt(np.sum(error_sq)*dx*dy)
    print("L2 Error: ", errorL2)

    # Ploting
    fig = plt.figure(figsize=(14, 5))

    # Plot Approximate Solution
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_appr, cmap=cm['viridis'], antialiased=True)
    ax1.set_title("Approximate Solution")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    fig.colorbar(surf1, ax=ax1, shrink=0.6, pad=0.08)

    # Plot True Solution
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z_true, cmap=cm['plasma'], antialiased=True)
    ax2.set_title("True Solution")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    fig.colorbar(surf2, ax=ax2, shrink=0.6, pad=0.08)

    # Plot difference
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(X, Y, Z_diff, cmap=cm['RdBu'], antialiased=True)
    ax3.set_title("Difference (Appr - True)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    fig.colorbar(surf3, ax=ax3, shrink=0.6, pad=0.08)

    plt.tight_layout()
    plt.show()
