import numpy as np
from src.utils import derivative_matrix, boundary_matrix, encoding

def solve_PDE(deg, x_z=None, y_z=None, x_m=None, y_m=None):
    GT = derivative_matrix.chebyshev_diff_matrix(deg=deg)
    GT_sq = GT @ GT
    I = np.identity(deg+1,dtype=float)

    A = np.kron(GT_sq,I) + np.kron(I,GT_sq)
    H = A.T @ A

    if x_z is not None:
        for xz in x_z:
            Bxz = boundary_matrix.zero_value_boundary_matrix(deg, x_z=xz)
            H += np.kron(Bxz.T @ Bxz,I)
    if x_m is not None:
        for xm in x_m:
            Bxm = boundary_matrix.zero_value_boundary_matrix(deg, x_z=xm)
            Bxm_GT = Bxm @ GT
            H += np.kron(Bxm_GT.T @ Bxm_GT,I)
    if y_z is not None:
        for yz in y_z:
            Byz = boundary_matrix.zero_value_boundary_matrix(deg, x_z=yz)
            H += np.kron(I,Byz.T @ Byz)
    if y_m is not None:
        for ym in y_m:
            Bym = boundary_matrix.zero_value_boundary_matrix(deg, x_z=ym)
            Bym_GT = Bym @ GT
            H += np.kron(I,Bym_GT.T @ Bym_GT)

    eigvals, eigvecs = np.linalg.eigh(H)
    psi_sol = eigvecs[:, 0]
    print("Spectral gap:", eigvals[1] - eigvals[0])
    return psi_sol

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n = 3
    deg = 2**n-1
    x_z = [-1.0,1.0]
    y_z = [-1.0]
    sol = lambda x,y: np.sin(np.pi*(x+1)/2)*np.sinh(np.pi*(y+1)/2)/np.sinh(np.pi)
#    sol = lambda x,y: np.cosh(np.pi*x/2)*np.sinh(np.pi*(y+1)/2)/np.sinh(np.pi)
    data_s = (0,1,np.sin(np.pi/2))

    psi_sol = solve_PDE(deg,x_z,y_z)
    print("Solution coefficients (Chebyshev basis):")
    print(psi_sol)

    tao_x = encoding.chebyshev_encoding(deg=deg, x=data_s[0])
    tao_y = encoding.chebyshev_encoding(deg=deg, x=data_s[1])
    tao_xy = np.kron(tao_x,tao_y)
    s_eta = data_s[2] / np.dot(tao_xy, psi_sol)
    print(s_eta**2, "\n")

    # Plot the solution
    x_plot = np.linspace(-1, 1, 100)
    y_plot = np.linspace(-1, 1, 100)
    f_plot = np.zeros((100,100))
    f_true = np.zeros((100,100))
    for i,x in enumerate(x_plot):
        tao_x = encoding.chebyshev_encoding(deg=deg, x=x)
        for j,y in enumerate(y_plot):
            tao_y = encoding.chebyshev_encoding(deg=deg, x=y)
            tao = np.kron(tao_x,tao_y)
            fj = s_eta * np.dot(tao, psi_sol)
            f_plot[i,j] = fj
            f_true[i,j] = sol(x,y)

    X, Y = np.meshgrid(x_plot,y_plot)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, f_plot.T, cmap='viridis', edgecolor='none')
    ax.set_title('Approximation f_plot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f_plot')
    fig.colorbar(surf, shrink=0.6, aspect=10)
    plt.tight_layout()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, f_true.T, cmap='viridis', edgecolor='none')
    ax.set_title('Analytical f_true')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f_true')
    fig.colorbar(surf, shrink=0.6, aspect=10)
    plt.tight_layout()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, np.abs(f_plot-f_true).T, cmap='viridis', edgecolor='none')
    ax.set_title('Error abs(f_plot-f_true)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('abs(f_plot-f_true)')
    ax.set_zlim(0, 1)
    fig.colorbar(surf, shrink=0.6, aspect=10)
    plt.tight_layout()

    plt.show()

