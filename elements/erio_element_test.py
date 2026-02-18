from src.utils import boundary_matrix, function_evaluation, equation_hamiltonian, interface_continuity, regularization, \
    meshing, sem_boundary_matrix
from src.utils import multivar_equation_parsing, multivar_equation_hamiltonian
import numpy as np
import sympy as sp

# Analytical solution
from scipy.special import lpmv

def test_ode_variable_coeff(n: int, num_elements: int):
    d = 2 ** n - 1
    d_out = 2 ** (n + 1) - 1

    x = sp.Symbol('x')
    f = sp.Function('f')(x)

    d1 = f.diff(x)
    d2 = f.diff(x, 2)

    l, m = 5, 1
    lhs = (1 - x ** 2) ** 2 * d2 - 2 * x * (1 - x ** 2) * d1 + (l * (l + 1) * (1 - x ** 2) - m ** 2) * f
    #lhs = (1 - x**2) * d2 - 2*x*d1 + (l*(l+1) - m**2/(1-x**2))*f
    #lhs = (x-1) * d2 - x * d1 + f - (x-1)**2
    print(f"Equation LHS: {lhs}")



    x_s = 1/3
    sol = lambda x: float(lpmv(m, l, x))
    #sol = lambda x: 1.5 * np.exp(x) - 0.125*x*(8*x+13) - 1
    data_s = (x_s, sol(x_s))
    print("Regular data point:", data_s)

    ## Create endpoints for the mesh
    nodes = np.linspace(-1, 1, num_elements + 1)
    endpoints = np.column_stack((nodes[:-1], nodes[1:]))

    H_diff_sem = equation_hamiltonian.sem_equation_hamiltonian(d, d_out, lhs, f, x, endpoints=endpoints, regular_data=None, regular_data_type='value')

    x_m = 0 #-0.195976
    HB_sem = sem_boundary_matrix.sem_boundary_hamiltonian(type='derivative', deg=d, deg_out=d_out, endpoints=endpoints, x=x_m)

    HC0_sem = interface_continuity.boundary_continuity_matrice('value', num_elements, d)
    HC1_sem = interface_continuity.boundary_continuity_matrice('derivative', num_elements, d)

    H_reg = regularization.regularization_matrix(d, p=1)
    H_reg_sem = np.kron(np.eye(num_elements), H_reg)

    H_sem =  H_diff_sem +  HB_sem + (HC0_sem + HC1_sem)*num_elements**2 + H_reg_sem * 0.05
    print("Hamiltonian size:", H_sem.shape)

    eigvals, eigvecs = np.linalg.eigh(H_sem)
    psi_sol = eigvecs[:, 0]
    print("Ground energy", eigvals[0])
    print("Spectral gap:", eigvals[1] - eigvals[0])
    print("Solution coefficients", np.reshape(psi_sol, (num_elements, -1)).round(2))

    f_s = function_evaluation.evaluate_sem_encoding(psi_sol=psi_sol, deg=d, deg_out=d_out, endpoints=endpoints,
                                                    x_eval_list=[data_s[0]], scaling_factor=1.0)[0]

    s_eta = data_s[1] / f_s
    print("Scaling^2:", s_eta ** 2)

    # Plot the solution
    import matplotlib.pyplot as plt

    x_plot = np.linspace(-1, 1, 10000)
    fQ_plot = []
    f_plot = []

    for xj in x_plot:
        yj = function_evaluation.evaluate_sem_encoding(psi_sol=psi_sol, deg=d, deg_out=d_out, endpoints=endpoints,
                                                       x_eval_list=np.array([xj]), scaling_factor=s_eta)[0]
        fQ_plot.append(yj)
        f_plot.append(sol(xj))
    plt.plot(x_plot, fQ_plot, c='red', label=r'$f^*_{Q}(x)$')
    plt.plot(x_plot, f_plot, '--', label=r'f^*(x)')
    plt.title(f"Solution to ODE: n={n}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()

    #return H_diff_sem, HB_sem, HC0_sem, HC1_sem

def test_inhom_ode(n: int, num_elements: int):
    d = 2 ** n - 1
    d_out = 2 ** (n+1) - 1

    x = sp.Symbol('x')
    f = sp.Function('f')(x)

    d1 = f.diff(x)
    d2 = f.diff(x, 2)

    lhs = (x-1) * d2 - x * d1 + f - (x-1)**2
    print(f"Equation LHS: {lhs}")

    x_s = -0.5
    sol = lambda x: float(1.5 * np.exp(x) - 0.125*x*(8*x+13) - 1)
    data_s = (x_s, sol(x_s))

    ## Create endpoints for the mesh
    nodes = np.linspace(-1, 1, num_elements + 1)
    endpoints = np.column_stack((nodes[:-1], nodes[1:]))

    H_diff_sem = equation_hamiltonian.sem_equation_hamiltonian(d, d_out, lhs, f, x, endpoints=endpoints, regular_data=data_s, regular_data_type='value')

    x_m = -0.195976
    HB_sem = sem_boundary_matrix.sem_boundary_hamiltonian(type='derivative', deg=d, deg_out=d_out, endpoints=endpoints, x=x_m)

    HC0_sem = interface_continuity.boundary_continuity_matrice('value', num_elements, d)
    HC1_sem = interface_continuity.boundary_continuity_matrice('derivative', num_elements, d)

    H_sem = H_diff_sem + HB_sem + (HC0_sem + HC1_sem)*num_elements**2
    print("Hamiltonian size:", H_sem.shape)

    eigvals, eigvecs = np.linalg.eigh(H_sem)
    psi_sol = eigvecs[:, 0]
    print("Ground energy", eigvals[0])
    print("Spectral gap:", eigvals[1] - eigvals[0])
    print("Solution coefficients", np.reshape(psi_sol, (num_elements, -1)).round(2))

    f_s = function_evaluation.evaluate_sem_encoding(psi_sol=psi_sol, deg=d, deg_out=d_out, endpoints=endpoints,
                                                    x_eval_list=[data_s[0]], scaling_factor=1.0)[0]
    print("Evaluated solution at regular data point:", f_s)

    s_eta = data_s[1] / f_s
    print("Scaling^2:", s_eta ** 2)


    # Plot the solution
    import matplotlib.pyplot as plt

    x_plot = np.linspace(-1, 1, 100000)
    fQ_plot = []
    f_plot = []

    for xj in x_plot:
        yj = function_evaluation.evaluate_sem_encoding(psi_sol=psi_sol, deg=d, deg_out=d_out, endpoints=endpoints,
                                                       x_eval_list=np.array([xj]), scaling_factor=s_eta)[0]
        fQ_plot.append(yj)
        f_plot.append(sol(xj))
    plt.plot(x_plot, fQ_plot, c='red', label=r'$f^*_{Q}(x)$')
    plt.plot(x_plot, f_plot, '--', label=r'f^*(x)')
    plt.title(f"Solution to ODE: n={n}, num_elements={num_elements}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()


def test_wave_equation(n: int, num_elements_for_dims: np.ndarray):
    d = 2 ** n - 1
    d_out = 2 ** n - 1

    x, y, t = sp.symbols('x y t')
    f = sp.Function('f')(x, y, t)

    d2x = f.diff(x, 2)
    d2y = f.diff(y, 2)
    d2t = f.diff(t, 2)

    c = 1.0
    lhs = d2t - c**2 * (d2x + d2y)
    print(f"Equation LHS: {lhs}")


    ## General setting
    Lx, Ly, T = 1.0, 1.0, 1.0
    mx, my = 2,3
    omega = np.pi * np.sqrt((mx / Lx) ** 2 + (my / Ly) ** 2)
    sol = lambda x, y, t: float(np.sin(mx * np.pi * x / Lx) * np.sin(my * np.pi * y / Ly) * np.cos(omega * t))

    coord_s = (0.25, 0.25, 0.5)
    data_s = (coord_s, sol(*coord_s))
    print("Regular data point:", data_s)

    ## Create endpoints for the 3D mesh
    x_nodes = np.linspace(0, Lx, num_elements_for_dims[0] + 1)
    y_nodes = np.linspace(0, Ly, num_elements_for_dims[1] + 1)
    t_nodes = np.linspace(0, 1.0, num_elements_for_dims[2] + 1)

    mesh = meshing.RectMesh([x_nodes, y_nodes, t_nodes])
    endpoints = mesh.bbox(np.arange(mesh.num_elems))

    print("Endpoints", endpoints)

    # Create Hamiltonian
    print(d, d_out, endpoints.shape, data_s)
    H_diff_sem = multivar_equation_hamiltonian.sem_multivar_equation_hamiltonian(d,
                                                                                 d_out,
                                                                                 diff_eq=lhs,
                                                                                 func=f,
                                                                                 vars=(x, y, t),
                                                                                 endpoints=endpoints,
                                                                                 regular_data=data_s,
                                                                                 regular_data_type='value')
    print("Hamiltonian size:", H_diff_sem.shape)

    # Boundary conditions (Dirichlet)
    # f(0,y,t) = f(Lx,y,t) = f(x,0,t) = f(x,Ly,t) = 0 for all t
    # Optional: f(x,y,0) = 0 for all x,y

    HB_sem = sem_boundary_matrix.sem_multivar_boundary_hamiltonian(deg=d, deg_out=d_out, endpoints=endpoints, type='value', coords=(0, None, None)) + \
             sem_boundary_matrix.sem_multivar_boundary_hamiltonian(deg=d, deg_out=d_out, endpoints=endpoints, type='value', coords=(Lx, None, None)) + \
             sem_boundary_matrix.sem_multivar_boundary_hamiltonian(deg=d, deg_out=d_out, endpoints=endpoints, type='value', coords=(None, 0, None)) + \
             sem_boundary_matrix.sem_multivar_boundary_hamiltonian(deg=d, deg_out=d_out, endpoints=endpoints, type='value', coords=(None, Ly, None)) + \
             sem_boundary_matrix.sem_multivar_boundary_hamiltonian(deg=d, deg_out=d_out, endpoints=endpoints, type='value', coords=(None, None, 0))


    HC0_sem = interface_continuity.multivar_boundary_continuity_matrix('value', num_elements_for_dims, d)
    HC1_sem = interface_continuity.multivar_boundary_continuity_matrix('derivative', num_elements_for_dims, d)

    print("Continuity matrices norm", np.linalg.norm(HC0_sem), np.linalg.norm(HC1_sem))

    M = sum(num_elements_for_dims)
    H_sem = H_diff_sem + HB_sem + (HC0_sem + HC1_sem)*M**2

    ## Solve the Hamiltonian
    eigvals, eigvecs = np.linalg.eigh(H_sem)
    psi_sol = eigvecs[:, 0]
    print("Ground energy", eigvals[0])
    print("Spectral gap:", eigvals[1] - eigvals[0])
    print("Solution coefficients", np.reshape(psi_sol, (mesh.num_elems, -1)).round(2))

    f_s = function_evaluation.evaluate_multivar_sem_encoding(psi_sol=psi_sol, deg=d, deg_out=d_out, endpoints=endpoints,
                                                    coords_eval_list=np.array([data_s[0]]), scaling_factor=1.0)[0]
    print("Evaluated solution at regular data point:", f_s)

    s_eta = data_s[1] / f_s
    print("Scaling^2:", s_eta ** 2)

    # # Plot the solution at t=T/2
    # import matplotlib.pyplot as plt
    # x_plot = np.linspace(0, Lx, 100)
    # y_plot = np.linspace(0, Ly, 100)
    # X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
    # fQ_plot = np.zeros_like(X_plot)
    #
    # for i in range(X_plot.shape[0]):
    #     for j in range(X_plot.shape[1]):
    #         coord_ij = (X_plot[i, j], Y_plot[i, j], T/2)
    #         fQ_plot[i, j] = function_evaluation.evaluate_multivar_sem_encoding(psi_sol=psi_sol, deg=d, deg_out=d_out, endpoints=endpoints,
    #                                                 coords_eval_list=np.array([coord_ij]), scaling_factor=s_eta)[0]
    # plt.contourf(X_plot, Y_plot, fQ_plot, levels=50, cmap='viridis')
    # plt.colorbar(label=r'$f^*_{Q}(x,y,T/2)$')
    # plt.title(f"Solution to Wave Equation at t=T/2: n={n}, num_elements={num_elements_for_dims}")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.grid()
    # plt.show()

    animate_wave_solution_sem(psi_sol=psi_sol, deg=d, deg_out=d_out, endpoints=endpoints, scaling_factor=s_eta, Lx=Lx, Ly=Ly, T=T, n=n, num_elements_for_dims=num_elements_for_dims, zlim=(-1.05, 1.05))
    return

def animate_wave_solution_sem(psi_sol,
                              deg,
                deg_out,
                endpoints,
                scaling_factor,
                Lx,
                Ly,
                T,
                n,
                num_elements_for_dims,
                nx=60,
                ny=60,
                n_frames=60,
                interval=100,
                zlim=None):
    """Animate the SEM solution as a 3D surface z = f(x,y,t) over t in [0, T].

    Notes
    -----
    - This is expensive: per frame, we evaluate the SEM solution on an (nx x ny) grid.
    - For speed, reduce nx/ny or n_frames.

    Parameters
    ----------
    zlim : tuple | None
        Optional (zmin, zmax) for a fixed z-axis range. If None, uses the first frame
        with a small margin.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from src.utils import function_evaluation

    # Grid (fixed)
    x_plot = np.linspace(0, Lx, nx)
    y_plot = np.linspace(0, Ly, ny)
    X_plot, Y_plot = np.meshgrid(x_plot, y_plot)

    def eval_field(t_val: float) -> np.ndarray:
        Z = np.zeros_like(X_plot, dtype=float)
        for i in range(X_plot.shape[0]):
            for j in range(X_plot.shape[1]):
                coord_ij = (X_plot[i, j], Y_plot[i, j], t_val)
                Z[i, j] = function_evaluation.evaluate_multivar_sem_encoding(
                    psi_sol=psi_sol,
                    deg=deg,
                    deg_out=deg_out,
                    endpoints=endpoints,
                    coords_eval_list=np.array([coord_ij]),
                    scaling_factor=scaling_factor,
                )[0]
        return Z

    t_vals = np.linspace(0.0, T, n_frames)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    Z0 = eval_field(t_vals[0])

    if zlim is None:
        zmin = float(np.min(Z0))
        zmax = float(np.max(Z0))
        if zmin == zmax:
            zmin -= 1.0
            zmax += 1.0
        else:
            pad = 0.05 * (zmax - zmin)
            zmin -= pad
            zmax += pad
    else:
        zmin, zmax = zlim

    norm = Normalize(vmin=zmin, vmax=zmax)
    cmap = cm.viridis

    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(zlim[0], zlim[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y,t)")

    surf = ax.plot_surface(
        X_plot, Y_plot, Z0,
        cmap=cmap,
        norm=norm,
        rstride=1, cstride=1,
        linewidth=0,
        antialiased=True
    )
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.6, label="f(x,y,t)")
    ax.set_title(
        f"Wave equation SEM surface | t={t_vals[0]:.3f}/{T:.3f} | n={n}, elems={num_elements_for_dims}"
    )

    def update(frame: int):
        nonlocal surf
        t_val = t_vals[frame]

        # Remove old surface (robust)
        if hasattr(surf, "remove"):
            surf.remove()
        else:
            ax.cla()
            ax.set_xlim(0, Lx)
            ax.set_ylim(0, Ly)
            ax.set_zlim(zlim[0], zlim[1])
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("f(x,y,t)")

        Z = eval_field(t_val)
        surf = ax.plot_surface(
            X_plot, Y_plot, Z,
            cmap=cmap,
            norm=norm,
            rstride=1, cstride=1,
            linewidth=0,
            antialiased=True
        )
        ax.set_title(
            f"Wave equation SEM surface | t={t_val:.3f}/{T:.3f} | n={n}, elems={num_elements_for_dims}"
        )
        return []

    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)
    plt.show()
    return ani

if __name__ == "__main__":
    #test_ode_variable_coeff(n=8, num_elements=3)
    #test_inhom_ode(n=2, num_elements=46)
    test_wave_equation(n=4, num_elements_for_dims=np.array([1, 1, 1]))
