from src.utils import function_evaluation, interface_continuity, meshing
from src.utils.basic_operators import basis_change, boundary_share
from src.utils.boundary_hamiltonian.general_boundary import build_general_boundary, sem_multivar_boundary
from src.utils.diffeq_hamiltonian import multivar_equation_hamiltonian
import numpy as np
import sympy as sp
import scipy
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os

def test_wave_propagation_2d(n: int, num_elements_for_dims: np.ndarray, sparse: bool=True, intg_cond_order: int=0, continuity_method: str='penalty'):
    """
    Solve the 2D wave equation with a point source using QSEM.
    Equation: u_tt = c^2 (u_xx + u_yy) + s(t) * delta(x-xs) * delta(y-ys)
    """
    d = 2 ** n - 1
    d_out = 2 ** (n) - 1

    ## General setting
    Lx, Ly, T = 1.0, 1.0, 2.0
    c_val = 0.5
    xs, ys = 0.5, 0.5
    
    # Gaussian approximation for delta function
    sigma = 0.05

    x, y, t = sp.symbols('x y t')
    u = sp.Function('u')(x, y, t)

    u_xx = u.diff(x, 2)
    u_yy = u.diff(y, 2)
    u_tt = u.diff(t, 2)

    # Wave equation LHS (no source term)
    lhs = u_tt - c_val**2 * (u_xx + u_yy) 
    print(f"Equation LHS: {lhs}")

    # Gaussian initial condition: u(x, y, 0) = u_init(x, y)
    # Similar to 1D: (x*(Lx-x)*y*(Ly-y)) / (xs*(Lx-xs)*ys*(Ly-ys)) * exp(-((x-xs)**2 + (y-ys)**2) / (2*sigma**2))
    u_init_expr = (x * (Lx - x) * y * (Ly - y)) / (xs * (Lx - xs) * ys * (Ly - ys)) * sp.exp(-((x - xs)**2 + (y - ys)**2) / (2 * sigma**2))
    func_init = sp.lambdify((x, y), u_init_expr, 'numpy')

    # Regular data point for pinning (at t=0, x=xs, y=ys)
    data_s = ((xs, ys, 0.0), 1.0) 
    print("Regular data point:", data_s)

    ## Create endpoints for the 3D mesh (x, y, t)
    x_nodes = np.linspace(0, Lx, num_elements_for_dims[0] + 1)
    y_nodes = np.linspace(0, Ly, num_elements_for_dims[1] + 1)
    t_nodes = np.linspace(0, T, num_elements_for_dims[2] + 1)

    mesh = meshing.RectMesh([x_nodes, y_nodes, t_nodes])

    # Create Hamiltonian
    print(f"Degree d={d}, d_out={d_out}, mesh_N={mesh.N}")

    if continuity_method == 'sharing':
        C = basis_change.optimal_basis_from_tau(d, kind="C1")
        C_joint = C
        for _ in range(1, mesh.n):
            C_joint = np.kron(C_joint, C)
    else:
        C = None
        C_joint = None

    H_diff_sem = multivar_equation_hamiltonian.sem_multivar_equation_hamiltonian(
        d, d_out, diff_eq=lhs, func=u, vars=(x, y, t), mesh=mesh,
        truncated_order=8, regular_data=data_s, regular_data_type='value',
        intg_cond_order=intg_cond_order, local_basis_transform=C,
        sparse=sparse, get_hamiltonian=True
    )
    print("Diff Hamiltonian size:", H_diff_sem.shape)

    # Initial value: u(x, y, 0) = u_init(x, y)
    ops_cache = {}
    HB_sem = sem_multivar_boundary(type='value', coords=(None, None, 0.), d=d, d_out=d_out, mesh=mesh, func=func_init,
                                   local_basis_transform=C, sparse=sparse, get_hamiltonian=True, ops_cache=ops_cache)

    # Initial velocity: u_t(x, y, 0) = 0
    HB_sem += build_general_boundary(type='derivative', coords=(None, None, 0), d=d, d_out=d_out, mesh=mesh,
                                     local_basis_transform=C, sparse=sparse, get_hamiltonian=True, ops_cache=ops_cache)

    # Dirichlet Boundary Conditions: u = 0 on all spatial boundaries
    # x = 0
    HB_sem += build_general_boundary(type='value', coords=(0., None, None), d=d, d_out=d_out, mesh=mesh,
                                     local_basis_transform=C, sparse=sparse, get_hamiltonian=True, ops_cache=ops_cache)
    # x = Lx
    HB_sem += build_general_boundary(type='value', coords=(Lx, None, None), d=d, d_out=d_out, mesh=mesh,
                                     local_basis_transform=C, sparse=sparse, get_hamiltonian=True, ops_cache=ops_cache)
    # y = 0
    HB_sem += build_general_boundary(type='value', coords=(None, 0., None), d=d, d_out=d_out, mesh=mesh,
                                     local_basis_transform=C, sparse=sparse, get_hamiltonian=True, ops_cache=ops_cache)
    # y = Ly
    HB_sem += build_general_boundary(type='value', coords=(None, Ly, None), d=d, d_out=d_out, mesh=mesh,
                                     local_basis_transform=C, sparse=sparse, get_hamiltonian=True, ops_cache=ops_cache)

    H_sem = H_diff_sem + HB_sem

    if continuity_method == 'penalty':
        HC0_sem = interface_continuity.multivar_boundary_continuity_matrix('value', num_elements_for_dims, d, sparse=sparse)
        HC1_sem = interface_continuity.multivar_boundary_continuity_matrix('derivative', num_elements_for_dims, d, sparse=sparse)
        H_sem += (HC0_sem + HC1_sem) * mesh.num_elems
    elif continuity_method == 'sharing':
        H_sem = boundary_share.apply_SHS(H_sem, mesh, d, num_funcs=1, num_vars=3, share_mode='C1', is_periodic=False)
        S = boundary_share.construct_interface_sharing(mesh, d, num_funcs=1, num_vars=3, share_mode='C1', is_periodic=False, sparse=sparse)
        H_sem += 10 * (scipy.sparse.eye(S.shape[0], format="csr") - S)

    print("Total Hamiltonian size:", H_sem.shape)

    if sparse:
        # Using a small negative sigma to find the ground state
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(H_sem, k=2, sigma=-1e-5, which='LM')
    else:
        eigvals, eigvecs = np.linalg.eigh(H_sem.toarray())
        eigvals, eigvecs = eigvals[:2], eigvecs[:, :2]

    psi_sol = eigvecs[:, 0]

    if continuity_method == 'sharing':
        psi_sol = boundary_share.apply_interface_sharing(psi_sol, mesh, d, num_funcs=1, num_vars=3, share_mode='C1', is_periodic=False)

    print("Ground energy:", eigvals[0])
    print("Spectral gap:", eigvals[1] - eigvals[0])

    f_s = function_evaluation.evaluate_multivar_sem_encoding(
        psi_sol=psi_sol, deg=d, deg_out=d_out, mesh=mesh, coords_eval_list=np.array([data_s[0]]), 
        local_basis_transform_joint=C_joint, scaling_factor=1.0
    )[0]
    print("Evaluated solution at regular data point:", f_s)

    s_eta = data_s[1] / f_s
    print("Scaling factor:", s_eta)

    # Animation of the wave over time
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x_plot = np.linspace(0, Lx, 50)
    y_plot = np.linspace(0, Ly, 50)
    X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
    
    # Initial surface
    surf = [ax.plot_surface(X_plot, Y_plot, np.zeros_like(X_plot), cmap='seismic', vmin=-0.5, vmax=0.5)]
    fig.colorbar(surf[0], ax=ax, shrink=0.5, aspect=10, label="Amplitude u(x, y, t)")
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u(x, y, t)")
    ax.set_zlim(-0.6, 0.6)
    
    title = ax.set_title(f"2D Wave Propagation (n={n}, continuity={continuity_method})")
    time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, color='black', fontweight='bold')

    def init():
        surf[0].remove()
        surf[0] = ax.plot_surface(X_plot, Y_plot, np.zeros_like(X_plot), cmap='seismic', vmin=-0.5, vmax=0.5)
        time_text.set_text('')
        return surf[0], time_text

    def update(frame):
        t_val = frame
        coords_eval = np.stack([X_plot.ravel(), Y_plot.ravel(), np.full(X_plot.size, t_val)], axis=1)
        u_vals = function_evaluation.evaluate_multivar_sem_encoding(
            psi_sol=psi_sol, deg=d, deg_out=d_out, mesh=mesh,
            coords_eval_list=coords_eval, local_basis_transform_joint=C_joint, scaling_factor=s_eta
        )
        u_grid = u_vals.reshape(X_plot.shape)
        
        surf[0].remove()
        surf[0] = ax.plot_surface(X_plot, Y_plot, u_grid, cmap='seismic', vmin=-0.5, vmax=0.5,
                                  linewidth=0, antialiased=True)
        
        time_text.set_text(f'Time: {t_val:.3f}s')
        return surf[0], time_text

    # Number of frames
    num_frames = 240
    t_frames = np.linspace(0, T, num_frames)

    ani = FuncAnimation(fig, update, frames=t_frames, init_func=init, blit=False)

    # Save the animation
    current_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(os.path.dirname(current_dir), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    save_path_gif = os.path.join(figures_dir, f"wave_propagation_2d_{continuity_method}.gif")
    print(f"Saving animation to {save_path_gif}...")
    try:
        ani.save(save_path_gif, writer='pillow', fps=10)
        print(f"Animation saved as {save_path_gif}")
    except Exception as e:
        print(f"Could not save GIF: {e}")
        plt.savefig(save_path_gif.replace('.gif', '.png'))
    
    plt.close()

if __name__ == "__main__":
    start = time.time()
    # Using small elements for testing
    print("=== Running with Penalty Method ===")
    test_wave_propagation_2d(n=3, num_elements_for_dims=np.array([4, 4, 4]), sparse=True, intg_cond_order=1, continuity_method='penalty')

    # print("=== Running with Sharing Method ===")
    # test_wave_propagation_2d(n=3, num_elements_for_dims=np.array([4, 4, 4]), sparse=True, intg_cond_order=1, continuity_method='sharing')

    print("\nElapsed time:", time.time() - start)
