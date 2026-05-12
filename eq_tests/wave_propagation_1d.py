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
import os

def test_wave_propagation_1d(n: int, num_elements_for_dims: np.ndarray, sparse: bool=True, intg_cond_order: int=0, continuity_method: str='penalty'):
    """
    Solve the 1D wave equation with a layered velocity model using QSEM.
    """
    d = 2 ** n - 1
    d_out = 2 ** (n+1) - 1

    ## General setting
    L, T = 1.0, 2.0
    c1, c2 = 1.0, 2.0
    x0, sigma = 0.25, 0.03

    x, t = sp.symbols('x t')
    u = sp.Function('u')(x, t)

    u_xx = u.diff(x, 2)
    u_tt = u.diff(t, 2)

    # Piecewise velocity model
    c_expr = sp.Piecewise((c1, x <= L/2), (c2, True))
    lhs = u_tt - c_expr**2 * u_xx
    print(f"Equation LHS: {lhs}")

    # Initial condition function
    u_init_expr = (x * (L - x)) / (x0 * (L - x0)) * sp.exp(-(x - x0)**2 / (2 * sigma**2))
    func_init = sp.lambdify(x, u_init_expr, 'numpy')

    # Regular data point for pinning (at t=0, x=x0)
    data_s = ((x0, 0.0), 1.0)
    print("Regular data point:", data_s)

    ## Create endpoints for the 2D mesh (x, t)
    x_nodes = np.linspace(0, L, num_elements_for_dims[0] + 1)
    t_nodes = np.linspace(0, T, num_elements_for_dims[1] + 1)

    mesh = meshing.RectMesh([x_nodes, t_nodes])

    # Create Hamiltonian
    print(f"Degree d={d}, d_out={d_out}, mesh_N={mesh.N}")

    if continuity_method == 'sharing':
        C = basis_change.optimal_basis_from_tau(d, kind="C1")
        C_joint = np.kron(C, C)
    else:
        C = None
        C_joint = None

    H_diff_sem = multivar_equation_hamiltonian.sem_multivar_equation_hamiltonian(
        d, d_out, diff_eq=lhs, func=u, vars=(x, t), mesh=mesh,
        truncated_order=8, regular_data=data_s, regular_data_type='value',
        intg_cond_order=intg_cond_order, local_basis_transform=C,
        sparse=sparse, get_hamiltonian=True
    )
    print("Diff Hamiltonian size:", H_diff_sem.shape)

    # Initial value: u(x, 0) = u_init(x)
    HB_sem = sem_multivar_boundary(type='value', coords=(None, 0.), d=d, d_out=d_out, mesh=mesh, func=func_init,
                                   local_basis_transform=C, sparse=sparse, get_hamiltonian=True)

    # Initial velocity: u_t(x, 0) = 0
    HB_sem += build_general_boundary(type='derivative', coords=(None, 0), d=d, d_out=d_out, mesh=mesh,
                                     local_basis_transform=C, sparse=sparse, get_hamiltonian=True)

    # Dirichlet Boundary Conditions: u(0, t) = 0, u(L, t) = 0
    HB_sem += build_general_boundary(type='value', coords=(0, None), d=d, d_out=d_out, mesh=mesh,
                                     local_basis_transform=C, sparse=sparse, get_hamiltonian=True)
    HB_sem += build_general_boundary(type='value', coords=(L, None), d=d, d_out=d_out, mesh=mesh,
                                     local_basis_transform=C, sparse=sparse, get_hamiltonian=True)

    H_sem = H_diff_sem + HB_sem @ HB_sem

    if continuity_method == 'penalty':
        HC0_sem = interface_continuity.multivar_boundary_continuity_matrix('value', num_elements_for_dims, d, sparse=sparse)
        HC1_sem = interface_continuity.multivar_boundary_continuity_matrix('derivative', num_elements_for_dims, d, sparse=sparse)
        H_sem += (HC0_sem + HC1_sem) * mesh.num_elems
    elif continuity_method == 'sharing':
        H_sem = boundary_share.apply_SHS(H_sem, mesh, d, num_funcs=1, num_vars=2, share_mode='C1', is_periodic=False)
        S = boundary_share.construct_interface_sharing(mesh, d, num_funcs=1, num_vars=2, share_mode='C1', is_periodic=False, sparse=sparse)
        H_sem += 10 * (scipy.sparse.eye(S.shape[0], format="csr") - S)

    print("Total Hamiltonian size:", H_sem.shape)

    if sparse:
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(H_sem, k=2, sigma=-0.1, which='LM')
    else:
        eigvals, eigvecs = np.linalg.eigh(H_sem.toarray())
        eigvals, eigvecs = eigvals[:2], eigvecs[:, :2]

    psi_sol = eigvecs[:, 0]

    if continuity_method == 'sharing':
        psi_sol = boundary_share.apply_interface_sharing(psi_sol, mesh, d, num_funcs=1, num_vars=2, share_mode='C1', is_periodic=False)

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
    fig, ax = plt.subplots(figsize=(10, 6))
    x_plot = np.linspace(0, L, 200)
    line, = ax.plot(x_plot, np.zeros_like(x_plot), lw=2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlim(0, L)
    ax.set_xlabel("Distance x")
    ax.set_ylabel("Amplitude u(x, t)")
    ax.set_title(f"1D Wave Propagation (n={n}, continuity={continuity_method})")
    ax.axvline(L/2, color='k', linestyle='--', alpha=0.5, label='Interface')
    ax.legend()
    ax.grid(True, alpha=0.3)

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        line.set_ydata(np.zeros_like(x_plot))
        time_text.set_text('')
        return line, time_text

    def update(frame):
        t_val = frame
        coords_eval = np.stack([x_plot, np.full_like(x_plot, t_val)], axis=1)
        u_vals = function_evaluation.evaluate_multivar_sem_encoding(
            psi_sol=psi_sol, deg=d, deg_out=d_out, mesh=mesh,
            coords_eval_list=coords_eval, local_basis_transform_joint=C_joint, scaling_factor=s_eta
        )
        line.set_ydata(u_vals)
        time_text.set_text(f'Time: {t_val:.3f}s')
        return line, time_text

    # Number of frames
    num_frames = 360
    t_frames = np.linspace(0, T, num_frames)

    ani = FuncAnimation(fig, update, frames=t_frames, init_func=init, blit=True)

    # Save the animation
    current_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(os.path.dirname(current_dir), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    save_path_gif = os.path.join(figures_dir, f"wave_propagation_1d_{continuity_method}.gif")
    print(f"Saving animation to {save_path_gif}...")
    try:
        ani.save(save_path_gif, writer='pillow', fps=15)
        print(f"Animation saved as {save_path_gif}")
    except Exception as e:
        print(f"Could not save GIF: {e}")
        # Fallback to saving static plot if animation fails
        plt.savefig(save_path_gif.replace('.gif', '.png'))
    
    plt.close()

if __name__ == "__main__":
    start = time.time()
    print("=== Running with Penalty Method ===")
    test_wave_propagation_1d(n=3, num_elements_for_dims=np.array([16, 32]), sparse=True, intg_cond_order=1, continuity_method='penalty')

    print("\n=== Running with Sharing Method ===")
    test_wave_propagation_1d(n=3, num_elements_for_dims=np.array([16, 32]), sparse=True, intg_cond_order=1, continuity_method='sharing')
    
    print("\nElapsed time:", time.time() - start)
