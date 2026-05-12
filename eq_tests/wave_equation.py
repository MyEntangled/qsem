from src.utils import function_evaluation, interface_continuity, regularization, meshing
from src.utils.basic_operators import basis_change, boundary_share
from src.utils.boundary_hamiltonian.general_boundary import build_general_boundary, sem_multivar_boundary
from src.utils.diffeq_hamiltonian import multivar_equation_hamiltonian
import numpy as np
import sympy as sp
import scipy
from src.utils.eigensolvers import cg_solver
import time
import matplotlib.pyplot as plt
import os

def test_wave_equation(n: int, num_elements_for_dims: np.ndarray, sparse: bool=True, intg_cond_order: int=0, continuity_method: str='penalty'):
    """
    Solve the wave equation using QSEM.
    
    Attributes:
        continuity_method (str): 'penalty' for Hamiltonian term for C0 and C1 continuity,
                                 'sharing' for local basis transform and interface sharing mechanism.
    """
    d = 2 ** n - 1
    d_out = 2 ** (n+1) - 1

    ## General setting
    Lx, Ly, T = 1.0, 1.0, 1.0
    mx, my = 2, 3
    omega = np.pi * np.sqrt((mx / Lx) ** 2 + (my / Ly) ** 2)

    x, y, t = sp.symbols('x y t')
    f = sp.Function('f')(x, y, t)

    d2x = f.diff(x, 2)
    d2y = f.diff(y, 2)
    d2t = f.diff(t, 2)

    c = 1.0
    lhs = d2t - c**2 * (d2x + d2y)
    print(f"Equation LHS: {lhs}")

    sol = lambda x, y, t: float(np.sin(mx * np.pi * x / Lx) * np.sin(my * np.pi * y / Ly) * np.cos(omega * t))
    func_init = lambda x, y: np.sin(mx * np.pi * x / Lx) * np.sin(my * np.pi * y / Ly)

    coord_s = (Lx/(2*mx), Ly/(2*my), np.pi/omega)
    data_s = (coord_s, sol(*coord_s))
    print("Regular data point:", data_s)

    ## Create endpoints for the 3D mesh
    x_nodes = np.linspace(0, Lx, num_elements_for_dims[0] + 1)
    y_nodes = np.linspace(0, Ly, num_elements_for_dims[1] + 1)
    t_nodes = np.linspace(0, 1.0, num_elements_for_dims[2] + 1)

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
        d, d_out, diff_eq=lhs, func=f, vars=(x, y, t), mesh=mesh,
        truncated_order=8, regular_data=data_s, regular_data_type='value',
        intg_cond_order=intg_cond_order, local_basis_transform=C,
        sparse=sparse, get_hamiltonian=True
    )
    print("Diff Hamiltonian size:", H_diff_sem.shape)

    # Initial value
    HB_sem = sem_multivar_boundary(type='value', coords=(None, None, 0.), d=d, d_out=d_out, mesh=mesh, func=func_init,
                                   local_basis_transform=C, sparse=sparse, get_hamiltonian=True)

    # Initial velocity
    HB_sem += build_general_boundary(type='derivative', coords=(None, None, 0), d=d, d_out=d_out, mesh=mesh,
                                     local_basis_transform=C, sparse=sparse, get_hamiltonian=True)

    # Fixed edge BC
    # HB_sem += build_general_boundary(type='value', coords=(0, None, None), d=d, d_out=d_out, mesh=mesh, local_basis_transform=C, sparse=sparse, get_hamiltonian=True)
    # HB_sem += build_general_boundary(type='value', coords=(Lx, None, None), d=d, d_out=d_out, mesh=mesh, local_basis_transform=C, sparse=sparse, get_hamiltonian=True)
    # HB_sem += build_general_boundary(type='value', coords=(None, 0, None), d=d, d_out=d_out, mesh=mesh, local_basis_transform=C, sparse=sparse, get_hamiltonian=True)
    # HB_sem += build_general_boundary(type='value', coords=(None, Ly, None), d=d, d_out=d_out, mesh=mesh, local_basis_transform=C, sparse=sparse, get_hamiltonian=True)

    H_sem = H_diff_sem + HB_sem

    if continuity_method == 'penalty':
        HC0_sem = interface_continuity.multivar_boundary_continuity_matrix('value', num_elements_for_dims, d, sparse=sparse)
        HC1_sem = interface_continuity.multivar_boundary_continuity_matrix('derivative', num_elements_for_dims, d, sparse=sparse)
        H_sem += (HC0_sem + HC1_sem) * mesh.num_elems
    elif continuity_method == 'sharing':
        H_sem = boundary_share.apply_SHS(H_sem, mesh, d, num_funcs=1, num_vars=3, share_mode='C1', is_periodic=False)
        S = boundary_share.construct_interface_sharing(mesh, d, num_funcs=1, num_vars=3, share_mode='C1', is_periodic=False, sparse=sparse)
        H_sem += 10 * (scipy.sparse.eye(S.shape[0], format="csr") - S)
    else:
        raise ValueError("Invalid continuity_method. Use 'penalty' or 'sharing'.")

    print("Total Hamiltonian size:", H_sem.shape)

    if sparse:
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(H_sem, k=2, sigma=-0.1, which='LM')

    else:
        eigvals, eigvecs = np.linalg.eigh(H_sem)
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
    print("Scaling^2:", s_eta ** 2)

    # Plot the solution at t0
    t0 = 0.

    x_plot = np.linspace(0, Lx, 10)
    y_plot = np.linspace(0, Ly, 10)
    X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
    fQ_plot = np.zeros_like(X_plot)

    for i in range(X_plot.shape[0]):
        for j in range(X_plot.shape[1]):
            coord_ij = (X_plot[i, j], Y_plot[i, j], t0)
            fQ_plot[i, j] = function_evaluation.evaluate_multivar_sem_encoding(
                psi_sol=psi_sol, deg=d, deg_out=d_out, mesh=mesh,
                coords_eval_list=np.array([coord_ij]), local_basis_transform_joint=C_joint, scaling_factor=s_eta
            )[0]
            
    plt.contourf(X_plot, Y_plot, fQ_plot, levels=50, cmap='viridis')
    plt.colorbar(label=r'$f^*_{Q}$')
    plt.title(f"Wave Function (t={t0}): n={n}, elems={num_elements_for_dims}, mode={continuity_method}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()

    # Save the plot in src/figures regardless of working directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(os.path.dirname(current_dir), 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    save_path = os.path.join(figures_dir, f"wave_equation_{continuity_method}.png")
    plt.savefig(save_path)
    print(f"Plot saved as {save_path}")
    plt.close()

if __name__ == "__main__":
    start = time.time()

    print("=== Running with Penalty Method ===")
    test_wave_equation(n=3, num_elements_for_dims=np.array([4, 4, 4]), sparse=True, intg_cond_order=1, continuity_method='penalty')
    
    print("\n=== Running with Sharing Method ===")
    test_wave_equation(n=3, num_elements_for_dims=np.array([4, 4, 4]), sparse=True, intg_cond_order=1, continuity_method='sharing')

    print("\nElapsed time:", time.time() - start)
