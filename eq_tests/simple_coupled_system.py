import numpy as np
import sympy as sp
import scipy
import time
import matplotlib.pyplot as plt
import os

from src.utils.diffeq_hamiltonian.vector_multivar_equation_hamiltonian import sem_vector_multivar_equation_hamiltonian
from src.utils.interface_continuity import vector_multivar_boundary_continuity_matrix
from src.utils.function_evaluation import evaluate_vector_multivar_sem_encoding
from src.utils.boundary_hamiltonian.general_boundary import sem_vector_multivar_boundary
from src.utils.meshing import RectMesh
from src.utils.eigensolvers.lanczos_solver import pylanczos_solve
from src.utils.basic_operators import basis_change, boundary_share

def test_coupled_ode_system(n: int, num_elements: int, sparse: bool=True, intg_cond_order: int=0, continuity_method: str='sharing'):
    """
    Solve a coupled 1D ODE system:
    u' + 2*pi*v = 4*pi*cos(2*pi*x)
    v' - 2*pi*u = -4*pi*sin(2*pi*x)
    
    Exact solutions:
    u(x) = sin(2*pi*x)
    v(x) = cos(2*pi*x)
    
    Attributes:
        continuity_method (str): 'penalty' for Hamiltonian term for C0 and C1 continuity,
                                 'sharing' for local basis transform and interface sharing mechanism.
    """
    d = 2 ** n - 1
    d_out = 2 ** (n + 3) - 1

    # 1. Define Symbols and Unknowns
    x = sp.Symbol('x')
    u = sp.Function('u')(x)
    v = sp.Function('v')(x)

    # 2. Define the System of Equations
    # These match the exact solution: u=sin(2pi x), v=cos(2pi x)
    # u' = 2pi cos(2pi x)
    # v' = -2pi sin(2pi x)
    # eq1: u' + 2pi v = 4pi cos(2pi x)
    # eq2: v' - 2pi u = -4pi sin(2pi x)
    eq1 = sp.diff(u, x) + 2*np.pi*v - 4*np.pi*sp.cos(2*sp.pi*x)
    eq2 = sp.diff(v, x) - 2*np.pi*u + 4*np.pi*sp.sin(2*sp.pi*x)

    print(f"Eq 1 LHS: {eq1}")
    print(f"Eq 2 LHS: {eq2}")

    # 3. Define Exact Solutions and Regularization Data
    sol_u = lambda x_val: float(np.sin(2*np.pi*x_val))
    sol_v = lambda x_val: float(np.cos(2*np.pi*x_val))

    # Point constraint for regularization
    x_s = 0.1
    data_s_u = ((x_s,), sol_u(x_s))
    data_s_v = ((x_s,), sol_v(x_s))
    reg_data_list = [data_s_u, data_s_v]

    print("Regular data points:", reg_data_list)

    # 4. Create mesh
    nodes = np.linspace(-1, 1, num_elements + 1)
    mesh = RectMesh([nodes]) 

    # 5. Continuity setup
    if continuity_method == 'sharing':
        C = basis_change.optimal_basis_from_tau(d, kind="C1")
    else:
        C = None

    # 6. Build Equation Hamiltonian
    H_diff_sem = sem_vector_multivar_equation_hamiltonian(
        d=d, d_out=d_out, diff_eqs=[eq1, eq2], funcs=[u, v], vars=[x],
        mesh=mesh, truncated_order=15, regular_data_list=reg_data_list,
        regular_data_type='value', intg_cond_order=intg_cond_order,
        local_basis_transform=C, sparse=sparse, get_hamiltonian=True
    )
    print("Diff Hamiltonian size:", H_diff_sem.shape)

    # 7. Build Boundary Conditions
    # We enforce u(-1) = sin(-2pi) = 0
    HB_u = sem_vector_multivar_boundary(
        type='value', coords=(-1.0,), d=d, d_out=d_out, mesh=mesh, 
        num_components=2, component_idx=0, local_basis_transform=C, 
        sparse=sparse, get_hamiltonian=True
    )
    
    H_sem = H_diff_sem + HB_u

    # 8. Handle Interface Continuity
    if continuity_method == 'penalty':
        HC0_sem = vector_multivar_boundary_continuity_matrix(type='value', M_list=[num_elements], num_components=2, deg=d, deg_out=d_out, sparse=sparse)
        HC1_sem = vector_multivar_boundary_continuity_matrix(type='derivative', M_list=[num_elements], num_components=2, deg=d, deg_out=d_out, sparse=sparse)
        H_sem += (HC0_sem + HC1_sem) * (num_elements ** 2)
    elif continuity_method == 'sharing':
        H_sem = boundary_share.apply_SHS(H_sem, mesh, d, num_funcs=2, share_mode='C1', is_periodic=False)
        S = boundary_share.construct_interface_sharing(mesh, d, num_funcs=2, share_mode='C1', is_periodic=False, sparse=sparse)
        if sparse:
            H_sem += (scipy.sparse.eye(S.shape[0], format="csr") - S) * 1000
        else:
            H_sem += (np.eye(len(S)) - S) * 1000
    else:
        raise ValueError("Invalid continuity_method. Use 'penalty' or 'sharing'.")

    print("Total Hamiltonian size:", H_sem.shape)

    # 9. Solve Eigenvalue Problem
    if sparse:
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(H_sem, k=2, sigma=-0.1, which='LM')

    else:
        eigvals, eigvecs = np.linalg.eigh(H_sem)
        eigvals, eigvecs = eigvals[:2], eigvecs[:, :2]
    psi_sol = eigvecs[:, 0]
    
    if continuity_method == 'sharing':
        psi_sol = boundary_share.apply_interface_sharing(psi_sol, mesh, d, num_funcs=2, share_mode='C1', is_periodic=False)

    print("Ground energy:", eigvals[0])
    print("Spectral gap:", eigvals[1] - eigvals[0])

    # 10. Evaluate & Find Scaling Factor
    # Evaluate at x_s to compute the scaling factors for each component
    f_s_unscaled = evaluate_vector_multivar_sem_encoding(
        psi_sol=psi_sol, deg=d, deg_out=d_out, num_components=2, mesh=mesh,
        coords_eval_list=np.array([[x_s]]), scaling_factor=None, local_basis_transform=C
    )[0]

    print("Evaluated solution at regular data point (unscaled):", f_s_unscaled)

    s_eta_u = data_s_u[1] / f_s_unscaled[0]
    s_eta_v = data_s_v[1] / f_s_unscaled[1]
    print(f"Calculated Scaling factor for u: {s_eta_u:.6f}")
    print(f"Calculated Scaling factor for v: {s_eta_v:.6f}")

    # 11. Plot Results
    x_plot = np.linspace(-1, 1, 500)
    coords_plot = x_plot.reshape(-1, 1)

    uv_eval = evaluate_vector_multivar_sem_encoding(
        psi_sol=psi_sol, deg=d, deg_out=d_out, num_components=2, mesh=mesh,
        coords_eval_list=coords_plot, scaling_factor=[s_eta_u, s_eta_v], local_basis_transform=C
    )

    u_Q = uv_eval[:, 0]
    v_Q = uv_eval[:, 1]
    u_exact = [sol_u(xv) for xv in x_plot]
    v_exact = [sol_v(xv) for xv in x_plot]

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, u_Q, 'r-', linewidth=2, label=r'SEM $u(x)$')
    plt.plot(x_plot, u_exact, 'k--', alpha=0.7, label=r'Exact $u(x) = \sin(2\pi x)$')
    plt.plot(x_plot, v_Q, 'b-', linewidth=2, label=r'SEM $v(x)$')
    plt.plot(x_plot, v_exact, 'gray', linestyle='--', alpha=0.7, label=r'Exact $v(x) = \cos(2\pi x)$')

    plt.title(f"Coupled 1D Vector ODE: n={n}, elems={num_elements}, mode={continuity_method}")
    plt.xlabel("x")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(os.path.dirname(current_dir), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    save_path = os.path.join(figures_dir, f"simple_coupled_system_{continuity_method}.png")
    plt.savefig(save_path)
    print(f"Plot saved as {save_path}")
    plt.close()

if __name__ == "__main__":
    start = time.time()

    print("\n=== Running with Penalty Method ===")
    test_coupled_ode_system(n=2, num_elements=300, sparse=True, intg_cond_order=0, continuity_method='penalty')

    print("=== Running with Sharing Method ===")
    test_coupled_ode_system(n=2, num_elements=300, sparse=True, intg_cond_order=0, continuity_method='sharing')
    
    print("\nElapsed time:", time.time() - start)
