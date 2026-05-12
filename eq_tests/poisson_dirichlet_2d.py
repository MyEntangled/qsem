import numpy as np
import sympy as sp
import scipy.sparse
import time
import matplotlib.pyplot as plt
import os

from src.utils import function_evaluation, interface_continuity, meshing
from src.utils.basic_operators import basis_change, boundary_share
from src.utils.boundary_hamiltonian.general_boundary import sem_multivar_boundary
from src.utils.diffeq_hamiltonian import multivar_equation_hamiltonian

from src.utils.eigensolvers import Solve_Lanczos as lanczos

def test_poisson_equation_2d(n: int, num_elements_for_dims: np.ndarray, sparse: bool=True, continuity_method: str='penalty'):
    """
    Solve the 2D Poisson Dirichlet equation 
    -\nabla u(x0, x1) = 13 pi^2 sin(2 pi x0) sin(3 pi x1) + 17 pi^2 sin(pi x0) sin(4 pi x1) - 9 x0 - 15 x1
    in [0, 1] x [0, 1].
    """
    d = 2 ** n - 1
    d_out = 2 ** (n+3) - 1

    x0, x1 = sp.symbols('x0 x1')
    u = sp.Function('u')(x0, x1)

    d2_x0 = u.diff(x0, 2)
    d2_x1 = u.diff(x1, 2)

    force_term = (13 * sp.pi**2 * sp.sin(2*sp.pi*x0) * sp.sin(3*sp.pi*x1) + 
                  17 * sp.pi**2 * sp.sin(sp.pi*x0) * sp.sin(4*sp.pi*x1) - 
                  9 * x0 - 15 * x1)

    lhs = -d2_x0 - d2_x1 - force_term
    print(f"Equation LHS: {lhs}")

    x_s = (0.0, 0.0)
    data_s = (x_s, 0.5)
    print("Regular data point:", data_s)

    x0_nodes = np.linspace(0, 1.0, num_elements_for_dims[0] + 1)
    x1_nodes = np.linspace(0, 1.0, num_elements_for_dims[1] + 1)
    mesh = meshing.RectMesh([x0_nodes, x1_nodes])

    if continuity_method == 'sharing':
        C = basis_change.optimal_basis_from_tau(d, kind="C1")
        C_joint = C
        for _ in range(1, mesh.n):
            C_joint = np.kron(C_joint, C)
    else:
        C = None
        C_joint = None

    H_diff_sem = multivar_equation_hamiltonian.sem_multivar_equation_hamiltonian(
        d, d_out, diff_eq=lhs, func=u, vars=(x0, x1), mesh=mesh,
        truncated_order=15, regular_data=data_s, regular_data_type='value',
        intg_cond_order=0, local_basis_transform=C,
        sparse=sparse, get_hamiltonian=True
    )
    print("Diff Hamiltonian size:", H_diff_sem.shape)

    # Boundary conditions
    # u(0, x1) = 1/2 + 5/2 x1^2
    ops_cache = {}
    HB_sem = sem_multivar_boundary(type='value', coords=(0.0, None), d=d, d_out=d_out, mesh=mesh, 
                                   func=lambda x1: 0.5 + 2.5 * x1**2,
                                   local_basis_transform=C, sparse=sparse, get_hamiltonian=True, ops_cache=ops_cache)
    # u(1, x1) = 2 + 5/2 x1^2
    HB_sem += sem_multivar_boundary(type='value', coords=(1.0, None), d=d, d_out=d_out, mesh=mesh, 
                                    func=lambda x1: 2.0 + 2.5 * x1**2,
                                    local_basis_transform=C, sparse=sparse, get_hamiltonian=True, ops_cache=ops_cache)
    # u(x0, 0) = 1/2 + 3/2 x0^3
    HB_sem += sem_multivar_boundary(type='value', coords=(None, 0.0), d=d, d_out=d_out, mesh=mesh, 
                                    func=lambda x0: 0.5 + 1.5 * x0**3,
                                    local_basis_transform=C, sparse=sparse, get_hamiltonian=True, ops_cache=ops_cache)
    # u(x0, 1) = 3 + 3/2 x0^2
    HB_sem += sem_multivar_boundary(type='value', coords=(None, 1.0), d=d, d_out=d_out, mesh=mesh, 
                                    func=lambda x0: 3.0 + 1.5 * x0**2,
                                    local_basis_transform=C, sparse=sparse, get_hamiltonian=True, ops_cache=ops_cache)

    H_sem = H_diff_sem + HB_sem

    if continuity_method == 'penalty':
        HC0_sem = interface_continuity.multivar_boundary_continuity_matrix('value', num_elements_for_dims, d, deg_out=d_out, sparse=sparse)
        HC1_sem = interface_continuity.multivar_boundary_continuity_matrix('derivative', num_elements_for_dims, d, deg_out=d_out, sparse=sparse)
        H_sem += (HC0_sem + HC1_sem) * mesh.num_elems
    elif continuity_method == 'sharing':
        H_sem = boundary_share.apply_SHS(H_sem, mesh, d, num_funcs=1, num_vars=2, share_mode='C1', is_periodic=False)
        S = boundary_share.construct_interface_sharing(mesh, d, num_funcs=1, num_vars=2, share_mode='C1', is_periodic=False, sparse=sparse)
        if sparse:
            H_sem += 10 * (scipy.sparse.eye(S.shape[0], format="csr") - S)
        else:
            H_sem += 10 * (np.eye(S.shape[0]) - S)
    else:
        raise ValueError("Invalid continuity_method. Use 'penalty' or 'sharing'.")

    print("Total Hamiltonian size:", H_sem.shape)

    if sparse:
        #eigvals, eigvecs = cg_solver.lobpcg_solve(H_sem, 2, 512, False)
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(H_sem, k=2, sigma=-0.1, which='LM')

        # x0 = np.ones(H_sem.shape[0])
        # max_iter = 1000
        # num_cores = 2
        # num_eigenvals = 2
        # find_max = False
        # result = lanczos.solve_lanczos(H_sem.data, H_sem.indices, H_sem.indptr, x0, max_iter, num_eigenvals, find_max, num_cores)
        # eigvals = [result[i][0] for i in range(num_eigenvals)]
        # eigvecs = np.asarray([result[i][1] for i in range(num_eigenvals)]).T

    else:
        eigvals, eigvecs = np.linalg.eigh(H_sem)
        eigvals, eigvecs = eigvals[:2], eigvecs[:, :2]
        
    psi_sol = eigvecs[:, 0]

    if continuity_method == 'sharing':
        psi_sol = boundary_share.apply_interface_sharing(psi_sol, mesh, d, num_funcs=1, num_vars=2, share_mode='C1', is_periodic=False)
        if np.linalg.norm(psi_sol) < 1 - 1e-5:
            print("Warning: Solution is not in the feasible subspace.")

    print("Ground energy:", eigvals[0])
    print("Spectral gap:", eigvals[1] - eigvals[0])

    f_s = function_evaluation.evaluate_multivar_sem_encoding(
        psi_sol=psi_sol, deg=d, deg_out=d_out, mesh=mesh, coords_eval_list=np.array([data_s[0]]), 
        local_basis_transform_joint=C_joint, scaling_factor=1.0
    )[0]
    print("Evaluated solution at regular data point:", f_s)

    s_eta = data_s[1] / f_s
    print("Scaling^2:", s_eta ** 2)

    # 3D Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x_plot = np.linspace(0, 1.0, 30)
    y_plot = np.linspace(0, 1.0, 30)
    X, Y = np.meshgrid(x_plot, y_plot)
    Z = np.zeros_like(X)
    
    # Evaluate over the grid
    coords_list = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            coords_list.append((X[i, j], Y[i, j]))
            
    eval_vals = function_evaluation.evaluate_multivar_sem_encoding(
        psi_sol=psi_sol, deg=d, deg_out=d_out, mesh=mesh,
        coords_eval_list=np.array(coords_list), local_basis_transform_joint=C_joint, scaling_factor=s_eta
    )
    
    idx = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = eval_vals[idx]
            idx += 1

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    ax.contour(X, Y, Z, zdir='z', cmap='viridis')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)
    
    ax.set_title(f"2D Poisson Solution (n={n}, elems={num_elements_for_dims}, {continuity_method})")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_zlabel("u(x0, x1)")

    # Save the plot in src/figures regardless of working directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(os.path.dirname(current_dir), 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    save_path = os.path.join(figures_dir, f"poisson_2d_{continuity_method}.png")
    plt.savefig(save_path)
    print(f"Plot saved as {save_path}")
    plt.close()

if __name__ == "__main__":
    start = time.time()

    print("=== Running with Penalty Method ===")
    test_poisson_equation_2d(n=4, num_elements_for_dims=np.array([4, 4]), sparse=True, continuity_method='penalty')
    
    print("\n=== Running with Sharing Method ===")
    test_poisson_equation_2d(n=4, num_elements_for_dims=np.array([4, 4]), sparse=True, continuity_method='sharing')

    print("\nElapsed time:", time.time() - start)

