import numpy as np
import sympy as sp
import scipy
import time
import matplotlib.pyplot as plt
import os

from src.utils import function_evaluation, interface_continuity, meshing
from src.utils.basic_operators import basis_change, boundary_share
from src.utils.boundary_hamiltonian.general_boundary import build_general_boundary
from src.utils.diffeq_hamiltonian import equation_hamiltonian
from src.utils.eigensolvers import cg_solver

from src.utils.eigensolvers import Solve_Lanczos as lanczos


def test_poisson_equation(n: int, num_elements: int, sparse: bool=True, continuity_method: str='penalty'):
    """
    Solve the Poisson Dirichlet equation -u'' = 100 cos(2*pi*x) cos(5*pi*x)
    in [0, 1] with u(0) = u(1) = 1.
    """
    d = 2 ** n - 1
    d_out = 2 ** (n+2) - 1

    x = sp.Symbol('x')
    f = sp.Function('f')(x)

    d2 = f.diff(x, 2)

    lhs = -d2 - 100 * sp.cos(2*sp.pi*x) * sp.cos(5*sp.pi*x)
    print(f"Equation LHS: {lhs}")

    sol = lambda x: 100 * (-58 + 116*x + 49*np.cos(3*np.pi*x) + 9*np.cos(7*np.pi*x)) / (882*np.pi**2) 

    x_s = 0.2
    data_s = (x_s, sol(x_s))
    print("Regular data point:", data_s)

    nodes = np.linspace(0, 1.0, num_elements + 1)
    mesh = meshing.RectMesh(nodes)

    if continuity_method == 'sharing':
        C = basis_change.optimal_basis_from_tau(d, kind="C1")
    else:
        C = None

    H_diff_sem = equation_hamiltonian.sem_equation_hamiltonian(
        d, d_out, diff_eq=lhs, func=f, var=x, mesh=mesh,
        truncated_order=15, regular_data=data_s, regular_data_type='value',
        local_basis_transform=C, sparse=sparse, get_hamiltonian=True
    )
    print("Diff Hamiltonian size:", H_diff_sem.shape)

    # Boundary conditions u(0) = 0, u(1) = 0
    HB_sem = build_general_boundary(type='value', coords=0.0, d=d, d_out=d_out, mesh=mesh, 
                                    local_basis_transform=C, sparse=sparse, get_hamiltonian=True)
    HB_sem += build_general_boundary(type='value', coords=1.0, d=d, d_out=d_out, mesh=mesh, 
                                     local_basis_transform=C, sparse=sparse, get_hamiltonian=True)

    H_sem = H_diff_sem + HB_sem

    if continuity_method == 'penalty':
        HC0_sem = interface_continuity.boundary_continuity_matrice('value', mesh.num_elems, d, deg_out=d_out, sparse=sparse)
        HC1_sem = interface_continuity.boundary_continuity_matrice('derivative', mesh.num_elems, d, deg_out=d_out, sparse=sparse)
        H_sem += (HC0_sem + HC1_sem) * mesh.num_elems
    elif continuity_method == 'sharing':
        H_sem = boundary_share.apply_SHS(H_sem, mesh, d, num_funcs=1, num_vars=1, share_mode='C1', is_periodic=False)
        S = boundary_share.construct_interface_sharing(mesh, d, num_funcs=1, num_vars=1, share_mode='C1', is_periodic=False, sparse=sparse)
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
        psi_sol = boundary_share.apply_interface_sharing(psi_sol, mesh, d, num_funcs=1, num_vars=1, share_mode='C1', is_periodic=False)

    if np.linalg.norm(psi_sol) < 1 - 1e-6:
        raise ValueError("The solution is not normalized: ", np.linalg.norm(psi_sol))

    print("Ground energy:", eigvals[0])
    print("Spectral gap:", eigvals[1] - eigvals[0])

    f_s = function_evaluation.evaluate_sem_encoding(
        psi_sol=psi_sol, deg=d, deg_out=d_out, mesh=mesh, x_eval_list=[data_s[0]], 
        scaling_factor=1.0, local_basis_transform=C
    )[0]
    print("Evaluated solution at regular data point:", f_s)

    s_eta = data_s[1] / f_s
    print("Scaling^2:", s_eta ** 2)

    x_plot = np.linspace(0, 1.0, 1000)
    fQ_plot = []
    f_plot = []

    for xj in x_plot:
        yj = function_evaluation.evaluate_sem_encoding(
            psi_sol=psi_sol, deg=d, deg_out=d_out, mesh=mesh,
            x_eval_list=np.array([xj]), scaling_factor=s_eta, local_basis_transform=C
        )[0]
        fQ_plot.append(yj)
        f_plot.append(sol(xj))
        
    plt.figure()
    plt.plot(x_plot, fQ_plot, c='red', label=r'$f^*_{Q}(x)$')
    plt.plot(x_plot, f_plot, '--', label=r'$f^*(x)$')
    plt.title(f"Poisson Function: n={n}, elems={num_elements}, mode={continuity_method}")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid()
    
    # Save the plot in src/figures regardless of working directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(os.path.dirname(current_dir), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    save_path = os.path.join(figures_dir, f"poisson_1d_{continuity_method}.png")
    plt.savefig(save_path)
    print(f"Plot saved as {save_path}")

    plt.close()

if __name__ == "__main__":
    start = time.time()

    print("=== Running with Penalty Method ===")
    test_poisson_equation(n=3, num_elements=6, sparse=True, continuity_method='penalty')
    
    print("\n=== Running with Sharing Method ===")
    test_poisson_equation(n=3, num_elements=6, sparse=True, continuity_method='sharing')

    print("\nElapsed time:", time.time() - start)