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

def test_variable_coeff_ode(n: int, num_elements: int, sparse: bool=True, continuity_method: str='sharing'):
    """
    Solve the variable coefficient ODE:
    (x-1) u'' - x u' + u - (x-1)**2 = 0
    in [-1, 1].
    
    Exact solution: u(x) = 1.5 * exp(x) - x**2 - 1.625*x - 1
    Boundary condition: u'(-0.195976) = 0
    Regular data: u(0) = 0.5
    """
    d = 2 ** n - 1
    d_out = 2 ** (n + 1) - 1

    x = sp.Symbol('x')
    f = sp.Function('f')(x)

    d1 = f.diff(x)
    d2 = f.diff(x, 2)

    # (x-1) u'' - x u' + u - (x-1)**2 = 0
    lhs = (x-1) * d2 - x * d1 + f - (x-1)**2
    print(f"Equation LHS: {lhs}")

    # Exact solution for verification and scaling
    sol = lambda x_val: float(1.5 * np.exp(x_val) - x_val**2 - 1.625*x_val - 1)
    
    x_s = 0.0
    data_s = (x_s, sol(x_s))
    print("Regular data point:", data_s)

    # Mesh setup
    nodes = np.linspace(-1, 1, num_elements + 1)
    mesh = meshing.RectMesh(nodes)

    if continuity_method == 'sharing':
        C = basis_change.optimal_basis_from_tau(d, kind="C1")
    else:
        C = None

    # Build Equation Hamiltonian
    H_diff_sem = equation_hamiltonian.sem_equation_hamiltonian(
        d, d_out, diff_eq=lhs, func=f, var=x, mesh=mesh,
        regular_data=data_s, regular_data_type='value',
        local_basis_transform=C, sparse=sparse, get_hamiltonian=True
    )
    print("Diff Hamiltonian size:", H_diff_sem.shape)

    # Internal point constraint: u'(x_m) = 0 where x_m = -0.195976
    x_m = -0.195976
    HB_sem = build_general_boundary(type='derivative', coords=x_m, d=d, d_out=d_out, mesh=mesh,
                                    local_basis_transform=C, sparse=sparse, get_hamiltonian=True)
    
    # In basis_recomb.py, this was cubed. We'll use a strong penalty instead for standard templates.
    # But to follow the original logic exactly, we could do HB_sem @ HB_sem @ HB_sem if it's dense.
    # For now, we'll just add it.
    H_sem = H_diff_sem + HB_sem

    # Handle Interface Continuity
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

    # Solve Eigenvalue Problem
    if sparse:
        # Use shift-invert to find the smallest eigenvalue
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(H_sem, k=2, sigma=-1e-3, which='LM')
    else:
        eigvals, eigvecs = np.linalg.eigh(H_sem)
        eigvals, eigvecs = eigvals[:2], eigvecs[:, :2]
        
    psi_sol = eigvecs[:, 0]

    if continuity_method == 'sharing':
        psi_sol = boundary_share.apply_interface_sharing(psi_sol, mesh, d, num_funcs=1, num_vars=1, share_mode='C1', is_periodic=False)

    print("Ground energy:", eigvals[0])
    print("Spectral gap:", eigvals[1] - eigvals[0])

    # Evaluate & Scale
    f_s = function_evaluation.evaluate_sem_encoding(
        psi_sol=psi_sol, deg=d, deg_out=d_out, mesh=mesh, x_eval_list=[data_s[0]], 
        scaling_factor=1.0, local_basis_transform=C
    )[0]
    print("Evaluated solution at regular data point:", f_s)

    s_eta = data_s[1] / f_s
    print("Scaling factor:", s_eta)

    # Plot Results
    x_plot = np.linspace(-1, 1, 500)
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
    plt.title(f"Variable Coeff ODE: n={n}, elems={num_elements}, mode={continuity_method}")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid()
    
    # Save the plot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(os.path.dirname(current_dir), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    save_path = os.path.join(figures_dir, f"variable_coeff_ode_{continuity_method}.png")
    plt.savefig(save_path)
    print(f"Plot saved as {save_path}")
    plt.close()

if __name__ == "__main__":
    start = time.time()

    print("=== Running with Penalty Method ===")
    test_variable_coeff_ode(n=3, num_elements=10, sparse=True, continuity_method='penalty')
    
    print("\n=== Running with Sharing Method ===")
    test_variable_coeff_ode(n=3, num_elements=10, sparse=True, continuity_method='sharing')

    print("\nElapsed time:", time.time() - start)
