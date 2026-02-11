from src.utils import boundary_matrix, function_evaluation, equation_hamiltonian, interface_continuity, regularization
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
    sol = lambda x: lpmv(m, l, x)
    #sol = lambda x: 1.5 * np.exp(x) - 0.125*x*(8*x+13) - 1
    data_s = (x_s, sol(x_s))
    print("Regular data point:", data_s)

    ## Create endpoints for the mesh
    nodes = np.linspace(-1, 1, num_elements + 1)
    endpoints = np.column_stack((nodes[:-1], nodes[1:]))

    H_diff_sem = equation_hamiltonian.sem_equation_hamiltonian(d, d_out, lhs, f, x, endpoints=endpoints, regular_data=None, regular_data_type='value')

    x_m = 0 #-0.195976
    HB_sem = boundary_matrix.sem_boundary_hamiltonian(type='derivative', deg=d, deg_out=d_out, endpoints=endpoints, x=x_m)

    HC0_sem = interface_continuity.boundary_continuity_matrice_1D_new('value', num_elements, d)
    HC1_sem = interface_continuity.boundary_continuity_matrice_1D_new('derivative', num_elements, d)

    H_reg = regularization.regularization_matrix(d, p=1)
    H_reg_sem = np.kron(np.eye(num_elements), H_reg)

    H_sem =  H_diff_sem +  HB_sem + (HC0_sem + HC1_sem)*num_elements**2 + H_reg_sem * 0.01
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

def test_inhom_ode(n, num_elements):
    d = 2 ** n - 1
    d_out = 2 ** (n+1) - 1

    x = sp.Symbol('x')
    f = sp.Function('f')(x)

    d1 = f.diff(x)
    d2 = f.diff(x, 2)

    lhs = (x-1) * d2 - x * d1 + f - (x-1)**2
    print(f"Equation LHS: {lhs}")

    x_s = 0.5
    sol = lambda x: float(1.5 * np.exp(x) - 0.125*x*(8*x+13) - 1)
    data_s = (x_s, sol(x_s))

    ## Create endpoints for the mesh
    nodes = np.linspace(-1, 1, num_elements + 1)
    endpoints = np.column_stack((nodes[:-1], nodes[1:]))

    H_diff_sem = equation_hamiltonian.sem_equation_hamiltonian(d, d_out, lhs, f, x, endpoints=endpoints, regular_data=data_s, regular_data_type='value')

    x_m = -0.195976
    HB_sem = boundary_matrix.sem_boundary_hamiltonian(type='derivative', deg=d, deg_out=d_out, endpoints=endpoints, x=x_m)

    HC0_sem = interface_continuity.boundary_continuity_matrice_1D_new('value', num_elements, d)
    HC1_sem = interface_continuity.boundary_continuity_matrice_1D_new('derivative', num_elements, d)

    H_sem = H_diff_sem + HB_sem + (HC0_sem + HC1_sem)
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
    plt.title(f"Solution to ODE: n={n}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()


#test_ode_variable_coeff(n=3, num_elements=3)
test_inhom_ode(n=2, num_elements=10)
