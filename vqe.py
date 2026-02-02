import os
os.environ["OMP_NUM_THREADS"] = "48"
os.environ["OPENBLAS_NUM_THREADS"] = "48"
os.environ["MKL_NUM_THREADS"] = "48"

import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import matplotlib.pyplot as plt
from utils import encoding, derivative_matrix
import time


def analytic_solution(x, y, t):
    omega = np.pi / np.sqrt(2)
    return np.sin(np.pi * (x + 1) / 2) * np.sin(np.pi * (y + 1) / 2) * np.cos(omega * t)

def initial_conditions(x, y):
    return np.sin(np.pi * (x + 1) / 2) * np.sin(np.pi * (y + 1) / 2)

def get_2d_cheb_coeffs(func, deg):
    nodes = np.cos(np.pi * (np.arange(deg + 1) + 0.5) / (deg + 1))
    X, Y = np.meshgrid(nodes, nodes, indexing='ij')
    F_vals = func(X, Y)

    coeffs_y = np.polynomial.chebyshev.chebfit(nodes, F_vals.T, deg).T
    coeffs = np.polynomial.chebyshev.chebfit(nodes, coeffs_y, deg)
    weights = np.array([derivative_matrix.get_weight(k, deg) for k in range(deg+1)])

    W_x = weights[:, np.newaxis]
    W_y = weights[np.newaxis, :]
    target_coeffs = coeffs / (W_x * W_y)
    return target_coeffs.flatten()

def compute_solution(psi, deg):
    tau_start = encoding.chebyshev_encoding(deg, -1.0)
    psi_tensor = psi.reshape((deg+1, (deg+1)**2))
    psi_t_start = tau_start @ psi_tensor
    f_coeffs = get_2d_cheb_coeffs(initial_conditions, deg)
    projection = np.dot(psi_t_start, f_coeffs)
    norm_psi_t = np.dot(f_coeffs, f_coeffs)
        
    if abs(projection) > 1e-12:
        scale = norm_psi_t / projection
        psi *= scale
    return psi

def compute_relative_L2(psi):
    t_vals, x_vals, y_vals = (np.linspace(-1,1,10), np.linspace(-1,1,10), np.linspace(-1,1,10))
    T_end = 3.0
    deg = 2**4 - 1
    psi_sol = compute_solution(psi, deg)
    wave_solution = np.zeros((len(t_vals), len(x_vals), len(y_vals)))
    for ti, t in enumerate(t_vals):
        for xi, x in enumerate(x_vals):
            for yi, y in enumerate(y_vals):
                tau_x = encoding.chebyshev_encoding(deg, x)
                tau_y = encoding.chebyshev_encoding(deg, y)
                tau_t = encoding.chebyshev_encoding(deg, t)
                tau_txy = np.kron(tau_t, np.kron(tau_x, tau_y))
                w_val = np.dot(tau_txy, psi_sol)
                wave_solution[ti, xi, yi] = w_val

    t_vals_original = (t_vals + 1) * (T_end / 2)
    true_solution = np.zeros((len(t_vals), len(x_vals), len(y_vals)))
    for ti, t in enumerate(t_vals_original):
        for xi, x in enumerate(x_vals):
            for yi, y in enumerate(y_vals):
                true_solution[ti, xi, yi] = analytic_solution(x, y, t)
    error = wave_solution - true_solution
    return np.linalg.norm(error) / np.linalg.norm(true_solution)

def get_classical_solution(H):
    eigvals, eigvecs = np.linalg.eigh(H)
    return eigvecs[:, 0]

def variational_ansatz(params, n_qubits):
    layers = len(params) // n_qubits
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    for l in range(layers):
        for i in range(n_qubits):
            qml.RY(params[l*n_qubits + i], i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

dev = qml.device("lightning.qubit", wires=12)

@qml.qnode(dev)
def vqe_circuit(params, H):
    variational_ansatz(params, n_qubits=12)
    return qml.expval(qml.Hermitian(H, wires=range(12)))

@qml.qnode(dev)
def compute_psi(params):
    variational_ansatz(params, n_qubits=12)
    return qml.state()

if __name__ == "__main__":
    H = np.load("wave_hamiltonian.npy")
    log = open("VQE_Log.log", "w")
    classical_sol = get_classical_solution(H)
    classical_l2 = compute_relative_L2(classical_sol)
    n_qubits = 12
    max_iter = 12000
    pnp.random.seed(42)
    for layers in [3, 4, 5, 6, 7]:
        opt = qml.AdamOptimizer(stepsize=0.01)
        params = pnp.random.uniform(-2*np.pi, 2*np.pi, size=(layers * n_qubits,), requires_grad=True)
        log.write("Starting new VQE Run at {}\n".format(time.asctime(time.localtime())))
        log.write(f"Starting VQE with {layers} layers.\n")
        for it in range(max_iter):
            params, cost_val = opt.step_and_cost(lambda v: vqe_circuit(v, H), params)
            log.write(f"Iteration {it+1}/{max_iter}: VQE Energy = {cost_val}\n")
            log.flush()
            if cost_val < 1e-3:
                log.write("Converged successfully.\n")
                break
        psi = compute_psi(params)
        vqe_l2 = compute_relative_L2(psi.real)
        log.write(f"Relative L2 Error from Classical Solution: {classical_l2}\n")
        log.write(f"Relative L2 Error of VQE Solution: {vqe_l2}\n")
        log.write("=="*40 + "\n")
    log.close()