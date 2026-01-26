import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import real_amplitudes
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorEstimator

from src.utils import derivative_matrix, boundary_matrix

def construct_ODE(deg, a,b,c,x_z):
    G_T = derivative_matrix.chebyshev_diff_matrix(deg=deg)

    A = a * G_T @ G_T + b * G_T + c * np.eye(deg+1)
    B = boundary_matrix.zero_value_boundary_matrix(deg=deg, x_z=x_z)
    #B_hat = boundary_matrix.zero_derivative_boundary_matrix(deg=3, x_m=0.0)

    T_A = A.T @ A
    T_B = B.T @ B

    H = T_A + T_B
    return H

n = 3
deg = 2**n - 1
a = 1.0
b = 4.0
c = 4.0
x_z = -1
data_s = (0,0.5)

# --- Step 1: Define your Hamiltonian Matrix ---
# Note: The matrix must be Hermitian and its size must be 2^n (2, 4, 8, 16...)
# Here we define a simple 4x4 matrix (representing a 2-qubit system)
# H_matrix = np.array([
#     [1, 0, 0, 0],
#     [0, 0, -1, 0],
#     [0, -1, 0, 0],
#     [0, 0, 0, 1]
# ])

H_matrix = construct_ODE(deg,a,b,c,x_z)

# --- Step 2: Convert Matrix to Quantum Operator ---
# This decomposes the matrix into Pauli strings (I, X, Y, Z)
qubit_op = SparsePauliOp.from_operator(H_matrix)
print(f"Operator Terms:\n{qubit_op}\n")

# --- Step 3: Configure the VQE Algorithm ---
# 1. Choose an Ansatz (the parameterized quantum circuit)
ansatz = real_amplitudes(num_qubits=n, reps=2, entanglement='linear')

# 2. Choose an Optimizer (Classical algorithm to tune parameters)
optimizer = COBYLA(maxiter=100)

# 3. Initialize the Estimator (Simulates the quantum expectation calculation)
estimator = StatevectorEstimator()

# 4. Setup VQE
vqe = VQE(estimator, ansatz, optimizer)

# --- Step 4: Run the Algorithm ---
result = vqe.compute_minimum_eigenvalue(qubit_op)

print(f"Ground State Energy (VQE): {result.eigenvalue.real:.5f}")

# --- Verification (Optional) ---
# Compare with the exact classical solution to check accuracy
eigenvalues = np.linalg.eigvalsh(H_matrix)
print(f"Ground State Energy (Exact): {min(eigenvalues):.5f}")
