import numpy as np
import pennylane as qml

from src.utils import derivative_matrix
from src.circuits import chebyshev_differentiation_circuit as cdc

np.set_printoptions(linewidth=200)

n = 2
num_qubits = 2*(n+3) + n
x = 0.5

ancs_0 = ["anc_00","anc_01"]
ancs_1 = ["anc_10","anc_11"]
anc_0 = "anc_02"
anc_1 = "anc_12"
wires_i0 = [f"i0{i}" for i in range(n)]
wires_i1 = [f"i1{i}" for i in range(n)]
wires_j = [f"j{i}" for i in range(n)]
wires = [*ancs_0,anc_0,*wires_i0,*ancs_1,anc_1,*wires_i1,*wires_j]

dev = qml.device("lightning.gpu", wires=wires)

@qml.qnode(dev)
def circuit1():
    cdc.U_G(n,ancs_0,wires_i0,anc_0,wires_j)
    cdc.U_G(n,ancs_1,wires_i1,anc_1,wires_j)
    return qml.state()

@qml.qnode(dev)
def circuit2():
    qml.X(wires_j[1])
    cdc.U_G(n,ancs_0,wires_i0,anc_0,wires_j)
    cdc.U_G(n,ancs_1,wires_i1,anc_1,wires_j)
    return qml.state()

@qml.qnode(dev)
def circuit3():
    qml.X(wires_j[0])
    cdc.U_G(n,ancs_0,wires_i0,anc_0,wires_j)
    cdc.U_G(n,ancs_1,wires_i1,anc_1,wires_j)
    return qml.state()

@qml.qnode(dev)
def circuit4():
    qml.X(wires_j[0])
    qml.X(wires_j[1])
    cdc.U_G(n,ancs_0,wires_i0,anc_0,wires_j)
    cdc.U_G(n,ancs_1,wires_i1,anc_1,wires_j)
    return qml.state()

state = circuit1().real
tensor = state.reshape([2]*num_qubits)
index = [0]*(num_qubits-n) + [slice(None)]*n
post = tensor[tuple(index)]
post_state1 = post.flatten()

state = circuit2().real
tensor = state.reshape([2]*num_qubits)
index = [0]*(num_qubits-n) + [slice(None)]*n
post = tensor[tuple(index)]
post_state2 = post.flatten()

state = circuit3().real
tensor = state.reshape([2]*num_qubits)
index = [0]*(num_qubits-n) + [slice(None)]*n
post = tensor[tuple(index)]
post_state3 = post.flatten()

state = circuit4().real
tensor = state.reshape([2]*num_qubits)
index = [0]*(num_qubits-n) + [slice(None)]*n
post = tensor[tuple(index)]
post_state4 = post.flatten()

#  print(qml.draw(circuit,max_length=200)(), "\n")
#
#  GT2_appr = qml.matrix(circuit)().real[0:2**n,0:2**n]
#  print(GT2_appr)

GT_true = derivative_matrix.chebyshev_diff_matrix(deg=2**n-1)
print(GT_true@GT_true)
norm_GT_S = max(np.linalg.norm(GT_true@GT_true.T,ord=1),
                np.linalg.norm(GT_true.T@GT_true,ord=1))

sub_norm_fact_true = norm_GT_S*3*2**(n-1)
print(post_state1)
print(post_state2)
print(post_state3)
print(post_state4)

matrix = np.array([post_state1,post_state2,post_state3,post_state4]).T
print(matrix)
print(post_state4[1]/post_state3[0])

#  print(f"\nQuantum State produced by the circuit for n={n}:",
#        f"\n{tao_appr}", "\n")
#  print(f"\nOriginal Quantum State for n={n}:",
#        f"\n{tao_true/np.sqrt(2)}", "\n")
#  print(f"\nSub-normalization factor (true, calc) for n={n}:",
#        f"\n({sub_norm_fact_true}, {sub_norm_fact})", "\n")
#  print(f"\nL1 Norm of Error for n={n}:",
#        f"\n{np.linalg.norm(tao_appr*sub_norm_fact-tao_true,ord=1)}", "\n")
