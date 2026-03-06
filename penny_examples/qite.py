import numpy as np
import scipy as sp
import pennylane as qml
from pathlib import Path
# import matplotlib.pyplot as plt

t = 8
func = lambda x: np.exp(-t*x)/np.exp(t)

data = np.load('qsp_phases.npz')
phiset_cosh_unmod = data['cosh']
phiset_sinh_unmod = data['sinh']

# current_path = Path(__file__).resolve()
# source_dir = current_path.parent.parent.parent / "hamiltonians"
# source_path = source_dir / "hamiltonian.npy"
# print(f"Hamiltonian is loaded from: {source_path}")
# H = np.load(source_path)
# eigvals, _ = np.linalg.eigh(H)
# print(f"Largest eigenvalue: {eigvals[-1]}")
# print(f"2nd smallest eigenvalue: {eigvals[1]}")
H = np.array([[0.8,0.1,0.1,0.0],
              [0.1,0.6,0.0,0.1],
              [0.1,0.0,0.4,0.1],
              [0.0,0.1,0.1,0.2]])
n_qubits = int(np.ceil(np.log2(len(H))))
eigvals, eigvecs = np.linalg.eigh(H)
exp_eig = func(eigvals)
print(f"\nEigenvalues: {eigvals}","\n")
print(f"\nexp(-{t}*eigval)/exp({t}): {exp_eig}","\n")

def transform_phases(phiset_in):
    phiset = phiset_in.copy()
    d = len(phiset)-1
    phiset[0] += phiset[d] + (d-1)*np.pi/2
    phiset[1:] -= np.pi/2
    return phiset[:-1]

phiset_cosh = transform_phases(phiset_cosh_unmod)
phiset_sinh = transform_phases(phiset_sinh_unmod)

def qsvt_phase(H,phiset,anc,regs):
    n = len(phiset)
    wires = [anc,*regs]
    if n%2:
        qml.RZ(-2*phiset[0],wires=anc)
        qml.BlockEncode(H,wires=wires)
        for i in range(1,n,2):
            qml.RZ(-2*phiset[i],wires=anc)
            qml.adjoint(qml.BlockEncode)(H,wires=wires)
            qml.RZ(-2*phiset[i+1],wires=anc)
            qml.BlockEncode(H,wires=wires)
    else:
        for i in range(0,n,2):
            qml.RZ(-2*phiset[i],wires=anc)
            qml.adjoint(qml.BlockEncode)(H,wires=wires)
            qml.RZ(-2*phiset[i+1],wires=anc)
            qml.BlockEncode(H,wires=wires)

def qsvt_exp(H,ancs,regs):
    anc1, anc2 = ancs
    qml.Hadamard(wires=anc1)
    qml.ctrl(qsvt_phase,
             control=anc1,
             control_values=[0])(H,phiset_cosh,anc2,regs)
    qml.ctrl(qsvt_phase,
             control=anc1,
             control_values=[1])(H,phiset_sinh,anc2,regs)
    qml.Hadamard(wires=anc1)
    qml.X(wires=anc1)

def im_qsvt_exp(H,ancs,regs):
    anc1, anc2, anc3 = ancs
    qml.Hadamard(wires=anc1)
    qml.ctrl(qsvt_exp,
             control=anc1,
             control_values=[0])(H,[anc2,anc3],regs)
    qml.ctrl(qml.adjoint(qsvt_exp),
             control=anc1,
             control_values=[1])(H,[anc2,anc3],regs)
    qml.Hadamard(wires=anc1)
    qml.X(wires=anc1)
    qml.RZ(np.pi,wires=anc1)

ancs = ["a0","a1","a2"]
regs = [f"q{i}" for i in range(n_qubits)]
dev = qml.device("lightning.gpu", wires=[*ancs,*regs])
@qml.qnode(dev)
def circuit():
    im_qsvt_exp(H,ancs,regs)
    return qml.state()

print(f"\nCalculated exp(-{t}H)/exp({t}) using QSVT:\n")
print(qml.matrix(circuit)()[:len(H),:len(H)].real,"\n")
print(f"\nCalculated exp(-{t}H)/exp({t}) using scipy",
      "with post normalization factor of 2:", "\n")
print(sp.linalg.expm(-8*H) / np.exp(8)/2,"\n")

# full_state = circuit()
# unnorm_ground_state = full_state[:32]
# 
# norm = np.linalg.norm(unnorm_ground_state)
# groud_state = unnorm_ground_state / norm
# 
# # print(groud_state.real)
# 
# # Save the ground state
# target_dir = current_path.parent
# target_path = target_dir / "ground_state.npy"
# np.save(target_path, groud_state.real)
# print(f"Ground state is saved to: {target_path}")
