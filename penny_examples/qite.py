import numpy as np
import pennylane as qml
from pathlib import Path
# import matplotlib.pyplot as plt

data = np.load('qsp_phases.npz')
phiset_cosh_unmod = data['cosh']
phiset_sinh_unmod = data['sinh']

H = np.load('hamiltonian.npy')
print(H.shape)
# H = np.array([[0.8,0.1,0.1,0.0],
#               [0.1,0.6,0.0,0.1],
#               [0.1,0.0,0.4,0.1],
#               [0.0,0.1,0.1,0.2]])

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

ancs = ["a1","a2","a3"]
regs = [f"q{i}" for i in range(1,6)]
dev = qml.device("lightning.gpu", wires=[*ancs,*regs])
@qml.qnode(dev)
def circuit():
    for reg in regs:
        qml.Hadamard(wires=reg)

    im_qsvt_exp(H,ancs,regs)
    return qml.state()

print(qml.matrix(circuit)()[:32,:32].real)

full_state = circuit()
unnorm_ground_state = full_state[:32]

norm = np.linalg.norm(unnorm_ground_state)
groud_state = unnorm_ground_state / norm

print(groud_state.real)

# Save the ground state
current_path = Path(__file__).resolve()
target_dir = current_path.parent
file_path = target_dir / "ground_state.npy"
np.save(file_path, groud_state.real)
print(f"Ground state is saved to: {file_path}")

