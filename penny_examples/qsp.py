import numpy as np
import pennylane as qml

data = np.load('qsp_phases.npz')
phiset_cosh = data['cosh']
phiset_sinh = data['sinh']

t = 8
func_exp = lambda x: np.exp(-t*x)/np.exp(t)

def phase_mat(phi):
    return np.array([[np.exp(phi*1j),0],[0,np.exp(-phi*1j)]])

def rotation_mat(a):
    diag = a
    off_diag = (1-a**2) ** (1/2) * 1j
    W = np.array([[diag,off_diag],[off_diag,diag]])
    return W

def qsp_phase(a,phiset,wire):
    W = rotation_mat(a)
    qml.RZ(-2*phiset[0],wires=wire)
    for phi in phiset[1:]:
        qml.QubitUnitary(W,wires=wire)
        qml.RZ(-2*phi,wires=wire)

def qsp_exp(a,anc,reg):
    qml.Hadamard(wires=anc)
    qml.ctrl(qsp_phase,control=anc,control_values=[0])(a,phiset_cosh,reg)
    qml.ctrl(qsp_phase,control=anc,control_values=[1])(a,phiset_sinh,reg)
    qml.Hadamard(wires=anc)
    qml.X(wires=anc)

def im_qsp_exp(a,ancs,reg):
    anc1, anc2 = ancs
    qml.Hadamard(wires=anc1)
    qml.ctrl(qsp_exp,control=anc1,control_values=[0])(a,anc2,reg)
    qml.ctrl(qml.adjoint(qsp_exp),control=anc1,control_values=[1])(a,anc2,reg)
    qml.Hadamard(wires=anc1)
    qml.X(wires=anc1)
    qml.RZ(np.pi,wires=anc1)

ancs = ["a1","a2"]
reg = ["q1"]
dev = qml.device("lightning.gpu", wires=[*ancs,*reg])
@qml.qnode(dev)
def circuit(a):
    im_qsp_exp(a,ancs,reg)
    return qml.state()

xs = np.linspace(-1,1,100)
ys_true = np.zeros(100)
ys_appr = np.zeros(100)

for i,x in enumerate(xs):
    ys_appr[i] = qml.matrix(circuit)(x)[0,0].real
    ys_true[i] = func_exp(x)/2

L1 = np.sum(np.abs(ys_appr-ys_true))
L2 = np.sum((ys_appr-ys_true)**2)

print(L1)
print(L2)

