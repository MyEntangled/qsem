import numpy as np
import pennylane as qml

from src.utils import derivative_matrix

def R_n(wires: qml.wires.Wires):
    """
    R_n: circuit block consisting of Multi-Controlled X gates. n specifies the number of qubits.

    Parameters
    ----------
    wires : pennylane.wires.Wires
        A list of wires or wire to be acted on

    Returns
    -------
    none :
        Applies R_n on inputted wires
    """

    wires_rev = wires[::-1]

    qml.PauliX(wires=wires_rev[0])
    qml.CNOT(wires=wires_rev[:2])
    for i in range(len(wires_rev)-2):
        qml.MultiControlledX(wires=wires_rev[:i+3])

def L_n(wires: qml.wires.Wires):
    """
    L_n: circuit block consisting of Multi-Controlled X gates. n specifies the number of qubits to be acted on.

    Parameters
    ----------
    wires : pennylane.wires.Wires
        A list of wires or wire to be acted on

    Returns
    -------
    none :
        Applies L_n on inputted wires
    """

    wires_rev = wires[::-1]

    for i in reversed(range(len(wires_rev)-2)):
        qml.MultiControlledX(wires=wires_rev[:i+3])
    qml.CNOT(wires=wires_rev[:2])
    qml.PauliX(wires=wires_rev[0])

def U_R(n,alpha,ancilla_0,wires_i,ancilla_1,wires_j):
    """
    U_R: circuit block consisting of 

    Parameters
    ----------
    wires : pennylane.wires.Wires
        A list of wires or wire to be acted on

    Returns
    -------
    none :
        Applies L_n on inputted wires
    """

    for k, wire_j in enumerate(reversed(wires_j)):
        theta_k = 2 * np.arcsin(-2**k * alpha)
        qml.CRY(theta_k, wires=[wire_j,ancilla_0])

    qml.Barrier()

    R_n(wires=wires_j)
    for i in reversed(range(1,n)):
        qml.ctrl(
            R_n,
            control=wires_i[i-1]
            )(wires=[ancilla_1,*wires_j[:i]])

def U_C(n,alpha,ancilla_0,wires_i,ancilla_1,wires_j):
    """
    U_C
    """
    for k, wire_j in enumerate(reversed(wires_j)):
        phi_k = 2 * np.arcsin(-2**k * alpha * (1/np.sqrt(2) - 1))
        qml.ctrl(
            qml.RY,
            control = [*wires_i,wire_j],
            control_values = [0]*n + [1]
            )(phi_k, wires=ancilla_0)

    qml.Barrier()

    L_n(wires=wires_j)
    for i in reversed(range(1,n)):
        qml.SWAP(wires=[wires_j[i-1],wires_j[i]])
    qml.SWAP(wires=[ancilla_1,wires_j[0]])
    for i in range(n):
        qml.SWAP(wires=[wires_i[i],wires_j[i]])

def U_G(n,ancs_0,wires_i,anc_1,wires_j):
    """
    U_G
    """
    GT = derivative_matrix.chebyshev_diff_matrix(deg=2**n-1)
    norm_G_S = max(np.linalg.norm(GT@GT.T,ord=1),np.linalg.norm(GT.T@GT,ord=1))
    alpha = 2/norm_G_S

    omega_c = 2 * np.arccos(np.sqrt(1/3))

    qml.RY(omega_c,ancs_0[0])
    qml.X(ancs_0[1])
    for wire_i in wires_i:
        qml.Hadamard(wire_i)

    qml.Barrier()

    qml.ctrl(U_R,
        control=ancs_0[0],
        control_values=[0]
        )(n,alpha,ancs_0[1],wires_i,anc_1,wires_j)

    qml.Barrier()

    qml.ctrl(U_C,
        control=ancs_0[0],
        control_values=[1]
        )(n,alpha,ancs_0[1],wires_i,anc_1,wires_j)

    qml.Barrier()

    qml.adjoint(qml.RY)(omega_c,ancs_0[0])
    for wire_i in wires_i:
        qml.Hadamard(wire_i)

if __name__ == "__main__":
    np.set_printoptions(linewidth=200)

    n = 3

    ancs_0 = ["anc_0","anc_1"]
    anc_1 = "anc_2"
    wires_i = [f"i{i}" for i in range(n)]
    wires_j = [f"j{i}" for i in range(n)]
    wires = [*ancs_0,*wires_i,anc_1,*wires_j]

    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev)
    def circuit():
        U_G(n,ancs_0,wires_i,anc_1,wires_j)
        return qml.state()

    print(f"\nThe Chebyshev Differentiation Circuit for n={n}:\n")
    print(qml.draw(circuit,max_length=200)(), "\n")

    GT_appr = qml.matrix(circuit)().real[0:2**n,0:2**n]

    GT_true = derivative_matrix.chebyshev_diff_matrix(deg=2**n-1)
    print(GT_true.T@GT_true)
    norm_GT_S = max(np.linalg.norm(GT_true@GT_true.T,ord=1),
                    np.linalg.norm(GT_true.T@GT_true,ord=1))

    sub_norm_fact = np.sum(GT_true@GT_true)/np.sum(GT_appr@GT_true)
    sub_norm_fact_true = norm_GT_S*3*2**(n-1)

    print(f"\nBlock-encoded matrix for n={n}:",
          f"\n{GT_appr}", "\n")
    print(f"\nOriginal matrix for n={n}:",
          f"\n{GT_true}", "\n")
    print(f"\nSub-normalization factor (true, calc) for n={n}:",
          f"\n({sub_norm_fact_true}, {sub_norm_fact})", "\n")
    print(f"\nL1 Norm of Error for n={n}:",
          f"\n{np.linalg.norm(GT_appr*sub_norm_fact-GT_true,ord=1)}", "\n")
