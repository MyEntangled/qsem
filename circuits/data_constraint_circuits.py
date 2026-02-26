import numpy as np
import pennylane as qml

from src.circuits import orthonormal_chebyshev_feature_map_circuit as ocfmc

def S_n(n: int, anc: qml.wires.Wires, wires: qml.wires.Wires):
    """
    S_n
    """
    qml.GlobalPhase(np.pi)
    qml.X(anc)
    qml.ctrl(qml.Z,
        control=wires,
        control_values=[0]*n
        )(wires=anc)
    qml.X(anc)


def U_B(n: int, x: float, ancs: qml.wires.Wires, wires: qml.wires.Wires):
    """
    U_B : The circuit implements a block-encoding of the matrix B_n with a sub-normalization factor of sqrt(2^(n+1)).

    Parameters
    ----------
    wires : pennylane.wires.Wires
        A list of wires or wire to be acted on

    Returns
    -------
    none :
        Applies R_n on inputted wires
    """
    qml.adjoint(ocfmc.U_tao)(n,x,ancs[1],wires)
    qml.Barrier()
    qml.Hadamard(ancs[0])
    qml.ctrl(S_n,control=ancs[0])(n,ancs[1],wires)
    qml.Hadamard(ancs[0])

def U_D(n: int, x: float, ancs: qml.wires.Wires, wires: qml.wires.Wires):
    """
    U_D : The circuit implements a block-encoding of the matrix D_n with a sub-normalization factor of sqrt(2^(n+1))/y_s.

    Parameters
    ----------
    wires : pennylane.wires.Wires
        A list of wires or wire to be acted on

    Returns
    -------
    none :
        Applies R_n on inputted wires
    """
    qml.adjoint(ocfmc.U_tao)(n,x,ancs[1],wires)
    qml.Barrier()
    qml.Hadamard(ancs[0])
    qml.ctrl(S_n,control=ancs[0])(n,ancs[1],wires)
    qml.Hadamard(ancs[0])

if __name__ == "__main__":
    np.set_printoptions(linewidth=200)

    n = 3
    x = 0.5

    ancs = ["anc_0","anc_1"]
    wires = [f"i{i}" for i in range(n)]

    dev = qml.device("default.qubit", wires=[*ancs,*wires])

    @qml.qnode(dev)
    def circuit():
        U_B(n,x,ancs,wires)
        return qml.state()

    print(f"\nThe Data Constraint Circuit U_B for n={n}:\n")
    print(qml.draw(circuit,max_length=200)(), "\n")

    B_appr = qml.matrix(circuit)().real[0:2**n,0:2**n]

    T = lambda n,x: np.cos(n*np.arccos(x))
    tao= np.array([T(i,x)/np.sqrt(2**(n-1)) for i in range(2**n)])
    tao[0] /= np.sqrt(2)
    init_state = np.array([1]+[0]*(2**n-1))
    B_true = np.sqrt(2**n) * np.einsum('i,j->ij',init_state,tao)

    sub_norm_fact = np.sum(B_true@B_true)/np.sum(B_appr@B_true)
    sub_norm_fact_true = np.sqrt(2**(n+1))

    print(f"\nBlock-encoded matrix for n={n}:",
          f"\n{B_appr}", "\n")
    print(f"\nOriginal matrix for n={n}:",
          f"\n{B_true}", "\n")
    print(f"\nSub-normalization factor (true, calc) for n={n}:",
          f"\n({sub_norm_fact_true}, {sub_norm_fact})", "\n")
    print(f"\nL1 Norm of Error for n={n}:",
          f"\n{np.linalg.norm(B_appr*sub_norm_fact-B_true,ord=1)}", "\n")
