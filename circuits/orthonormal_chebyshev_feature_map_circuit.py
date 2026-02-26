import numpy as np
import pennylane as qml

def U_tao(n: int, x: float, anc: qml.wires.Wires, wires: qml.wires.Wires):
    """
    U_tao : circuit prepares a purely real-valued quantum Chebyshev state |tao> with a sub-normalization factor of sqrt(2).

    Parameters
    ----------
    wires : pennylane.wires.Wires
        A list of wires or wire to be acted on

    Returns
    -------
    none :
        Applies R_n on inputted wires
    """

    qml.Hadamard(anc)

    for i,wire in enumerate(wires):
        phi = 2**n * np.arccos(x)/2**(i+1)
        qml.Hadamard(wire)
        qml.PhaseShift(1*phi, wires=wire)
        qml.ControlledPhaseShift(-2*phi, wires=[anc,wire])

    qml.ctrl(
        qml.RZ,
        control = wires,
        control_values = [0]*n
        )(-np.pi/2, wires=anc)

    qml.Hadamard(anc)

if __name__ == "__main__":
    np.set_printoptions(linewidth=200)

    n = 3
    x = 0.5

    anc = "anc_0"
    wires = [f"i{i}" for i in range(n)]

    dev = qml.device("default.qubit", wires=[anc,*wires])

    @qml.qnode(dev)
    def circuit():
        U_tao(n,x,anc,wires)
        return qml.state()

    print(f"\nThe Chebyshev Feature Map Circuit for n={n}:\n")
    print(qml.draw(circuit,max_length=200)(), "\n")

    init_state = np.array([1] + [0]*(2**n-1))
    U_tao = qml.matrix(circuit)().real[0:2**n,0:2**n]
    tao_appr = U_tao@init_state

    T = lambda n,x: np.cos(n*np.arccos(x))
    tao_true = np.array([T(i,x)/np.sqrt(2**(n-1)) for i in range(2**n)])
    tao_true[0] /= np.sqrt(2)

    sub_norm_fact = np.dot(tao_true,tao_true)/np.dot(tao_appr,tao_true)
    sub_norm_fact_true = np.sqrt(2)

    print(f"\nQuantum State produced by the circuit for n={n}:",
          f"\n{tao_appr}", "\n")
    print(f"\nOriginal Quantum State for n={n}:",
          f"\n{tao_true/np.sqrt(2)}", "\n")
    print(f"\nSub-normalization factor (true, calc) for n={n}:",
          f"\n({sub_norm_fact_true}, {sub_norm_fact})", "\n")
    print(f"\nL1 Norm of Error for n={n}:",
          f"\n{np.linalg.norm(tao_appr*sub_norm_fact-tao_true,ord=1)}", "\n")
