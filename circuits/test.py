import numpy as np
import pennylane as qml


def U2(wires):
    w1,w2 = wires

    qml.CNOT([w2,w1])
    qml.CZ([w1,w2])
    qml.CNOT([w2,w1])

    qml.CNOT([w1,w2])
    qml.CZ([w2,w1])
    qml.CNOT([w1,w2])

    qml.SWAP([w1,w2])


wires = ["w1","w2"]
dev = qml.device("default.qubit", wires=wires)
@qml.qnode(dev)
def circuit():
    U2(wires)
    return qml.state()

print(qml.draw(circuit,max_length=200)(), "\n")
Um = qml.matrix(circuit)().real
print(Um)
