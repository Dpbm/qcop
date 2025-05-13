import random
from math import pi
from abc import ABC, abstractclassmethod
from qiskit import QuantumCircuit
from qiskit.circuit import Gate as QiskitGate
from qiskit.circuit.library import (
    ZGate, 
    XGate, 
    HGate, 
    YGate, 
    RYGate, 
    RZGate, 
    RXGate,
    TGate, 
    SGate,
    CXGate,
    CZGate,
)


from constants import MAX_TOTAL_GATES, N_QUBITS


class Gate(ABC):
    @abstractclassmethod
    def get_random_gate(cls) -> QiskitGate:
        pass

class SingleQubitGate(Gate):
    rotation_gates = [
        RXGate,
        RZGate,
        RYGate
    ]
    simple_gates = [
        ZGate,
        XGate,
        HGate,
        YGate,
        TGate,
        SGate
    ]
    all_gates = [*rotation_gates, *simple_gates]

    @classmethod
    def get_random_gate(cls) -> QiskitGate:
        gate = random.choice(cls.all_gates)

        if gate in cls.rotation_gates:
            param = random.uniform(0, 2*pi)
            return gate(param)

        return gate()

class MultiQubitGate(Gate):
    gates = [
        CXGate,
        CZGate
    ]

    @classmethod
    def get_random_gate(cls) -> QiskitGate:
        gate = random.choice(cls.gates)
        return gate()



def get_random_circuit() -> QuantumCircuit:
    total_gates = random.randint(0,MAX_TOTAL_GATES)

    qc = QuantumCircuit(N_QUBITS)
    
    for _ in range(total_gates):
        add_single_qubit_gate = random.randint(0,1)

        if add_single_qubit_gate:
            qubit = [random.randint(0, N_QUBITS-1)]
            gate = SingleQubitGate.get_random_gate()
            qc.append(gate, qubit)
            continue

        qubits = random.sample(range(N_QUBITS), 2)
        gate = MultiQubitGate.get_random_gate()
        qc.append(gate, qubits)

    return qc

