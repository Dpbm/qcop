"""Generate random circuits by hand"""

import random
from math import pi
from abc import ABC, abstractmethod

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


class Gate(ABC):
    """interface for gates"""

    @classmethod
    @abstractmethod
    def get_random_gate(cls) -> QiskitGate:
        pass


class SingleQubitGate(Gate):
    """Handle single qubit gates"""

    rotation_gates = [RXGate, RZGate, RYGate]
    simple_gates = [ZGate, XGate, HGate, YGate, TGate, SGate]
    all_gates = [*rotation_gates, *simple_gates]

    @classmethod
    def get_random_gate(cls) -> QiskitGate:
        """Return an instance of a single qubit gate ready to use"""
        gate = random.choice(cls.all_gates)

        if gate in cls.rotation_gates:
            param = random.uniform(0, 2 * pi)
            return gate(param)

        return gate()


class MultiQubitGate(Gate):
    """Handle multi qubit gates"""

    gates = [CXGate, CZGate]

    @classmethod
    def get_random_gate(cls) -> QiskitGate:
        """Return an instance of a multi-qubit gate ready to use"""
        gate = random.choice(cls.gates)
        return gate()


def get_random_circuit(n_qubits: int, total_gates: int) -> QuantumCircuit:
    """Generate a random circuit based on the amount of qubits and gates."""

    total_gates = random.randint(0, total_gates)

    qc = QuantumCircuit(n_qubits)

    for _ in range(total_gates):
        add_single_qubit_gate = random.randint(0, 1)
        add_barrier = random.random() < 0.1  # 10% of chance

        if add_single_qubit_gate:
            qubit = [random.randint(0, n_qubits - 1)]
            gate = SingleQubitGate.get_random_gate()
            qc.append(gate, qubit)
        else:
            qubits = random.sample(range(n_qubits), 2)
            gate = MultiQubitGate.get_random_gate()
            qc.append(gate, qubits)

        if add_barrier:
            qc.barrier()

    return qc
