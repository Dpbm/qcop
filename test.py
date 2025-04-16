from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator
from qiskit.transpiler import generate_preset_pass_manager

sim = AerSimulator(method="statevector", device="GPU")
pm = generate_preset_pass_manager(backend=sim, optimization_level=0)

qc = random_circuit(3,10)
qc.measure_all()

isa_qc = pm.run([qc])[0]

result = sim.run(isa_qc, shots=1000).result()
print(result)
