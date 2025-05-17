from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Define a simple function to create a quantum circuit for order finding
def create_order_finding_circuit(n, a):
    # Number of qubits needed
    num_qubits = n.bit_length() + 1

    # Create a quantum circuit with the necessary number of qubits
    qc = QuantumCircuit(num_qubits, num_qubits)

    # Apply Hadamard gates to the first n qubits
    for qubit in range(n.bit_length()):
        qc.h(qubit)

    # Placeholder for controlled unitary operations
    # In a real implementation, this would involve modular exponentiation
    # Here, we just apply a simple controlled operation for demonstration
    qc.cx(0, n.bit_length())

    # Apply inverse Quantum Fourier Transform (QFT)
    qc.h(n.bit_length())

    # Measure the qubits
    qc.measure(range(num_qubits), range(num_qubits))

    return qc

# Example parameters
n = 15  # Example modulus
a = 7   # Example base

# Create the circuit
qc = create_order_finding_circuit(n, a)

# Simulate the circuit
backend = Aer.get_backend('qasm_simulator')
result = execute(qc, backend, shots=1024).result()
counts = result.get_counts()

# Plot the results
plot_histogram(counts)
plt.show() 
