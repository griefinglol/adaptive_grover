from grover import *
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram
from scipy import stats
from data_process import process

def find_preparation_gate(target_amplitudes):
    """
    Find the unitary matrix that transforms |0...0⟩ to the desired state.
    
    Args:
        target_amplitudes (list): List of complex amplitudes for the target state
    
    Returns:
        numpy.ndarray: The unitary matrix that performs the transformation
    """
    # Convert to numpy array and normalize
    target_state = target_amplitudes / np.linalg.norm(target_amplitudes)
    
    n_qubits = int(np.log2(len(target_state)))
    dim = 2**n_qubits
    
    # Create initial state |0...0⟩
    initial_state = np.zeros(dim)
    initial_state[0] = 1
    
    # Create a basis for the orthogonal complement
    basis = np.eye(dim)
    v1 = target_state
    
    # Gram-Schmidt process to create orthonormal basis
    Q = np.zeros((dim, dim), dtype=complex)
    Q[:, 0] = v1
    
    for i in range(1, dim):
        v = basis[:, i]
        for j in range(i):
            v = v - np.dot(Q[:, j].conj(), v) * Q[:, j]
        if np.linalg.norm(v) > 1e-10:  # Check if vector is non-zero
            v = v / np.linalg.norm(v)
        Q[:, i] = v
    
    # Construct unitary matrix
    U = Q
    
    return U

if __name__ == "__main__":
    heavy_tail_path = "data/heavy_tail_samples.txt"
    normal_path = "data/normal_distribution_samples.txt"
    quniform_path = "data/quasi_uniform_samples.txt"

    num_qubits = 5
    num_states = 2**num_qubits

    path_list = [heavy_tail_path, normal_path, quniform_path, None]
    disturion_step = []
    for path in path_list:
        if path is None:
            U = None
        else:
            _, amplitudes = process(path)
            U = find_preparation_gate(amplitudes)

        step_list = []
        for index in range(32):
            grover = GroversAlgorithm(search={index}, ugate = U)
            step_list.append(grover.run())

        disturion_step.append(step_list)

    with open('output/quantum_output.txt', 'w') as f:
        for item in disturion_step:
            f.write(str(item) + '\n')