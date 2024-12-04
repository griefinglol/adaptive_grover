from argparse import ArgumentParser, Namespace, BooleanOptionalAction
from qiskit import QuantumCircuit as qc
from qiskit import QuantumRegister as qr
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.result import Counts
from math import pi, sqrt
from heapq import nlargest
import numpy as np

class GroversAlgorithm:
    def __init__(self,
                 title: str = "Grover's Algorithm",
                 n_qubits: int = 5,
                 search: set[int] = { 3 },
                 shots: int = 1,
                 ugate: np.array = None,
                 print: bool = False) -> None:
        """
        Simulate Grover's algorithm using Qiskit's AerSimulator.

        Args:
            title (str, optional): Window title. 
            n_qubits (int, optional): Number of qubits. 
            search (set[int], optional): Set of nonnegative integers to search for using Grover's algorithm. 
            shots (int, optional): Amount of times the algorithm is simulated. Defaults to 1.
            print (bool, optional): Whether or not to print quantum circuit(s). 
        """
        # Parsing command line arguments
        self._parser: ArgumentParser = ArgumentParser(description = "Run grover's algorithm via command line", add_help = False)
        self._init_parser(title, n_qubits, search, shots, print)
        self._args: Namespace = self._parser.parse_args()

        # Set of nonnegative ints to search for
        self.search: set[int] = set(self._args.search)

        # Set of m N-qubit binary strings representing target state(s) (i.e. self.search in base 2)
        self._targets: set[str] = { f"{s:0{self._args.n_qubits}b}" for s in self.search }
        
        # N-qubit quantum register
        self._qubits: qr = qr(self._args.n_qubits, "qubit")

        # the quantum gate used for weighed data
        self._ugate: np.array = ugate
    
    def _init_uniform(self) -> None:
        if self._ugate is None:
            H = 1/np.sqrt(2) * np.array([[1, 1],
                            [1, -1]])
            U = H
            for _ in range(self._args.n_qubits - 1):  
                U = np.kron(U, H)
        
            self._ugate = U

    def _print_circuit(self, circuit: qc, name: str) -> None:
        """Print quantum circuit.

        Args:
            circuit (qc): Quantum circuit to print.
            name (str): Quantum circuit's name.
        """
        print(f"\n{name}:\n{circuit}")

    def _oracle(self, targets: set[str]) -> qc:
        """Mark target state(s) with negative phase.

        Args:
            targets (set[str]): N-qubit binary string(s) representing target state(s).

        Returns:
            qc: Quantum circuit representation of oracle.
        """
        # Create N-qubit quantum circuit for oracle
        oracle = qc(self._qubits, name = "Oracle")
        
        for target in targets:
            # Reverse target state since Qiskit uses little-endian for qubit ordering
            target = target[::-1]
            
            # Flip zero qubits in target
            for i in range(self._args.n_qubits):
                if target[i] == "0":
                    # Pauli-X gate
                    oracle.x(i)

            # Simulate (N - 1)-control Z gate
            # 1. Hadamard gate
            oracle.h(self._args.n_qubits - 1)

            # 2. (N - 1)-control Toffoli gate
            oracle.mcx(list(range(self._args.n_qubits - 1)), self._args.n_qubits - 1)

            # 3. Hadamard gate
            oracle.h(self._args.n_qubits - 1)

            # Flip back to original state
            for i in range(self._args.n_qubits):
                if target[i] == "0":
                    # Pauli-X gate
                    oracle.x(i)

        # Display oracle, if applicable
        if self._args.print: self._print_circuit(oracle, "ORACLE")

        return oracle

    def _diffuser(self) -> qc:
        """Amplify target state(s) amplitude, which decreases the amplitudes of other states
        and increases the probability of getting the correct solution (i.e. target state(s)).

        Returns:
            qc: Quantum circuit representation of diffuser (i.e. Grover's diffusion operator).
        """
        # Create N-qubit quantum circuit for diffuser
        diffuser = qc(self._qubits, name = "Diffuser")

        # first u gate
        diffuser.unitary(np.transpose(self._ugate), self._qubits)

        # Oracle with all zero target state
        diffuser.append(self._oracle({"0" * self._args.n_qubits}), list(range(self._args.n_qubits)))

        # second u gate
        diffuser.unitary(self._ugate, self._qubits)
        
        # Display diffuser, if applicable
        if self._args.print: self._print_circuit(diffuser, "DIFFUSER")
        
        return diffuser

    def _grover(self) -> qc:
        """Create quantum circuit representation of Grover's algorithm,
        which consists of 4 parts: (1) state preparation/initialization,
        (2) oracle, (3) diffuser, and (4) measurement of resulting state.
        
        Steps 2-3 are repeated an optimal number of times (i.e. Grover's
        iterate) in order to maximize probability of success of Grover's algorithm.

        Returns:
            qc: Quantum circuit representation of Grover's algorithm.
        """
        # Create N-qubit quantum circuit for Grover's algorithm
        grover = qc(self._qubits, name = "Grover Circuit")

        # init u gate
        self._init_uniform()
        
        # Intialize qubits with weight 
        grover.unitary(self._ugate, self._qubits) 
        
        # # Apply barrier to separate steps
        grover.barrier()

        # Apply oracle and diffuser (i.e. Grover operator) optimal number of times
        for _ in range(int((pi / 4) * sqrt((2 ** self._args.n_qubits) / len(self._targets)))):
            grover.append(self._oracle(self._targets), list(range(self._args.n_qubits)))
            grover.append(self._diffuser(), list(range(self._args.n_qubits)))
        
        # Measure all qubits once finished
        grover.measure_all()

        # Display grover circuit, if applicable
        if self._args.print: self._print_circuit(grover, "GROVER CIRCUIT")
        
        return grover

    def run(self) -> int:
        """
        Run Grover's algorithm simulation.
        """
        # Simulate Grover's algorithm locally
        backend = AerSimulator(method = "density_matrix")

        # Generate optimized grover circuit for simulation
        transpiled_circuit = transpile(self._grover(), backend, optimization_level = 2)

        # Run Grover's algorithm simulation 
        simulation = backend.run(transpiled_circuit, shots = self._args.shots)
        results = simulation.result().get_counts()
        num_itr = int((pi / 4) * sqrt((2 ** self._args.n_qubits) / len(self._targets)))

        step = 1 

        while(not (list(results.keys())[0] == next(iter(self._targets)))):
            simulation = backend.run(transpiled_circuit, shots = self._args.shots)
            results = simulation.result().get_counts()
            step = step + 1

        return step * num_itr

    def _init_parser(self,
                     title: str,
                     n_qubits: int,
                     search: set[int],
                     shots: int,
                     print: bool) -> None:
        """
        Helper method to initialize command line argument parser.

        Args:
            title (str): Window title.
            n_qubits (int): Number of qubits.
            search (set[int]): Set of nonnegative integers to search for using Grover's algorithm.
            shots (int): Amount of times the algorithm is simulated.
            fontsize (int): Histogram's font size.
            print (bool): Whether or not to print quantum circuit(s).
            combine_states (bool): Whether to combine all non-winning states into 1 bar labeled "Others" or not.
        """
        self._parser.add_argument("-H, --help",
                                  action = "help",
                                  help = "show this help message and exit")

        self._parser.add_argument("-T, --title",
                                  type = str,
                                  default = title,
                                  dest = "title",
                                  metavar = "<title>",
                                  help = f"window title (default: \"{title}\")")

        self._parser.add_argument("-n, --n-qubits",
                                  type = int,
                                  default = n_qubits,
                                  dest = "n_qubits",
                                  metavar = "<n_qubits>",
                                  help = f"number of qubits (default: {n_qubits})")

        self._parser.add_argument("-s, --search",
                                  default = search,
                                  type = int,
                                  nargs = "+",
                                  dest = "search",
                                  metavar = "<search>",
                                  help = f"nonnegative integers to search for with Grover's algorithm (default: {search})")

        self._parser.add_argument("-S, --shots",
                                  type = int,
                                  default = shots,
                                  dest = "shots",
                                  metavar = "<shots>",

                                  help = f"amount of times the algorithm is simulated (default: {shots})")
        
        self._parser.add_argument("-u, --ugate",
                                  default = None,
                                  type = np.array,
                                  nargs = "+",
                                  dest = "ugate",
                                  metavar = "<ugate>",
                                  help = f"the unitary gate which represent weight of data")

        self._parser.add_argument("-p, --print",
                                  action = BooleanOptionalAction,
                                  type = bool,
                                  default = print,
                                  dest = "print",
                                  help = f"whether or not to print quantum circuit(s) (default: {print})")


if __name__ == "__main__":
    step_list = []
    for index in range(32):
        grover = GroversAlgorithm(search={index})
        step_list.append(grover.run())
    print(step_list)