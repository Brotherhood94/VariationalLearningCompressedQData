import wandb
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from qiskit import *
from qiskit.circuit.library import *
from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider import *
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from qiskit_aer.noise import NoiseModel
from qiskit.quantum_info import Statevector, state_fidelity, purity
from qiskit.converters import circuit_to_dag
from multiprocessing import Pool, cpu_count
import json

import qiskit.qasm2
import os
import warnings
from flip_flop_state_preparation import flip_flop_state_preparation
os.environ['OPENBLAS_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')


WAND_ENABLED = False
WAND_PROJECT_NAME = 'tests_variational_learning_compressed_qdata'


def init_wandb(noisy, n_records, n_features, circuit_width, n_layers, su2_gates, max_epochs, learning_method, init_parameters, num_parameters, num_extending_qubits, optimization_level, seed):
    """
    Initialize Weights and Biases (wandb) for logging hyperparameters and run metadata.

    Args:
        noisy (bool): Flag indicating whether the simulation is noisy or not.
        n_records (int): Number of records in the dataset.
        n_features (int): Number of features in the dataset.
        circuit_width (int): Width of the quantum circuit.
        n_layers (int): Number of layers in the quantum circuit.
        su2_gates (int): Number of SU(2) gates in the quantum circuit.
        max_epochs (int): Maximum number of epochs for training.
        learning_method (str): Optimization method for training.
        init_parameters (numpy.ndarray): Initial parameters for the ansatz.
        num_parameters (int): Number of parameters in the ansatz.
        num_extending_qubits (int): Number of extending qubits in the ansatz.
        optimization_level (int): Optimization level for transpiling the circuit.
        seed (int): Random seed for reproducibility.
    """
    wandb.init(
        project=WAND_PROJECT_NAME,
        config={
            "learning_method": learning_method,
            "max_epochs": max_epochs,
            "noisy": noisy,
            "n_records": n_records,
            "n_features": n_features,
            "circuit_width": circuit_width,
            "n_layers": n_layers,
            "su2_gates": su2_gates,
            "init_parameters": init_parameters,
            "num_parameters": num_parameters,
            "num_extending_qubits": num_extending_qubits,
            "optimization_level": optimization_level,
            "seed": seed
        }
    )


def get_ansatz(su2_gates, n_layers, num_extending_qubits, abstract_circuit):
    """
    Get the ansatz circuit for variational learning.

    Args:
        su2_gates (int): Number of SU(2) gates in the ansatz.
        n_layers (int): Number of layers in the ansatz.
        num_extending_qubits (int): Number of extending qubits in the ansatz.
        abstract_circuit (QuantumCircuit): The abstract circuit for state preparation.

    Returns:
        QuantumCircuit: The ansatz circuit.
    """
    ansatz = EfficientSU2(abstract_circuit.num_qubits + num_extending_qubits,
                          su2_gates=su2_gates, reps=n_layers)
    return ansatz


def get_circuit_stats(circuit):
    """
    Get statistics of a quantum circuit.

    Args:
        circuit (QuantumCircuit): The quantum circuit.

    Returns:
        int: Depth of the circuit.
        int: Size of the circuit.
        dict: Count of each operation in the circuit.
        DAGCircuit: Directed Acyclic Graph representation of the circuit.
    """
    return circuit.depth(), circuit.size(), circuit.count_ops(), circuit_to_dag(circuit)


def run_backend(circuit, model_instance, method='density_matrix', optimization_level=0):
    """
    Run a quantum circuit on a backend.

    Args:
        circuit (QuantumCircuit): The quantum circuit to run.
        model_instance (BaseBackend): The backend instance.
        method (str, optional): Method for simulation. Defaults to 'density_matrix'.
        optimization_level (int, optional): Optimization level for transpiling the circuit. Defaults to 0.

    Returns:
        numpy.ndarray: Density matrix of the final state.
        QuantumCircuit: Transpiled circuit.
    """
    circuit.save_density_matrix()
    noise_model = NoiseModel.from_backend(model_instance)
    coupling_map = model_instance.configuration().coupling_map
    basis_gates = noise_model.basis_gates

    backend = AerSimulator(method=method, noise_model=noise_model,
                           coupling_map=coupling_map, basis_gates=basis_gates)

    transpiled_circuit = transpile(
        circuit, backend, optimization_level=optimization_level, coupling_map=coupling_map)

    density_matrix = backend.run(transpiled_circuit).result().data()[
        'density_matrix']
    return density_matrix, transpiled_circuit


def compute_loss(paramaters, abstract_circuit, ansatz, num_extending_qubits, noisy_instance, fault_tolerant_instance, noisy):
    """
    Compute the loss function for variational learning.

    Args:
        paramaters (numpy.ndarray): Parameters for the ansatz.
        abstract_circuit (QuantumCircuit): The abstract circuit for state preparation.
        ansatz (QuantumCircuit): The ansatz circuit.
        num_extending_qubits (int): Number of extending qubits in the ansatz. (NOT USED)
        noisy_instance (BaseBackend): The noisy backend instance.
        fault_tolerant_instance (BaseBackend): The fault-tolerant backend instance.
        noisy (bool): Flag indicating whether the simulation is noisy or not.

    Returns:
        float: The computed loss value.
    """
    binded_ansatz = ansatz.assign_parameters(paramaters)
    density_matrix_ansatz, _ = run_backend(binded_ansatz.copy(), noisy_instance) if noisy else run_backend(
        binded_ansatz.copy(), fault_tolerant_instance)

    initialized_adjoint_circuit = QuantumCircuit(binded_ansatz.num_qubits)
    initialized_adjoint_circuit.set_density_matrix(density_matrix_ansatz)
    initialized_adjoint_circuit.append(
        abstract_circuit.inverse(), initialized_adjoint_circuit.qubits[:])

    density_matrix_initialized_adjoint_circuit, _ = run_backend(
        initialized_adjoint_circuit.copy(), fault_tolerant_instance)

    ket0 = Statevector.from_label('0'*initialized_adjoint_circuit.num_qubits)

    infidelity = 1 - \
        state_fidelity(ket0, density_matrix_initialized_adjoint_circuit)
    purity_val = purity(density_matrix_initialized_adjoint_circuit).real

    wandb.log({'infidelity': infidelity, 'purity': purity_val}
              ) if WAND_ENABLED else None

    print('Infidelity: {}, Purity: {}'.format(infidelity, purity_val))
    return infidelity


def train_ansatz(init_parameters, abstract_circuit, ansatz, num_extending_qubits, noisy_instance, fault_tolerant_instance, learning_method, max_epochs, noisy):
    """
    Train the ansatz circuit using variational learning.

    Args:
        init_parameters (numpy.ndarray): Initial parameters for the ansatz.
        abstract_circuit (QuantumCircuit): The abstract circuit for state preparation.
        ansatz (QuantumCircuit): The ansatz circuit.
        num_extending_qubits (int): Number of extending qubits in the ansatz.
        noisy_instance (BaseBackend): The noisy backend instance.
        fault_tolerant_instance (BaseBackend): The fault-tolerant backend instance.
        learning_method (str): Optimization method for training.
        max_epochs (int): Maximum number of epochs for training.
        noisy (bool): Flag indicating whether the simulation is noisy or not.

    Returns:
        QuantumCircuit: The trained ansatz circuit.
        numpy.ndarray: The best parameters found during training.
        float: The final infidelity value.
    """
    out = minimize(compute_loss,
                   x0=init_parameters,
                   method=learning_method,
                   tol=0.001,
                   options={'maxiter': max_epochs},
                   args=(abstract_circuit, ansatz, num_extending_qubits,
                         noisy_instance, fault_tolerant_instance, noisy)
                   )
    best_parameters = out.x
    final_infidelity = out.fun
    print('Final infidelity: '+str(final_infidelity))
    return ansatz.assign_parameters(best_parameters), best_parameters, final_infidelity


def run_test(state_preparation, n_records, n_features, n_layers, su2_gates, num_extending_qubits, max_epochs, noisy_instance, fault_tolerant_instance, learning_method, optimization_level, noisy, seed, folder):
    """
    Run a test for variational learning.

    Args:
        n_records (int): Number of records in the dataset.
        n_features (int): Number of features in the dataset.
        n_layers (int): Number of layers in the ansatz.
        su2_gates (int): Number of SU(2) gates in the ansatz.
        num_extending_qubits (int): Number of extending qubits in the ansatz.
        max_epochs (int): Maximum number of epochs for training.
        noisy_instance (BaseBackend): The noisy backend instance.
        fault_tolerant_instance (BaseBackend): The fault-tolerant backend instance.
        learning_method (str): Optimization method for training.
        optimization_level (int): Optimization level for transpiling the circuit.
        noisy (bool): Flag indicating whether the simulation is noisy or not.
        seed (int): Random seed for reproducibility.
        folder (str): Folder path for saving the results.
    """
    print('Running: '+str(n_records)+'_'+str(n_features)+'_' +
          str(n_layers)+'_'+str(su2_gates)+'_'+str(num_extending_qubits)+'_'+str(seed)+'_'+str(noisy_instance.backend_name))

    folder = folder+'/noisy_learning_'+str(noisy)+'/noisy_instance_'+str(noisy_instance.backend_name)+'/learning_method_'+learning_method+'/n_layers_'+str(n_layers)+'/su2_gates_'+str(su2_gates)+'/num_extending_qubits_'+str(
        num_extending_qubits)+'/n_records_'+str(n_records)+'/n_features_'+str(n_features)+'/seed_'+str(seed)+'/optimization_level_'+str(optimization_level)+'/'

    Path(folder).mkdir(parents=True, exist_ok=True)
    files_in_folder = len(list(Path(folder).glob('*')))
    if files_in_folder >= 1:
        print('Already Done. Skipping: '+folder)
        return

    np.random.seed(seed)
    X, _ = make_classification(
        n_samples=n_records, n_features=n_features, n_informative=n_features, n_redundant=0, n_classes=2, n_clusters_per_class=1, random_state=seed)
    X = preprocessing.normalize(X, axis=1)

    # Compute the grounf truth density matrix
    abstract_circuit = state_preparation(X)

    # Get the ansatz circuit
    ansatz = get_ansatz(su2_gates=su2_gates, n_layers=n_layers, num_extending_qubits=num_extending_qubits,
                        abstract_circuit=abstract_circuit)
    init_parameters = np.random.normal(0, 1, ansatz.num_parameters)

    init_wandb(noisy=noisy, n_records=n_records, n_features=n_features, circuit_width=abstract_circuit.num_qubits, n_layers=n_layers, su2_gates=su2_gates, max_epochs=max_epochs, learning_method=learning_method,
               init_parameters=init_parameters, num_parameters=ansatz.num_parameters, num_extending_qubits=num_extending_qubits, optimization_level=optimization_level, seed=seed) if WAND_ENABLED else None

    # train the ansatz
    learned_ansatz, best_parameters, final_infidelity = train_ansatz(init_parameters=init_parameters, abstract_circuit=abstract_circuit.copy(), ansatz=ansatz.copy(), num_extending_qubits=num_extending_qubits,
                                                                     noisy_instance=noisy_instance, fault_tolerant_instance=fault_tolerant_instance, learning_method=learning_method, max_epochs=max_epochs, noisy=noisy)
    wandb.finish() if WAND_ENABLED else None

    density_matrix_fault_tolerant_original, transpiled_original_circuit_fault_tolerant = run_backend(
        abstract_circuit.copy(), fault_tolerant_instance, optimization_level=optimization_level)

    density_matrix_noisy_original, transpiled_original_circuit_noisy = run_backend(
        abstract_circuit.copy(), noisy_instance, optimization_level=optimization_level)

    density_matrix_noisy_learned_ansazt, transpiled_learned_ansatz_noisy = run_backend(
        learned_ansatz.copy(), noisy_instance, optimization_level=optimization_level)

    density_matrix_fault_tolerant_learned_ansazt, _ = run_backend(
        learned_ansatz.copy(), fault_tolerant_instance, optimization_level=optimization_level)

    fidelity_fault_tolerant_original_noisy_original = state_fidelity(
        density_matrix_fault_tolerant_original, density_matrix_noisy_original)

    fidelity_fault_tolerant_original_noisy_ansatz = state_fidelity(
        density_matrix_fault_tolerant_original, density_matrix_noisy_learned_ansazt)

    fidelity_noisy_original_noisy_ansatz = state_fidelity(
        density_matrix_noisy_original, density_matrix_noisy_learned_ansazt)

    fidelity_fault_tolerant_original_fault_tolerant_ansatz = state_fidelity(
        density_matrix_fault_tolerant_original, density_matrix_fault_tolerant_learned_ansazt)

    depth_abstract, size_abstract, count_ops_abstract, dag_abstract = get_circuit_stats(
        abstract_circuit)

    depth_fault_tolerant, size_fault_tolerant, count_ops_fault_tolerant, dag_fault_tolerant = get_circuit_stats(
        transpiled_original_circuit_fault_tolerant)

    depth_noisy_original, size_noisy_original, count_ops_noisy_original, dag_noisy_original = get_circuit_stats(
        transpiled_original_circuit_noisy)

    depth_noisy_learned_ansatz, size_noisy_learned_ansatz, count_ops_noisy_learned_ansatz, dag_noisy_learned_ansatz = get_circuit_stats(
        transpiled_learned_ansatz_noisy)

    df = pd.DataFrame()
    data = {'noisy': noisy,
            'learning_method': learning_method,
            'n_layers': n_layers,
            'su2_gates': su2_gates,
            'n_records': n_records,
            'n_features': n_features,
            'ansatz_total_number_of_qubits_original': learned_ansatz.num_qubits,
            'num_extending_qubits': num_extending_qubits,
            'init_parameters': init_parameters.tolist(),
            'best_parameters': best_parameters.tolist(),
            'num_parameters': learned_ansatz.num_parameters,
            'final_infidelity_optimizer': final_infidelity,
            'total_number_of_qubits': abstract_circuit.num_qubits,
            'fidelity_ft_original_noisy_original': fidelity_fault_tolerant_original_noisy_original if noisy else -1,
            # same of final infidelity
            'fidelity_ft_original_noisy_ansatz': fidelity_fault_tolerant_original_noisy_ansatz if noisy else -1,
            'fidelity_noisy_original_noisy_ansatz': fidelity_noisy_original_noisy_ansatz if noisy else -1,
            # same of final infidelity
            'fidelity_ft_original_ft_ansatz': fidelity_fault_tolerant_original_fault_tolerant_ansatz if not noisy else -1,
            'hardware_backend': noisy_instance.backend_name,
            'hardware_basis_gates': noisy_instance.configuration().basis_gates,
            'hardware_coupling_map': noisy_instance.configuration().coupling_map,
            'fault_tolerant_backend': fault_tolerant_instance.name,
            'fault_tolerant_basis_gates': fault_tolerant_instance.configuration().basis_gates,
            'fault_tolerant_coupling_map': fault_tolerant_instance.configuration().coupling_map,
            'depth_fault_tolerant': depth_fault_tolerant,
            'size_fault_tolerant': size_fault_tolerant,
            'count_ops_fault_tolerant': json.dumps(count_ops_fault_tolerant),
            'depth_hardware_original': depth_noisy_original,
            'size_hardware_original': size_noisy_original,
            'count_ops_hardware_original': json.dumps(count_ops_noisy_original),
            'layout_hardware_original': transpiled_original_circuit_noisy.layout.final_index_layout(filter_ancillas=True),
            'depth_hardware_ansatz': depth_noisy_learned_ansatz,
            'size_hardware_ansatz': size_noisy_learned_ansatz,
            'count_ops_hardware_ansatz': json.dumps(count_ops_noisy_learned_ansatz),
            'layout_hardware_ansatz': transpiled_learned_ansatz_noisy.layout.final_index_layout(filter_ancillas=True),
            'depth_abstract': depth_abstract,
            'size_abstract': size_abstract,
            'count_ops_abstract': json.dumps(count_ops_abstract),
            'abstract_circuit': qiskit.qasm2.dumps(abstract_circuit),
            'fault_tolerant_original': qiskit.qasm2.dumps(transpiled_original_circuit_fault_tolerant),
            'hardware_original': qiskit.qasm2.dumps(transpiled_original_circuit_noisy),
            'hardware_ansatz': qiskit.qasm2.dumps(transpiled_learned_ansatz_noisy),
            'optimization_level': optimization_level,
            'seed': seed,
            }
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_csv(folder + 'data.dataframe', index=False)


if __name__ == '__main__':

    optimization_level = 0
    max_epochs = 400

    folder = './Results_variational_learning_compressed_qdata/'

    fault_tolerant_instance = AerSimulator()
    noisy_instance = Fake20QV1()

    learning_methods = ['COBYLA']
    seed_list = range(10, 160)
    noisy_learning = [True, False]
    n_layers_list = [0, 1, 2, 3, 4]
    su2_gates_list = [['ry']]
    num_extending_qubits_list = [0]  # not using
    n_records_list = [2, 4, 8]
    state_preparations = [flip_flop_state_preparation]

    tasks_params = []
    for state_preparation in state_preparations:
        for noisy in noisy_learning:
            for learning_method in learning_methods:
                for seed in seed_list:
                    for n_layers in n_layers_list:
                        for su2_gates in su2_gates_list:
                            for num_extending_qubits in num_extending_qubits_list:
                                for n_records in n_records_list:
                                    for n_features in [2**(i) for i in range(int(np.log2(n_records)))]:
                                        tasks_params.append([state_preparation, n_records, n_features, n_layers, su2_gates, num_extending_qubits, max_epochs,
                                                            noisy_instance, fault_tolerant_instance, learning_method, optimization_level, noisy, seed, folder])
    with Pool(10) as p:
        p.starmap(run_test, tasks_params)
