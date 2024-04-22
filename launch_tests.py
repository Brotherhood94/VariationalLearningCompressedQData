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
os.environ['OPENBLAS_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')


WAND_ENABLED = False
WAND_PROJECT_NAME = 'tests_variational_learning_compressed_qdata'


def init_wandb(noisy, n_records, n_features, circuit_width, n_layers, su2_gates, max_epochs, learning_method, init_parameters, num_parameters, num_extending_qubits, optimization_level, seed):
    wandb.init(
        # set the wandb project where this run will be logged
        project=WAND_PROJECT_NAME,

        # track hyperparameters and run metadata
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


def flip_flop_state_preparation(X: np.ndarray):
    n_records, n_features = X.shape
    n_qubits_per_records = int(np.ceil(np.log2(n_records)))
    if n_features > 1:
        n_qubits_per_feature = int(np.ceil(np.log2(n_features)))
    qindex_records = QuantumRegister(n_qubits_per_records, 'm')
    if n_features > 1:
        qindex_features = QuantumRegister(n_qubits_per_feature, 'n')
    qdata = QuantumRegister(1, 'd')
    if n_features > 1:
        ff_circuit = QuantumCircuit(qindex_records, qindex_features, qdata)
    else:
        ff_circuit = QuantumCircuit(qindex_records, qdata)
    ff_circuit.h(qindex_records)
    if n_features > 1:
        ff_circuit.h(qindex_features)
    for index_record, record in enumerate(X):
        for index_feature, feature in enumerate(record):
            bin_index_record = "{0:b}".format(
                index_record).zfill(n_qubits_per_records)
            if n_features > 1:
                bin_index_feature = "{0:b}".format(
                    index_feature).zfill(n_qubits_per_feature)
                ctrl_state = bin_index_feature + bin_index_record
                CRYGate = RYGate(2*np.arcsin(feature)).control(len(qindex_records) +
                                                               len(qindex_features), ctrl_state=ctrl_state)
                ff_circuit.append(
                    CRYGate, qindex_records[:] + qindex_features[:] + qdata[:])
            else:
                ctrl_state = bin_index_record
                CRYGate = RYGate(
                    2*np.arcsin(feature)).control(len(qindex_records), ctrl_state=ctrl_state)
                ff_circuit.append(
                    CRYGate, qindex_records[:] + qdata[:])
    return ff_circuit


def get_ansatz(su2_gates, n_layers, num_extending_qubits, abstract_circuit):
    ansatz = EfficientSU2(abstract_circuit.num_qubits + num_extending_qubits,
                          su2_gates=su2_gates, reps=n_layers)
    return ansatz


def get_circuit_stats(circuit):
    return circuit.depth(), circuit.size(), circuit.count_ops(), circuit_to_dag(circuit)


def run_backend(circuit, model_instance, method='density_matrix', optimization_level=0):
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
    binded_ansatz = ansatz.assign_parameters(paramaters)
    density_matrix_ansatz, _ = run_backend(binded_ansatz.copy(), noisy_instance) if noisy else run_backend(
        binded_ansatz.copy(), fault_tolerant_instance)

    initialized_adjoint_circuit = QuantumCircuit(binded_ansatz.num_qubits)
    initialized_adjoint_circuit.set_density_matrix(density_matrix_ansatz)
    # not using extending qubits
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
    # final_num_of_epochs = out.nita # not available in COBYLA
    return ansatz.assign_parameters(best_parameters), best_parameters, final_infidelity


def run_test(n_records, n_features, n_layers, su2_gates, num_extending_qubits, max_epochs, noisy_instance, fault_tolerant_instance, learning_method, optimization_level, noisy, seed, folder):
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

    abstract_circuit = flip_flop_state_preparation(X)

    ansatz = get_ansatz(su2_gates=su2_gates, n_layers=n_layers, num_extending_qubits=num_extending_qubits,
                        abstract_circuit=abstract_circuit)
    init_parameters = np.random.normal(0, 1, ansatz.num_parameters)

    init_wandb(noisy=noisy, n_records=n_records, n_features=n_features, circuit_width=abstract_circuit.num_qubits, n_layers=n_layers, su2_gates=su2_gates, max_epochs=max_epochs, learning_method=learning_method,
               init_parameters=init_parameters, num_parameters=ansatz.num_parameters, num_extending_qubits=num_extending_qubits, optimization_level=optimization_level, seed=seed) if WAND_ENABLED else None

    learned_ansatz, best_parameters, final_infidelity = train_ansatz(init_parameters=init_parameters, abstract_circuit=abstract_circuit.copy(), ansatz=ansatz.copy(), num_extending_qubits=num_extending_qubits,
                                                                     noisy_instance=noisy_instance, fault_tolerant_instance=fault_tolerant_instance, learning_method=learning_method, max_epochs=max_epochs, noisy=noisy)
    wandb.finish() if WAND_ENABLED else None

    # Running fault tolerant instance with original circuit
    density_matrix_fault_tolerant_original, transpiled_original_circuit_fault_tolerant = run_backend(
        abstract_circuit.copy(), fault_tolerant_instance, optimization_level=optimization_level)

    # Running noisy instance with original circuit
    density_matrix_noisy_original, transpiled_original_circuit_noisy = run_backend(
        abstract_circuit.copy(), noisy_instance, optimization_level=optimization_level)

    # Running noisy instance with learned ansatz
    density_matrix_noisy_learned_ansazt, transpiled_learned_ansatz_noisy = run_backend(
        learned_ansatz.copy(), noisy_instance, optimization_level=optimization_level)

    # Running fault_tolerant instance with learned ansatz
    density_matrix_fault_tolerant_learned_ansazt, _ = run_backend(
        learned_ansatz.copy(), fault_tolerant_instance, optimization_level=optimization_level)

    # Compute fidelities

    # NOISY:
    fidelity_fault_tolerant_original_noisy_original = state_fidelity(
        density_matrix_fault_tolerant_original, density_matrix_noisy_original)
    print('Fidelity Fault Tolerant (original) vs Noisy (original):' +
          str(fidelity_fault_tolerant_original_noisy_original))

    fidelity_fault_tolerant_original_noisy_ansatz = state_fidelity(
        density_matrix_fault_tolerant_original, density_matrix_noisy_learned_ansazt)
    print('Fidelity Fault Tolerant (original) vs Noisy (ansatz):' +
          str(fidelity_fault_tolerant_original_noisy_ansatz))

    fidelity_noisy_original_noisy_ansatz = state_fidelity(
        density_matrix_noisy_original, density_matrix_noisy_learned_ansazt)
    print('Fidelity Fault Tolerant (original) vs Noisy (ansatz):' +
          str(fidelity_noisy_original_noisy_ansatz))

    # NOT NOISY
    fidelity_fault_tolerant_original_fault_tolerant_ansatz = state_fidelity(
        density_matrix_fault_tolerant_original, density_matrix_fault_tolerant_learned_ansazt)
    print('Fidelity Fault Tolerant (original) vs Fault Tolerant (ansatz):' +
          str(fidelity_fault_tolerant_original_fault_tolerant_ansatz))


    depth_abstract, size_abstract, count_ops_abstract, dag_abstract = get_circuit_stats(
        abstract_circuit)
    print(depth_abstract, size_abstract, count_ops_abstract)

    depth_fault_tolerant, size_fault_tolerant, count_ops_fault_tolerant, dag_fault_tolerant = get_circuit_stats(
        transpiled_original_circuit_fault_tolerant)
    print(depth_fault_tolerant, size_fault_tolerant, count_ops_fault_tolerant)

    depth_noisy_original, size_noisy_original, count_ops_noisy_original, dag_noisy_original = get_circuit_stats(
        transpiled_original_circuit_noisy)
    print(depth_noisy_original, size_noisy_original, count_ops_noisy_original)

    depth_noisy_learned_ansatz, size_noisy_learned_ansatz, count_ops_noisy_learned_ansatz, dag_noisy_learned_ansatz = get_circuit_stats(
        transpiled_learned_ansatz_noisy)
    print(depth_noisy_learned_ansatz, size_noisy_learned_ansatz,
          count_ops_noisy_learned_ansatz)

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
            # should be the same of final infidelity
            'fidelity_ft_original_noisy_ansatz': fidelity_fault_tolerant_original_noisy_ansatz if noisy else -1,
            'fidelity_noisy_original_noisy_ansatz': fidelity_noisy_original_noisy_ansatz if noisy else -1,
            # should be the same of final infidelity
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

    folder = './tests_result'

    fault_tolerant_instance = AerSimulator()
    noisy_instance = Fake20QV1()

    learning_methods = ['COBYLA']
    seed_list = range(10, 160)
    noisy_learning = [True, False]
    n_layers_list = [0, 1, 2, 3, 4]
    su2_gates_list = [['ry']]
    num_extending_qubits_list = [0]  # not using
    n_records_list = [2, 4, 8]

    tasks_params = []
    for noisy in noisy_learning:
        for learning_method in learning_methods:
            for seed in seed_list:
                for n_layers in n_layers_list:
                    for su2_gates in su2_gates_list:
                        for num_extending_qubits in num_extending_qubits_list:
                            for n_records in n_records_list:
                                for n_features in [2**(i) for i in range(int(np.log2(n_records)))]:
                                    tasks_params.append([n_records, n_features, n_layers, su2_gates, num_extending_qubits, max_epochs,
                                                        noisy_instance, fault_tolerant_instance, learning_method, optimization_level, noisy, seed, folder])
    with Pool(1) as p:
        p.starmap(run_test, tasks_params)

