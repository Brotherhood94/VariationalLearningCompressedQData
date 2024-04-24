from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RYGate
import numpy as np


def flip_flop_state_preparation(X: np.ndarray):
    """
    Prepares a quantum circuit for the flip-flop state based on the input data.

    Args:
        X (np.ndarray): Input data as a numpy array.

    Returns:
        QuantumCircuit: The prepared quantum circuit for the flip-flop state.
    """
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
