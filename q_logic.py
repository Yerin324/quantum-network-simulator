# q_logic.py
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

def generate_sbm_network(nodes=8, p_in=0.8, p_out=0.1):
    """SBM 기반의 네트워크와 초기 상태 행렬을 생성합니다."""
    sizes = [nodes // 2, nodes - (nodes // 2)]
    probs = [[p_in, p_out], [p_out, p_in]]
    
    G = nx.stochastic_block_model(sizes, probs, seed=42)
    pos = nx.spring_layout(G, seed=42)
    adj_matrix = nx.to_numpy_array(G)
    
    # 초기 상태 벡터 설정 (0번 인덱스만 1)
    S_0 = np.zeros(nodes)
    S_0[0] = 1
    
    return G, pos, adj_matrix, S_0

def build_quantum_life_circuit(adj_matrix, current_state, alpha, gamma):
    """인접 행렬과 현재 상태를 기반으로 양자 회로를 생성합니다."""
    num_nodes = len(adj_matrix)
    qc = QuantumCircuit(num_nodes)
    
    # 1. 초기 상태 설정
    for i in range(num_nodes):
        if current_state[i] == 1:
            qc.x(i)
    qc.barrier()
    
    # 2. 확산 (Controlled-Ry)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i][j] == 1:
                qc.cry(alpha, i, j)
                qc.cry(alpha, j, i)
    qc.barrier()
    
    # 3. 얽힘 (Rzz) - 독립적인 비교군을 만들기 위해 gamma=0일 때는 생략
    if gamma > 0:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if adj_matrix[i][j] == 1:
                    qc.rzz(gamma, i, j)
        qc.barrier()
        
    qc.measure_all()
    return qc

def get_next_state_from_counts(counts, num_nodes):
    """측정 결과에서 확률적으로 다음 세대 상태를 샘플링합니다."""
    bitstrings = list(counts.keys())
    frequencies = list(counts.values())
    total_shots = sum(frequencies)
    
    probs = [f / total_shots for f in frequencies]
    chosen_bitstring = np.random.choice(bitstrings, p=probs)
    
    reversed_bitstring = chosen_bitstring[::-1]
    return np.array([int(bit) for bit in reversed_bitstring])

def run_simulation(adj_matrix, initial_state, alpha, gamma, generations=10, shots=1024):
    """지정된 세대만큼 회로를 반복 실행하여 시계열 히스토리 행렬을 반환합니다."""
    simulator = AerSimulator()
    num_nodes = len(adj_matrix)
    
    history = [initial_state.copy()]
    current_state = initial_state.copy()
    
    for t in range(1, generations + 1):
        qc = build_quantum_life_circuit(adj_matrix, current_state, alpha, gamma)
        job = simulator.run(qc, shots=shots)
        counts = job.result().get_counts()
        
        next_state = get_next_state_from_counts(counts, num_nodes)
        history.append(next_state)
        current_state = next_state
        
    return np.array(history)