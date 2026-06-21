import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# --- 상태 매핑 상수 ---
STATE_S = 0  # 00 (Susceptible / 정상)
STATE_I = 1  # 01 (Infected / 감염)
STATE_R = 2  # 10 (Recovered / 회복 및 면역)

def generate_sbm_network(nodes=8, p_in=0.8, p_out=0.1):
    """더미 테스트용 SBM 기반 네트워크를 생성합니다."""
    sizes = [nodes // 2, nodes - (nodes // 2)]
    probs = [[p_in, p_out], [p_out, p_in]]
    
    G = nx.stochastic_block_model(sizes, probs, seed=42)
    pos = nx.spring_layout(G, seed=42)
    adj_matrix = nx.to_numpy_array(G)
    
    return G, pos, adj_matrix

def build_sir_quantum_circuit(adj_matrix, current_states, alpha, beta, gamma):
    """SIR 모델을 위한 1세대 진화 양자 회로를 빌드합니다."""
    num_nodes = len(adj_matrix)
    num_qubits = num_nodes * 2 # 노드당 2 큐비트
    
    qc = QuantumCircuit(num_qubits)
    
    # 1. 상태 초기화
    for node_idx in range(num_nodes):
        state = current_states[node_idx]
        q_idx_1 = node_idx * 2      
        q_idx_2 = node_idx * 2 + 1  
        if state == STATE_I: 
            qc.x(q_idx_2)
        elif state == STATE_R: 
            qc.x(q_idx_1)
    qc.barrier()
    
    # 2. 감염 및 회복 이벤트
    for i in range(num_nodes):
        state_i = current_states[i]
        q_idx_1_i = i * 2
        q_idx_2_i = i * 2 + 1
        
        if state_i == STATE_I:
            # [감염: S -> I]
            for j in range(num_nodes):
                if adj_matrix[i][j] == 1 and current_states[j] == STATE_S:
                    q_idx_2_j = j * 2 + 1
                    qc.cry(alpha, q_idx_2_i, q_idx_2_j)
                    
            # [회복: I -> R]
            qc.ry(beta, q_idx_1_i) 
            qc.cx(q_idx_1_i, q_idx_2_i)
    qc.barrier()
    
    # 3. 양자 얽힘 (집단 감염/회복 상관관계)
    if gamma > 0:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if adj_matrix[i][j] == 1:
                    q_idx_2_i = i * 2 + 1
                    q_idx_2_j = j * 2 + 1
                    qc.rzz(gamma, q_idx_2_i, q_idx_2_j)
    qc.barrier()
    
    qc.measure_all()
    return qc

def get_next_sir_state_from_counts(counts, num_nodes):
    """측정 결과를 확률적으로 샘플링하고 S, I, R 상태로 변환합니다."""
    bitstrings = list(counts.keys())
    frequencies = list(counts.values())
    total_shots = sum(frequencies)
    
    probs = [f / total_shots for f in frequencies]
    chosen_bitstring = np.random.choice(bitstrings, p=probs)
    reversed_bitstring = chosen_bitstring[::-1] # Qiskit Endianness 보정
    
    next_states = []
    for i in range(num_nodes):
        chunk = reversed_bitstring[i*2 : i*2+2]
        if chunk == '00':
            next_states.append(STATE_S)
        elif chunk == '01':
            next_states.append(STATE_I)
        else: # '10' 또는 '11'(노이즈 억제용)은 회복으로 간주
            next_states.append(STATE_R)
            
    return next_states

def run_sir_simulation(adj_matrix, initial_state, alpha, beta, gamma, generations=10, shots=1024):
    """지정된 세대만큼 양자 SIR 시뮬레이션을 반복 실행합니다."""
    simulator = AerSimulator()
    num_nodes = len(adj_matrix)
    
    history = [initial_state.copy()]
    current_state = initial_state.copy()
    
    for t in range(1, generations + 1):
        qc = build_sir_quantum_circuit(adj_matrix, current_state, alpha, beta, gamma)
        job = simulator.run(qc, shots=shots)
        counts = job.result().get_counts()
        
        next_state = get_next_sir_state_from_counts(counts, num_nodes)
        history.append(next_state)
        current_state = next_state
        
    return history