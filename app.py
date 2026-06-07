import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import time
from scipy.stats import entropy
from q_logic import generate_sbm_network, run_simulation


st.set_page_config(page_title="Quantum Network Simulator", layout="wide")
st.title("🌐 양자 얽힘 기반 네트워크 확산 대시보드")
st.markdown("자신의 네트워크 데이터(인접 행렬 CSV)를 업로드하여 확산 과정을 시뮬레이션해 보세요.")

# --- 1. 사이드바: 데이터 업로드 및 파라미터 설정 ---
st.sidebar.header("📁 1. 네트워크 데이터 입력")

# CSV 파일 업로더 추가
uploaded_file = st.sidebar.file_uploader("인접 행렬 CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
    # 사용자가 파일을 업로드한 경우: CSV를 Numpy 배열로 변환
    df = pd.read_csv(uploaded_file, header=None) # 헤더가 없는 순수 행렬 형태라고 가정
    adj_matrix = df.to_numpy()
    total_nodes = len(adj_matrix)
    
    # NetworkX 그래프 및 레이아웃 생성
    G = nx.from_numpy_array(adj_matrix)
    pos = nx.spring_layout(G, seed=42)
    st.sidebar.success(f"✅ {total_nodes}개의 노드를 가진 네트워크 로드 완료!")
else:
    # 업로드한 파일이 없는 경우: 기본 더미 데이터(SBM) 사용
    st.sidebar.info("업로드된 파일이 없어 기본 SBM 네트워크를 사용합니다.")
    total_nodes = 8
    G, pos, adj_matrix, _ = generate_sbm_network(nodes=total_nodes)


st.sidebar.header("⚙️ 2. 시뮬레이션 설정")
# 초기 활성화 노드 직접 선택 UI 추가
initial_active = st.sidebar.multiselect(
    "초기 활성 노드(Seed) 선택", 
    options=list(range(total_nodes)), 
    default=[0]
)

# 초기 상태 벡터(S_0) 생성
S_0 = np.zeros(total_nodes)
for node in initial_active:
    S_0[node] = 1

generations = st.sidebar.slider("총 세대 수", 5, 20, 10)
alpha_param = st.sidebar.slider("확산 강도 (Alpha)", 0.0, np.pi, np.pi/4, step=0.1)
gamma_param = st.sidebar.slider("얽힘 강도 (Gamma)", 0.0, np.pi, np.pi/4, step=0.1)

# --- 2. 양자 백엔드 실행 로직 ---
# 입력 데이터가 바뀌면 캐시를 무효화하도록 함수의 인자로 adj_matrix와 S_0를 받게 수정
@st.cache_data(show_spinner="Qiskit 양자 회로 시뮬레이션 중...")
def fetch_quantum_data_dynamic(_adj_matrix, _S_0, gen, alpha, gamma):
    history_indep = run_simulation(_adj_matrix, _S_0, alpha, gamma=0.0, generations=gen)
    history_entangled = run_simulation(_adj_matrix, _S_0, alpha, gamma=gamma, generations=gen)
    return history_indep, history_entangled

# 시뮬레이션 실행버튼 (데이터가 클 경우를 대비해 수동 실행 유도)
if st.sidebar.button("🚀 시뮬레이션 실행"):
    # session_state에 결과 저장
    st.session_state['history_indep'], st.session_state['history_entangled'] = fetch_quantum_data_dynamic(
        adj_matrix, S_0, generations, alpha_param, gamma_param
    )

# --- 3. 대시보드 렌더링 ---
# 시뮬레이션 결과가 존재할 때만 화면에 표시
# --- 3. 대시보드 렌더링 (자동 재생 기능 추가) ---
if 'history_entangled' in st.session_state:
    history_indep = st.session_state['history_indep']
    history_entangled = st.session_state['history_entangled']

    st.subheader("타임라인 뷰어 (자동 재생)")
    
    # 레이아웃 분할: 좌측은 슬라이더, 우측은 재생/정지 버튼
    col_slider, col_btn1, col_btn2 = st.columns([3, 1, 1])
    
    with col_slider:
        current_gen = st.slider("세대(Generation) 확인", 0, generations, 0, step=1, key="gen_slider")
    
    with col_btn1:
        play_button = st.button("▶️ 자동 재생")
        
    with col_btn2:
        stop_button = st.button("⏹️ 정지 (초기화)")

    # 애니메이션이 렌더링될 빈 컨테이너(Placeholder) 생성
    plot_container = st.empty()
    status_container = st.empty()
    
    # 상태 텍스트 초기화
    status_container.markdown(f"**현재 상태 (Gen {current_gen}):** `{history_entangled[current_gen]}`")

    def plot_network_step(G, pos, state_vector):
        """네트워크 그래프 렌더링 함수 (기존과 동일)"""
        node_colors = ['#ff4b4b' if state == 1 else '#1f77b4' for state in state_vector]
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f'N {node}')
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="bottom center", hoverinfo='text', marker=dict(color=node_colors, size=30, line_width=2))
        fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, margin=dict(b=20,l=5,r=5,t=20), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=400))
        return fig

    # 기본 화면 렌더링 (재생 버튼을 누르지 않았을 때 슬라이더 값에 따라 변경)
    plot_container.plotly_chart(
    plot_network_step(G, pos, history_entangled[current_gen]), 
    use_container_width=True, 
    key="base_chart_view"  # 고유 이름표 달기
)
    # --- 애니메이션 자동 재생 로직 ---
    if play_button:
        # 0세대부터 사용자가 설정한 총 세대(generations)까지 순차적으로 렌더링
        for t in range(generations + 1):
            # 상태 텍스트 업데이트
            status_container.markdown(f"**현재 상태 (Gen {t}):** `{history_entangled[t]}`")
            # 그래프 업데이트
            fig = plot_network_step(G, pos, history_entangled[t])
            # 빈 컨테이너에 덮어쓰기 방식으로 차트 렌더링 (애니메이션 효과)
            plot_container.plotly_chart(
                fig, 
                use_container_width=True, 
                key=f"anim_chart_frame_{t}"  # 프레임마다 다른 이름표 달기
            )
            # 0.5초 대기 (프레임 속도 조절)
            time.sleep(0.5)

    if stop_button:
        # 정지 버튼을 누르면 0세대로 리셋
        st.rerun()

    # --- 통계 비교 분석 차트 ---
    st.markdown("---")
    st.subheader("📊 고전(독립) 전파 vs 양자 얽힘 전파 비교")
    ratio_indep = np.sum(history_indep, axis=1) / total_nodes
    ratio_entangled = np.sum(history_entangled, axis=1) / total_nodes
    eps = 1e-9
    prob_indep = (np.sum(history_indep, axis=1) + eps) / (total_nodes + eps * generations)
    prob_entangled = (np.sum(history_entangled, axis=1) + eps) / (total_nodes + eps * generations)
    kl_div_scores = [entropy([prob_entangled[t], 1-prob_entangled[t]], [prob_indep[t], 1-prob_indep[t]]) for t in range(len(prob_indep))]

    col1, col2 = st.columns(2)
    with col1:
        fig_speed = go.Figure()
        fig_speed.add_trace(go.Scatter(y=ratio_indep, mode='lines+markers', name='Classical (γ=0)', line=dict(color='blue', dash='dash')))
        fig_speed.add_trace(go.Scatter(y=ratio_entangled, mode='lines+markers', name='Quantum (γ>0)', line=dict(color='red', width=3)))
        fig_speed.update_layout(title="위상 전이 및 활성화 속도", xaxis_title="Generation", yaxis_title="Activation Ratio", yaxis=dict(range=[0, 1.05]), height=350, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig_speed, use_container_width=True)

    with col2:
        fig_kl = go.Figure()
        fig_kl.add_trace(go.Bar(x=list(range(len(kl_div_scores))), y=kl_div_scores, marker_color='purple'))
        fig_kl.update_layout(title="KL Divergence (정보량 차이)", xaxis_title="Generation", yaxis_title="Nats", height=350, showlegend=False)
        st.plotly_chart(fig_kl, use_container_width=True)
else:
    st.info("좌측 사이드바에서 데이터와 파라미터를 설정한 후 '🚀 시뮬레이션 실행' 버튼을 눌러주세요.")