import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import time
from q_logic import STATE_S, STATE_I, STATE_R, generate_sbm_network, run_sir_simulation

st.set_page_config(page_title="Quantum SIR Simulator", layout="wide")
st.title("🌐 양자 얽힘 기반 SIR 전염병 확산 대시보드")
st.markdown("자신의 네트워크 데이터(인접 행렬 CSV)를 업로드하고 양자 얽힘이 적용된 전파-회복 모델을 시뮬레이션해 보세요.")

# --- 1. 사이드바: 데이터 입력 및 파라미터 ---
st.sidebar.header("📁 1. 네트워크 입력")
uploaded_file = st.sidebar.file_uploader("인접 행렬 CSV 업로드", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=None)
    adj_matrix = df.to_numpy()
    total_nodes = len(adj_matrix)
    G = nx.from_numpy_array(adj_matrix)
    pos = nx.spring_layout(G, seed=42)
    st.sidebar.success(f"✅ {total_nodes}개 노드 로드 완료!")
else:
    st.sidebar.info("기본 10개 노드 SBM 네트워크를 사용합니다.")
    total_nodes = 10
    G, pos, adj_matrix = generate_sbm_network(nodes=total_nodes)

st.sidebar.header("⚙️ 2. 시뮬레이션 설정")
# 다중 초기 감염자 선택 UI
initial_active = st.sidebar.multiselect(
    "최초 감염 노드(Seed) 선택", 
    options=list(range(total_nodes)), 
    default=[0]
)

# 초기 상태 벡터(S_0) 생성: 모두 S(0)로 두고 선택된 노드만 I(1)로 변경
S_0 = [STATE_S] * total_nodes
for node in initial_active:
    S_0[node] = STATE_I

generations = st.sidebar.slider("총 세대 수", 5, 30, 15)
alpha_param = st.sidebar.slider("감염률 (Alpha)", 0.0, np.pi, np.pi/4, step=0.1)
beta_param = st.sidebar.slider("회복률 (Beta)", 0.0, np.pi, np.pi/8, step=0.1)
gamma_param = st.sidebar.slider("얽힘 강도 (Gamma)", 0.0, np.pi, np.pi/6, step=0.1)

# --- 2. 양자 시뮬레이션 실행 ---
@st.cache_data(show_spinner="Qiskit 양자 회로 시뮬레이션 중...")
def fetch_sir_data(_adj_matrix, _initial_state, gen, alpha, beta, gamma):
    return run_sir_simulation(_adj_matrix, _initial_state, alpha, beta, gamma, generations=gen)

if st.sidebar.button("🚀 시뮬레이션 실행"):
    st.session_state['history'] = fetch_sir_data(adj_matrix, S_0, generations, alpha_param, beta_param, gamma_param)

# --- 3. 대시보드 렌더링 (타임라인 및 애니메이션) ---
if 'history' in st.session_state:
    history_data = st.session_state['history']

    st.subheader("타임라인 뷰어 (자동 재생)")
    col_slider, col_btn1, col_btn2 = st.columns([3, 1, 1])
    
    with col_slider:
        current_gen = st.slider("세대(Generation) 확인", 0, generations, 0, step=1, key="gen_slider")
    with col_btn1:
        play_button = st.button("▶️ 자동 재생")
    with col_btn2:
        stop_button = st.button("⏹️ 정지 (초기화)")

    plot_container = st.empty()
    status_container = st.empty()

    def plot_network_step(G, pos, state_vector):
        """SIR 상태에 따라 노드 색상을 렌더링하는 그래프 함수"""
        color_map = {STATE_S: '#1f77b4', STATE_I: '#ff4b4b', STATE_R: '#2ca02c'}
        node_colors = [color_map[state] for state in state_vector]
        
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
            
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="bottom center", 
            hoverinfo='text', marker=dict(color=node_colors, size=30, line_width=2)
        )
        fig = go.Figure(data=[edge_trace, node_trace], 
                        layout=go.Layout(showlegend=False, margin=dict(b=20,l=5,r=5,t=20),
                                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), 
                                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=400))
        return fig

    # 기본 및 애니메이션 렌더링
    if play_button:
        for t in range(generations + 1):
            status_container.markdown(f"**현재 상태 (Gen {t}):** `{history_data[t]}`")
            fig = plot_network_step(G, pos, history_data[t])
            plot_container.plotly_chart(fig, use_container_width=True, key=f"anim_{t}")
            time.sleep(0.5)
    else:
        status_container.markdown(f"**현재 상태 (Gen {current_gen}):** `{history_data[current_gen]}`")
        plot_container.plotly_chart(plot_network_step(G, pos, history_data[current_gen]), use_container_width=True, key="base_chart")
        
    if stop_button:
        st.rerun()

    # --- 4. SIR 통계 대시보드 ---
    st.markdown("---")
    st.subheader("📊 SIR 확산 커브 및 상태 분포")

    count_S, count_I, count_R = np.zeros(generations+1), np.zeros(generations+1), np.zeros(generations+1)
    
    for t in range(generations + 1):
        state_array = np.array(history_data[t])
        count_S[t] = np.sum(state_array == STATE_S)
        count_I[t] = np.sum(state_array == STATE_I)
        count_R[t] = np.sum(state_array == STATE_R)

    ratio_S = count_S / total_nodes
    ratio_I = count_I / total_nodes
    ratio_R = count_R / total_nodes
    x_generations = list(range(generations + 1))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**1. 시간에 따른 SIR 커브**")
        fig_sir_line = go.Figure()
        fig_sir_line.add_trace(go.Scatter(x=x_generations, y=ratio_S, mode='lines+markers', name='Susceptible (파랑)', line=dict(color='#1f77b4', width=2)))
        fig_sir_line.add_trace(go.Scatter(x=x_generations, y=ratio_I, mode='lines+markers', name='Infected (빨강)', line=dict(color='#ff4b4b', width=3)))
        fig_sir_line.add_trace(go.Scatter(x=x_generations, y=ratio_R, mode='lines+markers', name='Recovered (초록)', line=dict(color='#2ca02c', width=2)))
        fig_sir_line.update_layout(xaxis_title="Generation", yaxis_title="Ratio", yaxis=dict(range=[0, 1.05]), height=400, legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
        st.plotly_chart(fig_sir_line, use_container_width=True)

    with col2:
        st.markdown("**2. 세대별 상태 누적 비율**")
        fig_sir_area = go.Figure()
        fig_sir_area.add_trace(go.Scatter(x=x_generations, y=ratio_R, mode='lines', name='Recovered', line=dict(width=0.5, color='#2ca02c'), stackgroup='one', fillcolor='rgba(44, 160, 44, 0.5)'))
        fig_sir_area.add_trace(go.Scatter(x=x_generations, y=ratio_I, mode='lines', name='Infected', line=dict(width=0.5, color='#ff4b4b'), stackgroup='one', fillcolor='rgba(255, 75, 75, 0.5)'))
        fig_sir_area.add_trace(go.Scatter(x=x_generations, y=ratio_S, mode='lines', name='Susceptible', line=dict(width=0.5, color='#1f77b4'), stackgroup='one', fillcolor='rgba(31, 119, 180, 0.5)'))
        fig_sir_area.update_layout(xaxis_title="Generation", yaxis_title="Cumulative Ratio", yaxis=dict(range=[0, 1.0]), height=400, legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
        st.plotly_chart(fig_sir_area, use_container_width=True)

    st.markdown("---")
    peak_infection_gen = np.argmax(ratio_I)
    peak_infection_ratio = ratio_I[peak_infection_gen] * 100
    st.write(f"**💡 시뮬레이션 인사이트:**")
    st.write(f"- 감염 절정기(Peak): **{peak_infection_gen}세대** (전체 노드의 **{peak_infection_ratio:.1f}%** 동시 감염)")
    st.write(f"- 최종 면역/회복 비율: 전체 노드의 **{ratio_R[-1] * 100:.1f}%**")

else:
    st.info("좌측 사이드바에서 데이터와 파라미터를 설정한 후 '🚀 시뮬레이션 실행' 버튼을 눌러주세요.")