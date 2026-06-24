import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import time
import base64
from q_logic import STATE_S, STATE_I, STATE_R, generate_sbm_network, run_sir_simulation

def svg_img(svg_str, width="100%"):
    b64 = base64.b64encode(svg_str.encode("utf-8")).decode("utf-8")
    st.markdown(
        f'<img src="data:image/svg+xml;base64,{b64}" style="width:{width}; border-radius:8px; margin:8px 0;"/>',
        unsafe_allow_html=True,
    )

SIR_DIAGRAM_SVG = '''<svg viewBox="0 0 620 300" xmlns="http://www.w3.org/2000/svg" style="background:#0e1117; border-radius:10px; font-family:sans-serif;">
  <line x1="60" y1="240" x2="580" y2="240" stroke="#4a5568" stroke-width="1.5"/>
  <line x1="60" y1="30"  x2="60"  y2="240" stroke="#4a5568" stroke-width="1.5"/>
  <text x="14" y="135" fill="#9ca3af" font-size="11" transform="rotate(-90,14,135)" text-anchor="middle">인구 비율</text>
  <text x="320" y="272" fill="#9ca3af" font-size="11" text-anchor="middle">시간 (세대)</text>
  <text x="54" y="244" fill="#6b7280" font-size="10" text-anchor="end">0</text>
  <text x="54" y="139" fill="#6b7280" font-size="10" text-anchor="end">0.5</text>
  <text x="54" y="34"  fill="#6b7280" font-size="10" text-anchor="end">1.0</text>
  <line x1="57" y1="135" x2="63" y2="135" stroke="#4a5568" stroke-width="1"/>
  <line x1="57" y1="32"  x2="63" y2="32"  stroke="#4a5568" stroke-width="1"/>
  <path d="M60,32 C120,32 160,33 200,55 C240,80 260,130 290,175 C320,218 360,235 430,238 C480,239 540,240 580,240"
        fill="none" stroke="#008CFF" stroke-width="2.5"/>
  <path d="M60,238 C100,236 140,220 180,185 C210,158 225,120 240,100 C255,80 265,75 280,80 C295,85 310,100 330,130 C355,165 390,210 440,232 C490,238 540,240 580,240"
        fill="none" stroke="#FF003C" stroke-width="2.5"/>
  <path d="M60,240 C120,240 160,239 200,237 C240,233 270,220 300,200 C330,178 360,148 400,118 C430,95 470,68 520,50 C545,42 565,37 580,35"
        fill="none" stroke="#16AF44" stroke-width="2.5"/>
  <rect x="455" y="48" width="118" height="82" rx="6" fill="#1a1f2e" stroke="#2d3748" stroke-width="1"/>
  <line x1="465" y1="68"  x2="485" y2="68"  stroke="#008CFF" stroke-width="2.5"/>
  <text x="491" y="72" fill="#e2e8f0" font-size="11">Susceptible (S)</text>
  <line x1="465" y1="90"  x2="485" y2="90"  stroke="#FF003C" stroke-width="2.5"/>
  <text x="491" y="94" fill="#e2e8f0" font-size="11">Infected (I)</text>
  <line x1="465" y1="112" x2="485" y2="112" stroke="#16AF44" stroke-width="2.5"/>
  <text x="491" y="116" fill="#e2e8f0" font-size="11">Recovered (R)</text>
  <line x1="265" y1="30" x2="265" y2="240" stroke="#FF003C" stroke-width="1" stroke-dasharray="4,4" opacity="0.5"/>
  <text x="269" y="48" fill="#FF003C" font-size="10" opacity="0.8">감염 최고점 (Peak)</text>
  <text x="320" y="18" fill="#e2e8f0" font-size="13" font-weight="600" text-anchor="middle">SIR 모델 — 시간에 따른 인구 비율 변화</text>
</svg>'''

CIRCUIT_DIAGRAM_SVG = '''<svg viewBox="0 0 720 310" xmlns="http://www.w3.org/2000/svg" style="background:#0e1117; border-radius:10px; font-family:monospace;">
  <text x="360" y="22" fill="#e2e8f0" font-size="13" font-weight="600" text-anchor="middle" font-family="sans-serif">Q-SIR 양자 회로 구성 (2노드 예시 - 하이브리드 모델)</text>
  <text x="8" y="68"  fill="#9ca3af" font-size="11">q₀</text>
  <text x="8" y="80"  fill="#6b7280" font-size="9" font-family="sans-serif">노드A 상태</text>
  <text x="8" y="118" fill="#9ca3af" font-size="11">q₁</text>
  <text x="8" y="130" fill="#6b7280" font-size="9" font-family="sans-serif">노드A 감염</text>
  <text x="8" y="178" fill="#9ca3af" font-size="11">q₂</text>
  <text x="8" y="190" fill="#6b7280" font-size="9" font-family="sans-serif">노드B 상태</text>
  <text x="8" y="228" fill="#9ca3af" font-size="11">q₃</text>
  <text x="8" y="240" fill="#6b7280" font-size="9" font-family="sans-serif">노드B 감염</text>
  <line x1="48" y1="68"  x2="678" y2="68"  stroke="#2d3748" stroke-width="1.5"/>
  <line x1="48" y1="118" x2="678" y2="118" stroke="#2d3748" stroke-width="1.5"/>
  <line x1="48" y1="178" x2="678" y2="178" stroke="#2d3748" stroke-width="1.5"/>
  <line x1="48" y1="228" x2="678" y2="228" stroke="#2d3748" stroke-width="1.5"/>
  <line x1="148" y1="34" x2="148" y2="248" stroke="#2d3748" stroke-width="1" stroke-dasharray="2,3"/>
  <line x1="310" y1="34" x2="310" y2="248" stroke="#2d3748" stroke-width="1" stroke-dasharray="2,3"/>
  <line x1="440" y1="34" x2="440" y2="248" stroke="#2d3748" stroke-width="1" stroke-dasharray="2,3"/>
  <line x1="570" y1="34" x2="570" y2="248" stroke="#2d3748" stroke-width="1" stroke-dasharray="2,3"/>
  <rect x="52" y="56" width="88" height="24" rx="3" fill="#1a2744" stroke="#3b82f6" stroke-width="1"/>
  <text x="96" y="72" fill="#93c5fd" font-size="10" text-anchor="middle" font-family="sans-serif">INIT STATE</text>
  <rect x="52" y="106" width="88" height="24" rx="3" fill="#1a2744" stroke="#3b82f6" stroke-width="1"/>
  <text x="96" y="122" fill="#93c5fd" font-size="10" text-anchor="middle" font-family="sans-serif">INIT STATE</text>
  <rect x="52" y="166" width="88" height="24" rx="3" fill="#1a2744" stroke="#3b82f6" stroke-width="1"/>
  <text x="96" y="182" fill="#93c5fd" font-size="10" text-anchor="middle" font-family="sans-serif">INIT STATE</text>
  <rect x="52" y="216" width="88" height="24" rx="3" fill="#1a2744" stroke="#3b82f6" stroke-width="1"/>
  <text x="96" y="232" fill="#93c5fd" font-size="10" text-anchor="middle" font-family="sans-serif">INIT STATE</text>
  <text x="229" y="38" fill="#fbbf24" font-size="10" text-anchor="middle" font-family="sans-serif">① 회복 &amp; 감염 (RY, CX)</text>
  <text x="375" y="38" fill="#a78bfa" font-size="10" text-anchor="middle" font-family="sans-serif">② 얽힘 RZZ(γ)</text>
  <text x="505" y="38" fill="#f87171" font-size="10" text-anchor="middle" font-family="sans-serif">③ 간섭 RX(γ/2)</text>
  <rect x="160" y="166" width="45" height="24" rx="4" fill="#1a1a2e" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="182.5" y="182" fill="#c4b5fd" font-size="10" text-anchor="middle" font-family="sans-serif">RY(β)</text>
  <circle cx="230" cy="178" r="5" fill="#8b5cf6"/>
  <line x1="230" y1="183" x2="230" y2="220" stroke="#8b5cf6" stroke-width="1.5"/>
  <circle cx="230" cy="228" r="9" fill="none" stroke="#8b5cf6" stroke-width="1.5"/>
  <line x1="221" y1="228" x2="239" y2="228" stroke="#8b5cf6" stroke-width="1.5"/>
  <line x1="230" y1="219" x2="230" y2="237" stroke="#8b5cf6" stroke-width="1.5"/>
  <rect x="210" y="106" width="65" height="24" rx="4" fill="#0d1f0d" stroke="#16AF44" stroke-width="1.5"/>
  <text x="242.5" y="122" fill="#86efac" font-size="9" text-anchor="middle" font-family="sans-serif">RY(α_eff)</text>
  <rect x="349" y="106" width="52" height="24" rx="4" fill="#2a0d0d" stroke="#FF003C" stroke-width="1.5"/>
  <text x="375" y="122" fill="#fca5a5" font-size="10" text-anchor="middle" font-family="sans-serif">RZZ(γ)</text>
  <rect x="349" y="216" width="52" height="24" rx="4" fill="#2a0d0d" stroke="#FF003C" stroke-width="1.5"/>
  <text x="375" y="232" fill="#fca5a5" font-size="10" text-anchor="middle" font-family="sans-serif">RZZ(γ)</text>
  <line x1="375" y1="130" x2="375" y2="216" stroke="#FF003C" stroke-width="1.5" stroke-dasharray="3,2"/>
  <rect x="477" y="106" width="55" height="24" rx="4" fill="#2e1a1a" stroke="#f87171" stroke-width="1.5"/>
  <text x="504.5" y="122" fill="#fca5a5" font-size="9" text-anchor="middle" font-family="sans-serif">RX(γ/2)</text>
  <rect x="477" y="216" width="55" height="24" rx="4" fill="#2e1a1a" stroke="#f87171" stroke-width="1.5"/>
  <text x="504.5" y="232" fill="#fca5a5" font-size="9" text-anchor="middle" font-family="sans-serif">RX(γ/2)</text>
  <rect x="620" y="56"  width="44" height="24" rx="4" fill="#1e1e1e" stroke="#6b7280" stroke-width="1"/>
  <text x="642" y="72"  fill="#9ca3af" font-size="10" text-anchor="middle" font-family="sans-serif">MEAS</text>
  <rect x="620" y="106" width="44" height="24" rx="4" fill="#1e1e1e" stroke="#6b7280" stroke-width="1"/>
  <text x="642" y="122" fill="#9ca3af" font-size="10" text-anchor="middle" font-family="sans-serif">MEAS</text>
  <rect x="620" y="166" width="44" height="24" rx="4" fill="#1e1e1e" stroke="#6b7280" stroke-width="1"/>
  <text x="642" y="182" fill="#9ca3af" font-size="10" text-anchor="middle" font-family="sans-serif">MEAS</text>
  <rect x="620" y="216" width="44" height="24" rx="4" fill="#1e1e1e" stroke="#6b7280" stroke-width="1"/>
  <text x="642" y="232" fill="#9ca3af" font-size="10" text-anchor="middle" font-family="sans-serif">MEAS</text>
  <text x="360" y="270" fill="#6b7280" font-size="10" text-anchor="middle" font-family="sans-serif">큐비트 상태 인코딩:  00 = 정상(S)  ·  01 = 감염(I)  ·  10 = 회복(R)  ·  노드당 2큐비트 할당</text>
  <text x="360" y="288" fill="#4a5568" font-size="9" text-anchor="middle" font-family="sans-serif">α_eff = 고전 계산된 유효 감염률  ·  β = 회복률(Beta)  ·  γ = 얽힘 강도(Gamma)</text>
</svg>'''

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="Q-SIR: 양자 기반 복잡계 시뮬레이터", layout="wide")

# --- 2. 인트로 ---
st.title("Q-SIR: 양자 얽힘으로 사회적 확산을 시뮬레이션하다")
st.markdown("""
2008년 금융위기 당시 리먼 브라더스의 파산은 불과 며칠 만에 전 세계 금융 시스템을 마비시켰습니다.
코로나19 초기, 마스크 품절 소식이 SNS에 퍼지자 수십만 명이 동시에 마트로 몰렸습니다.
이 두 사건의 공통점은 **개인들이 서로를 지켜보다가 어느 임계점에서 한꺼번에 같은 행동을 취했다**는 것입니다.

기존의 확산 모델(고전 SIR)은 이런 현상을 설명하지 못합니다. A가 감염될 확률과 B가 감염될 확률을 **독립적으로** 계산하기 때문에, 사람들이 서로 눈치를 보며 동시에 행동하는 **집단 동조**를 담아낼 수 없습니다.

이 **Q-SIR 시뮬레이터**는 그 문제를 양자 회로의 **얽힘(Entanglement)** 으로 풀어냅니다. 서로 연결된 노드(사람) 사이에 **Rzz 게이트**를 적용해 큐비트를 얽힘 상태로 묶으면, 한 노드의 상태 변화가 연결된 노드 전체에 동시적으로 영향을 주는 상관관계가 만들어집니다. 여기에 **간섭(Interference) 효과**를 추가하여 복잡계의 군집 행동과 급격한 임계 전이를 최종 모사합니다.
""")

# --- 3. 개념 가이드 & 모델 비교 ---
with st.expander("처음 오셨나요? 핵심 개념 가이드 (클릭하여 펼치기)", expanded=False):

    st.markdown("### 이 앱이 다루는 문제")
    st.markdown("""
    전염병, 금융 위기, 가짜뉴스 확산에는 공통된 패턴이 있습니다. 한동안 잠잠하다가 어느 순간을 기점으로 **폭발적으로 전체에 퍼진다**는 것입니다.
    기존 컴퓨터 모델(고전 SIR)은 각 개인의 감염 여부를 독립적인 확률로 따로따로 계산합니다.
    그러나 현실의 사람들은 주변을 보고 따라 행동합니다. 이런 **집단 동조(Herd Behavior)** 는 독립 확률 가정으로는 재현하기 어렵습니다.

    이 시뮬레이터는 그 상관관계를 **양자 얽힘과 위상 간섭**으로 표현합니다.
    """)

    st.divider()
    st.markdown("### 핵심 용어 사전")

    st.markdown("**SIR 모델이란?**")
    st.markdown("""
    네트워크의 각 노드(사람)를 세 가지 상태로 분류해 확산을 추적하는 통계 역학 모델입니다.
    감염병 분석뿐 아니라 금융 리스크 전파, 정보 확산 연구에도 널리 쓰입니다.
    - **S (Susceptible, 정상):** 아직 감염되지 않은 상태
    - **I (Infected, 감염):** 현재 감염된 상태. 주변 S 노드에 영향을 줍니다
    - **R (Recovered, 회복):** 감염 후 회복되어 더 이상 전파하지 않는 상태
    """)
    svg_img(SIR_DIAGRAM_SVG)

    st.markdown("**양자 얽힘(Quantum Entanglement)이란?**")
    st.markdown("""
    두 개 이상의 양자(큐비트)가 서로 연결되어, 한쪽의 상태가 결정되는 순간 나머지의 상태도 즉시 영향을 받는 현상입니다.
    물리적으로 아무리 멀리 떨어져 있어도 이 상관관계는 유지됩니다.

    이 시뮬레이터에서 양자 얽힘은 **사람들 사이의 심리적·사회적 연결**을 모사합니다.
    얽힘이 강할수록 한 노드의 감염이 연결된 노드 전체에 동시적으로 영향을 미칩니다.
    """)

    st.markdown("**Rzz 게이트란?**")
    st.markdown("""
    두 큐비트 사이에 얽힘을 만드는 양자 논리 게이트입니다.
    이 시뮬레이터에서는 서로 연결된 노드(사람) 쌍마다 Rzz 게이트를 적용합니다.
    사이드바의 **얽힘 강도(Gamma)** 슬라이더가 이 게이트의 회전각을 제어합니다.
    값이 클수록 노드 간 상관관계가 강해져 집단 감염 양상이 나타납니다.
    """)

    st.divider()
    st.markdown("### 양자 회로 설계 방식")
    st.markdown("""
    이 시뮬레이터에서 각 노드(사람)는 **2개의 큐비트**로 표현됩니다.
    2개의 큐비트는 00(정상), 01(감염), 10(회복) 세 가지 상태를 인코딩합니다.

    한 세대(Generation)가 진행될 때 회로는 하이브리드 제어 알고리즘을 거쳐 3단계로 동작합니다.

    1. **회복 및 감염 전파** — 감염자 상태의 노드는 RY(β) + CX 조합을 통해 일정 확률로 회복 상태로 이동합니다. 정상 노드는 고전 컴퓨터로 주변 감염자 수를 사전에 분석하여 상쇄 효과를 방지한 RY(α_eff) 회전각을 적용해 안전하게 감염을 구현합니다.
    2. **양자 얽힘** — 연결된 노드 쌍 전체에 Rzz(γ) 게이트를 적용해 이웃 간 사회적 상관관계를 상태 위상(Phase)에 기록합니다.
    3. **간섭 변환** — 마지막으로 모든 감염 대상 비트에 RX(γ/2) 게이트를 추가 인가하여, Rzz로 인해 비틀어진 위상 차이를 현실의 측정 가능한 확률 진폭(Amplitude) 분포로 섞어내어 집단 동조 효과를 추출합니다.
    """)
    svg_img(CIRCUIT_DIAGRAM_SVG)

    st.divider()
    st.markdown("### 기존 SIR 모델 vs Q-SIR 모델의 차이")
    col_vs1, col_vs2 = st.columns(2)
    with col_vs1:
        st.info("**기존 모델 (독립 확률 기반)**")
        st.markdown("""
        * **독립적 확률:** A가 감염될 확률과 B가 감염될 확률을 분리해서 계산합니다.
        * **완만한 확산:** 순차적으로 퍼지기 때문에 비교적 예측 가능한 곡선을 그립니다.
        * **한계:** 다수가 한순간에 동조 행동(Herd Behavior)을 보이며 동시다발적으로 무너지는 현상은 잘 설명하지 못합니다.
        """)
    with col_vs2:
        st.error("**Q-SIR 모델 (얽힘 기반)**")
        st.markdown("""
        * **상관관계 반영:** Rzz 게이트를 통해 얽힌 노드들은 서로의 상태에 영향을 주고받습니다.
        * **급격한 임계 전이:** 일정 수준까지는 완만하다가, 특정 임계치를 넘으면 짧은 시간에 감염이 크게 늘어나는 양상을 보입니다.
        * **활용 예시:** 금융 시스템 리스크나 정보 확산처럼, 상관관계가 강한 시스템의 급변점을 분석하는 데 참고할 수 있습니다.
        """)
st.divider()

# --- 4. 사이드바: 시뮬레이션 컨트롤 패널 ---
with st.sidebar:
    st.header("시뮬레이션 컨트롤 패널")
    st.caption("아래의 3단계를 순서대로 조작하여 직접 결과를 확인해 보세요.")
    
    st.divider()

    st.subheader("Step 1. 시나리오 프리셋 선택")
    scenario = st.radio(
        "분석할 현상의 테마를 먼저 골라주세요.",
        ["기본 사용자 설정", "전염병 집단 발병", "금융 뱅크런 연쇄 도산", "가짜뉴스 에코체임버"],
        index=1 
    )

    if scenario == "전염병 집단 발병":
        st.success("[전염병 시나리오] 밀집 공간에서 접촉 빈도가 높아, 특정 임계점에서 폭발적인 집단 감염이 일어나는 상황을 모사합니다.")
        lbl_alpha, val_alpha = "바이러스 전파력 (Alpha)", np.pi/3
        lbl_beta, val_beta = "치료 및 격리 속도 (Beta)", np.pi/10
        lbl_gamma, val_gamma = "밀집도 및 집단 감염 강도 (Gamma)", np.pi/2.5
        help_alpha = "바이러스 자체의 전염성입니다."
        help_beta = "감염자를 찾아 격리하거나 치료하는 속도입니다."
        help_gamma = "특정 집단 내에서 사람들이 동시에 바이러스에 노출되는 환경적 얽힘 강도입니다."
        
    elif scenario == "금융 뱅크런 연쇄 도산":
        st.warning("[뱅크런 시나리오] 군중 동조 심리(Gamma)가 강해, 남들이 돈을 빼는 것을 보고 연쇄적인 대규모 예금 인출이 일어납니다.")
        lbl_alpha, val_alpha = "공포 심리 자극 (Alpha)", np.pi/4
        lbl_beta, val_beta = "당국의 시장 안정화 (Beta)", np.pi/8
        lbl_gamma, val_gamma = "군중 동조 심리 (Gamma)", np.pi/2.0
        help_alpha = "루머나 악재가 예금자에게 주는 초기 공포감입니다."
        help_beta = "예금자 보호 선언 등 당국의 개입 및 진화 속도입니다."
        help_gamma = "'남들이 돈을 빼니까 나도 빼야 한다'는 군중 심리의 압박감(얽힘)입니다."
        
    elif scenario == "가짜뉴스 에코체임버":
        st.error("[가짜뉴스 시나리오] 알고리즘의 반복 노출과 확증 편향(Gamma)으로 인해 팩트체크가 무력화되며 일제히 퍼지는 현상입니다.")
        lbl_alpha, val_alpha = "알고리즘 노출도 (Alpha)", np.pi/2.5
        lbl_beta, val_beta = "팩트체크 및 차단 (Beta)", np.pi/12
        lbl_gamma, val_gamma = "에코체임버 동조 효과 (Gamma)", np.pi/3
        help_alpha = "자극적인 콘텐츠가 피드에 뜰 확률입니다."
        help_beta = "플랫폼이 허위 정보를 식별하고 제재하는 속도입니다."
        help_gamma = "같은 성향의 유저들이 맹목적으로 정보를 공유하고 휩쓸리는 동조 현상입니다."
        
    else:
        st.info("[기본 설정] 자유롭게 파라미터를 조절하여 양자 회로의 동작을 테스트합니다.")
        lbl_alpha, val_alpha = "감염률 (Alpha)", np.pi/4
        lbl_beta, val_beta = "회복률 (Beta)", np.pi/8
        lbl_gamma, val_gamma = "얽힘 강도 (Gamma)", np.pi/3
        help_alpha = "값이 클수록 감염 노드가 인접 노드에 주는 영향이 커집니다."
        help_beta = "감염 상태에서 회복 상태로 넘어가는 속도입니다."
        help_gamma = "Rzz 게이트 및 Rx 간섭 게이트에 적용되어 얽힘 강도를 결정합니다."

    st.divider()

    st.subheader("Step 2. 네트워크 환경 구성")
    uploaded_file = st.file_uploader("인접 행렬 CSV (옵션)", type=["csv"], help="사용자 정의 네트워크를 업로드합니다. 없으면 10개 노드의 기본 SBM 네트워크를 사용합니다.")
    
    with st.expander("인접 행렬(CSV) 작성 가이드"):
        st.markdown("""
        인접 행렬(Adjacency Matrix)은 사람(노드) 간의 연결 상태를 숫자로 나타낸 표입니다.
        - **1**: 두 사람이 연결됨 (감염/정보 전파 가능)
        - **0**: 연결되지 않음
        
        **[작성 예시]** (엑셀이나 메모장에서 첫 행/열 이름 없이 숫자만 쉼표로 구분)
        ```text
        0,1,0
        1,0,1
        0,1,0
        ```
        """)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=None)
        adj_matrix = df.to_numpy()
        total_nodes = len(adj_matrix)
        
        if total_nodes > 12:
            st.error(f"메모리 한계 초과: 현재 {total_nodes}개의 노드({total_nodes*2} 큐비트)를 입력하셨습니다. 12개 이하만 가능합니다.")
            st.stop()
            
        G = nx.from_numpy_array(adj_matrix)
        st.success(f"{total_nodes}개 노드 데이터 로드 완료")
    else:
        st.info("기본 네트워크(10개 노드) 적용됨")
        total_nodes = 10
        G, _, adj_matrix = generate_sbm_network(nodes=total_nodes)

    pos_3d = nx.spring_layout(G, dim=3, seed=42)

    st.divider()

    st.subheader("Step 3. 세부 조건 및 파라미터 튜닝")
    initial_active = st.multiselect(
        "최초 확산 발원지 (Seed Node)", 
        options=list(range(total_nodes)), 
        default=[0, 1],
        help="0번 세대에서 확산을 시작할 노드입니다."
    )

    S_0 = [STATE_S] * total_nodes
    for node in initial_active:
        S_0[node] = STATE_I

    generations = st.slider("총 관찰 세대 수 (Time Steps)", 5, 40, 20)
    
    st.markdown("**양자 모델 파라미터**")
    st.caption("Step 1에서 선택한 시나리오에 맞춰 초기값이 세팅되었습니다. 슬라이더를 움직여 세부 조정이 가능합니다.")
    
    alpha_param = st.slider(lbl_alpha, 0.0, np.pi, float(val_alpha), step=0.1, help=help_alpha)
    beta_param = st.slider(lbl_beta, 0.0, np.pi, float(val_beta), step=0.1, help=help_beta)
    gamma_param = st.slider(lbl_gamma, 0.0, np.pi, float(val_gamma), step=0.1, help=help_gamma)

    st.markdown("---")
    run_btn = st.button("양자 시뮬레이션 가동", use_container_width=True, type="primary")

# --- 5. 백엔드 연동 및 시각화 ---
@st.cache_data(show_spinner="IBM Quantum Qiskit 회로를 구성하고 연산을 수행 중입니다...")
def fetch_sir_data(_adj_matrix, _initial_state, gen, alpha, beta, gamma):
    return run_sir_simulation(_adj_matrix, _initial_state, alpha, beta, gamma, generations=gen)

if run_btn:
    st.session_state['history'] = fetch_sir_data(adj_matrix, S_0, generations, alpha_param, beta_param, gamma_param)
    st.session_state['history_classical'] = fetch_sir_data(adj_matrix, S_0, generations, alpha_param, beta_param, 0.0)
    st.session_state['gamma_used'] = gamma_param

if 'history' in st.session_state:
    history_data = st.session_state['history']
    history_classical = st.session_state.get('history_classical')
    
    tab_sim, tab_stat = st.tabs(["3D 확산 애니메이션 뷰어", "분석 리포트"])

    with tab_sim:
        st.markdown("#### 네트워크 내 감염 확산 시각화")
        st.caption("우측의 재생 버튼을 누르면 확산 과정이 애니메이션으로 재생됩니다. 마우스로 드래그하면 3D 공간을 회전할 수 있습니다.")
        
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([4, 1, 1])
        with col_ctrl1:
            current_gen = st.slider("타임라인 스크롤", 0, generations, 0, step=1, key="gen_slider", label_visibility="collapsed")
        with col_ctrl2:
            play_button = st.button("3D 자동 재생", use_container_width=True)
        with col_ctrl3:
            stop_button = st.button("리셋", use_container_width=True)

        metric_container = st.empty()
        plot_container = st.empty()

        def draw_3d_network(G, pos_3d, state_vector):
            color_map = {STATE_S: "#008CFF", STATE_I: '#FF003C', STATE_R: "#16AF44"}
            node_sizes = [25 if state == STATE_I else 15 for state in state_vector]

            edge_x, edge_y, edge_z = [], [], []
            for edge in G.edges():
                x0, y0, z0 = pos_3d[edge[0]]
                x1, y1, z1 = pos_3d[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_z.extend([z0, z1, None])

            edge_trace = go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                line=dict(color='rgba(255,255,255,0.2)', width=2),
                hoverinfo='none', mode='lines'
            )

            state_groups = {
                STATE_S: dict(name='정상 (S)', color='#008CFF', size=15),
                STATE_I: dict(name='감염 (I)', color='#FF003C', size=25),
                STATE_R: dict(name='회복 (R)', color='#16AF44', size=15),
            }
            node_traces = []
            for state, style in state_groups.items():
                xs, ys, zs, texts = [], [], [], []
                for node in G.nodes():
                    if state_vector[node] == state:
                        x, y, z = pos_3d[node]
                        xs.append(x); ys.append(y); zs.append(z)
                        texts.append(f'Node {node}')
                if not xs:
                    xs, ys, zs, texts = [None], [None], [None], ['']
                node_traces.append(go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode='markers', name=style['name'],
                    hoverinfo='text', text=texts,
                    marker=dict(color=style['color'], size=style['size'], line=dict(color='white', width=1), opacity=0.9)
                ))

            fig = go.Figure(data=[edge_trace] + node_traces)
            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=0),
                scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor='rgba(0,0,0,0)'),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500,
                legend=dict(x=0, y=1, font=dict(color='#E2E8F0'), bgcolor='rgba(0,0,0,0)')
            )
            return fig

        def render_frame(t):
            current_state = np.array(history_data[t])
            s_cnt, i_cnt, r_cnt = np.sum(current_state == STATE_S), np.sum(current_state == STATE_I), np.sum(current_state == STATE_R)
            prev_i_cnt = np.sum(np.array(history_data[max(0, t-1)]) == STATE_I)
            delta_i = i_cnt - prev_i_cnt
            delta_str = f"▲ {abs(delta_i)}" if delta_i > 0 else (f"▼ {abs(delta_i)}" if delta_i < 0 else "")
            delta_color = "#ef4444" if delta_i > 0 else ("#22c55e" if delta_i < 0 else "#9ca3af")

            with metric_container.container():
                st.markdown(f"""
                <div style="display:flex; gap:12px; margin-bottom:8px;">
                    <div style="flex:1; background:#1e1e2e; border-radius:8px; padding:16px 20px;">
                        <div style="font-size:13px; color:#9ca3af; margin-bottom:6px;">진행 세대</div>
                        <div style="font-size:28px; font-weight:600; color:#e2e8f0;">Gen {t}</div>
                    </div>
                    <div style="flex:1; background:#0d2a4a; border-left:4px solid #008CFF; border-radius:8px; padding:16px 20px;">
                        <div style="font-size:13px; color:#008CFF; margin-bottom:6px;">Susceptible (정상)</div>
                        <div style="font-size:28px; font-weight:600; color:#e2e8f0;">{s_cnt}</div>
                    </div>
                    <div style="flex:1; background:#3a0a14; border-left:4px solid #FF003C; border-radius:8px; padding:16px 20px;">
                        <div style="font-size:13px; color:#FF003C; margin-bottom:6px;">Infected (감염자)</div>
                        <div style="font-size:28px; font-weight:600; color:#e2e8f0;">
                            {i_cnt} <span style="font-size:14px; color:{delta_color};">{delta_str}</span>
                        </div>
                    </div>
                    <div style="flex:1; background:#0a2e14; border-left:4px solid #16AF44; border-radius:8px; padding:16px 20px;">
                        <div style="font-size:13px; color:#16AF44; margin-bottom:6px;">Recovered (회복)</div>
                        <div style="font-size:28px; font-weight:600; color:#e2e8f0;">{r_cnt}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            fig = draw_3d_network(G, pos_3d, history_data[t])
            plot_container.plotly_chart(fig, use_container_width=True, key=f"anim_{t}")

        if play_button:
            for t in range(generations + 1):
                render_frame(t)
                time.sleep(0.3)
        else:
            render_frame(current_gen)

        if stop_button:
            st.rerun()

    with tab_stat:
        # 데이터 처리
        count_S, count_I, count_R = np.zeros(generations+1), np.zeros(generations+1), np.zeros(generations+1)
        for t in range(generations + 1):
            state_array = np.array(history_data[t])
            count_S[t], count_I[t], count_R[t] = np.sum(state_array == STATE_S), np.sum(state_array == STATE_I), np.sum(state_array == STATE_R)

        ratio_S, ratio_I, ratio_R = count_S / total_nodes, count_I / total_nodes, count_R / total_nodes
        x_generations = list(range(generations + 1))
        
        peak_gen = np.argmax(ratio_I)
        peak_ratio = ratio_I[peak_gen] * 100

        delta_I = np.diff(ratio_I) * 100
        max_delta = np.max(delta_I) if len(delta_I) > 0 else 0
        max_delta_gen = np.argmax(delta_I) + 1 if len(delta_I) > 0 else 0

        if history_classical is not None:
            count_S_c, count_I_c, count_R_c = np.zeros(generations+1), np.zeros(generations+1), np.zeros(generations+1)
            for t in range(generations + 1):
                state_array_c = np.array(history_classical[t])
                count_S_c[t], count_I_c[t], count_R_c[t] = np.sum(state_array_c == STATE_S), np.sum(state_array_c == STATE_I), np.sum(state_array_c == STATE_R)
            ratio_S_c, ratio_I_c, ratio_R_c = count_S_c / total_nodes, count_I_c / total_nodes, count_R_c / total_nodes
            peak_gen_c = np.argmax(ratio_I_c)
            peak_ratio_c = ratio_I_c[peak_gen_c] * 100
            diff_ratio = peak_ratio - peak_ratio_c
        else:
            peak_ratio_c = 0
            peak_gen_c = 0
            diff_ratio = 0

        st.markdown("### 시뮬레이션 심층 분석 리포트")
        st.caption("수치 결과가 현실 시나리오에서 어떤 의미를 가지는지, 양자 얽힘(집단 동조)이 확산에 미친 영향을 심층 해석합니다.")
        
        st.markdown("#### 1. 핵심 지표 요약")
        cmp_m1, cmp_m2, cmp_m3, cmp_m4 = st.columns(4)
        cmp_m1.metric("Q-SIR 최대 감염 비율", f"{peak_ratio:.1f}%", help=f"제 {peak_gen}세대에 도달")
        if history_classical is not None:
            cmp_m2.metric("고전 모델 최대 감염", f"{peak_ratio_c:.1f}%", help=f"제 {peak_gen_c}세대에 도달")
            cmp_m3.metric("동조 효과에 의한 초과치", f"{diff_ratio:+.1f}%p", help="군중 동조(Gamma)로 인해 추가로 발생한 감염/인출 비율")
        else:
            cmp_m2.metric("고전 모델 최대 감염", "N/A")
            cmp_m3.metric("동조 효과 초과치", "N/A")
        cmp_m4.metric("순간 최대 확산 속도", f"▲ {max_delta:.1f}%p / 세대", help=f"제 {max_delta_gen}세대에 발생 (1세대 만에 급증한 비율)")

        st.divider()

        st.markdown("#### 2. 동조 현상 및 임계 전이 분석")
        
        threshold_velocity = 15.0 
        threshold_diff = 10.0     

        is_velocity_critical = max_delta >= threshold_velocity
        is_peak_critical = (diff_ratio >= threshold_diff) if history_classical else False

        if is_velocity_critical or is_peak_critical:
            st.error(f"**[경고] 임계 전이(Phase Transition) 발생 감지**")
            st.markdown(f"> **데이터 해석:** 제 {max_delta_gen}세대를 기점으로 단 1세대 만에 수치가 **{max_delta:.1f}%p 폭증**하며 시스템이 임계점을 넘었습니다. "
                        f"각 노드가 독립적으로 행동했다면 {peak_ratio_c:.1f}% 수준에서 방어되었겠지만, "
                        f"현재 설정된 동조 강도(Gamma={st.session_state.get('gamma_used', 0):.2f})가 연쇄 반응의 도화선이 되어 전체의 **{peak_ratio:.1f}%까지 초과 확산**되었습니다.")

            st.markdown("**현실 시나리오 적용 시사점:**")
            if "전염병" in scenario:
                st.markdown("- **의료 체계 붕괴 위험:** 밀폐/밀집된 환경에서 방역망이 뚫리면서 통제 불능의 **동시다발적 집단 감염(Super-spreading)**이 발생했습니다.\n"
                            "- **대응 방안:** 확산세가 꺾이기 전에 병상(Beta)이 마비될 수 있습니다. 현재의 치료 속도로는 역부족이므로, 물리적인 봉쇄(Lockdown)를 통해 사람 간의 얽힘(접촉 강도) 자체를 차단해야 합니다.")
            elif "뱅크런" in scenario:
                st.markdown("- **유동성 고갈 및 연쇄 도산:** 초기 루머(Alpha)는 작았을지라도, 불안감이 얽혀(Gamma) **'남들이 빼니까 나도 당장 빼야 한다'는 군중 심리가 폭발**했습니다.\n"
                            "- **대응 방안:** 제 {max_delta_gen}세대 부근에서 예금자들의 뱅크런이 집중되었습니다. 현재 당국이 제공하는 예금자 보호나 유동성 지원(Beta) 속도로는 시장 붕괴를 막지 못했습니다. 보다 강력한 지급 보증 선언이 즉각적으로 필요합니다.")
            elif "가짜뉴스" in scenario:
                st.markdown("- **에코체임버 폭발:** 특정 커뮤니티나 인플루언서를 중심으로 형성된 확증 편향이 임계점을 넘었습니다.\n"
                            "- **대응 방안:** 플랫폼의 팩트체크 및 필터링(Beta)이 개입하기도 전에, 유사한 성향을 가진 유저들이 일제히 정보를 공유(Gamma)하여 진실 여부와 무관하게 기정사실로 굳어져 버렸습니다. 알고리즘 노출 구조 자체의 개편이 필요합니다.")
            else:
                st.markdown("- 노드 간 얽힘(상관관계)이 폭발하여 시스템 전체의 상태가 순식간에 붕괴되는 복잡계의 **'눈사태(Avalanche)' 현상**이 확인되었습니다. 파라미터 조정이 필요합니다.")
                
        else:
            st.success(f"**[안정] 점진적이고 통제 가능한 확산**")
            st.markdown(f"> **데이터 해석:** 급격한 집단 동조나 폭발적인 임계 전이는 관찰되지 않았습니다. 최대 확산 속도가 세대당 {max_delta:.1f}%p 수준으로 억제되고 있으며, "
                        f"고전적인 독립 확률 모델의 궤적과 큰 차이가 없습니다. 얽힘(Gamma)의 힘이 확산을 주도하지 못했습니다.")

            st.markdown("**현실 시나리오 적용 시사점:**")
            if "전염병" in scenario:
                st.markdown("- 방역 및 격리 조치(Beta)가 효과적으로 작동하여 감염 재생산지수가 안정적으로 유지되고 있습니다.\n"
                            "- 집단 내 밀접 접촉에 의한 대규모 집단 감염(Pandemic)은 방어된 상태입니다.")
            elif "뱅크런" in scenario:
                st.markdown("- 부분적인 예금 인출이나 부실 우려 노출은 있었으나, 시장 전체의 패닉(Gamma)으로 전이되지 않았습니다.\n"
                            "- 금융 당국의 빠른 개입(Beta)과 건전성에 대한 신뢰가 예금자들의 동조 인출을 성공적으로 차단했습니다.")
            elif "가짜뉴스" in scenario:
                st.markdown("- 특정 허위 정보가 일부 노출(Alpha)되었으나, 유저들이 맹목적으로 동조하고 확산시키기 전에 플랫폼의 팩트체크(Beta)가 적절히 작동했습니다.\n"
                            "- 심각한 여론 왜곡이나 에코체임버 형성을 막아냈습니다.")

        st.divider()

        st.markdown("#### 3. 세부 확산 추이 시각화")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("**시간에 따른 SIR 확산 곡선**")
            fig_sir_line = go.Figure()
            fig_sir_line.add_trace(go.Scatter(x=x_generations, y=ratio_S, mode='lines+markers', name='정상 (S)', line=dict(color='#008CFF', width=2)))
            fig_sir_line.add_trace(go.Scatter(x=x_generations, y=ratio_I, mode='lines+markers', name='감염 (I)', line=dict(color='#FF003C', width=4)))
            fig_sir_line.add_trace(go.Scatter(x=x_generations, y=ratio_R, mode='lines+markers', name='회복 (R)', line=dict(color='#16AF44', width=2)))
            fig_sir_line.update_layout(xaxis_title="Generation", yaxis_title="인구 비율", yaxis=dict(range=[0, 1.05]), height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#E2E8F0'))
            st.plotly_chart(fig_sir_line, use_container_width=True)

        with col_c2:
            st.markdown("**세대별 상태 누적 비율 (Area Chart)**")
            fig_sir_area = go.Figure()
            fig_sir_area.add_trace(go.Scatter(x=x_generations, y=ratio_R, mode='lines', name='회복 (R)', line=dict(width=0.5, color='#16AF44'), stackgroup='one', fillcolor='rgba(22, 175, 68, 0.7)'))
            fig_sir_area.add_trace(go.Scatter(x=x_generations, y=ratio_I, mode='lines', name='감염 (I)', line=dict(width=0.5, color='#FF003C'), stackgroup='one', fillcolor='rgba(255, 0, 60, 0.7)'))
            fig_sir_area.add_trace(go.Scatter(x=x_generations, y=ratio_S, mode='lines', name='정상 (S)', line=dict(width=0.5, color='#008CFF'), stackgroup='one', fillcolor='rgba(0, 140, 255, 0.5)'))
            fig_sir_area.update_layout(xaxis_title="Generation", yaxis_title="누적 비율", yaxis=dict(range=[0, 1.0]), height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#E2E8F0'))
            st.plotly_chart(fig_sir_area, use_container_width=True)

        st.divider()

        st.markdown("#### 4. 고전 모델(독립 확률, Gamma=0)과의 비교")
        st.caption("같은 조건에서 얽힘(Gamma)만 0으로 설정했을 때의 결과를 겹쳐서 보여줍니다. 옅은 색이 현재 Q-SIR 결과, 진한 점선이 고전 모델입니다.")

        if history_classical is not None:
            fig_overlay = go.Figure()

            fig_overlay.add_trace(go.Scatter(x=x_generations, y=ratio_S, mode='lines', name='정상 (S) · Q-SIR', line=dict(color='#008CFF', width=2), opacity=0.25))
            fig_overlay.add_trace(go.Scatter(x=x_generations, y=ratio_I, mode='lines', name='감염 (I) · Q-SIR', line=dict(color='#FF003C', width=5), opacity=0.3))
            fig_overlay.add_trace(go.Scatter(x=x_generations, y=ratio_R, mode='lines', name='회복 (R) · Q-SIR', line=dict(color='#16AF44', width=2), opacity=0.25))

            fig_overlay.add_trace(go.Scatter(x=x_generations, y=ratio_S_c, mode='lines+markers', name='정상 (S) · 고전', line=dict(color='#008CFF', width=2, dash='dot')))
            fig_overlay.add_trace(go.Scatter(x=x_generations, y=ratio_I_c, mode='lines+markers', name='감염 (I) · 고전', line=dict(color='#FF003C', width=3, dash='dot')))
            fig_overlay.add_trace(go.Scatter(x=x_generations, y=ratio_R_c, mode='lines+markers', name='회복 (R) · 고전', line=dict(color='#16AF44', width=2, dash='dot')))

            fig_overlay.update_layout(
                xaxis_title="Generation", yaxis_title="인구 비율", yaxis=dict(range=[0, 1.05]),
                height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E2E8F0'), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
            )
            st.plotly_chart(fig_overlay, use_container_width=True)
        else:
            st.warning("비교 데이터가 없습니다. 사이드바에서 시뮬레이션을 다시 실행해 주세요.")

        st.divider()
        df_data = {"Generation": x_generations, "Susceptible": count_S, "Infected": count_I, "Recovered": count_R}
        if history_classical is not None:
            df_data.update({
                "Susceptible_Classical": count_S_c,
                "Infected_Classical": count_I_c,
                "Recovered_Classical": count_R_c,
            })
        df_result = pd.DataFrame(df_data)
        csv = df_result.to_csv(index=False).encode('utf-8-sig')
        st.download_button(label="시뮬레이션 결과 데이터 다운로드 (CSV 추출)", data=csv, file_name='q_sir_simulation_result.csv', mime='text/csv', use_container_width=True)

else:
    st.info("좌측 사이드바 패널에서 설정을 완료한 후, [양자 시뮬레이션 가동] 버튼을 눌러주세요.")