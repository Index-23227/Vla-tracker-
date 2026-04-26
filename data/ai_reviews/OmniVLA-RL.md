# OmniVLA-RL: A Vision-Language-Action Model with Spatial Understanding and Online RL

> **한 줄 요약**: Reasoning/spatial/action 전문가를 Mix-of-Transformers로 통합하고, flow matching을 SDE로 재정식화한 Flow-GSPO를 GSPO와 결합하여 VLA의 공간 이해와 online RL 안정성을 동시에 끌어올리려는 시도.

---

## 1. 배경 및 동기

- 기존 VLA는 공간 인지가 부정확하고 멀티모달 융합이 suboptimal, RL fine-tuning 시 불안정성이 고질적 문제
- 하나의 monolithic transformer에 reasoning, spatial grounding, action generation을 모두 맡기면 expert 특화가 어려움
- Flow matching 기반 action head를 RL로 업데이트할 때 policy gradient와의 호환성 확보가 challenging

---

## 2. 방법론

### Mix-of-Transformers (MoT) 아키텍처
Reasoning expert, spatial expert, action expert를 별도의 transformer 모듈로 분리하여 각 modality/기능에 특화된 표현을 학습. 전문가 간 synergy를 통해 공간 인지와 multimodal fusion 정확도를 개선.

### Flow-GSPO
Flow matching을 Stochastic Differential Equation(SDE) process로 재정식화하고, 이를 Group Segmented Policy Optimization(GSPO)과 결합. SDE 형태의 샘플링이 RL 탐색에 필요한 stochasticity를 제공하면서 policy gradient 계산이 tractable해지는 구조.

---

## 3. 실험 결과

### LIBERO (Table 1)

| 방법 | Spatial | Object | Goal | Long | Avg |
|------|--------:|-------:|-----:|-----:|----:|
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| F1 | 98.2 | 97.8 | 95.4 | 91.3 | 95.7 |
| **OmniVLA-RL** | **99.2** | **99.2** | **98.5** | **93.5** | **97.6** |

- LIBERO 4개 suite 모두 1위, 평균 **97.6%** — 기존 SOTA π0.5 대비 +0.7pp.
- 대표적으로 LIBERO-Long에서 +1.1pp 향상 (compounding error에 민감한 long-horizon 태스크에서 robust).

### LIBERO-Plus Ablation (Table 2)

| 설정 | 성공률 | Δ vs SFT |
|------|-------:|---------:|
| SFT only | 41.2% | — |
| + PPO | 78.7% | +37.5 |
| + GRPO | 65.7% | +24.5 |
| **+ Flow-GSPO (본 논문)** | **80.3%** | **+39.1** |
| w/o Spatial Expert | 32.9% | −8.3 |

- Flow-GSPO이 PPO/GRPO를 각각 +1.6pp, +14.6pp 상회.
- Spatial Expert 제거 시 41.2 → 32.9 (-8.3pp): 공간 이해가 MoT 구조의 핵심 기여.

### 실험 하이퍼파라미터

- Action chunk H=16, K=10 denoising steps
- Flow-GSPO: group G=8, 클리핑 ε=0.2, KL β=0.01, σ_max=0.1
- 200 RL 업데이트 스텝, rollout buffer 10 스텝마다 갱신

---

## 4. 한계 및 미해결 문제

1. **Sim-only 검증** (저자 직접 인정, Sec 7): "primarily been validated within high-fidelity simulation environments" — 실물 로봇 sim-to-real gap 미검증.
2. **World model 부재** (저자 직접 인정): long-horizon 구조적 reasoning을 위한 dedicated world model 없음.
3. **MoT 오버헤드**: 3개 expert 분리 시 파라미터/추론 비용 증가 가능성, Mixture-of-Experts sparsity 활용 여부 불명확.
4. **Code/weights 미공개**: 논문에 GitHub URL 또는 release 언급 없음.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★☆☆ — Flow matching의 SDE 재정식화 + GSPO 결합은 신선하지만 MoE-style VLA 아이디어는 선행 연구 존재 |
| **Practical impact** | ★★★☆☆ — LIBERO 수치 및 real-robot 검증 공개 여부에 따라 재평가 필요 |

---

## 6. 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Flow matching을 SDE로 바꾸는 이유는? | Deterministic ODE 형태로는 on-policy RL에 필요한 action distribution log-prob 계산/탐색이 어려움. SDE는 stochasticity와 tractable score-based policy gradient를 동시에 제공. |
| 2 | MoT가 기존 MoE-VLA와 다른 점은? | Token-level routing 대신 reasoning/spatial/action이라는 기능 축으로 expert를 분리. 라우팅 불안정성을 회피하는 대신 expert 개수가 고정됨. |
| 3 | GSPO의 Group Segmented는 무엇을 segmenting? | Paper Sec 4.4 확인: action block A_t를 sequence-level unit으로 취급하고 각 block 내에서 group sampling + segmented optimization. Token-level GRPO가 importance ratio 설계 결함으로 collapse하는 문제를 action-block 수준에서 회피. |

<!-- VERIFIED: pdf -->
