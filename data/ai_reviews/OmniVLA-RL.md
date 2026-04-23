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

> 논문 PDF 미검증 (abstract-only). 구체 수치는 paper 참조 필요.

- Abstract상 LIBERO / LIBERO-Plus에서 평가 수행됨이 언급됨. 정량 수치는 PDF 확인 후 업데이트 예정.

---

## 4. 한계 및 미해결 문제

1. **MoT 오버헤드**: 3개 expert 분리 시 파라미터/추론 비용 증가 가능성, Mixture-of-Experts sparsity 활용 여부 불명확.
2. **Flow-GSPO 일반성**: 다른 action head (autoregressive, diffusion)와 비교했을 때 이득의 원천이 SDE 재정식화인지 GSPO인지 분리되지 않음.
3. **Spatial expert ground-truth**: 공간 이해를 감독하는 신호(depth, 3D annotation 등)가 abstract에서 언급되지 않아 unclear.

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
| 3 | GSPO의 Group Segmented는 무엇을 segmenting? | Action chunk 또는 trajectory segment 단위로 advantage를 그룹화하여 variance를 줄이는 variant로 추정. PDF 확인 필요. |

<!-- VERIFIED: abstract-only -->
