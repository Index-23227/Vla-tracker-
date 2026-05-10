# VLA-ATTC: Adaptive Test-Time Compute for VLA Models with Relative Action Critic Model 세미나 리뷰

## 1. 연구 배경 및 핵심 아이디어

VLA-ATTC(Li et al., 2026, arXiv:2605.01194, University of Sydney / Central South University / Southeast University / UNSW / SenseTime Research, ICML 2026)는 기존 VLA 모델의 "fast, instinctive" 추론 paradigm이 복잡하거나 모호한 상황에서 catastrophic failure를 일으키는 한계를 지적하고, **adaptive test-time compute(TTC)** 로 이를 극복하는 plug-and-play framework를 제안한다.

핵심 동기는 LLM의 System 1 → System 2 deliberation 흐름을 VLA에 도입하되, (i) sequential CoT 방식이 가지는 fine-tuning/annotation 비용 문제와 (ii) 기존 parallel best-of-N + critic 방식이 가지는 indiscriminate, high-cost, unstable absolute scoring 문제를 모두 해결하는 것이다.

세 가지 challenge를 명시적으로 정의:
1. 상황별 난이도를 정량화하여 deliberation을 **adaptive**하게 trigger
2. multi-action sampling의 computational overhead 최소화
3. 가볍지만 정확한 critic model 설계

이를 위해 (a) DTW 기반 "Cognitive Clutch" uncertainty quantifier, (b) shared pre-fill을 통한 효율적 candidate sampling, (c) pairwise comparison 기반 **Relative Action Critic(RAC)** 모델, (d) ODE step 변조를 이용한 자동 preference pair curation pipeline을 통합한다.

## 2. 아키텍처: Cognitive Clutch + Relative Action Critic

VLA-ATTC는 base VLA(논문에서는 PI0와 PI0.5 사용)를 수정하지 않는 **plug-and-play wrapper**다.

**Cognitive Clutch (uncertainty quantification)**: 매 timestep t에서 동일한 VLM context Ct를 공유하면서 두 개의 random seed z1, z2로 action chunk a1, a2를 생성한다. 두 chunk(각각 길이 H의 pose sequence) 간의 일관성을 Dynamic Time Warping(DTW) cumulative cost matrix Γ(i,j)로 계산하고 Γ(H,H)를 uncertainty score Ut로 사용한다. 이 Ut가 threshold τ(K-th percentile, default 80th)를 넘으면 TTC phase trigger.

**Efficient Parallel Sampling**: VLM forward pass(pre-filling)가 PI0 기준 86ms 중 59ms를 차지하는 비대칭 비용 구조를 활용. pre-fill은 1회만 수행하고 action head만 N번 stochastic sampling — 여러 candidate를 거의 무료로 얻는다.

**Relative Action Critic (RAC)**: VLM backbone과 같은 depth L의 Transformer. 입력은 4개의 MLP를 거친 (Action_i, Action_j, Action_diff = i−j, proprioceptive state). 각 layer가 **3-branch attention**으로 구성:
- Self-Attention: RAC 내부 representation
- Raw Cross-Attention: 동일 layer l의 VLM raw features F_vlm^l
- Query Cross-Attention: VLM에 추가된 Nq=8개 learnable query token의 distilled features F_query^l (learnable scalar gate g_l로 modulate)

세 출력은 concat → FFN+LN+residual로 fuse. 최종 layer 출력은 MLP+sigmoid로 "p(a_i ≻ a_j)"를 예측, focal loss로 학습.

**Tournament Selection**: N개 candidate를 pairwise하게 토너먼트 형태로 reduce(논문 Algorithm 1) — log N round로 single best action 선택.

## 3. 학습 데이터 및 자동 Preference Pair Curation

RAC 학습용 preference dataset을 **fully automated**로 생성한다는 점이 핵심 기여 중 하나.

**Conditional Flow-Matching 원리 활용**: flow-matching action head는 Gaussian prior → expert distribution으로 가는 vector field v(a_τ, τ, Ct)를 학습한다. ODE를 τ=1→0으로 풀 때 사용하는 integration step 수 N_steps에 따라 sample quality가 graceful하게 저하됨.

**Curation pipeline**: expert dataset의 (o_t, a_expert_t) 각각에 대해
- a_good_t: N_high steps로 ODE 풀이 (high quality)
- a_bad_t: N_low steps로 ODE 풀이 (low quality)

선호 순서 a_expert ≻ a_bad, a_good ≻ a_bad를 생성하고, positional bias 방지를 위해 symmetric pair(label inversion)도 추가. 실험에서는 (N_high, N_low) = ⟨10,3⟩ 및 ⟨9,2⟩ mixed pair 사용.

**RAC 학습 step**: 30,000 step. 추가 hyperparameter: τ = 80th percentile, N_q = 8, N_candidates = 16.

이 pipeline의 타당성은 RQ3 mechanism analysis(Table 7, Pour water 실제 robot)에서 확인 — N_steps=10/5/1일 때 success rate 54%/42%/18%로 점진적 degradation을 보이며 "semantically coherent but precision-error" 양상으로 random noise가 아님을 입증.

## 4. 핵심 실험 결과: LIBERO-LONG 및 실제 로봇

**환경**: LIBERO-LONG(시뮬), Agilex Piper Arm(실제 로봇). 각 task 50회 반복 평가.

**Base 모델**: PI0, PI0.5(둘 다 flow-matching action head 기반 SOTA VLA).

**LIBERO-LONG (Table 1, Average over 10 tasks, %)**:
| Method | Success Rate |
|--------|--------------|
| Robomonkey | 56.5 |
| PI0 (baseline) | 82.8 |
| PI0 + VLA-ATTC (Full, no clutch) | **92.2 (+9.4)** |
| PI0 + VLA-ATTC (with clutch) | 90.6 (+7.8) |
| PI0.5 (baseline) | 90.6 |
| PI0.5 + VLA-ATTC (Full) | **95.4 (+4.8)** |
| PI0.5 + VLA-ATTC (with clutch) | 94.0 (+3.4) |

특히 가장 어려운 "Both pots on stove" task에서 PI0 40% → 58%(+18%), PI0.5 54% → 68%(+14%)로 catastrophic failure를 효과적으로 줄임. **Long suite SOTA failure rate를 50% 이상 감소**.

**Real-world (Agilex Piper, Table 2, %)**:
| Task | PI0 | PI0+ATTC(Full) | PI0.5 | PI0.5+ATTC(Full) |
|------|-----|----------------|-------|------------------|
| Stack cubes | 46 | 62 | 50 | 60 |
| Pour water | 50 | 66 | 54 | 68 |
| Sweep rubbish | 42 | 62 | 52 | 60 |
| **Average** | 46 | **63.3 (+17.3)** | 52 | **62.7 (+10.7)** |

**Ablation (Table 3, threshold τ)**: K-th percentile을 0% → 80%로 늘려 deliberation 비율을 줄여도 PI0.5 기준 95.4% → 94.0%로 거의 손실 없음. → "어려운 상황은 sparse하다"는 근거이자 cognitive clutch 정확도 입증.

**Ablation (Table 4, candidate 수 N)**: PI0.5 기준 N=4/8/16/32 → 92.0/93.8/94.0/95.2%. N=16이 cost-effective sweet spot.

**Ablation (Figure 5, RAC 구조)**: PI0.5+ATTC 기준 Full RAC 94% → w/o Learnable Query 92.4% → w/o Action Difference 93.2% → w/o Learnable Weight 93.6%. Learnable Query 제거가 가장 큰 손실.

**Mechanism (Table 6, uncertainty validation)**: 1,000개 observation pair에 대해 4명 human expert annotation을 ground truth로 비교. DTW(N=2) 89.2% agreement, MPD(N=4) 90.1%, MPD(N=8) 90.4% — N을 4/8로 늘려도 <1.5% 개선뿐이라 N=2가 cost-effective.

**Efficiency (Table 5, 평균 control frequency)**: PI0/0.5 baseline 23.3Hz → VLA-ATTC(clutch on) **20.8Hz** (실시간 가능) → VLA-ATTC(Full) 12.1Hz → Robomonkey 1.5Hz. Clutch 덕분에 baseline 대비 11% latency 손실로 +4.8~17.3% 성능 gain 확보.

## 5. 비교: Test-Time Scaling for VLA 계열에서의 위상

논문은 VLA용 deliberation을 sequential vs parallel로 나눠 비교:

**Sequential deliberation**: ECoT, CoT-VLA, RoboMamba, PI0.5, ChatVLA(2), Hume, OneTwoVLA — base VLA의 fine-tuning과 CoT data annotation이 필수. text reasoning과 action 생성 사이 최적화 충돌 존재.

**Parallel deliberation**: 
- VLA-Reasoner(Guo et al., 2025): world model + MCTS, 막대한 cost와 world model fidelity 한계
- RoboMonkey(Kwok et al., 2025), Steering Generalists(Nakamoto et al., 2024): 대규모 외부 critic model로 best-of-N 선택, indiscriminate deliberation으로 비효율, absolute scoring instability

VLA-ATTC의 차별점:
1. **No fine-tuning of base VLA** — plug-and-play
2. **Adaptive trigger** via DTW cognitive clutch — 평균 latency 손실 최소화
3. **Pairwise relative scoring** — absolute value estimation의 instability 회피, lightweight critic만으로 SOTA 갱신
4. **Auto-curated preference data** — flow-matching ODE step 변조로 human annotation 불필요
5. **Shared pre-fill amortization** — N=16 sampling이 거의 free

Table 1, Table 2, Table 5에서 RoboMonkey 대비 성능(56.5 → 95.4 LIBERO-LONG, 26 → 63.3 real-world avg)과 control frequency(1.5Hz → 20.8Hz)를 모두 압도.

## 6. 평가 및 시사점

**강점**:
- Frozen base VLA 위에 얹는 wrapper 구조로 **PI0와 PI0.5 모두에 generalize** — 향후 다른 flow-matching VLA에도 직접 적용 가능
- Cognitive clutch 도입으로 "deliberation은 비싸다" 라는 통념을 깨고 20.8Hz의 실시간성 유지
- Pairwise relative loss는 absolute value 학습의 noise를 제거 — RL value learning에서 잘 알려진 문제를 우아하게 회피
- 자동 preference curation은 cross-embodiment scaling 가능성을 시사

**한계 및 의문점**:
- 평가가 **LIBERO-LONG 단일 suite**에 국한 — Spatial/Object/Goal 4-suite avg 비교가 없어 다양한 task family에서의 generalization은 미검증
- Base가 flow-matching VLA에 한정 — autoregressive(OpenVLA류)나 diffusion(Octo류)로의 transfer는 ODE-step preference curation pipeline 재설계 필요
- RAC가 VLM과 동일 depth L Transformer라 "lightweight"라고 표현하지만 절대 parameter 수, FLOPs 표는 제시되지 않음
- Cognitive clutch threshold τ가 K-th percentile로 calibration 필요 — task distribution이 바뀌면 재calibration 가능성
- Real-world는 3 task만 평가 — 더 다양한 contact-rich/long-horizon manipulation 검증 필요
- "Will open-source" 명시했으나 현재 시점 code/weights 공개 URL 부재

**시사점**:
VLA-ATTC는 robotics에서 inference-time scaling이 단순히 "더 많이 sample" 하는 것이 아니라 (a) **어디서**(uncertainty-driven trigger), (b) **무엇을**(relative pairwise critic), (c) **어떻게**(amortized pre-fill, log N tournament) 라는 세 축을 동시 최적화해야 함을 보여준다. 특히 flow-matching의 ODE solver 자체가 무료 quality knob을 제공한다는 관찰은 향후 자동 preference data 생성, RLAIF, distillation 등 광범위한 후속 연구로 확장될 잠재력이 크다.

<!-- VERIFIED: pdf -->
