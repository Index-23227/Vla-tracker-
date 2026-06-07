# CTRL-STEER: Closed-Loop Neural Activation Control in Vision-Language-Action Models

> **한 줄 요약**: OpenVLA의 LLaMA-2 FFN 내부에서 mechanistic interpretability로 motion-aligned 뉴런 10개를 선택한 뒤, 고정 계수 대신 **PID + PPO closed-loop controller**가 매 timestep마다 steering 계수 α_t를 동적으로 조절하여, height/speed steering을 수행하면서도 task success를 보존하는 학습-불필요(controller만 학습) test-time framework. LIBERO 4-suite에서 RL controller가 height SR 73.88% / speed SR 76.12%로 unsteered baseline(71.37%) 대비 개선, 같은 조건의 static steering(C=20)은 1.8~27.4%로 붕괴.

---

## 1. 배경 및 동기

### 기존 activation steering의 구조적 한계

- **Häon et al. (CoRL 2025)** [12]는 OpenVLA의 FFN 뉴런 중 "up/down/left/right" 등 단일 feature에 대응되는 폴리세만틱 뉴런을 찾고, 그 활성을 **고정 스칼라 α** 로 곱해 행동을 편향시키는 mechanistic-interpretability 기반 steering을 제안.
- 그러나 (i) α가 시간에 따라 변하지 않는 **open-loop**, (ii) task state가 시간에 따라 진화하는 embodied control과 불일치 → over-correction, oscillation, **task success 붕괴** 발생.
- 본문 핵심 수치: static C=20에서 unsteered 71.37% → height steering 27.37%, speed steering **1.8%** (LIBERO 4-suite 평균).

### 핵심 질문

- "**시간에 따라 변하는 control 개념(speed, smoothness)**을 단일 timestep forward로는 측정조차 못 하는 VLA를 어떻게 steering 할 것인가?"
- "신경망 내부 활성을 직접 manipulate하면서도 task success를 잃지 않을 수 있는가?"

저자 답: **decouple representation from regulation** — 뉴런(representation)은 그대로 두되, intervention 강도(regulation)를 closed-loop controller로 다이내믹하게 조절.

📌 [Figure 1 삽입] — 'pick up the book and place it in the back compartment of the caddy' 태스크에서 unsteered는 낮은 trajectory, static steering은 높지만 task fail, closed-loop는 높이 유지 + 성공.

---

## 2. 방법론 심층 분석

### 2.1 Mechanistic Interpretability: 뉴런 선택

Transformer FFN을 key-value memory로 해석 [10]:

$$\mathrm{FFN}^\ell(r^\ell) = \sum_{i=1}^{d_m} m_i^\ell\, v_i^\ell$$

각 value vector $v_i^\ell \in \mathbb{R}^d$를 LM head $W_{out} \in \mathbb{R}^{|V| \times d}$로 vocabulary logit space에 사영 → softmax로 확률 분포 $p_i^\ell$ 산출 → top-k=20 토큰의 embedding을 확률 가중평균하여 **semantic embedding** $\mathrm{sem}_i^\ell$ 정의.

$$\mathrm{sem}_i^\ell = \sum_{w \in \mathrm{TopK}(p_i^\ell)} p_i^\ell(w)\, e(w)$$

**Concept-to-neuron matching**: motion concept C에 대해 representative token 집합 $T_C = \{\text{up, down, left, right, forward, backward}\}$ 사용. 각 token w에 대해 모든 FFN 뉴런의 sem embedding과 cosine similarity로 k=5 NN 검색 → union으로 후보 set S 형성 → 사람이 promoted token을 inspect해서 polysemantic 뉴런 필터링 → 최종 **10개 뉴런** 보유.

> ❓ **예상 질문**: |V|=32000인데 top-k=20에서 의미 안 흐려지는가?
> **답변**: 확률 가중평균이라 dominant token이 sem embedding을 지배. 또한 manual polysemantic filtering이 안전망. 다만 "수동 필터링"이 limitation으로 명시됨.

### 2.2 Steering as Closed-Loop Control

기존 open-loop 식:

$$\tilde{m}_i^\ell = \begin{cases} \alpha & \text{if } (\ell, i) \in S \\ m_i^\ell & \text{otherwise} \end{cases}$$

CTRL-STEER는 α를 시간 가변 벡터 $\alpha_t \in \mathbb{R}^k$로 대체. 개념 추적 오차 $e_t = c^* - c_t$ 정의 (c*는 목표값, 예: height steering이면 $c^* = 2 h_0$).

### 2.3 PID Controller

$$\alpha_t^{PID} = K_P\, e_t + K_I \sum_{\tau=0}^{t} e_\tau + K_D\, (e_t - e_{t-1})$$

- **Proportional (KP=4.0)**: 순간 편차 비례 보정
- **Integral (KI=0.5)**: 누적 편차 보정 — base VLA가 steering 목적과 충돌하는 trained dynamic을 갖기 때문에 steady-state bias를 점진적으로 상쇄
- **Derivative (KD=1.0)**: 급격한 활성 변화 → trajectory 진동 억제

PID는 **scalar** 신호를 모든 10 뉴런에 동일하게 적용. $\alpha_t^{PID} \in [0, 20]$ 클리핑.

### 2.4 RL Controller (PPO)

PID는 reactive — long-horizon task success를 명시적으로 최적화하지 않는다. 그래서 PPO 정책 $\pi_\theta$ 도입:

**State**:
$$s_t = [a_t,\ \Delta a_t,\ \alpha_{t-1},\ t/T]$$

($a_t \in \mathbb{R}^k$: 10 뉴런 활성, $\Delta a_t$: 변화량, $\alpha_{t-1}$: 직전 steering, $t/T$: 정규화 episode progress)

**Action**: $\alpha_t^{RL} = \pi_\theta(s_t) \in \mathbb{R}^k$ — **각 뉴런 별** 계수 (PID와 달리 per-neuron).

**Reward**:
$$r_t = r_{\mathrm{steer}}(t) + \lambda\, r_{\mathrm{task}}$$

λ ∈ {100, 200, 500, 1000} sweep. **task별 별도 policy 학습** (per-task RL — limitation).

**중요**: RL policy는 PID controller가 출력한 trajectory로 **warm-start**되어 학습. Random init 시 성능 붕괴 (Sec 4 ablation).

> ❓ **예상 질문**: per-neuron control이 진짜 필요한가? PID도 충분히 좋지 않은가?
> **답변**: Table 3/4 비교 — height SR: PID 71.0% vs RL 73.88%, speed SR: PID 72.5% vs RL 76.12%. RL이 +2.9~3.6%p 우위. 특히 LIBERO-Long speed에서 PID 59.5% → RL 66.5%로 long-horizon에서 격차가 커진다 — long-horizon planning 효과.

### 2.5 Implementation Details

- Base VLA: **OpenVLA** (7B, LLaMA-2 + Prismatic VLM [14]) — suite별 fine-tuned
- 추가 검증: **X-VLA** [33] — transferability 검증용
- PID: KP=4.0, KI=0.5, KD=1.0, control horizon 20
- PPO: horizon T=920 (warm-up 20 + execution 900)
- 추론 overhead: PID 0.2021 s/step, RL 0.2094 s/step (OpenVLA 0.1869 s/step 대비 ~8~12% 증가)

---

## 3. 데이터 전략

이 논문은 **신규 데이터셋이나 학습 데이터를 사용하지 않는다**. PID controller는 hyperparameter 튜닝(KP, KI, KD)만, RL controller는 LIBERO 환경에서 PPO rollout만 수집. Base VLA는 OpenVLA 공개 weight 그대로.

> ❓ **예상 질문**: RL은 task별로 학습한다는데 generalization은?
> **답변**: 명시적으로 limitation으로 표기됨. "RL controller is trained per task." Future work로 task-agnostic controller 제안.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Base VLA | OpenVLA 7B (LLaMA-2 + Prismatic), 별도 X-VLA |
| 선택 뉴런 수 | 10 (k-NN k=5, manual filtering 후) |
| Representative tokens | {up, down, left, right, forward, backward} |
| PID gains | KP=4.0, KI=0.5, KD=1.0 |
| Steering coef bound | [0, 20] |
| RL algorithm | PPO |
| RL horizon | T=920 steps (20 warm-up + 900 exec) |
| Height target c* | 2 × initial end-effector height |
| Speed target c* | 30 cm/s |
| Speed threshold s_thr | 20 cm/s |
| λ sweep | 100, 200, 500, 1000 |
| Inference cost | +8~12% step time, GPU 동일 (14.26 GB) |

---

## 5. 실험 설계 및 평가 프로토콜

**Benchmark**:
1. **LIBERO** 4-suite (Goal, Object, Spatial, Long [libero-10])
2. **BridgeData V2 pick-and-place** evaluated on **SimplerEnv**
3. **X-VLA on LIBERO-Goal** — transferability 검증

**Concept**:
- **Height** (state-based 검증용)
- **Speed** (temporal control concept — 단일 forward로 측정 불가, 따라서 closed-loop의 가장 큰 motivation)

**Metric**:
- Steering: end-effector mean height, 95th percentile height, **AAT (Area Above Threshold)**; mean speed, **SAT (Speed Above Threshold)** with $s_{thr}$=20 cm/s
- Task: **SR (Success Rate %)**

**Rollouts**: 각 task 20 deterministic rollouts (seed 고정).

---

## 6. 실험 결과 심층 분석

### 6.1 LIBERO Height Steering (Table 3, SR %)

| Suite | Unsteered OpenVLA | Static C=20 [12] | **PID** | **RL** |
|-------|------------------|-----------------|--------|-------|
| Long | 58.00 | 10.00 | 54.00 | **57.00** |
| Goal | 77.50 | 41.00 | 79.00 | **82.00** |
| Object | 72.00 | 33.50 | 76.00 | **77.00** |
| Spatial | 78.00 | 25.00 | 75.00 | **79.50** |
| **Avg** | **71.37** | **27.37** | **71.00** | **73.88** |

- Static C=20은 일부 suite에서 AAT(steering 강도)는 올라가나 **SR이 71→27%로 급락** — open-loop의 한계 노출.
- RL controller가 **steering을 하면서 unsteered보다 SR을 더 높임** (+2.51%p). LIBERO-Goal에서 82%로 unsteered 77.5% 대비 +4.5%p.

### 6.2 LIBERO Speed Steering (Table 4, SR %)

| Suite | Unsteered | C=20 | PID | **RL** |
|-------|-----------|------|-----|-------|
| Long | 58.00 | **1.50** | 59.50 | **66.50** |
| Goal | 77.50 | **2.50** | 76.00 | **83.00** |
| Object | 72.00 | **2.50** | 76.50 | **76.50** |
| Spatial | 78.00 | **1.00** | 78.00 | **78.50** |
| **Avg** | 71.37 | **1.88** | 72.50 | **76.12** |

- **Speed가 height보다 closed-loop이 더 중요** — C=20 평균 1.88%로 거의 완전 붕괴 (temporal concept이 단일 forward로 grounded되지 않기 때문).
- RL이 unsteered 대비 **+4.75%p**, LIBERO-Goal에서는 83% vs 77.5% (+5.5%p) — 가장 큰 격차.

### 6.3 BridgeData V2 / SimplerEnv (Table 2)

| Method | AAT (height) | SR (Height) | Avg speed | SR (Speed) |
|--------|--------------|------------|-----------|-----------|
| Unsteered | 39.15 | 40.0% | 16.93 | 40.0% |
| Static (20) | 47.96 | 12.5% | 25.18 | 14.9% |
| PID | 47.28 | 46.0% | 18.90 | 45.8% |
| **RL** | **48.13** | **47.6%** | **21.38** | **48.9%** |

- RL이 **AAT/avg-speed 모두 static과 동급**으로 끌어올리면서 **SR을 4배 가까이 보존** (12.5→47.6%).
- BridgeData V2 분포 shift에서도 동작 — OpenVLA 자체가 LIBERO 외 데이터로 fine-tuning이 안 된 zero-shot 상황에서도 controller가 robust.

### 6.4 X-VLA Transferability (Table 1, LIBERO-Goal)

| Method | Height | 95th % | AAT | SR |
|--------|--------|--------|-----|-----|
| Unsteered X-VLA | 1.041 | 1.140 | 161.58 | 59% |
| Static C=20 | 1.020 | 1.073 | 626.74 | 25% |
| PID | 1.046 | 1.087 | 183.35 | 60% |
| RL | 1.037 | 1.123 | **400.33** | 60% |

- X-VLA에서도 동일 패턴 — RL이 AAT(400)를 unsteered(161) 대비 2.5배 키우면서 SR 유지. Static은 AAT 626이지만 SR 25%로 붕괴.

> ❓ **예상 질문**: X-VLA가 OpenVLA와 다른 backbone인데 같은 뉴런 선택 절차로 되는가?
> **답변**: 본문에서는 framework transferability를 보였지만 X-VLA 내부 뉴런 선택 detail은 supplementary. FFN-기반 transformer라면 일반화 가능하다는 주장이나, 정확한 절차 검증은 미흡.

---

## 7. Ablation 분석

### 7.1 PID Initialization for RL (Table 5, Speed Steering)

| Suite | RL + PID warm-start | RL (no PID) | Δ |
|-------|--------------------|-----------|----|
| Long | 66.50 | 57.00 | **-9.5** |
| Goal | 83.00 | 78.00 | -5.0 |
| Object | 76.50 | 72.00 | -4.5 |
| Spatial | 78.50 | 77.00 | -1.5 |

- PID warm-start 없으면 LIBERO-Long에서 **9.5%p 손실** — long-horizon task일수록 stable init이 필수.
- 저자 주장: PID가 RL training을 "meaningful initialization"으로 grounding.

### 7.2 PID vs RL 비교 (Tables 3, 4 종합)

- Height: PID 71.0% vs RL 73.88% (+2.88)
- Speed: PID 72.5% vs RL 76.12% (+3.62)
- RL이 일관되게 우월 — non-linear per-neuron policy의 expressiveness.

### 7.3 Inference Cost (Table 6)

| Method | time/step (s) | Peak GPU (GB) |
|--------|--------------|--------------|
| OpenVLA | 0.1869 | 14.26 |
| C=5 | 0.1961 | 14.26 |
| C=20 | 0.1974 | 14.26 |
| PID | 0.2021 | 14.26 |
| RL | 0.2094 | 14.26 |

- RL의 overhead는 **12%**에 불과. VLA forward pass(~95%)가 dominant — controller는 negligible.

---

## 8. 관련 연구 비교

| 방법 | Steering 방식 | Task feedback | Per-neuron | Long-horizon planning | LIBERO Speed SR |
|------|--------------|--------------|------------|---------------------|----------------|
| Häon et al. [12] | Fixed α, open-loop | ✗ | ✗ | ✗ | 1.88% (C=20) |
| Khan et al. [16] | Sparse latent dir | ✗ | △ | ✗ | — |
| Representation Eng. [22, 36] | LLM only, open-loop | ✗ | ✗ | ✗ | N/A |
| **CTRL-STEER (PID)** | Closed-loop scalar | ✓ (error) | ✗ | ✗ | **72.5%** |
| **CTRL-STEER (RL)** | Closed-loop vector | ✓ (reward) | **✓** | **✓** (PPO) | **76.12%** |

### 핵심 차이

- 기존 activation steering은 **모두 open-loop, fixed α** — 본 논문이 **closed-loop control theory ↔ mechanistic interpretability**의 첫 결합 시도.
- Diffusion policy의 test-time guidance(예: Wagenmaker [29], Yuan [31])와는 다른 노선 — 본 논문은 token-based VLA의 FFN 내부 활성 자체를 manipulate.

---

## 9. 한계 및 미해결 문제

저자 명시:
1. **RL controller가 task별 학습** — task-agnostic generalization 없음.
2. **Neuron identification에 manual filtering** — automated disentanglement 미해결.
3. **소수 concept(height, speed)만 평가** — multi-concept simultaneous steering 미검증.

추가 비판점:
4. **OpenVLA 7B(token-based)에 특화** — flow-matching(pi0), diffusion policy 같은 다른 action head paradigm으로 transfer 가능성 불분명.
5. **선택 뉴런 10개의 robustness** — random seed에 따라 선택이 바뀌는지, layer 분포가 어떤지(본문은 "latter half" 정도만 언급) sensitivity 부족.
6. **Static C=20이 1.8%로 붕괴**한다는 비교가 unfair할 수 있음 — C=5는 SR ~50~80% 유지하면서 AAT도 어느 정도 향상. fair comparison은 "C를 task별 최적값으로 tuning"한 결과 (Appendix에는 있다고 명시)이지만 main table에서는 C=20만 highlighted.
7. **Real-world 평가 부재** — 모든 결과가 simulation (LIBERO + SimplerEnv). Real robot에서 sensor noise 하에서 PID/RL controller의 안정성 미검증.

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Mechanistic interpretability + classical feedback control의 깔끔한 결합. Activation steering의 첫 closed-loop formulation. |
| **Technical depth** | ★★★☆☆ — PID는 잘 알려진 기법, PPO도 표준. 새로움은 framework 설계에 있음. |
| **Experimental rigor** | ★★★☆☆ — LIBERO 4-suite + BridgeData/SimplerEnv + X-VLA cross-validation은 충분. 단 real-world 부재, RL task-agnostic 부재. |
| **Practical impact** | ★★★★☆ — 12% overhead로 fine-tuning 없이 controllable behavior — deployment 친화. Concept-level interpretable intervention. |
| **Writing quality** | ★★★★☆ — Open-loop vs closed-loop의 trade-off가 명확히 시각화됨. 수식 깔끔. |

**강점**: (1) Static steering이 SR 1.8%로 붕괴하는 dramatic counterexample 제시 — closed-loop 필요성 강력하게 입증. (2) 12% overhead로 base VLA 재학습 없이 dynamic behavioral 조절 가능. (3) X-VLA로 backbone transferability 검증.

**약점**: (1) RL이 task-specific — generalization 부족. (2) Neuron 선택에 manual filtering 의존. (3) Simulation only, multi-concept 미검증, action-head paradigm 일반화 미검증.

---

## 11. 예상 날카로운 세미나 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | "PID gains (4.0, 0.5, 1.0)을 어떻게 결정? Task-by-task tuning이 필요한가?" | 본문은 보수적으로 선택했다고만 명시. Sensitivity analysis 부재 — 다른 task로 가면 재튜닝 필요할 가능성. |
| 2 | "10 뉴런 선택이 random seed sensitive 한가? Reproducibility?" | 본문에서 layer 분포 ("latter half") 외 detail 부족. Manual filtering 자체가 reproducibility 저해. |
| 3 | "왜 OpenVLA discrete token에서만 효과적? pi0 flow-matching에서는?" | 명시적으로 검증 안 됨. Token-based action head에 FFN steering이 직접 영향. Flow-matching은 latent 공간이 달라 동일 절차 일반화 불확실. |
| 4 | "Reward의 λ sweep {100, 200, 500, 1000}이 너무 크게 grid함. 최적값?" | 본문에서 best 값을 task별로 선택했다고 함 — sweep 자체가 hyperparameter tuning cost. |
| 5 | "Static C=20과 비교는 unfair 아닌가? C=5는 SR 70%대 유지하면서 약한 steering 가능." | Main table은 strong-intervention failure case 강조용. Appendix에 task별 optimal C 비교 있다고 명시. |
| 6 | "RL이 per-task training이면 LIBERO 40 task × 2 concept = 80 policy 학습? 실용성?" | 정확. Future work로 task-agnostic 제안. 현재로서는 control 개념을 미리 fix해두고 deploy하는 시나리오 한정. |
| 7 | "PID warm-start 없으면 LIBERO-Long에서 9.5%p drop. PID가 사실상 main contribution 아닌가?" | 부분적으로 맞음 — RL 단독으로는 unstable. PID는 controller로서는 inferior(SR 72.5 vs 76.1)지만 stable initialization 제공자로서는 critical. |
| 8 | "Mechanistic interpretability 부분이 실제로 필요한가? Random 10 FFN 뉴런에 PID/RL 적용하면?" | 본문에 random baseline 없음. 핵심 비교가 빠진 부분. Häon et al.의 단일 feature 방식 vs 본 논문의 concept set 방식만 비교됨. |
| 9 | "Real-world에서 sensor noise/latency가 PID gain 안정성에 영향?" | Real robot 평가 없음 — 가장 큰 미해결 문제. |
| 10 | "Multi-concept 동시 steering (height + speed)에서 PID 항이 서로 간섭하지 않는가?" | 본문에서 미평가. Future work로 명시. 직관적으로 reward 가중치 conflict가 예상됨. |

---

## 12. 결론

CTRL-STEER는 **VLA의 mechanistic interpretability와 closed-loop control theory를 처음으로 통합**한 framework로, fixed-coefficient steering의 근본적인 task-success 붕괴 문제를 해결한다. PID로 stable한 reactive control을, PPO로 long-horizon task-aware steering을 제공하며, 12%의 미미한 inference overhead로 base VLA 재학습 없이 dynamic behavioral modulation을 가능케 한다. 다만 (i) per-task RL training, (ii) manual neuron filtering, (iii) simulation-only 평가, (iv) token-based VLA 외 일반화 미검증이 향후 과제다. Activation-level interpretable intervention을 closed-loop으로 가져온다는 방향성은 multi-concept control, real-world deployment, safety-critical robotics에서 의미 있는 후속 연구의 출발점이 될 것이다.

<!-- VERIFIED: pdf -->
