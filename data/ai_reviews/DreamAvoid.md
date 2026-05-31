# DreamAvoid: Critical-Phase Test-Time Dreaming to Avoid Failures in VLA Policies

> **한 줄 요약**: 임의의 base VLA(π0.5, GR00T-N1.6)에 plug-in되는 test-time intervention으로, DINOv2 기반 Dream Trigger가 위험 phase를 감지하면 flow matching을 ODE→SDE로 변환해 K개 후보 action chunk를 생성하고, 증류된 DreamDojo-2B world model로 각 후보를 imagination roll-out하여 Robometer value model이 선택. 실세계 4 task 평균 48.8% → 72.5% (+23.7%p), LIBERO 96.5% → 97.8%, SimplerEnv도 일관된 향상.

---

## 1. 배경 및 동기

### Base VLA의 본질적 약점

- 현재 VLA는 **single-shot deterministic inference** — flow matching이라 해도 ODE의 결정적 결과 1개만 사용
- Critical phase(plug insertion, screw alignment 등 정밀 접촉)에서 작은 오차가 catastrophic 실패로 증폭됨 — pi0.5가 screw insertion 30%, charger plugging 47.5%로 무너지는 부분
- 더 큰 base 모델로 해결하려면 cost가 폭증 → **test-time compute을 어디에 어떻게 쓸 것인가**의 문제

### 기존 test-time 접근의 한계

- **Best-of-N sampling**: 단순 후보 생성은 가능하나 어느 후보가 좋은지 판단하는 oracle 부재
- **Always-on planning** (예: GPC-RANK): 매 step planning은 7-9초/step으로 실용 불가
- **Failure detection** 단독: 감지 후 무엇을 할지가 빠짐 — 단순 freeze는 contact-rich task에서 위험

### 핵심 질문

- **언제(when)** dream할 것인가 — critical phase 자동 감지 가능?
- **무엇(what)**을 dream할 것인가 — 다양한 candidate action을 어떻게 sampling?
- **어떻게(how) 평가**할 것인가 — world model imagination + value model이 진짜로 실패를 회피?

📌 [Figure 1 삽입] — DreamAvoid 파이프라인: Dream Trigger → SDE Action Proposer (K candidates) → DreamDojo Rollout → Robometer Score → Best Action 선택

---

## 2. 방법론 심층 분석

### 2.1 Dream Trigger — When

- **Frozen DINOv2-ViT-B/14**가 각 카메라 frame을 인코딩, 시간 차원으로 average pooling
- Proprioceptive state와 concat 후 **3-layer MLP**가 "현재가 critical phase인가" 확률 출력
- Polling 주파수 ~1Hz, latency 50ms — base policy(93±5ms) 옆에서 병렬 실행
- Threshold $\gamma$는 task-dependent, soft sigmoid $\sigma((t - t_{crit})/\beta)$로 ±β frame 부드러운 전이

> ❓ **예상 질문**: 1Hz polling이면 critical phase 진입을 놓치는 latency가 최대 1초 — contact-rich task에서 너무 늦지 않나?
> **답변**: 이것이 trigger 설계의 핵심 trade-off. 더 빠른 polling은 false positive를 늘려 dream 비용이 폭증. Soft sigmoid window가 hard threshold보다 미스 보정 — phase가 임박할수록 dream 확률이 점진 증가. 그래도 reactive plug 삽입 등 0.5초 이내 결정 task엔 한계.

### 2.2 Action Proposer — What (ODE → SDE 변환)

표준 flow matching 추론은 deterministic probability flow ODE:

$$dx_t = v_\theta(x_t, t)\,dt$$

DreamAvoid는 이를 **stochastic SDE**로 변환:

$$dx_t = v_\theta(x_t, t)\,dt + \sigma(\rho)\,dW_t$$

같은 noise schedule이지만 noise injection $\sigma(\rho)$로 **K개 다양한 trajectory** 생성. K 값:
- Real-world: K=8
- LIBERO: K=4
- SimplerEnv: K=16

SDE noise $\sigma$ 값:
- Real-world: 0.05–0.1 (조심스러운 다양화)
- Simulation: 0.3–0.5 (공격적 다양화)

> ❓ **예상 질문**: ODE를 SDE로 바꾸면 marginal distribution이 보존되나? Best-of-K이 ODE 단독보다 우수하다는 근거는?
> **답변**: Flow matching의 SDE 변환은 score-based diffusion과 동일한 이론(Anderson 1982, Song 2021)으로 marginal $p_t(x)$가 보존됨. K=8에서 분산이 있는 sampling이 deterministic ODE 1개보다 mode coverage가 넓음. Ablation에서 SDE 72.5% vs Repeated ODE 57.5% — 같은 K로도 SDE가 우수.

### 2.3 Dream Evaluator — How

두 구성요소:

**(a) Distilled World Model (DreamDojo-2B)**
- 원본 DreamDojo-2B를 **Self Forcing paradigm**으로 autoregressive student로 증류
- 입력: 현재 state + candidate action chunk
- 출력: rollout된 미래 frame (latent)
- 증류로 inference latency 단축, multi-candidate roll-out이 실용 시간에 가능

**(b) Robometer Value Model**
- Rollout latent에서 "task progress" 점수 회귀
- Loss: Huber regression + ranking loss
- **핵심 학습 데이터 구성**:
  - 20%: terminal success trajectories
  - **40%: boundary/failure** — 실패 직전·실패 trajectory를 명시적 supervision
  - 40%: steady progression
- Priority-based batch sampling으로 value model이 routine progression에 과적합되지 않음

> ❓ **예상 질문**: World model rollout이 hallucination되면 value가 의미 없지 않은가?
> **답변**: 정확히 핵심 위험. Robometer가 real trajectory + dream rollout 모두에 joint training되므로 dream-domain shift를 어느 정도 보정. 그러나 OOD context에서 DreamDojo 자체가 부정확하면 value도 잘못된 후보를 고를 위험 — 논문이 명시한 미해결 limitation 중 하나.

### 2.4 전체 파이프라인

```
[매 step] base policy → action chunk
[1Hz parallel] Dream Trigger 확률 p
   if p > γ:
     [critical phase 진입]
     PID-hold 자세 유지 (제어 끊김 방지)
     K개 SDE candidate 생성
     each: DreamDojo rollout → Robometer score
     best 선택 → 실행
```

**Latency**:
- Base policy: 93±5ms (normal phase)
- Dream Trigger: 50±5ms (parallel)
- Critical-phase intervention 전체: **2133±110ms**
- 비교: always-on GPC-RANK 7.6–9.7초 → DreamAvoid 2.7–7.0초로 ~3× 빠름

---

## 3. Hyperparameter

| 항목 | Real-world | LIBERO | SimplerEnv |
|------|-----------|--------|-----------|
| Action chunk H | 50 | 10 | (명시 안 됨) |
| Candidates K | 8 | 4 | 16 |
| SDE noise $\sigma$ | 0.05–0.1 | 0.3–0.5 | (sim 범위) |
| Base policy | π0.5 | π0.5 | GR00T-N1.6 |
| Trials | 40/task | 200/task | - |

> ❓ **예상 질문**: Real-world와 simulation의 SDE noise가 6×차이. 왜?
> **답변**: Real-world는 candidate 간 차이가 너무 크면 안전성 위험(예: 다른 방향으로 plug 삽입 시 핀 파손). 작은 σ로 base ODE 주변만 탐색. Sim은 risk-free라 공격적 다양화로 mode coverage 극대화. 실용적이나 hyperparameter sensitivity가 큰 신호.

---

## 4. 실험 결과 심층 분석

### 4.1 Real-world 4 task (40 trials/task) — 핵심 결과

| Task | π0.5 (base) | DreamAvoid-ABL | Δ |
|------|------------|----------------|---|
| Cup Sleeving | 62.5% | **90.0%** | +27.5 |
| Charger Plugging | 47.5% | **67.5%** | +20.0 |
| Cap Opening | 55.0% | **80.0%** | +25.0 |
| Screw Insertion | 30.0% | **52.5%** | +22.5 |
| **Average** | **48.8%** | **72.5%** | **+23.7** |

해석:
- **모든 task에서 +20%p 이상 향상** — 일관된 효과
- 가장 어려운 Screw Insertion에서도 ~75% 상대 향상
- 그러나 절대 점수는 여전히 52.5% — completely solve는 아님

### 4.2 LIBERO (200 trials/task)

| Suite | π0.5 (base) | DreamAvoid-ABL |
|-------|------------|----------------|
| Spatial | (개별 미보고) | 99.0% |
| Object | (개별 미보고) | 98.5% |
| Goal | (개별 미보고) | 99.0% |
| Long | (개별 미보고) | 94.5% |
| **Avg** | **96.5%** | **97.8%** |

LIBERO는 이미 saturated → 향상 폭 작음 (+1.3%p). 그러나 가장 어려운 Long suite 94.5%로 base 대비 향상.

### 4.3 SimplerEnv

| Dataset | GR00T-N1.6 | DreamAvoid-ABL |
|---------|-----------|----------------|
| Bridge | 59.9% | **63.6%** (+3.7) |
| Fractal | 76.4% | **80.7%** (+4.3) |

- 다른 base policy(GR00T-N1.6)에서도 작동 — generalization 입증
- Sim 환경에서 향상 폭이 real보다 작음(가설: sim의 critical phase가 real만큼 많지 않음)

### 4.4 Ablation (Real-world)

| 변형 | Avg |
|------|-----|
| **DA-ABL (Full)** | **72.5%** |
| DA-Vanilla (no autonomous boundary learning) | 66.9% |
| SDE 대신 Repeated ODE | 57.5% |
| Random candidate selection (no value) | 45.0% |
| Base π0.5 | 48.8% |

핵심 인사이트:
- **Random selection 45% < Base 48.8%**: value model 없이 그냥 candidate sampling만 하면 오히려 base보다 나쁨. World model + value model이 결정적
- **SDE 72.5% vs Repeated ODE 57.5%**: 같은 K로 deterministic 반복은 다양성 부족
- **Autonomous boundary learning +5.6%p**: 실패 데이터의 명시적 활용이 가치 큼

---

## 5. 학습 데이터 구성 (중요)

| 구성요소 | 데이터 |
|---------|--------|
| Base π0.5 | 100 human teleoperation demonstrations (성공 trajectory만) |
| Dream Trigger | (논문 명시 부족) |
| World Model (DreamDojo distillation) | base DreamDojo-2B + Self Forcing |
| Robometer Value | $\mathcal{D}_{teleop} \cup \mathcal{D}_{online}$, priority-based sampling (20% terminal success / 40% boundary-failure / 40% steady progression), joint training on real + dream trajectories |

**$\mathcal{D}_{online}$의 출처**: deployment 중 자율적으로 수집되는 success/failure trajectory. 즉 시스템이 운영되면서 value model이 지속 개선되는 closed loop.

> ❓ **예상 질문**: Value model을 dream trajectory로 학습하면 dream-real domain shift에 cascade되지 않는가?
> **답변**: Real + dream joint training이 정확히 그 문제에 대한 답. Real trajectory가 anchor 역할로 dream의 distributional drift를 잡음. 그러나 base policy가 deploy되어야 $\mathcal{D}_{online}$이 모이는 chicken-and-egg가 있어 초기 buffer는 teleop만.

---

## 6. 한계 및 미해결 문제

### 방법론적

1. **Latency 2133ms intervention**: critical phase에서 2초간 base 정지(PID-hold) 후 dream → 사람에게는 부자연스러운 멈춤, 또한 task의 시간 제약이 강하면 (떨어지는 물체 잡기 등) 적용 불가
2. **1Hz Dream Trigger polling**: 최대 1초 detection 지연 — reactive critical phase 놓칠 가능성
3. **World Model 신뢰도**: DreamDojo가 OOD에서 hallucinate하면 value가 잘못된 후보 선택. 논문은 inevitable limitation으로 인정
4. **K=8 candidate가 충분한 mode coverage인가**: SDE로 다양성 확보하지만 8개 sample이 7-DoF × 50-step action space에서 sparse. 진정한 multimodality 보장 어려움
5. **Hyperparameter sensitivity**: σ가 real 0.05–0.1, sim 0.3–0.5로 6× 차이 — task별 tuning 필요

### Attribution

- 72.5% 중 어느 정도가 SDE sampling(다양성)이고 어느 정도가 value model(선택)인가? Ablation에서 random selection 45% → value 추가 72.5%로 value가 27.5%p 기여. SDE 자체 효과는 ODE 57.5%와 비교 시 +15%p.
- DreamDojo distillation의 비용이 미보고 — Self Forcing 학습 자체가 큰 cost일 수 있음

### Real-world 평가의 polish

- 4개 task는 다양성 있으나 모두 정밀 삽입/조작 → critical phase가 명확한 task. 비-critical task(pick-place 등)에서 효과 미검증 (LIBERO에서 +1.3%p는 시사점)
- Failure mode 정성 분석 없음 — DreamAvoid가 실패할 때 왜 실패하는지?

---

## 7. 관련 연구 비교

| 방법 | When | What | How | Real-world |
|------|------|------|-----|-----------|
| Base π0.5 | always | 1 ODE | direct | 48.8% |
| Best-of-N (no value) | always | K samples | random | 45% (오히려 하락) |
| GPC-RANK (always-on planning) | every step | candidates | world model + value | 높음, 7-9초 latency |
| **DreamAvoid** | **critical phase only** | **K SDE** | **dream + value** | **72.5%, 2-7초** |

### 차이의 본질

- **선택적 활성화**: critical phase에만 dream하므로 latency 비용을 적재적소에 집중
- **SDE 다양화**: deterministic policy를 stochastic으로 변환하는 최소 비용 trick
- **Boundary learning**: 실패 데이터를 학습 신호로 활용 — 대부분의 imitation learning이 무시하는 부분

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Critical-phase dreaming framework는 새로움. 개별 구성요소(SDE, world model, value model)는 기존이나 결합 방식이 신선 |
| **Technical depth** | ★★★★☆ — ODE→SDE 변환, Self Forcing distillation, priority sampling이 모두 sound |
| **Experimental rigor** | ★★★★★ — Real-world 4 task × 40 trials, LIBERO 200 trials/task, 2가지 base policy, 4개 ablation — 매우 충실 |
| **Practical impact** | ★★★★☆ — 48.8 → 72.5%는 실용적 의미가 큰 향상. 다만 2초 latency가 적용 범위를 제한 |
| **Writing quality** | ★★★★☆ — 구성요소 분리가 명확 |

**강점**: VLA의 test-time compute을 "어디에 쓸 것인가"라는 본질적 질문을 critical-phase 한정으로 답한 첫 plug-and-play 프레임워크. Real-world +23.7%p는 매우 인상적이며 ablation으로 각 구성요소의 기여가 깔끔히 분리. **약점**: 2초 intervention latency와 1Hz trigger polling이 reactive task에 부적합. DreamDojo의 OOD hallucination이 가장 큰 unresolved risk이며 σ hyperparameter의 task-dependent tuning이 사용성을 떨어뜨림.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Critical-phase intervention 2초 동안 PID hold하면 contact-rich task에서 force build-up 위험은? | 큰 우려. Plug가 살짝 박혀 있는 상태에서 2초 멈추면 부품 손상 가능. 논문은 "position-holding state with PID control … maintains contact forces"로 처리한다 주장하나 구체 안전 marginal 검증 부재. |
| 2 | DreamDojo hallucination이 value의 입력이 되면 가짜 future를 좋다고 평가하지 않는가? | Real + dream joint training으로 완화하나 근본 해결은 아님. 논문이 명시한 limitation. World model OOD가 가장 큰 미해결 risk. |
| 3 | K=8 후보가 7-DoF × 50-step action space에서 충분한가? | 이론적으로 부족. SDE로 다양성을 확보하지만 mode coverage 보장은 어려움. K 증가는 latency 선형 증가 → 8이 trade-off 점. |
| 4 | LIBERO +1.3%p는 너무 작은데 LIBERO도 critical phase가 있는가? | LIBERO는 대부분 pick-place로 critical phase가 드묾 → Trigger가 거의 안 발동. Dreamavoid의 가치는 contact-rich real task에 집중. |
| 5 | Robometer value model의 ground truth는 무엇인가? Self-supervised? | "Task progress" 회귀 + ranking. Ground truth는 terminal success(0/1) 및 expert demonstration의 progress estimate. 정확한 progress label 정의가 본문에서 다소 모호. |
| 6 | Autonomous boundary learning이 deploy되면서 진짜 실패를 수집해야 하는데 초기 deploy의 안전성은? | Chicken-and-egg. 초기에는 teleop demo만으로 학습된 value 사용 → 실패 수집되며 점진 개선. 안전 critical 환경(의료, 자율주행) 초기 배포엔 risk. |
| 7 | σ noise가 real 0.05–0.1, sim 0.3–0.5로 6× 차이. 일반화 가능한 hyperparameter rule이 있는가? | 명확한 rule 없음. Task 안전성(real)과 mode coverage 필요성(sim)의 직관적 trade-off. 일반화엔 task별 calibration 필요. |
| 8 | Best-of-N (random selection)이 base보다 낮은 45%인 이유? | 무선택 sampling은 noise만 추가하므로 base의 ODE 1개보다 평균 quality가 떨어짐. Value model이 진짜 가치. |
| 9 | GR00T-N1.6에서도 작동(SimplerEnv)했다고 하나 다른 action representation의 정책에서 일반화 보장은? | π0.5와 GR00T-N1.6 모두 flow matching action head — SDE 변환의 전제. AR-token 정책(OpenVLA original)에는 직접 적용 불가, ODE→SDE 변환 자체가 무의미. |
| 10 | DreamDojo distillation의 비용이 미보고된 것은 critical limitation 아닌가? | Yes. Self Forcing 학습은 비용이 클 가능성 — 2B world model을 student로 다시 학습. 전체 시스템 비용 평가에 누락된 부분. |
| 11 | 진짜 새로운 기여는 SDE인가 value model인가 critical phase trigger인가? | Ablation에 따르면: (a) Critical phase trigger: 측정 안 됨, (b) SDE: ODE 대비 +15%p, (c) Value model: random 대비 +27.5%p, (d) Boundary learning: vanilla 대비 +5.6%p. **Value model이 가장 큰 단일 기여**. |
| 12 | Real-world 4 task만 — 일반화 범위는? | Contact-rich precision insertion에 강함. 다른 카테고리(deformable, multi-object rearrangement, mobile manipulation)에선 미검증. |

<!-- VERIFIED: pdf -->
