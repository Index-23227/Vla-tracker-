# ConSFT: Preserving Foundational Capabilities in Flow-Matching VLAs through Conservative SFT

> **한 줄 요약**: Flow-matching VLA(π0, π0.5, GR00T-N1.6-3B)에서 downstream task fine-tuning 시 발생하는 **catastrophic forgetting** 을 해결하기 위해, 손실에 신뢰도-기반 게이팅을 가한 **Conservative SFT** 목적함수 J = sg[exp(−L/τ)] · L 를 제안. 참조 네트워크/replay buffer 없이 trust-region 영감을 받은 self-distillation으로 parameter sparsity를 유지. **Vanilla SFT 대비 prior capability 평균 +20%p 보존**. Shanghai AI Laboratory.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- Flow-matching VLA(π0, π0.5)는 OXE 등 대규모 pretraining으로 풍부한 **foundational manipulation skill** 보유.
- Downstream task(LIBERO-Spatial, RoboTwin-Indep)에서 SFT 시 **target task 성능은 오르나 prior capabilities 급격 손실** (catastrophic forgetting): LIBERO-Object/Goal 등에서 −30%p 이상.
- 기존 forgetting 완화 기법:
  - **EWC, KL-regularization**: Fisher information / KL penalty 필요 → 추가 계산.
  - **Experience Replay**: 사전학습 data를 fine-tuning에 일부 섞음 → data 저장/접근 필요.
  - **LoRA**: parameter footprint 작으나 expressivity 제한.
  - **Reference network**: 사전학습 model을 parallel copy로 유지 → 메모리 2배.

### 핵심 질문
- **참조 네트워크나 replay buffer 없이, 단지 손실 형태만 바꿔서 catastrophic forgetting을 줄일 수 있는가?**
- **그 메커니즘이 trust-region 방법의 통찰("over-confident update를 억제")로 해석 가능한가?**

📌 [Figure 1 삽입] — 각 fine-tuning 방법(SFT/LoRA/ER/KL/ConSFT)의 target vs. prior capability tradeoff curve.

---

## 2. 방법론 심층 분석

### 2.1 손실 함수

표준 SFT loss L_SFT (flow-matching: noise/velocity MSE)에 대해:
```
J_ConSFT(θ) = sg[exp(−L_SFT(θ) / τ)] · L_SFT(θ)
```
- **sg[·]**: stop-gradient. 즉 exp(−L/τ) 항은 forward에선 scaling factor지만 backward gradient는 통과시키지 않음.
- **τ (temperature)**: 신뢰도-기반 게이팅 강도.

### 2.2 직관: Trust-Region with Confidence Gating
- **L_SFT 가 큰 sample** (모델 신뢰도 낮음): exp(−L/τ) ≈ 0 → effective gradient ≈ 0 → update 억제.
- **L_SFT 가 작은 sample** (모델이 이미 잘 fitting): exp(−L/τ) ≈ 1 → 정상적 update.
- 결과: 모델이 "확신하는 sample"에만 update → **conservative** update → over-fitting과 forgetting 동시 억제.

> ❓ **예상 질문**: 큰 loss sample을 무시하면 어려운 task는 영원히 학습 못 하는 것 아닌가?
> **답변**: 핵심은 τ의 dynamic scaling. 초기 τ_start=0.003에서 시작해 점진적으로 증가(scaling κ=25.0) → 학습 초반엔 잘 맞는 sample만 학습, 후반엔 어려운 sample도 점진적으로 합류. Curriculum-like effect.

### 2.3 Hyperparameter
| 변수 | 값 |
|------|-----|
| τ_start | 0.003 |
| κ (scaling factor) | 25.0 |
| ω_min (lower bound on effective weight) | 0.001 |
| Learning rate | 2.5×10⁻⁵ |
| Global batch size | 1024 |

### 2.4 Sparse Parameter Updates
- 신뢰도 게이팅의 부수적 효과: 매 step에서 일부 sample만 effective gradient를 가짐 → **parameter update가 자연스럽게 sparse**.
- 저자들은 Section 5.4에서 mechanistic 분석으로 update의 sparsity가 prior capabilities 보존과 인과적으로 연결됨을 시사.

📌 [Figure 2 삽입] — Update sparsity over training steps; ConSFT vs vanilla SFT 비교.

---

## 3. 데이터 전략

| Domain | Target task | Prior capability tasks |
|--------|------------|----------------------|
| LIBERO | Spatial | Object, Goal |
| RoboTwin | Indep | Single-Arm, Coord |

- Target task에서 fine-tune, prior task에서 retention 측정.
- Replay/reference data 없음 — 알고리즘이 self-contained.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Hardware | 8 × NVIDIA A100 80GB |
| Parallelism | FSDP |
| Optimizer | AdamW (presumed) |
| Hosts | π0, π0.5, GR00T-N1.6-3B |
| LR | 2.5e-5 |
| Batch | 1024 (global) |

---

## 5. 실험 결과 심층 분석

### 5.1 LIBERO (π0 host, Table 2)

| 메트릭 | Target (Spatial) | Prior Object | Prior Goal | Prior Avg |
|--------|-----------------|-------------|-----------|----------|
| Vanilla SFT | ~0.90 | ~0.06 (↓0.52) | ~0.30 (↓0.10) | ~0.18 |
| **ConSFT** | **0.90 ± 0.067** | **0.32 ± 0.033 (↓0.26)** | **0.35 ± 0.034 (↓0.05)** | **0.34 ± 0.024 (↓0.15)** |

→ Target 성능은 vanilla SFT와 동등, **prior retention은 +16%p 이상**.

### 5.2 LIBERO (π0.5, GR00T host)

| Host | Target Spatial | Prior Avg |
|------|---------------|----------|
| π0.5 + ConSFT | **1.00 ± 0.000** | 0.43 (↓0.34) |
| GR00T + ConSFT | 0.63 ± 0.108 | 0.59 (↓0.30) |

- π0.5에서 target 100% 달성하면서도 prior 43% 유지.
- GR00T는 target task 자체에 약함(0.63) — 모델 capacity 한계로 추정.

### 5.3 RoboTwin (π0 host)

| 메트릭 | Target Indep | Prior Single-Arm | Prior Coord | Prior Avg |
|--------|-------------|-----------------|------------|----------|
| **ConSFT** | **0.60 ± 0.110** | **0.43 (↓0.14)** | **0.13 (↓0.17)** | **0.28 (↓0.16)** |

### 5.4 Baseline 비교 (Table 3)
ConSFT는 LIBERO/RoboTwin 두 도메인 모두에서 **highest prior retention** 달성:
- LIBERO prior avg: 34% (vs vanilla SFT ~18%)
- RoboTwin prior avg: 28%

LoRA / Experience Replay / KL-Regularization와 비교에서도 **target 손실 없이 retention만 추가로 얻는** 유일한 방법.

### 5.5 Ablation (Section 5.4)
- τ_start 변화: 너무 작으면(0.0001) 학습 자체가 stall, 너무 크면(0.1) vanilla SFT와 유사 — 0.003이 sweet spot.
- κ 변화: prior retention과 비례. κ↑ → 더 보수적.
- ω_min: numerical stability를 위한 floor — 성능 영향은 미미.

---

## 6. 관련 연구 비교

| 방법 | Reference net | Replay | Architecture 변경 | Forgetting 완화 |
|------|--------------|--------|------------------|----------------|
| Vanilla SFT | × | × | × | × (catastrophic) |
| LoRA | × | × | ✓ (LoRA adapter) | △ (expressivity 제한) |
| Experience Replay | × | ✓ | × | ○ (data 의존) |
| KL-Regularization | ✓ | × | × | △ (overhead) |
| EWC | × | × | × (Fisher 저장) | △ |
| **ConSFT** | **×** | **×** | **× (loss만 변경)** | **○ (+20%p avg)** |

핵심 차별점: **architecture / data / 추가 메모리 변경 없이** loss 함수만 수정.

---

## 7. 한계 및 미해결 문제

1. **τ scheduling 의존성**: τ_start와 κ의 적절한 값이 task에 따라 다를 수 있음. 자동 튜닝 기제 부재.
2. **Long-tail task에서의 우려**: exp(−L/τ) 게이팅은 "잘 fitting 되는 sample"에 update를 집중시킴 → 매우 어려운(rare) sample은 끝까지 무시될 위험. Curriculum이 충분히 reach해야 함.
3. **Target task 성능 한계**: 일부 케이스(π0 LIBERO-Spatial 0.90, π0.5 1.00)에서는 target 성능이 우수하나, GR00T-N1.6-3B에서는 target 0.63 — 모든 host에서 vanilla SFT를 따라잡지는 못함.
4. **이론적 보증 부재**: trust-region 영감이라 명시되어 있으나, exp(−L/τ) 게이팅이 어떤 trust-region constraint와 정확히 대응되는지 mathematical proof 없음.
5. **Code 미공개**: Shanghai AI Lab 출처지만 release 없음 → 재현 어려움.
6. **Prior retention 절대값은 여전히 낮음**: π0 LIBERO prior avg 0.34는 vanilla보다 우월하나, 사전학습 baseline (no fine-tune) 대비 여전히 큰 손실. 즉 **forgetting을 줄였을 뿐 완전 해결은 아님**.
7. **Single target task 가정**: multi-task fine-tuning에서의 작동 미검증.

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — exp(−L/τ) 게이팅의 단순함이 매력적. trust-region 해석은 신선하나 이론 미완. |
| **Technical depth** | ★★★☆☆ — 손실 한 줄 수정이 핵심. Sparsity mechanistic 분석은 좋으나 깊이 보강 필요. |
| **Experimental rigor** | ★★★★☆ — 3개 host(π0, π0.5, GR00T) × 2개 domain(LIBERO, RoboTwin) × 5개 baseline 비교 풍부 |
| **Practical impact** | ★★★★★ — 코드 한 줄 수정으로 +20%p retention. 즉시 채택 가능. |
| **Writing quality** | ★★★★☆ — 명확한 motivation, Appendix에 hyperparameter 충실 |

**강점**: Loss 함수 한 줄 수정만으로 catastrophic forgetting을 평균 20%p 줄임. Reference network/replay buffer 없이 작동해 추가 메모리 부담 없음. Trust-region 영감이 elegant. **약점**: τ scheduling이 task-dependent, code 미공개, 그리고 prior retention 절대값은 여전히 0.34 수준으로 완벽 보존과는 거리가 멈.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | sg[exp(−L/τ)]에서 stop-gradient가 왜 중요한가? | 만약 stop-gradient 없이 gradient가 흐르면 exp(−L/τ) 항이 L 자체에 의존해 음의 gradient를 만들 수 있음(loss 줄이려고 weighting을 키우는 ill-posed update). stop-gradient는 weighting을 "선택된 scalar"로 고정. |
| 2 | exp(−L/τ)는 사실상 importance weighting인데, 기존 weighted SFT와 무엇이 다른가? | 표준 weighted SFT는 weight를 외부에서 미리 정의(data difficulty / inverse frequency). ConSFT는 **model의 현재 신뢰도**로 weight를 동적 결정 → self-paced learning에 가까움. |
| 3 | Trust-region method와의 정확한 mathematical correspondence는? | 논문은 정성적 영감만 제시 — 엄밀히는 PPO나 TRPO의 KL 제약과 대응되지 않음. exp(−L/τ)는 오히려 **focal loss의 변형**에 가까움(잘 맞는 sample에 더 weight). |
| 4 | τ=0.003은 매우 작은 값인데 numerical stability는 괜찮은가? | L_SFT가 ~1.0 정도일 때 exp(−1/0.003)≈0 underflow 발생. ω_min=0.001이 floor 역할. 또한 κ=25.0으로 τ가 점진 증가 → 후반엔 stable. |
| 5 | π0.5 LIBERO-Spatial 100%가 의심스럽지 않은가? | 표 2에서 ±0.000 — 모든 seed에서 100% 달성. LIBERO-Spatial은 비교적 쉬운 task이고 π0.5 capacity가 충분. 다만 over-fitting 가능성. |
| 6 | GR00T-N1.6-3B에서 target 0.63인 이유? | GR00T는 humanoid focused로 LIBERO 같은 desktop manipulation에 sub-optimal. Capacity가 아니라 distribution mismatch. |
| 7 | RoboTwin Coord 0.13는 매우 낮은데 forgetting 완화가 의미 있는가? | Coord(dual-arm coordination)는 본질적으로 어려운 task. vanilla SFT는 ~0.0로 추정 → 0.13은 상대적으로 큰 보존. 다만 절대값은 사용가능 수준 미달. |
| 8 | Vanilla SFT에서 forgetting이 정말 그렇게 심한가? | 표 2 기준 prior avg ↓0.30~↓0.34 — 즉 사전학습 baseline 대비 30-34%p 손실. Catastrophic이라 부르기 적절. |
| 9 | LoRA와 직접 비교하면 어떠한가? | 표 3에서 LoRA는 target 성능은 ConSFT와 동등하나 prior retention에서 ConSFT가 우위. Architecture 변경 없는 점도 ConSFT 이점. |
| 10 | 왜 flow-matching VLA에 한정하는가? | 알고리즘 자체는 architecture-agnostic. 다만 저자가 π0/π0.5/GR00T에서만 검증 → autoregressive VLA(OpenVLA, π0-FAST)에서의 효과는 미검증. |

<!-- VERIFIED: pdf -->
