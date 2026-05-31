# Pre-VLA: Preemptive Runtime Verification for Reliable Vision-Language-Action and World-Model Rollouts 세미나 리뷰

> **한 줄 요약**: VLA 정책(RynnVLA-002, 개선된 WorldVLA/Chameleon 기반)이 산출한 action chunk를 **실행/world-model 상상 직전**에 평가하는 경량 dual-branch verifier(분류 head=safety confidence, 회귀 head=advantage). Focal + MSE + soft-threshold 다중 손실로 1.28M chunk 샘플을 학습하고, 추론 시 dual-mode preemptive resampling scheduler가 저품질 chunk를 기각/재샘플. **LIBERO 4-suite closed-loop 평균 30.79% → 37.62%** (+6.83pp), chunk당 forward 183.9 ms.

---

## 1. 배경 및 동기

대규모 VLA와 generative world model은 long-horizon embodied intelligence를 빠르게 발전시키고 있지만, 학습 기반 행동 생성에는 본질적 불확실성이 남는다:

- **실제 실행 측면**: 잘못된 action은 충돌/물체 손상 등 물리적 실패를 유발.
- **World-model 측면**: 상상된 rollout이 잘못된 action으로 시작하면, 이후 imagined trajectory 전체가 hallucinated chain을 따라 누적 오차를 키운다.

Pre-VLA의 출발점은 "행동을 실제로 실행하거나 상상에 넣기 *전(preemptive)*에, 후보 chunk가 안전하고 가치 있는지 사전 판단하는 runtime verifier가 필요하다"는 시스템 안전 관점이다.

---

## 2. 방법론 심층 분석

### 2.1 백본 — Improved WorldVLA

Pre-VLA는 새 백본을 만들지 않고 **개선된 WorldVLA**를 base policy로 사용한다. WorldVLA는 **Chameleon** vision-language model 위에 action/state 토크나이저를 추가해 **text + image + state + action을 단일 discrete token sequence로 표현**한다. 이로써 VLA 정책과 world model이 같은 시퀀스 공간을 공유 — Pre-VLA는 이 공유 구조 위에 verification 모듈을 얹는다.

### 2.2 Modality-Aware Pooling

Hidden state $H_t$에서 각 모달리티 토큰을 골라내기 위해, 미리 정의된 modality mask $M_m$ ($m \in \{\text{text}, \text{image}, \text{state}, \text{action}\}$)을 사용해 **Masked Mean Pooling**을 수행:

$$
z_m = \frac{\sum_i M_m[i] \cdot H_t[i]}{\sum_i M_m[i]}
$$

이렇게 만든 4개의 modality-specific 표현 $z_\text{text}, z_\text{image}, z_\text{state}, z_\text{action}$이 verifier head 입력이 된다.

### 2.3 Dual-Branch Verifier Head

| Branch | 출력 | 구조 |
|---|---|---|
| Classification | $p_t \in [0,1]$ (sigmoid) — **safety confidence** | 2-layer FFN |
| Regression | $\hat A^K_t \in \mathbb R$ — **critic-derived advantage** | 2-layer FFN |

두 head 모두 가볍게 설계되어 verifier overhead를 최소화.

### 2.4 Multi-Task 손실

심각한 class imbalance(positive 95% : negative 5%)와 경계 근처 라벨 불안정성을 다루기 위해:

- **Focal Loss** ($\alpha_\text{focal}=0.25, \beta_\text{focal}=2$): 쉬운 positive를 down-weight하고 rare hard negative에 집중
- **MSE Regression**: $\|\hat A^K_t - \tilde A^K_t\|^2_2$ (Eq. 10)
- **Soft-threshold Loss** (Eq. 11): $s_t = \sigma\!\big((\tilde A^K_t - \tau_A) / \tau_\text{temp}\big)$ — boundary calibration

$$
\mathcal L = \mathcal L_\text{cls} + \lambda_\text{soft} \mathcal L_\text{soft} + \lambda_\text{reg} \mathcal L_\text{reg}, \quad \lambda_\text{soft}=0.2, \lambda_\text{reg}=0.05
$$

### 2.5 Dual-Mode Preemptive Resampling Scheduler

추론 시:

1. VLA가 chunk를 산출 → verifier가 $p_t$와 $\hat A^K_t$ 계산
2. classification threshold(-0.21)보다 낮으면 **기각**하고 재샘플링 요청
3. 최대 retry 횟수 $N=5$ 안에서 동작 — compute budget 초과 방지
4. warmup 단계($T_w=20$)에서는 verifier 사용 보류

이중 모드(dual-mode)는 시뮬레이션의 **실제 실행 경로**와 world model의 **상상 경로** 모두에 동일한 logic을 적용한다.

---

## 3. 데이터셋 및 평가 프로토콜

- **학습 데이터**: 1,284,485 action chunk 샘플, 초기 positive:negative 비율 **95:5**
- **train/test split**: 92:8
- **dynamic batch sampling**: 매 배치에 negative 30% 강제 — class imbalance 보정
- **하드웨어**:
  - 학습: 8× NVIDIA H20D (~141GB each), DDP, per-GPU batch 512, 10 epoch, **총 8d 16h 43m**
  - 추론: 16× NVIDIA RTX 5090 (32GB each)
- **평가**: LIBERO 4 suite, closed-loop success rate

---

## 4. 실험 결과

### 4.1 Suite-Level Closed-Loop (Table III)

| Suite | RynnVLA-002 SR | **Pre-VLA SR** | Random Resamp | RynnVLA Steps | Pre-VLA Steps | Verifier fwd (ms) |
|---|---|---|---|---|---|---|
| LIBERO-Spatial | 0.3948 | **0.4566** | ~0.40 | 116.75 | **101.55** | 187.6 |
| LIBERO-Object | 0.0114 | **0.0395** | ~0.01 | 299.10 | **290.90** | 181.9 |
| LIBERO-Goal | 0.7225 | **0.8126** | ~0.72 | 166.15 | **164.50** | 178.1 |
| LIBERO-10 (Long) | 0.1029 | **0.1961** | ~0.10 | 267.25 | **262.28** | 188.0 |
| **Avg** | **30.79%** | **37.62%** | ~30.79% | — | — | **183.9** |

- **Random resampling은 거의 baseline과 동일** → 향상은 verifier의 신호 덕분이지, 단순 재시도 덕분이 아니다.
- **태스크 완수 step 수도 감소**: Spatial 116.75 → 101.55 (−13%), Object 299.10 → 290.90, 모든 suite에서 fewer steps.
- LIBERO-Object의 0.0114 → 0.0395는 절대 수치 자체가 낮음 — base RynnVLA-002가 Object suite에서 매우 약함을 시사. Pre-VLA는 verifier로 일부 개선했지만 4%대.

### 4.2 Pass Rate 분리 (Table II)

| Trajectory 유형 | Mean | Min | Max |
|---|---|---|---|
| **실패 trajectory** | 0.5167 | 0.4333 | 0.6036 |
| **성공 trajectory** | 0.8424 | 0.7931 | 0.9048 |

verifier가 성공 경로의 chunk를 84%로 통과시키고, 실패 경로의 chunk를 52%까지만 통과시킴 — 약 30pp의 분리력.

### 4.3 Offline Verifier Discrimination (Table IV)

| 메트릭 | 값 |
|---|---|
| F1 | **0.8303** |
| Accuracy | **0.9542** |
| Invalid precision | 0.7200 |
| Invalid recall | **0.9800** |
| False pass rate | **0.0200** |
| False reject rate | 0.0491 |

**False pass rate 2%**: 실제로 위험한 chunk가 verifier를 통과해 실행될 확률이 2%로 낮음.

### 4.4 평균 Resampling 시도 횟수 (Table III)

| Suite | Avg attempts |
|---|---|
| LIBERO-Goal | 1.4216 |
| LIBERO-Spatial | 2.0038 |
| LIBERO-Object | 1.3696 |
| LIBERO-10 | 1.2841 |

대부분 1-2회의 재샘플링 안에서 통과 — compute budget 부담이 비교적 적음.

---

## 5. Ablation 분석 (Table IV)

| 방법 | F1 | Accuracy | Invalid Precision | Invalid Recall | False Pass ↓ | False Reject ↓ |
|---|---|---|---|---|---|---|
| Raw Score Thresholding | 0.4112 | 0.7006 | 0.2651 | 0.9163 | 0.0837 | 0.3271 |
| Imbalance-aware Loss | 0.5078 | 0.7900 | 0.3466 | 0.9500 | 0.0500 | 0.2306 |
| **Pre-VLA (Full)** | **0.8303** | **0.9542** | **0.7200** | **0.9800** | **0.0200** | **0.0491** |

- Raw thresholding → Imbalance-aware: F1 0.41 → 0.51 (+10pp)
- + Soft-threshold + multi-task: F1 0.51 → **0.83** (+32pp) — soft-threshold calibration이 핵심
- False reject rate가 32.7% → 4.9%로 떨어지는 점이 실용 측면에서 결정적 (불필요한 재샘플링 감소)

---

## 6. 한계 및 의의

**한계**:
- **LIBERO-Object 절대 SR이 3.95%** — RynnVLA-002 base policy 자체가 Object suite에서 매우 약함. Pre-VLA가 verifier로 chunk를 거른다고 해도 base policy가 좋은 chunk를 *애초에* 만들지 못하면 한계가 명확.
- **RynnVLA-002 / WorldVLA에 강결합** — OpenVLA, π₀ 등 다른 백본에서의 plug-in 효과 미검증.
- **Verifier 자체의 파라미터 수와 학습 cost** 정확히 미보고 ("light 2-layer FFN", "8d 16h on 8×H20"만 명시).
- **World-model rollout에서의 정량적 error reduction** 수치가 abstract/본문에 명확하게 보고되지 않음 — 정성적 효과만 강조.
- Code/repository 공개 안됨 → 재현 어려움.
- 평균 **per-step inference time 1097.73 ms** — verifier overhead 183.9 ms는 작아 보이지만, 전체 step time에 resampling 1-2회 × VLA forward가 누적되어 실시간 제어와는 거리가 있음.

**의의**:
- "VLA = 정책 + runtime verifier"라는 시스템 관점은 안전한 실세계 배포에 직접 기여하는 관점.
- **183.9 ms verifier overhead로 LIBERO 4-suite 평균 +6.83pp**는 경량 verifier의 비용-이득이 매력적임을 보여줌.
- Multi-task loss(Focal + MSE + soft-threshold) 각각의 기여를 ablation으로 분해 — 단순 thresholding 대비 F1 2× 향상.
- World-model imagined rollout과 실제 실행 모두에 동일한 verifier를 적용하는 dual-mode 디자인은 Dreamer 류 world model + VLA의 결합 흐름에 잘 맞는 ingredient.
- 1.28M chunk 학습 데이터셋과 95:5 imbalance 처리 방법은 후속 verifier 연구의 reference로 활용 가능.

---

## 7. 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 |
|---|---|---|
| 1 | LIBERO-Object 3.95%는 너무 낮은 게 아닌가? | Base RynnVLA-002가 Object suite에서 1.14%로 처음부터 매우 약함. Pre-VLA의 가치는 절대 SR보다 +Δ로 보아야 함. 그래도 4%대는 실용 영역 밖 |
| 2 | False pass rate 2%는 안전한가? | 50 trial당 1 fail 정도 — 시뮬에서는 OK지만 실로봇에서는 그 1번이 충돌이면 큰 비용. 안전 critical 환경에서는 더 낮은 threshold 필요 |
| 3 | Random resampling baseline과 +Δ는 어느 정도인가? | 표에 "~baseline"으로만 표기 — 즉 random은 30.79%와 사실상 같음. 향상은 verifier signal 덕분이라는 강한 근거 |
| 4 | Verifier 학습이 8d 16h on 8×H20인데 worth it인가? | 학습은 1회성, 추론은 +183.9 ms/chunk. base policy 재학습보다 훨씬 저렴 |
| 5 | World-model rollout 실험의 정량 수치는? | abstract/본문에 명확한 error accumulation 감소 수치 부재 — 한계 |
| 6 | Verifier가 다른 VLA backbone에서도 동작하나? | RynnVLA-002 / WorldVLA의 unified token sequence에 강결합. 다른 backbone은 modality-aware pooling 구조부터 변경 필요 |
| 7 | Focal loss + soft-threshold 중 더 중요한 것은? | Ablation Table IV: Imbalance-aware(Focal) F1 0.51 → Full(soft-threshold 포함) 0.83 — **soft-threshold가 핵심** (+32pp) |
| 8 | per-step 1097ms는 실시간 제어가 가능한가? | 1 Hz 수준 → real-time closed-loop은 어렵고, slow manipulation 시나리오용. 가속화는 future work |

<!-- VERIFIED: pdf -->
