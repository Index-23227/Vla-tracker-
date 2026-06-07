# ELAN4D: Embodiment-Centric 4D Supervision for Vision-Language-Action Models via Plug-and-Play Adaptation

> **한 줄 요약**: 외부 트래커 없이 로봇 자체의 proprioception(joint angle)과 forward kinematics만으로 **로봇 keypoint의 미래 변위(4D track)** 를 생성하고, 이를 ControlNet 스타일의 **gradient-isolated 보조 branch** 로 pi_0 / pi_0.5 backbone에 plug-and-play 주입하여 LIBERO 97.0% / LIBERO-Plus +14.0% / 실제 환경 spatial generalization +50%p를 달성한 embodiment-centric 4D supervision 프레임워크.

---

## 1. 배경 및 동기

### 기존 VLA의 구조적 한계
- pi_0 / pi_0.5, OpenVLA 등 SOTA VLA는 **future RGB pixel** 또는 **2D point track** 을 보조 supervision으로 사용
- 그러나 (a) 외부 tracker(CoTracker, TAPIR 등)는 GPU 수 시간을 소모하고, (b) full-scene track은 robot이 아닌 distractor / 카메라 motion까지 포함하여 noisy signal을 만든다
- 또한 query token 방식으로 VLM이 직접 4D를 예측하게 하면 backbone representation이 drift 되어 **본업(action prediction) 성능이 떨어진다**

### 핵심 질문
- **로봇 자신의 미래(4D)** 만 supervision으로 쓰면 충분한가? Scene 전체를 추적하지 않아도 되는가?
- 4D 신호를 **backbone을 건드리지 않고** plug-in 할 수 있는가?

📌 [Figure 1 삽입] — ELAN4D 아키텍처: PaliGemma + Action Expert + ControlNet residual branch + Track Decoder

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

ELAN4D는 pi_0 / pi_0.5 (PaliGemma VLM + flow-matching action expert)를 **건드리지 않고** 다음 두 모듈만 추가:

1. **Control Branch (b_psi)**: backbone hidden state u_t를 stop-gradient한 뒤 residual feature C_t를 산출
2. **Track Decoder**: Point MLP + Control MLP + Fusion MLP → **H x K x 3** 변위 예측 (H: future horizon, K: keypoint 수)

Residual 융합:

```
u_tilde_t = u_t + Proj(C_t),   C_t = b_psi(sg(u_t))
```

`Proj`는 **zero-initialized** projection — 학습 초기에 backbone 출력을 그대로 보존하고 점진적으로 4D 정보를 주입

### 2.2 4D 신호 생성: 외부 tracker 제로

- 로봇 URDF + joint angle q_t → **forward kinematics** → keypoint 위치 p_t^k ∈ R^3
- Future horizon H 동안의 **변위** Δp_{t+h}^k = p_{t+h}^k − p_t^k 가 supervision target
- LIBERO K=8, RoboTwin2.0 K=14, real-world K=7

> ❓ **예상 질문**: 외부 tracker(CoTracker) 대비 cost 차이는?
> **답변**: full-scene track은 일반적으로 episode당 ~4 GPU-hours가 필요한 반면, ELAN4D의 robot-only track은 **CPU에서 ~1분**. 데이터 규모가 커질수록 격차가 압도적이다.

### 2.3 손실 함수

```
L = L_act + lambda_track * L_track,   lambda_track = 0.1
```

- L_act: flow-matching action loss (pi_0 / pi_0.5 원본 유지)
- L_track: predicted vs FK-derived displacement L2

### 2.4 Gradient Isolation의 의미

`sg(u_t)` (stop-gradient)와 zero-init projection의 조합은 다음을 보장:

- L_track의 gradient는 **control branch만** 갱신
- L_act의 gradient는 backbone과 action expert를 정상적으로 갱신
- **backbone CKA가 baseline과 거의 동일** (ablation에서 검증) → representation drift 차단

> ❓ **예상 질문**: query token으로 VLM이 직접 4D를 예측하면 안 되는가?
> **답변**: 가능하지만 본 논문 ablation에서 **66.8% (-6.8%p)** 로 오히려 악화. CKA 분석상 backbone의 task-relevant feature가 4D 예측 쪽으로 잠식되어 action 성능이 깎인다.

---

## 3. 데이터 전략

| 환경 | 데이터 | 규모 | Keypoints K |
|------|--------|------|-------------|
| LIBERO | LIBERO-90 / Spatial / Object / Goal / Long | ~2K demos | 8 |
| LIBERO-Plus | OOD 변형 (camera, background, distractor 등) | LIBERO 동일 + perturbation | 8 |
| RoboTwin 2.0 | Bimanual manipulation | 다양한 태스크 | 14 |
| Real-world | Visual robustness / spatial generalization / temporal reasoning | 직접 수집 | 7 |

핵심 포인트: ELAN4D는 **시뮬레이션-only**, **bimanual**, **real-world** 세 도메인 모두에서 동일한 recipe로 동작 — plug-and-play 주장의 근거

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Steps | 30K |
| Optimizer | AdamW, LR 2.5e-5 |
| Batch Size | 64 |
| Hardware | 8 x NVIDIA GH200 |
| Backbone | PaliGemma + Action Expert (pi_0 / pi_0.5) |
| lambda_track | 0.1 |

> ❓ **예상 질문**: 8 GH200으로 30K step이면 얼마나 가벼운 추가 비용인가?
> **답변**: pi_0.5 baseline과 동일 수준 — 즉 **거의 free**. Control branch는 lightweight MLP-based로 added param이 backbone 대비 무시할 수준.

---

## 5. 실험 결과 심층 분석

### 5.1 LIBERO (Table 2)

| 모델 | Spatial | Object | Goal | Long | **Avg** |
|------|---------|--------|------|------|---------|
| pi_0 baseline | - | - | - | - | 94.2 |
| **ELAN4D(pi_0)** | - | - | - | - | **95.0** (+0.8) |
| pi_0.5 baseline | - | - | - | - | 96.9 |
| **ELAN4D(pi_0.5)** | 97.4 | 99.0 | 96.4 | 95.2 | **97.0** (+0.1) |

- 표준 LIBERO에서는 baseline이 이미 saturated (~97%) → 절대값 차이는 작지만 **LIBERO-Long에서 +6.6%p** 의 의미 있는 향상

### 5.2 LIBERO-Plus (Table 1, OOD) — **핵심 결과**

| 모델 | Overall |
|------|---------|
| pi_0 baseline | 53.6 |
| **ELAN4D(pi_0)** | **67.6** (+14.0) |
| pi_0.5 baseline | 73.6 |
| **ELAN4D(pi_0.5)** | **78.2** (+4.6) |

- camera / background perturbation에서 가장 큰 gain
- "본 supervision이 task-irrelevant visual nuisance에 대한 robustness를 부여한다"는 핵심 주장의 정량적 근거

### 5.3 RoboTwin 2.0 (Bimanual)

| 모델 | Avg |
|------|-----|
| pi_0 baseline | 12 |
| ELAN4D(pi_0) | 15 (+3) |
| pi_0.5 baseline | 32 |
| **ELAN4D(pi_0.5)** | **37 (+5)** |

- Dump Bin (+12), Lift Pot (+10) 등 양손 협조가 중요한 태스크에서 유의미한 gain

### 5.4 Real-world

| 평가 영역 | baseline | ELAN4D |
|----------|---------|--------|
| Visual Robustness | 50% | **80%** |
| Spatial Generalization | 15% | **65%** |
| Temporal Reasoning | 5% | **45%** |

- Spatial generalization **+50%p**, temporal reasoning **+40%p** — 시뮬레이션 saturation을 뚫고 real에서 **dramatically** 효과 발휘
- 단, real 평가 episode 수가 작아 통계적 신뢰구간은 불명확

---

## 6. Ablation 분석

### 6.1 4D supervision이 진짜 원인인가? (parameter ablation)

| 설정 | LIBERO-Plus |
|------|-------------|
| baseline (pi_0.5) | 73.6 |
| Control branch만 추가, L_track 제거 | 73.3 |
| **ELAN4D (control branch + L_track)** | **78.2** |

- **순수 파라미터 증가 효과 = 0** → gain은 4D signal에서 나온다

### 6.2 VLM이 직접 4D 예측 vs control branch

| 설정 | LIBERO-Plus |
|------|-------------|
| Query token 방식 (backbone이 직접 4D 예측) | 66.8 |
| **Control branch (gradient-isolated)** | **78.2** |

- query token 방식은 baseline보다도 **악화** (-6.8%p)
- CKA 분석에서 backbone representation이 변형됨을 확인 → **gradient isolation이 결정적**

### 6.3 Whole-scene track vs robot-only track

| Track 종류 | LIBERO-Plus | Cost |
|-----------|-------------|------|
| Whole-scene (privileged) | 79.3 | ~4 GPU-hours/episode |
| **Robot-only (ELAN4D)** | **78.2** | **~1 CPU-min/episode** |

- whole-scene은 GT-level 정보를 추가했음에도 **+1.1%p** 차이만 발생 → **robot motion이 supervision의 핵심**임을 시사

### 6.4 데이터 효율

| Data ratio | baseline | ELAN4D |
|-----------|---------|--------|
| 20% | ~60 | 75.0 |
| 40% | ~67 | - |
| 60% | ~71 | - |
| 80% | - | - |

- ELAN4D@20% ≈ pi_0.5@30% (1.5배 데이터 효율)

---

## 7. 관련 연구 비교

| 모델 | 4D supervision | Source | Backbone modification |
|------|---------------|--------|----------------------|
| ATM | 2D point track | CoTracker | end-to-end (drift 위험) |
| Im2Flow2Act | 2D flow | RAFT | end-to-end |
| ManiTrack | 2D point track | CoTracker | end-to-end |
| **ELAN4D** | **3D robot keypoint displacement** | **Forward kinematics (free)** | **gradient-isolated plug-in** |

핵심 차이: (1) 3D + future, (2) 외부 모델 의존 0, (3) backbone preservation 보장 — 동시에 만족하는 유일한 방법

---

## 8. 한계 및 미해결 문제

1. **Backbone 의존성**: 모든 실험이 pi_0 / pi_0.5에서만 수행. OpenVLA, RT-2, RDT-1B에도 transferable 한지 확인 안 됨
2. **Multi-robot 일반화**: 모든 실험이 단일 robot embodiment 또는 bimanual 한 종에서 수행. cross-embodiment에서 forward kinematics + URDF 의존이 어떻게 작동하는지 불분명
3. **Real-world episode 수**: 20 episode 수준의 평가는 +50%p gain의 통계적 신뢰성을 약화
4. **Track horizon H의 선택**: H를 어떻게 선택했는지, sensitivity는 어떤지 본문에서 명확하지 않음
5. **Saturation 영역**: 표준 LIBERO 97% 영역에서는 gain이 0.1%p로 미미 — saturation에서 더 가치를 보이지 못함
6. **Open-loop vs closed-loop**: pi_0.5 기반이므로 closed-loop이긴 하나 long-horizon real task에서의 stability 분석 부재

### Attribution 우려
- LIBERO-Plus의 +14.0%p가 정말 "4D" 때문인지, 단순한 "robot-centric inductive bias" 때문인지 분리 불가
- LIBERO 표준에서 marginal gain은 supervision이 OOD에서만 유효함을 시사 — 이는 dataset의 saturation 가능성을 더 의심하게 만든다

---

## 9. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — forward kinematics를 supervision으로 사용한 점은 단순하면서 강력 |
| **Technical depth** | ★★★★☆ — gradient isolation + ControlNet-style residual의 결합이 잘 설계됨 |
| **Experimental rigor** | ★★★★☆ — ablation이 충실 (parameter, gradient isolation, whole-scene 비교, data scaling 모두 포함) |
| **Practical impact** | ★★★★★ — plug-and-play, free supervision, real-world에서 50%p gain |
| **Writing quality** | ★★★★☆ — 표 구성과 비교가 명확 |

**강점**: 외부 tracker 의존을 제거하고도 OOD robustness를 크게 끌어올린 점. 본질적으로 "robot은 자기 자신의 미래를 가장 잘 안다"는 통찰을 단순한 FK 한 줄로 supervision으로 변환. **약점**: backbone scope가 PaliGemma 계열에 국한되었고, real-world 평가 통계 신뢰성이 약하다. 표준 LIBERO에서 saturation으로 인해 절대값 효과가 작아 보이는 것도 marketing에 불리.

---

## 10. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | gradient isolation이 핵심이면 그냥 다 freeze 하면 안 되나? | backbone freeze는 action 학습 자체를 막음. ELAN4D는 L_act에는 gradient flow를 허용하되 L_track에서만 차단 — selective isolation |
| 2 | K=8 keypoint는 어떻게 선택했는가? | 본문 명시는 URDF의 주요 link/joint frame을 따른다. Sensitivity ablation은 부재 — 단점 |
| 3 | RT-2, OpenVLA에도 작동하는가? | 미검증. flow-matching action expert를 가진 모델(pi_0 계열)에 더 자연스럽게 결합되는 구조 |
| 4 | forward kinematics가 noisy joint encoder에서 잘 작동하는가? | LIBERO/RoboTwin은 sim → noiseless. Real-world (Franka 등)에서는 encoder noise 영향 미보고 |
| 5 | whole-scene track이 +1.1%p만 좋다면, scene track 방법이 over-engineered였다는 뜻인가? | 부분적으로 맞음. ELAN4D는 "robot motion만으로도 거의 동일 효과"를 보임 — scene track의 비용 대비 효과 비판 |
| 6 | LIBERO 97%에서 +0.1%p는 의미 있나? | 표준 LIBERO에서는 의미 적음. 가치는 LIBERO-Plus와 real-world에 있음 |
| 7 | Real-world spatial gen 15→65는 너무 극적인데 신뢰 가능? | episode 수 작음. 재현성 검증 필요 |
| 8 | lambda_track=0.1 민감도는? | 본문 sweep 부재. 0.1이 "small auxiliary" 직관이라는 정도만 정당화됨 |

<!-- VERIFIED: pdf -->
