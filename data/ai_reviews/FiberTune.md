# FiberTune: Preserving Action-Fiber Visual Residuals in Vision-Language-Action Fine-Tuning

> **한 줄 요약**: VLA fine-tuning이 "action-supervised loss로는 제약되지 않는 action-equivalent visual 구조(색·카테고리·distractor·미래 관련 객체 상태)"를 붕괴시킨다는 점을 *local action fiber* 개념으로 형식화하고, 온라인 linear action probe로 action-predictive 방향을 필터링한 잔차 표현 Rpf에 대해 (i) 동결된 RADIO teacher에 cosine 정렬, (ii) effective-rank prior를 부여하는 **추론-time overhead 0**의 학습-time 보조 손실 FiberTune을 제안. 6개 통제 simulation 설정(CALVIN ABC->D, LIBERO x pi0.5/OpenVLA-OFT x general/adapted)과 물리 SO-101에서 task-loss-only baseline을 모두 능가; 대표 결과는 CALVIN adapted SR(5) 61.4% -> 72.1%(+10.7pp) 및 SO-101 held-out green block 46.9% -> 62.5%(+15.6pp).

---

## 1. 배경 및 동기

### 기존 VLA fine-tuning의 구조적 빈틈
- RT-1/RT-2, OpenVLA, OpenVLA-OFT, Octo, CogACT, pi0/pi0.5는 모두 **action-supervised fine-tuning**(continuous action regression / action chunking / flow-matching denoising 등)으로 사전학습 VL 표현을 로봇 제어로 적응시킨다.
- 그러나 action loss는 **predicted action을 바꾸는 방향만 제약**한다. 같은 short-horizon end-effector 명령을 유발하는 visual 변동(객체 색, 카테고리, 배경, 인접 distractor, 미래 관련 상태)은 task loss에 1차 gradient를 받지 못하고 fine-tuning 도중 **압축/붕괴**할 수 있다.
- 선행 연구(BlindVLA, MAPS, Spatial Forcing, RS-CL)는 "VLA fine-tuning이 시각 표현을 퇴화시킨다"는 현상을 보고했으나, **어떤 시각 구조를 보존해야 하는가**에 대한 sharp한 답은 없었다.

### 핵심 질문
- Action-predictive 성분을 *걸러낸 뒤* 남는 잔차 표현(action fiber에 tangent한 방향)을 명시적으로 보존하면 fine-tuning 행동이 개선되는가?
- 그렇다면 (i) teacher 정렬과 (ii) 잔차 spread 유지 중 어느 쪽이 dominant한가?

---

## 2. 방법론 심층 분석

### 2.1 Local Action Fiber 형식화

샘플 (v, l, a)와 중간 visual-token 표현 R_theta(v,l) in R^{Tv x d}에 대해, pooled feature R_bar 주변에서 local prediction map h_theta를 1차 선형화하면

```
h_theta(R + dR) ~= h_theta(R) + J_theta * d_r_bar
```

- Row(J_theta) = task loss가 직접 제약하는 fiber-crossing 방향.
- Null(J_theta) = 1차 task-loss 변화 없는 **action fiber에 tangent한 잔차** 방향 U_theta = R * (I - J^T (J J^T)^+ J).
- U_theta에 색·카테고리·distractor 등 action-equivalent 시각 구조가 존재.

### 2.2 Probe-Filtered Residual (이상 잔차의 계산 가능 근사)

대형 비선형 VLA의 Jacobian을 정확히 매 step 계산하는 것은 비현실적 -> **online linear action probe**로 1차 row-space만 추정.

```
W* = argmin_W E || A - W * R_bar ||^2   (detached pooled visual-token features, no-bias)
P_probe = W*^T (W* W*^T + eta * I)^-1 W*   (ridge, eta = 1e-4)
R_pf = R_theta (I - P_probe)               (probe-filtered residual)
```

- Probe는 자신의 action-regression objective로만 업데이트되며 정책 gradient에는 잡히지 않음.
- P_probe는 채널 단위 필터로 모든 visual token 위치에 공유 적용(transformer hidden state의 channel-wise 구조와 일치).
- *Proposition*: 모집단에서 R_bar~MVN, A = J0 * R_bar + eps(noise mean-zero), E[R_bar R_bar^T]가 full rank이면 ridge-less probe row space는 J0와 일치하며, ridge eta -> 0일 때 finite-sample probe가 모집단 필터로 수렴.

### 2.3 Teacher-Conditioned Residual Alignment (Lalign)

BlindVLA식 frozen teacher + fixed random adapter 프로토콜을 차용하되 *대상*을 R_pf로 바꿈:

```
Y_theta = norm(C_T * g(R_pf))            (token-centering C_T = I - 1/Tv * 11^T, row-norm)
Z_tilde = norm(C_T * Z),  Z = tau(V)     (frozen teacher RADIO c-radio v3-l)
L_align = -E_t[ cos(Y_theta,t, Z_tilde_t) ]
```

- Teacher: **RADIO c-radio v3-l** (Heinrich et al., 2025) 동결.
- Adapter g: per-token 독립 MLP, hidden 2048, 무작위 초기화 후 고정. 학습되지 않음. Token-axis centering이 평균 방향이 아닌 **token-relative 구조**에 cosine을 적용하도록 강제.
- vMF directional likelihood로 cosine 형태가 정당화됨: log p(Y|Z,kappa) = kappa * Y^T Z + const.

### 2.4 Effective-Rank Residual Prior (Lrank)

Alignment만으로는 H(U_theta | A, L) 항을 보장하지 못함 -- teacher와 상관된 소수 방향에 집중되어도 cosine은 만족됨. 따라서 batch-token 잔차 covariance의 spectral entropy를 정규화:

```
{lambda_i}_{i=1..K} = covariance eigenvalues of R_pf (before g)
p_i = lambda_i / sum_j lambda_j
H_shape(R_pf) = -(1 / log K) * sum_i p_i log p_i
L_rank = -H_shape(R_pf)
```

K = min(|B|*Tv - 1, d). Log-K 정규화로 차원-불변. VICReg / Barlow Twins의 covariance-spread anti-collapse 계열이지만 **action-filtered 잔차에만** 적용되는 것이 차별점.

### 2.5 최종 학습 손실

```
L = L_task + lambda_align * L_align + lambda_rank * L_rank
```

- Coefficient pair (lambda_align, lambda_rank) = (0.0125, 0.00625) (CALVIN/LIBERO), (0.003125, 0.0015625) (SO-101).
- Auxiliary 1k step warmup; 3k~6k(또는 10k~20k) step 사이 0.5배로 decay.
- Calibration rule: rho_aux = (lambda_align |L_align| + lambda_rank |L_rank|) / L_task의 tail value를 약 3%~6%로 유지.

### 2.6 추론 시점
- Teacher tau, probe P_probe, adapter g, 보조 손실 모두 **제거**. 배포되는 정책 가중치는 base VLA와 정확히 동일한 시그니처 -- 추가 latency / 메모리 0.

> **예상 질문**: 이게 새로운 VLA 모델인가 아니면 fine-tuning method인가?
> **답변**: Fine-tuning method이자 deliverable은 fine-tuned 정책(pi0.5+FiberTune, OpenVLA-OFT+FiberTune). 추론 그래프는 base VLA와 동일하나 학습 결과가 다르므로 leaderboard 등재 대상.

---

## 3. 데이터 전략

| 데이터셋 | 분량 | 용도 |
|---------|-----|-----|
| CALVIN ABC training split | (표준) | pi0.5 CALVIN fine-tuning |
| LIBERO 4-suite (Spatial/Object/Goal/LIBERO-10) | 10 tasks x 50 ep / suite | pi0.5 및 OpenVLA-OFT LIBERO fine-tuning |
| SO-101 90-demo set | 90 demos, yellow/orange/purple ID, green held-out | 물리 pick-place fine-tuning |

핵심: **데이터는 baseline과 완전히 동일**. 차이는 보조 손실뿐 -- 행동 차이가 FiberTune 자체에서 비롯됨을 보장하는 controlled protocol.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Visual-token layer (pi0.5) | prefix-tower layer 8 |
| Visual-token layer (OpenVLA-OFT) | layer 16 |
| Teacher | RADIO c-radio v3-l (frozen) |
| Adapter g | MLP hidden 2048, per-token, frozen after random init |
| Probe ridge eta | 1e-4 |
| Batch / accum / GPU (pi0.5 CALVIN/LIBERO/SO-101) | 8 / 4 또는 8 / 1~2 |
| Budget (CALVIN, LIBERO pi0.5, SO-101) | 6,000 steps |
| Budget (OpenVLA-OFT LIBERO) | 20k (general) / 15k (adapted) |
| LR schedule | 2.5e-5 -> 2.5e-6 cosine + 1k warmup (대부분), 5e-5 -> 5e-6 (LIBERO general), 1e-4 constant (OFT) |
| Hardware | >=48GB NVIDIA GPUs |

Source-policy provenance: lerobot/pi05_base, lerobot/pi05_libero, RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT, openvla/openvla-7b, moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10.

---

## 5. 실험 설계 및 평가 프로토콜

### 5.1 통제(controlled) 비교 규약
모든 비교에서 **starting checkpoint, training data, optimization budget, evaluator, model-selection rule**을 baseline과 FiberTune 사이 고정. FiberTune-specific 선택(layer, span, lambda)은 backbone별 결과 수집 *전에* 결정 후 고정.

### 5.2 시뮬레이션 벤치마크
- **CALVIN ABC->D**: 1000 chain-evaluation trials, RLinf/OpenPI bridge, avg. completed subtasks + SR1~SR5.
- **LIBERO**: 4-suite x 10 tasks x 50 episodes = 2000 episodes / 비교.

### 5.3 물리 평가 (SO-101)
- 128 trials/arm = 32 trials x 4 colors. Yellow/orange/purple = ID, **green = held-out OOD**.
- Single-task pick-place ("pick up the {color} block and put it into the black box").
- Stage breakdown: contact / lift / task success.

### 5.4 표현 진단(diagnostic)
- **Linear CKA** (Kornblith et al., 2019): probe-filtered residual R_pf와 매칭된 teacher feature 간 정렬도, adapter g **이전**에서 측정.
- **Participation-ratio effective rank**: d_eff = (sum lambda_i)^2 / sum lambda_i^2. 학습 손실의 spectral entropy와는 **다른** 통계 -- 보조 학습이 본 측정량을 직접 최적화하지 않음을 보장.
- 진단용 probe는 4-fold cross-fit(CALVIN/pi0.5-LIBERO) 또는 in-sample(OFT). Diagnostic target: 7D per-step action.

---

## 6. 주요 결과 정량 비교

### 6.1 Main Behavior Table (Table 1)

| 벤치마크 | 정책 | 초기화 | 메트릭 | Baseline | **FiberTune** | 이득 |
|---------|------|-------|-------|----------|---------------|------|
| CALVIN ABC->D | pi0.5 | general | Avg. seq. len. | 0.796 | **1.012** | +0.216 |
| CALVIN ABC->D | pi0.5 | general | SR(5) (%) | 0.4 | **1.3** | +0.9 |
| CALVIN ABC->D | pi0.5 | adapted | Avg. seq. len. | 3.837 | **4.116** | +0.279 |
| CALVIN ABC->D | pi0.5 | adapted | SR(5) (%) | 61.4 | **72.1** | **+10.7** |
| LIBERO | pi0.5 | general | SR (%) | 92.35 | **93.35** | +1.00 |
| LIBERO | pi0.5 | adapted | SR (%) | 95.75 | **97.10** | +1.35 |
| LIBERO | OpenVLA-OFT | general | SR (%) | 42.50 | **48.95** | **+6.45** |
| LIBERO | OpenVLA-OFT | adapted | SR (%) | 95.35 | **96.15** | +0.80 |

6/6 통제 설정에서 baseline 능가. Wilson 95% CI 모두 분리됨(예: CALVIN adapted SR(5) Baseline [58.3, 64.4] vs FiberTune [69.2, 74.8]).

### 6.2 LIBERO Per-suite (Table C.6, pi0.5 adapted seed 1000)

| Suite | Spatial | Object | Goal | LIBERO-10 | Avg |
|------|--------|-------|-----|----------|-----|
| Baseline | 96.4 | 98.0 | 95.6 | 93.0 | 95.75 |
| FiberTune | 97.8 | 98.8 | 97.8 | 94.0 | **97.10** |

3-seed pooled (7/42/1000): Baseline 96.12% vs FiberTune 96.82% (+0.70 pp).

### 6.3 SO-101 물리 결과 (128 trials/arm)

| 항목 | Baseline | **FiberTune** | 이득 |
|-----|----------|---------------|------|
| Contact | 93.8% | **96.9%** | +3.1 |
| Lift | 81.2% | **82.8%** | +1.6 |
| Task success (all colors) | 72.7% | **78.1%** | **+5.5** |
| ID colors | 81.2% | **83.3%** | +2.1 |
| **OOD green (held-out)** | 46.9% | **62.5%** | **+15.6** |

OOD green의 큰 폭 개선은 author의 action-fiber 가설(색이 action-equivalent 잔차에 산다)과 직접 부합.

### 6.4 표현 진단 (Table 3)

| 설정 | Method | 행동 | Residual CKA | Residual eff. rank |
|------|--------|-----|--------------|-------------------|
| CALVIN/pi0.5/adapted | Baseline | 3.837 | 0.202 | 8.35 |
| CALVIN/pi0.5/adapted | **FiberTune** | 4.116 | **0.467** | **58.84** |
| LIBERO/pi0.5/adapted | Baseline | 95.75 | 0.194 | 6.37 |
| LIBERO/pi0.5/adapted | **FiberTune** | 97.10 | **0.414** | **44.88** |
| LIBERO/OFT/general | Baseline | 42.50 | 0.380 | 39.89 |
| LIBERO/OFT/general | **FiberTune** | 48.95 | **0.717** | **52.26** |

6/6 설정에서 residual CKA와 effective rank가 동시에 상승 -- 행동 개선이 잔차 보존과 일치한다는 증거.

---

## 7. Ablation 분석 (Table 4, CALVIN)

| Variant | Teacher signal | Rank prior | Avg seq | SR(5)% |
|--------|----------------|-----------|---------|--------|
| Baseline | none | no | 3.837 | 61.4 |
| Full-token align (BlindVLA-style) | full tokens | no | 3.685 | 58.3 |
| Full-token + rank | full tokens | yes | 3.956 | 65.6 |
| Residual align only | pf-residual | no | 3.980 | 66.2 |
| **Residual rank only** | none | yes | 4.064 | **70.5** |
| **FiberTune (full)** | pf-residual | yes | **4.116** | **72.1** |

핵심 발견:
- **Full-token alignment 단독은 baseline보다 낮음** (3.685 < 3.837) -- BlindVLA식 보존 압력을 CALVIN 장기적 fine-tuning에 그대로 적용하면 in-distribution long-horizon 실행에 필요한 방향까지 제약함.
- **Rank prior 단독이 dominant component** (3.837 -> 4.064; SR5 +9.1pp). Action-orthogonal 잔차 subspace의 spread 유지가 가장 큰 동인.
- Residual alignment는 incremental하게 +0.05 Avg / +1.6 SR(5)pp를 추가.
- Probe filtering이 alignment의 부호를 바꾼다(full-token은 -0.152, pf-residual은 +0.143)는 점이 *action-fiber 절단*의 본질적 가치를 시사.

---

## 8. 한계 / 단점

저자가 명시한 한계:
1. **학습-time overhead**: Frozen teacher + online probe + auxiliary losses가 fine-tuning 동안 추가됨(inference는 불변).
2. **1차 pooled-feature 근사**: R_pf는 local action-fiber 잔차의 1차 + pooled-feature 근사. Action prediction이 강하게 비선형이거나 visual token 위치가 action에 이질적으로 기여할 때 근사가 깨질 수 있음.
3. **Benchmark-specific 고정 coefficient**: rho_aux를 3~6%로 유지하는 단기 calibration이 필요하며 벤치마크 간 자동 전이 안 됨.
4. **물리 증거의 폭이 제한적**: 단일 task(SO-101 pick-place), 단일 manipulator. 더 다양한 실세계 조건에서의 failure mode는 미평가.

리뷰어 추가 코멘트:
- Auxiliary coefficient의 ratio 2:1(align:rank)이 어떻게 도출됐는지에 대한 systematic search가 없음(short calibration이라고만 명시).
- RADIO 외 다른 teacher(DINOv2, SigLIP 등) 비교 부재.
- "Layer 8 (pi0.5) / Layer 16 (OFT)" 선택의 sensitivity 분석은 부록에서도 명시적이지 않음.

---

## 9. 후속/관련 작업과의 연결

- **BlindVLA (Kachaev et al., 2025)**: Frozen-teacher + fixed-adapter cosine 정렬 프로토콜 차용. 차이는 (i) 정렬 대상이 R_pf로 잔차화, (ii) rank prior 추가.
- **MAPS (Huang et al., 2025)**: Module-wise proximity scheduling으로 VL 표현 보존. 전체 표현에 작용; FiberTune은 잔차에만 작용.
- **Spatial Forcing (Li et al., 2025)**: VLA 특성을 spatial foundation 표현에 정렬. Teacher space가 다름.
- **RS-CL (Kim et al., 2025)**: Robot-state-aware contrastive regularization. State metric 기반; FiberTune은 visual residual 기반.
- **VICReg / Barlow Twins**: Anti-collapse covariance regularization 계열의 잔차 적용판.
- **RankMe (Garrido et al., 2023)**: Effective rank diagnostic 채용.
- **LangForce (Lian et al., 2026)**: VLA의 latent action query에 대한 conditional information 분해 -- FiberTune의 H(U|Z,A,L) 분해와 같은 흐름.

---

## 10. 학문적 의의

- **표현 보존 문제를 "조건부 상호정보 보존"으로 명료화**: I(U; Z_c | A, L) 분해(잔차 spread + teacher predictability)가 design principle을 제공.
- **"무엇을 보존할지"의 sharpening**: 'visual representation을 보존하라'에서 'action-orthogonal residual을 보존하라'로 정밀화 -- 전체 token 보존은 ID long-horizon 실행과 충돌할 수 있음을 ablation으로 입증.
- **추론-비용 0의 fine-tuning regularizer 패턴**: 임의의 backbone(pi0.5, OpenVLA-OFT) 위에 plug-in 가능. Cross-architecture 일반성을 6/6 settings로 보임.
- **물리 OOD 일반화 증거**: 같은 fine-tuning data로 held-out 색에서 +15.6pp는 representation collapse가 deployment-critical하다는 직접 증거.

---

## 11. 산업적 함의

- **Inference budget 변동 0**: 기존 VLA 배포 파이프라인(pi0.5, OpenVLA-OFT 등) 그대로 사용 가능. 학습 단계에만 RADIO teacher(추가 forward pass 1회) + 작은 MLP + 선형 probe를 더하면 됨.
- **OOD 색 일반화 효과**가 큼 -- pick-and-place 류 서비스 로봇/제조 자동화에서 *데이터 수집 없이* deployment-time 색 변동에 강건성 추가 여지.
- 한 backbone에서 결정한 lambda를 다른 backbone에 그대로 옮기지 못함은 운영 측면 단점 -- 새로운 모델/도메인마다 calibration run 필요.
- **Probe + adapter + teacher = 메모리 헤더**: 학습 GPU >=48GB 권장. 양자화/LoRA와의 호환성은 미평가.

---

## 12. 결론

FiberTune은 "action-supervised fine-tuning이 action-equivalent visual 구조를 1차 무 gradient 영역으로 방치한다"는 점을 *local action fiber* 관점으로 형식화하고, 이를 막기 위한 **추론 무비용, 학습-time 단독** 보조 손실(probe-filtered residual cosine alignment + spectral-entropy rank prior)을 제안한다. 

핵심 contribution은 세 가지:
1. **이론**: action-fiber 분해 + I(U; Z_c | A, L) 정보론적 보존 원리.
2. **방법**: online linear probe로 action-predictive 방향을 추정·필터링하여 잔차 R_pf에만 teacher alignment와 rank prior를 적용 -- BlindVLA식 전체 정렬의 ID 실행 충돌을 회피.
3. **실증**: CALVIN, LIBERO(pi0.5 + OpenVLA-OFT 두 architecture, general + adapted 두 초기화), 물리 SO-101 모든 통제 설정에서 task-loss baseline을 능가; ablation으로 rank prior가 dominant, probe filtering이 alignment의 sign을 바꾼다는 점 확인; CKA/effective rank diagnostic이 행동 개선과 동행.

가장 인상적인 단일 결과는 **CALVIN ABC->D SR(5) 61.4% -> 72.1% (+10.7pp)**와 **SO-101 held-out green 46.9% -> 62.5% (+15.6pp)** -- 동일 데이터·동일 budget으로 representation-only regularization이 장기-horizon 실행과 색-OOD 일반화에 모두 의미 있는 영향을 줄 수 있음을 보인 사례.

향후 방향: (i) auxiliary coefficient의 cross-benchmark 자동 balancing, (ii) 비선형 / token-위치별 이질적 action map에서의 high-order action-fiber projector, (iii) 다양한 visual teacher 비교 및 real-world failure mode characterization.

<!-- VERIFIED: pdf -->
