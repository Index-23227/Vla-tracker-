# RotVLA: Rotational Latent Action for Vision-Language-Action Model

> **한 줄 요약**: 기존 Latent Action Model(LAM)이 의존하던 *이산 VQ-VAE 양자화*를 **SO(n) 위의 연속 회전 잠재 action**으로 대체하고 **triplet frame learning** 으로 frame-reconstruction 붕괴를 차단한 뒤, InternVL3.5-1B + 24-layer DiT 기반 flow-matching expert에 latent + 실제 action을 함께 디노이즈시키는 *unified action expert*로 확장하여, **1.7B 파라미터 / 1700+ 시간 pretraining**으로 LIBERO 98.2%, RoboTwin2.0 89.6%/88.5% (clean/randomized)를 달성한 Peking University · Xiaomi Robotics · CASIA의 VLA.

---

## 1. 배경 및 동기

### Latent Action Model(LAM)의 구조적 문제
VLA pretraining에서 heterogeneous 데이터(서로 다른 embodiment, action space)를 통합하기 위한 유력 패러다임이 LAM이다. 그러나 기존 LAM은 거의 모두 **VQ-VAE 류 이산 양자화** encode/decode 파이프라인에 의존하며, 저자들은 이로부터 세 가지 결함이 발생한다고 지적한다.

1. **Trivial reconstruction 붕괴** — 디코더가 latent를 통하지 않고도 target frame을 그대로 복원하는 shortcut을 학습 (논문 [10] CLAM에서 보고).
2. **표현력 부족** — 이산화는 본질적으로 연속인 physical action의 dynamics를 표현하기 어려움.
3. **물리적 무의미성** — scale·composition 같은 물리적 구조가 latent 공간에 부재.

### 핵심 질문
- *"latent action을 SO(n)이라는 연속 회전군 위에 모델링하면, 세 문제를 동시에 해결할 수 있는가?"*
- *"잠재 action을 단순한 supervision이 아니라 high-level planner로 활용했을 때, 다운스트림 control 성능이 정말 개선되는가?"*

📌 [Figure 1·2 삽입] — 기존 LAM(discrete VQ)와 RotVLA(연속 SO(n) + triplet)의 대비

---

## 2. 방법론 심층 분석

RotVLA의 학습은 세 단계로 구성된다.

### 2.1 Stage I — 연속 회전 잠재 action 학습

**SoftVQ로 연속성 확보.** VQ-VAE를 **SoftVQ** [11] 로 교체. soft categorical 분포가 코드북 위 여러 codeword를 가중평균으로 결합하여, codebook 구조를 유지하면서도 latent를 연속화한다.

**SO(n) 사영.** Latent action을 그대로 n×n 회전 행렬로 강제하면 표현력이 과도하게 제약된다는 것을 실험적으로 발견하고, 저자들은 먼저 unconstrained 행렬 M = UΣV^T 를 예측한 뒤 **SVD-based orthogonal projection**으로 가장 가까운 회전 행렬을 얻는 우회로를 택한다.

$$\text{Proj}(M) = U\,\text{diag}(1, 1, \ldots, \det(UV^\top))\,V^\top$$

이 사영은 orthogonality와 unit determinant를 보장하면서도 학습 중 expressive flexibility를 유지한다. n은 16으로 설정 (n=8/32 대비 LIBERO 평균 97.3 → 98.2).

**Triplet frame learning.** Trivial reconstruction을 차단하기 위해 세 프레임 (I_t, I_{t+1}, I_{t+2}) 을 동시에 사용한다.
- 단일-step: ẑ_{t→t+1}, ẑ_{t+1→t+2} 두 transition을 각각 reconstruct (L_single).
- **합성 step**: 동일 프레임 쌍(I_t, I_t)에서 batch-wise mean으로 identity element z_I를 추정하고, $z^{comp}_{t\to t+2} = z_{t+1\to t+2}\cdot z_I^{-1}\cdot z_{t\to t+1}$ 로 두-step 합성 action을 만들어 I_{t+2}를 한 번에 예측 (L_comp).
- 최종 손실: $\mathcal{L}_{triplet} = \mathcal{L}_{single} + \mathcal{L}_{comp} + \mathcal{L}_{soft}$ (KL).

> ❓ **예상 질문**: SO(n) 사영 대신 회전 표현을 직접 학습하지 못한 이유는?
> **답변**: 저자들이 명시적으로 보고 — "회전 행렬 형태를 직접 강제하면 representational capacity가 과도하게 제약"되기 때문. unconstrained M을 먼저 학습한 뒤 사영을 적용하는 것이 학습 안정성과 표현력의 trade-off에서 우월. abs EEF action도 Rotate6D representation [50] 으로 두어 latent와 자연스럽게 정합.

> ❓ **예상 질문**: Triplet의 효과를 정량적으로 어떻게 확인했나?
> **답변**: Table 2에서 imagined-frame error Δ가 Recon-Only(UniVLA) 0.0037 → Triplet 0.0048로 증가. 즉 동일 latent를 다른 frame I_{t+1}에 적용했을 때 결과가 *덜 비슷해야* 진짜 motion을 학습했다는 뜻이고, Triplet 쪽이 그 gap이 명확히 크다. Ablation에서도 Cont. SO(n) 단독은 97.0%인데 Triplet을 더한 RotVLA는 98.2% (+1.2pp).

### 2.2 Stage II — VLA Pretraining

- **VLM backbone**: InternVL3.5-1B (304M ViT + 752M LLM).
- **Action expert**: 24-layer Diffusion Transformer (DiT) [49], 305M.
- **LAM**: 290M (frozen after Stage I).
- 총 1.7B 파라미터.

Latent action z_{t→t+1} ∈ R^{n×n}을 n-차원 action chunk(horizon n)로 취급하고, flow-matching 목적함수 $L^{FM}_{LA}$ 로 velocity field $v_\theta(z_\tau, \tau, h)$ 를 학습. h는 VLM의 vision+language token 표현.

### 2.3 Stage III — RotVLA Finetuning (Unified Action Expert)

**핵심 디자인.** 다운스트림에서 flow-matching expert를 latent + 실제 robot action을 **동시에 디노이즈**하는 단일 expert로 확장. joint variable $x = (a, z_{t\to t+1})$ 에 대해

$$\mathcal{L}^{FM}_{LA-RA} = \mathbb{E}\big[\|v_\theta(x_\tau, \tau, h) - (x - x_0)\|^2\big]$$

**Structured attention.** 정보 흐름을 명시적으로 제약:
- latent action token → vision/language token만 attend (Pretraining과 정합).
- robot action token → latent action token + vision/language token 모두 attend.

이로써 latent가 "planner", action이 "controller" 역할로 분리되며, pretraining의 cross-embodiment action semantic이 유지된다.

> ❓ **예상 질문**: w/o. Planner ablation에서 latent를 떼어내면 성능이 얼마나 떨어지는가?
> **답변**: Table 4 — 평균 98.2 → 96.5 (특히 LIBERO-Long 96.4 → 93.2). latent를 단순한 pretraining supervision이 아니라 *runtime planner*로 유지하는 것이 long-horizon에 결정적.

---

## 3. 데이터 전략

| 카테고리 | 비율 | 주요 데이터 |
|---------|------|------------|
| Open X-Embodiment | 31.43% | BC-Z, Fractal, Bridge, FMB, … (단일팔 22종) |
| RoboMIND | 29.06% | UR5e, Franka Panda, Tien Kung (이중팔/덱스) |
| RoboCOIN | 24.36% | RMC-AIDA-L, AgileX Cobot Magic, Galaxea R1 Lite 등 |
| AGIBOT-beta | 10.78% | AgiBot G1 dual-arm |
| Ego4D | 4.37% | 인간 시점 영상 |

**Embodiment 비율**: 단일팔 47.44% / 이중팔 31.01% / 덱스 17.18% / 인간 4.37%.
총 **1700+ 시간**.

데이터 스케일링 ablation (Fig. 9): pretrain 없이 LIBERO 94.20% → 40% robot data 97.20 → full robot 98.00 → + Ego4D 98.20.

> ❓ **예상 질문**: Ego4D 인간 영상이 정말 도움이 되는가?
> **답변**: 정량적으로 +0.2pp(LIBERO) / +0.7pp~+1.7pp(RoboTwin). 인간 데이터의 marginal gain이 크지 않은 것은 사실이나, **LARY benchmark** Table 3에서 RotVLA가 인간 행동 분류에서 LAPA-DINOv3(64.19) 대비 74.33%로 더 높음 — Ego4D 없이 학습한 RotVLA*도 58.13%로 LAPA-DINOv3에 근접. 즉 LAM 자체가 인간/로봇 횡단 motion semantic을 효과적으로 학습.

---

## 4. 시스템/학습 세부

| 단계 | Steps | Batch | LR / Optimizer | Hardware |
|------|-------|-------|----------------|----------|
| Stage I (LAM) | 200k | 256 | AdamW lr=1e-4, wd=0.01, constant | 8× H200, 50h |
| Stage II (Pretrain) | 200k | 256 | AdamW lr=1e-4, wd=0.001, 5k warmup, cosine | 8× H200 |
| Stage III (Finetune) | LIBERO 80k / RoboTwin 120k / Real 40k | 128 | 동일 + grad clip 1.0 | 8× H200 |

- VLM backbone에는 감소된 LR을 부여하여 catastrophic forgetting 방지 (X-VLA recipe).
- Action expert는 처음 5k step 동안 VLM frozen 상태로 warm-up.
- 추론 latency: NVIDIA H20 server에서 **79 ms/step** (pi0.5는 61 ms).

---

## 5. 실험 결과 심층 분석

### 5.1 LIBERO (Table 1)

| Model | Size | Spatial | Object | Goal | Long | **Avg** |
|-------|------|---------|--------|------|------|---------|
| OpenVLA | 7B | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| OpenVLA-OFT | 7B | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| π₀ | 3B | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| π₀.5 | 3B | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| GR00T-N1.6 | 3B | 97.7 | 98.5 | 97.5 | 94.4 | 97.0 |
| UniVLA* | 9B | 95.4 | 98.8 | 93.6 | 94.0 | 95.4 |
| X-VLA | 0.9B | 98.2 | 98.6 | 97.6 | 97.8 | 98.1 |
| StarVLA | 4B | 97.8 | 98.6 | 96.2 | 93.8 | 96.6 |
| **RotVLA** | **1.7B** | **98.2** | **99.6** | **98.4** | 96.4 | **98.2** |

- 1.7B로 7B OpenVLA-OFT(97.1), 9B UniVLA(95.4)를 능가.
- Object/Goal에서 최고치 (각 99.6, 98.4).
- Long(96.4)에서는 X-VLA(97.8)에 1.4pp 뒤지지만, X-VLA는 LIBERO 4 suite 평균에서 98.1로 RotVLA와 거의 동률 (0.9B → 1.7B 로 사이즈가 두 배 가까이 큰 점이 단점).

### 5.2 RoboTwin 2.0 (Table 1 / Appendix Table 6, 50 tasks)

| Setting | RotVLA | StarVLA | LingBot-VLA | Motus | π₀.5 |
|---------|--------|---------|-------------|-------|------|
| Clean | **89.6** | 88.2 | 88.6 | 88.7 | 82.7 |
| Random | **88.5** | 88.3 | 86.7 | 87.0 | 76.8 |

- 50개 dual-arm 태스크 중 RotVLA가 100% 도달한 태스크: Adjust Bottle, Grab Roller, Handover Mic, Place Cans Plasticbox, Shake Bottle, Place Empty Cup, Place Object Stand 등 7개.
- 약점 태스크: **Open Microwave(clean 36%, rand 12%), Hanging Mug(44/52%), Click Alarmclock(47/46%), Turn Switch(46/46%)** — 모두 정밀 조작 또는 회전 기반 mechanism이 결합된 task. 평균 randomized이 1.1pp만 떨어진 것은 robust성 측면에서 강점.

### 5.3 Real-world (Fig. 3, ARX R5 dual-arm)

| Task | π₀.5 Clean | π₀.5 Rand | **RotVLA Clean** | **RotVLA Rand** |
|------|-----------|-----------|------------------|-----------------|
| Pick and Place | 93.3 | 73.3 | **93.3** | **90.0** |
| Put and Close | 86.7 | 66.7 | **96.7** | **90.0** |
| Stack Three Cups | 56.7 | 33.3 | **66.7** | **60.0** |

- 가장 큰 gap이 **randomized** 세팅 (배경 변경 + distractor object). Pick&Place에서 73.3→90.0, Stack에서 33.3→60.0.
- 저자 해석: 연속 SO(n) latent가 배경 외관이 아닌 high-level motion semantic을 학습.

### 5.4 LARY linear probing (Table 3)

| Model | Regression MSE↓ Avg | Classification Acc↑ Avg |
|-------|--------------------|-----------------------|
| LAPA | 0.97 | 19.13 |
| UniVLA | 0.87 | 18.82 |
| villa-X | 0.87 | 23.85 |
| LAPA-DINOv3 | 0.63 | 45.62 |
| **RotVLA*** | 0.27 | 59.70 |
| **RotVLA** | **0.20** | **70.98** |

- AGIBOT/RoboCOIN/Ego4D 제외한 RotVLA*도 기존 SOTA를 압도. LAM 자체의 표현력이 뛰어남을 시사.

---

## 6. Ablation 분석 (Table 4)

| Variant | Spatial | Object | Goal | Long | Avg |
|---------|---------|--------|------|------|-----|
| w/o. Pretrain | 94.8 | 97.8 | 94.2 | 89.8 | 94.2 |
| w/o. Planner (latent runtime 제거) | 96.0 | 99.6 | 97.2 | 93.2 | 96.5 |
| n = 8 | 97.2 | 99.4 | 98.2 | 94.4 | 97.3 |
| n = 32 | 98.2 | 99.6 | 97.4 | 94.0 | 97.3 |
| Discrete LAM | 95.4 | 99.4 | 96.6 | 89.6 | 95.3 |
| Continuous (no SO(n)) | 96.6 | 99.4 | 97.2 | 93.6 | 96.7 |
| Cont. + SO(n), no Triplet | 97.0 | 99.2 | 97.6 | 94.2 | 97.0 |
| **RotVLA (full)** | **98.2** | **99.6** | **98.4** | **96.4** | **98.2** |

- **Latent design 기여 분해**: Discrete → Continuous (+1.4pp) → +SO(n) (+0.3pp) → +Triplet (+1.2pp). 가장 큰 단일 기여는 **연속화** 자체이며, SO(n) 구조 단독만으로는 marginal — 합성 supervision(triplet)이 SO(n)의 가치를 드러낸다는 점이 흥미롭다.
- **Pretrain 효과**: 4pp (94.2 → 98.2).
- **Runtime planner 효과**: 1.7pp (96.5 → 98.2), 특히 Long에서 93.2 → 96.4 (+3.2pp).

---

## 7. 한계 및 미해결 문제

1. **회전 표현의 직접 학습이 불가능**. SO(n) 사영을 통해 우회했지만, "사영 후 회전"이 정말 quaternion/Lie algebra 기반 학습보다 우월한지에 대한 비교 부재. n도 8/16/32 sweep만 있고 더 큰 n은 미실험.
2. **다른 강력한 LAM 변종과의 ablation 빈약**. Discrete vs Continuous는 비교했지만 LAPA, UniVLA의 specific design 요소(예: skill-aware quantization)와의 head-to-head 부재.
3. **추론 latency가 pi0.5 대비 18ms (29%) 더 느림**. Real-time control 측면에서는 X-VLA(0.9B)·StarVLA가 더 매력적일 수 있음.
4. **Real-world 평가가 ARX R5 하나에 국한**. Cross-embodiment를 그토록 강조하는데, 실제 실험은 한 platform.
5. **Code/weight 미공개** (project page만 존재).
6. **Pretraining 비용**. 8× H200 × 50시간 LAM + 200k step VLA pretraining은 작은 연구실에서 재현 어려움.

---

## 8. 관련 연구 비교

| Model | Latent type | Backbone | Real action repr. | LIBERO Avg |
|-------|------------|----------|------------------|-----------|
| LAPA | Discrete VQ | LLaMA | Token | - |
| UniVLA | Discrete (task-centric) | 9B | Token | 95.4 |
| villa-X | Discrete + annotated | 3B | Mixed | 90.1 |
| Motus | Continuous (world model) | 8B | Latent | - / RoboTwin 88.7 |
| **RotVLA** | **Continuous SO(n) + Triplet** | **InternVL3.5-1B + DiT** | **abs EEF + Rotate6D** | **98.2** |

### 핵심 차별점
- **SO(n) + Triplet의 결합**이 RotVLA만의 고유한 디자인. 다른 continuous LAM(예: CLAM, CoMo)은 가우시안 잠재만 사용 → 합성성(compositionality)이 자연스럽게 정의되지 않음.
- **abs EEF + Rotate6D**라는 robot action 표현이 latent SO(n)과 representation 호환성 측면에서 정합.

---

## 9. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Continuous SO(n) latent는 LAM 계열에서 명확한 새로움 |
| **Technical depth** | ★★★★☆ — Triplet learning + Unified expert + Structured attention의 결합이 체계적 |
| **Experimental rigor** | ★★★★☆ — LIBERO/RoboTwin/Real/LARY 4축 평가, ablation도 충실 |
| **Practical impact** | ★★★★☆ — 1.7B로 SOTA — 다만 latency 가 pi0.5 보다 29% 느림 |
| **Writing quality** | ★★★★☆ — 명확하고 figure가 잘 정리됨 |

**강점**: latent action을 단순 supervision이 아닌 *runtime planner*로 활용하는 디자인 + 회전군 사용으로 합성성 확보가 개념적으로 매우 깔끔. Discrete → Continuous → +SO(n) → +Triplet 으로 ablation이 단계적으로 기여를 분해해 보여줌. 7B OpenVLA-OFT, 9B UniVLA를 1.7B로 능가하는 효율성.

**약점**: SO(n) 구조 자체의 기여는 marginal(+0.3pp)이며, 진짜 이득은 *Triplet 합성 supervision*에서 나옴. 즉 핵심 contribution이 "회전군"이라기보다 "합성 가능한 연속 latent + 합성 supervision"으로 재해석 가능. 또한 ARX R5 한 플랫폼만의 real-world 평가는 cross-embodiment 주장에 비해 빈약.

---

## 10. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | SO(n)가 정말 필요한가? Continuous + Triplet만으로 충분하지 않나? | Cont. (no SO) 96.7 vs Cont.+SO(n)+Triplet 98.2 — 차이는 1.5pp. SO(n)이 *합성 가능성*을 정의하기 때문에 Triplet의 z_{t→t+1}·z_I^{-1}·z_{t+1→t+2} 곱셈 연산이 가능. Continuous Gaussian latent에서는 이 합성이 자연스럽게 정의되지 않음 |
| 2 | n=16이 왜 최적인가? 더 큰 n은? | n=8(97.3), 16(98.2), 32(97.3) — sweet spot이 16. n=32 미만에서 부족한 표현력, n=32 초과는 optimization difficulty + 계산비용 증가 |
| 3 | LAM이 290M밖에 안 되는데 정말 효과적인가? | Table 3 LARY에서 RotVLA*(290M LAM, AGIBOT/RoboCOIN/Ego4D 제외)도 LAPA(7B) 대비 압도. 즉 LAM의 효율은 사이즈가 아니라 *triplet 구조*에서 옴 |
| 4 | 추론 latency 79ms는 closed-loop control에 적합한가? | 13Hz 정도 → fine manipulation 가능 수준. 다만 pi0.5(61ms, 16Hz)보다 18ms 느림 — high-frequency dexterous control에는 부적합 |
| 5 | RoboTwin Open Microwave 36% 같은 극저 성능은 왜? | Multi-stage 정밀 trajectory + 회전축 정렬 결합 task. Action representation이 abs EEF + Rotate6D인데, microwave door처럼 hinge constraint를 명시적으로 표현하지 못함. Latent SO(n)이 free-form rotation에 강하나 *constrained* rotation에는 한계 |
| 6 | Real-world stack three cups가 66.7%로 낮은 이유? | 3단계 long-horizon dual-arm task. Latent planner가 sub-goal 결정에서 안정적이나 dual-arm coordination 정밀도에서 frozen pretrained checkpoint의 한계가 드러남. pi0.5(56.7)보다는 향상 |
| 7 | Triplet learning 외에 trivial reconstruction을 막는 다른 방법은? | CLAM(연속 + variational), Genie(InfoNCE 기반 contrastive)가 대안. 저자들은 비교 부재. Triplet의 강점은 합성성 검증이 학습 시점에 자연스럽게 일어난다는 점 |
| 8 | Identity element z_I 추정의 안정성은? | batch-wise mean을 SO(n)에 사영. 데이터에 stationary frame 쌍(I_t, I_t)이 충분히 포함되어야 z_I가 의미를 가짐 — robot에 stationary frame이 많은 데이터에서 잘 작동. human video Ego4D에서는 stationary frame 비율이 낮을 수 있어 z_I 추정이 noisy할 가능성 |
| 9 | Pretrain 없는 RotVLA(94.2)와 pretrain 있는 RotVLA(98.2)의 gap이 작은 이유? | LIBERO는 in-domain finetuning이 강력. 진짜 pretrain 효과는 *cross-domain transfer*에서 드러나며 (Real-world randomized에서 pi0.5 대비 +16~27pp), LARY zero-shot probing에서도 압도적 |
| 10 | OpenVLA-OFT(97.1)와 1.1pp 차이는 통계적으로 유의한가? | 평균 success rate에서 1.1pp는 LIBERO 40 task × 50 rollout = 2000 trial에서 의미 있는 차이. 다만 표준편차/seed variance가 보고되지 않아 완전 검증 불가 |

<!-- VERIFIED: pdf -->
