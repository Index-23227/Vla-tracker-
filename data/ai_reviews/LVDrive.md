# LVDrive: Latent Visual Representation Enhanced Vision-Language-Action Autonomous Driving Model

> **한 줄 요약**: EVA-02-L vision encoder + QT-Former + Vicuna v1.5(LoRA) + 경량 vision decoder ViS_θ를 결합하여, **256 token/frame의 latent future scene**을 VQGAN-ImageNet feature로 supervise하고 두 단계 trajectory decoding으로 future-aware planning을 수행 — Bench2Drive base split에서 **Driving Score 80.71 / SR 58.26%**로 UniDrive-WM 대비 +1.49 / +1.84 향상을 달성한 HKUST·Xiaomi EV의 latent world-model VLA(arXiv 2605.22089, 2026.05).

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 자율주행 VLA 모델은 보통 (i) **pixel-level future generation**(world model로 RGB image 예측) 또는 (ii) **autoregressive token generation**(scene description을 자연어로 생성) 둘 중 하나에 의존
- (i) Pixel reconstruction은 비용이 막대 — high-resolution image generation에 많은 compute 필요, 또한 task-irrelevant detail(차량 색상, 광원 변화)에 capacity를 낭비
- (ii) Autoregressive token generation은 token-by-token decoding → 추론 latency가 큼 (closed-loop 자율주행에 부적합)

### 핵심 질문
- **Future scene을 pixel 대신 latent space에서 예측하면 효율과 성능 모두 개선되는가?**
- **Future-aware reasoning을 single forward pass로 통합할 수 있는가?**

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

```
Multi-camera input → EVA-02-L encoder → QT-Former
                                            ↓
                  [Vicuna v1.5 + LoRA] ←——— vision tokens
                            ↓
                    LLM hidden states
                       ↙        ↘
                  ViS_θ          Trajectory Decoder
              (future latent)    (two-stage)
                  ↓                   ↓
        VQGAN-ImageNet         predicted trajectory
        supervision (16384
        codebook, 256 tokens
        per future frame)
```

### 2.2 Latent Future Representation

핵심 통찰: **pixel reconstruction을 latent feature reconstruction으로 대체**.
- 각 future frame에 대해 LLM hidden state에서 **256 token**을 ViS_θ가 생성
- 이 token들은 **VQGAN-ImageNet pretrained encoder**의 feature와 매칭되도록 학습
- VQGAN의 codebook size: 16384 (rich semantic vocabulary)

> ❓ **예상 질문**: 왜 DINOv3나 MoVQGAN이 아닌 VQGAN-ImageNet인가?
> **답변**: Ablation에서 VQGAN-ImageNet supervision이 alternatives 대비 우월. 추측: VQGAN의 discrete codebook이 driving scene의 semantic primitive(차량, 차선, 신호등)에 잘 매핑되며, ImageNet pretraining이 객체 인식에 강함.

### 2.3 Two-stage Trajectory Decoding

- **Stage 1**: 초기 trajectory 예측(coarse)
- **Stage 2**: latent future representation을 conditioning으로 trajectory 정제(refinement)
- 두 단계가 모두 single forward pass에 포함 → autoregressive 대비 빠름

> ❓ **예상 질문**: Two-stage decoding의 효과는?
> **답변**: Dev10 ablation에서 Driving Score 60.43 → 82.39로 +21.96. 핵심 성능 driver.

### 2.4 손실 함수

$$\mathcal{L} = \mathcal{L}_{\text{vis}} + \mathcal{L}_{\text{plan}} + \mathcal{L}_{\text{plan\_r}} + \mathcal{L}_{\text{qt}} + \mathcal{L}_{\text{ce}}$$

- $\mathcal{L}_{\text{vis}}$: latent future feature 예측(VQGAN supervision)
- $\mathcal{L}_{\text{plan}}$: stage 1 trajectory
- $\mathcal{L}_{\text{plan\_r}}$: stage 2 refined trajectory
- $\mathcal{L}_{\text{qt}}$: QT-Former feature extraction
- $\mathcal{L}_{\text{ce}}$: cross-entropy(token generation)

---

## 3. 데이터 및 학습

| 항목 | 값 |
|------|----|
| Dataset | Bench2Drive base split |
| Training clips | ~950 |
| Scenes / weathers / towns | 44 / 23 / 12 |
| Future supervision | 6 future frames |
| GPU | 32x NVIDIA H20 (96GB) |
| Epochs | 6 (end-to-end) |
| Backbone fine-tuning | LoRA on Vicuna v1.5 |

> ❓ **예상 질문**: 950 trajectory로 충분한가?
> **답변**: Bench2Drive의 표준 base split. CARLA의 다양한 interactive scene이 포함되어 manifold coverage는 보장. 그러나 real-world 일반화는 별도 검증 필요.

---

## 4. 실험 결과 심층 분석

### 4.1 Bench2Drive Closed-Loop (Base Set)

| Metric | UniDrive-WM-AR | **LVDrive** | Δ |
|--------|---------------|------------|---|
| Driving Score | 79.22 | **80.71** | **+1.49** |
| Success Rate (%) | 56.42 | **58.26** | **+1.84** |
| Efficiency | — | 155.77 | — |
| Comfort | — | 14.34 | — |
| L2 error (3s, m) | 0.63 | 0.63 | 0 |

### 4.2 Multi-Ability Breakdown

| Ability | LVDrive (%) |
|---------|-------------|
| Traffic Sign | 74.21 |
| Merging | 39.74 |
| Overtaking | 68.89 |
| Emergency Brake | 71.67 |
| Give Way | **20.00** ⚠️ |
| **Mean** | **54.90** |

- **Give Way에서 20%는 명확한 약점** — 사회적 협력 행동(다른 차량에게 우선 양보)에서 취약
- Merging도 39.74로 평균 이하 — interactive scenario에서 부족
- Traffic Sign, Emergency Brake는 상대적으로 robust

### 4.3 Inference 속도

- LVDrive: ~2.03 s/step
- Baseline(아마 autoregressive variant): ~0.93 s/step

⚠️ **2 초/스텝은 closed-loop 자율주행에 부적합** — real-time deployment를 위한 추가 최적화 필요.

### 4.4 Ablation (Dev10)

- Two-stage decoding 제거: 82.39 → 60.43 (−21.96 DS) ← **결정적**
- VQGAN-ImageNet vs DINOv3 / MoVQGAN: VQGAN-ImageNet이 모두 우월

---

## 5. 한계 및 미해결 문제

### 5.1 성능 한계
1. **Give Way 20%, Merging 39.74%**: 사회적·상호작용 시나리오에서 약함 — 다른 driver의 의도 추론이 부족
2. **Bench2Drive L2 error 0.63m가 baseline과 동일** — open-loop trajectory 정확도 자체는 unchanged. 향상이 closed-loop 평가에만 국한
3. **Inference 2.03s/step**: closed-loop control rate는 0.5Hz — 실차 deployment 비현실적

### 5.2 평가 범위
1. **nuScenes 평가 부재**: 일반적으로 자율주행 VLA는 nuScenes open-loop도 평가하나, LVDrive는 Bench2Drive(CARLA)만 평가 → real-world generalization 의문
2. **Base split만 평가**: Bench2Drive의 더 큰 split에서의 성능은 미보고
3. **단일 city/weather 일반화 검증 부재**: 44 scene을 train/eval split했는지, held-out city/weather가 있는지 명시 부족

### 5.3 방법론
1. **VQGAN-ImageNet은 driving domain pretrain이 아님**: ImageNet 기반이라 driving-specific feature(차선, 신호등)에 sub-optimal일 가능성. Driving-specific tokenizer가 더 적합할 수 있음
2. **6 future frames만 예측**: 장기 planning(10초+)에서 부족할 수 있음
3. **Parameter count 미보고**: Vicuna 7B인지 13B인지 불명확

---

## 6. 관련 연구 비교

| 모델 | Future prediction | Trajectory decoding | Bench2Drive DS |
|------|------------------|---------------------|---------------|
| VAD | ✗ | single-stage | ~40-50 |
| ThinkDrive | autoregressive text | autoregressive | — |
| UniDrive-WM | pixel-level world model | single-stage | 79.22 |
| **LVDrive** | **latent (VQGAN)** | **two-stage** | **80.71** |

핵심 차이:
- **유일하게 latent space에서 future scene을 예측**
- Pixel 생성의 compute overhead 회피

---

## 7. 강점 및 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★☆☆ — Latent future prediction은 새로우나 핵심 idea는 점진적 |
| **Technical depth** | ★★★★☆ — Two-stage decoding과 VQGAN supervision 조합 |
| **Experimental rigor** | ★★★☆☆ — Bench2Drive만 평가, nuScenes 부재 |
| **Practical impact** | ★★☆☆☆ — 2 s/step inference로 real-time 불가 |
| **Open access** | ★★☆☆☆ — Code 미공개 |

**강점**: Pixel reconstruction의 비효율을 latent prediction으로 해결하는 합리적 접근. **약점**: Give Way / Merging 등 interactive scenario에서 약하고, inference latency가 deployment에 비현실적.

---

## 8. 예상 날카로운 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 왜 nuScenes 평가가 없는가? | 미언급. Bench2Drive(CARLA closed-loop)만 평가 — real-world 검증 부재 |
| 2 | Give Way 20%의 의미는? | 다른 driver 의도 추론·사회적 협력 시나리오에서 약함. Trajectory만 보고 다른 차량의 미래 행동을 추론하지 못함 |
| 3 | 2.03 s/step inference로 closed-loop 가능한가? | Bench2Drive simulator는 wall-clock과 분리된 evaluation 가능. 그러나 실차 배포는 불가능 |
| 4 | VQGAN-ImageNet이 driving domain에 맞는가? | Ablation에서 alternatives보다 우월하나, driving-specific tokenizer는 미시도 |
| 5 | Two-stage decoding 제거시 −22 DS — 그렇다면 latent future prediction의 contribution은? | Two-stage가 핵심. Latent future representation 자체의 ablation이 부재해 attribution 어려움 |
| 6 | Vicuna v1.5는 7B인가 13B인가? | 미보고. Parameter count 명시 부재 |
| 7 | 32×H20 GPU로 6 epoch — 학습 cost는? | 약 1500 GPU-hour 추정. Resource-intensive |
| 8 | UniDrive-WM 대비 +1.49는 statistically significant한가? | Confidence interval 미보고. CARLA evaluation의 stochasticity 고려 시 marginal일 수 있음 |

<!-- VERIFIED: pdf -->
