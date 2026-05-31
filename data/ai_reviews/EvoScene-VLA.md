# EvoScene-VLA: Evolving Scene Beliefs Inside the Action Decoder for Chunked Robot Control

> **한 줄 요약**: 청크 단위 VLA 정책이 청크 내부의 접촉/가림/물체 이동을 반영하지 못하는 본질적 약점을, action decoder 내부에 흐르는 *recurrent scene prefix*로 해결. Qwen2.5-VL-3B 백본 위에 Joint Action-Scene Denoising(flow-matching) + 학습 전용 Scene Predictor + Geometric Anchor(depth & 3D foundation teacher)를 결합. RoboTwin 2.0 31-task에서 fixed 87.2→89.1%, randomized 86.1→88.5% (+1.9~2.4pp), Galaxea R1-Lite 실로봇 42% (baseline 37.3%).

---

## 1. 배경 및 동기

### 청크 VLA의 구조적 blind-spot
- 청크 단위 VLA는 한 번의 VLM 호출로 여러 스텝의 action을 예측 — 단일 시각 관측만을 chunk 시작 시점의 조건으로 사용
- 그러나 로봇 action 자체가 **접촉, 가림(occlusion), 물체 이동**을 유발 → 다음 시각 업데이트가 도착하기 전에 장면 기하가 이미 변함
- 기존 *Spatial VLA*는 current frame의 기하만 강화하고, *Temporal VLA*는 과거 프레임을 누적할 뿐 — **action에 의해 갱신되는 장면 사전(prior)을 청크 사이에 유지하는 메커니즘은 부재**

### 핵심 질문
- **Chunk 경계를 가로질러 흐르는 latent scene state를 어떻게 표현/업데이트할 것인가?**
- **이 state가 추론 시 추가 비용을 거의 들이지 않으면서 학습 시 geometric supervision을 받을 수 있는가?**

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

```
Chunk t:
    [Observation slots (per-view)] --+
                                     +--> Action decoder (Joint Denoising)
    [Prior slots from chunk t-1] ----+         |
                                               |--> next action chunk
                                               |--> next scene prefix (prior for chunk t+1)
```

| 컴포넌트 | 역할 | 학습/추론 |
|---------|------|----------|
| Qwen2.5-VL-3B-Instruct | VLM backbone (D=2048) | 추론 사용 |
| Observation slots | 각 view에서 fresh visual evidence 추출 | 추론 사용 |
| Prior slots | 이전 청크의 action-updated scene state 상속 | 추론 사용 |
| Action decoder | 다음 action chunk + scene update 동시 denoising | 추론 사용 |
| **Scene Predictor** | 미래 scene-token 타깃 supervision | **학습 전용** |
| **Geometric Anchor** | Depth(local) + 3D foundation model(global) teacher | **학습 전용** |

### 2.2 Joint Action-Scene Denoising

핵심 메커니즘: Action expert가 **action chunks와 scene tokens를 동시에 co-denoise**한다 (flow-matching 기반).

- Prior slots는 이전 청크의 action-updated scene state를 inherit
- 새 관측이 들어오면 observation slots이 prior slots를 correction
- Action decoder는 동시에:
  1. 다음 action chunk
  2. 다음 청크의 prior가 될 compact scene update

### 2.3 학습 보조 모듈 (배포 시 제거)

**Scene Predictor**:
- 미래 scene-token에 대한 self-supervised target 공급
- "다음 chunk에 어떤 scene이 펼쳐질지" 예측하는 능력을 scene tokens에 inject

**Geometric Anchor (two-level)**:
- *Local*: frozen depth model teacher → scene slot이 깊이 정보 보존
- *Global*: frozen 3D foundation model teacher → 장면의 global 3D 구조 정렬

> ❓ **예상 질문**: 학습 보조가 제거되면 deployment에서 scene update의 quality는 어떻게 유지되는가?
> **답변**: 보조 모듈은 학습 중 scene tokens가 "깊이/3D"를 표현하도록 *shape* 한 뒤 제거됨. Inference에서 scene tokens 자체는 이미 정렬된 표현을 학습한 상태로 동작 → 비용은 들지 않지만 학습된 inductive bias는 보존.

### 2.4 학습 셋업

| 항목 | 값 |
|------|---|
| Hardware | 8× A800 GPUs |
| Batch size | 256 |
| Learning rate | 1×10⁻⁴ |
| Total steps | 20,000 |

---

## 3. 데이터 전략

- **RoboTwin 2.0**: 31개 language-conditioned manipulation task (fixed + randomized 환경)
- **Real robot (Galaxea R1-Lite)**: indoor-cleaning 3 task subset, 439 episodes, ~9 hours 수집

---

## 4. 실험 결과 심층 분석

### 4.1 RoboTwin 2.0 (31 tasks)

| Method | Clean Avg | Randomized Avg |
|--------|-----------|----------------|
| π₀.5 | 81.2 | 75.9 |
| LingBot-VLA | 85.3 | 84.1 |
| LingBot-VLA* (depth-augmented) | 87.2 | 86.1 |
| **EvoScene-VLA** | **89.1** | **88.5** |
| **Δ vs strongest baseline** | **+1.9** | **+2.4** |

- 31-task 평균에서 90% 근접 — RoboTwin 2.0에서 매우 강력
- Randomized에서 +2.4pp 격차가 fixed의 +1.9pp보다 큼 → **scene prior가 환경 변동(randomization)에 더 도움**됨을 시사

### 4.2 Real Robot (Galaxea R1-Lite, indoor-cleaning)

| Task | EvoScene-VLA | Baseline |
|------|--------------|----------|
| Mirror | 29% | – |
| Sink | 51% | – |
| Cutting-board | 46% | – |
| **Avg** | **42.0%** | **37.3%** |

- Mirror가 가장 낮은 29% — 반사면이 visual evidence를 교란
- Sink 51%, Cutting-board 46% — 일반적 실내 조작 성공률

---

## 5. Ablation 분석 (RoboTwin-5Task)

| Variant | Clean | Randomized |
|---------|-------|-----------|
| LingBot-VLA* (baseline) | 87.8 | 84.6 |
| + L_pred & L_rep | 89.3 | 86.2 |
| + L_geo | 90.1 | 86.5 |
| **+ Prior recurrence** | **90.8** | **87.8** |

### 해석
1. **+L_pred & L_rep** (Scene Predictor + scene representation loss): +1.5/+1.6pp — scene token이 future-aware해지는 것의 이득
2. **+L_geo** (Geometric Anchor): +0.8/+0.3pp — depth/3D teacher의 추가 효과는 modest
3. **+Prior recurrence**: +0.7/+1.3pp — chunk 간 prior가 실제 흐르는 것의 최종 효과

→ **Scene Predictor가 가장 큰 기여**, Prior recurrence가 randomized 환경에서 추가 이득

---

## 6. 관련 연구 비교

| 모델 | Inter-chunk scene state | Geometric supervision | Inference 비용 |
|------|------------------------|----------------------|---------------|
| π₀.5 | ✗ | ✗ | Standard |
| Spatial VLA | △ (현재 frame만) | △ | Standard |
| Temporal VLA | △ (frame stack) | ✗ | + frame stack |
| **EvoScene-VLA** | **✓ (recurrent prefix)** | **✓ (training only)** | **Standard** |

핵심 차이: **action decoder 내부**에 상태가 흐르고, geometric supervision은 학습에만 사용 → 배포 비용 증가 없음.

---

## 7. 한계 및 미해결 문제

1. **Real robot 성능 격차**: 42% (avg)는 sim 89%와 큰 격차 — sim-to-real gap이 상당. Mirror 29%는 반사면에 매우 취약
2. **Baseline 비교 범위**: π₀.5, LingBot-VLA(*) 외 다른 backbone(π₀, OpenPi 등)과의 head-to-head 부재
3. **LIBERO/CALVIN/SimplerEnv 부재**: RoboTwin 2.0과 실로봇에 한정 → cross-benchmark 일반화 검증 부족
4. **Scene prefix의 해석성**: prefix가 정말 "scene geometry"를 표현하는지에 대한 시각화/probing 부재
5. **Code 미공개**: 재현 부담
6. **Scene update의 표현력**: "compact" scene update의 차원·정확도가 명시되지 않음. 너무 작으면 정보 손실, 너무 크면 추론 비용
7. **31-task 평균 89%는 이미 saturation**: 추가 1.9pp 향상의 통계적 유의성과 분산이 명시되지 않음

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| Novelty | ★★★★☆ — "Action decoder 내부에 scene state가 흐른다"는 비교적 단순한 아이디어를 깔끔히 정형화 |
| Technical depth | ★★★★☆ — Joint Action-Scene Denoising + 학습 전용 보조의 조합이 체계적 |
| Experimental rigor | ★★★☆☆ — RoboTwin 2.0 중심; 다른 benchmark 부재가 아쉬움 |
| Practical impact | ★★★★☆ — 추론 비용 unchanged + 평균 +1.9~2.4pp는 적용 부담이 적음 |
| Writing quality | ★★★★☆ — 문제 정의가 직관적 |

**강점**: 청크 VLA의 본질적 약점("청크 내 환경 변화 무시")을 직접 짚고, **action decoder 안에 상태가 흐른다**는 단순 설계로 +1.9~2.4pp 이득. 보조 모듈을 inference에서 제거하므로 비용 증가 거의 없음. **약점**: Real robot 격차(sim 89% vs real 42%) — scene prefix가 sim-to-real gap을 완화하지는 못함. Mirror 29%는 반사면에 취약. RoboTwin 2.0 외 cross-benchmark 검증 부재.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 |
|---|------|----------|
| 1 | Inference에서 Scene Predictor와 Geometric Anchor를 제거한 후 scene token 품질은 어떻게 유지? | 학습 중 보조 supervision이 scene token의 표현을 *shape*. Inference에서는 그 표현이 frozen weight에 내재화되어 보조 없이도 활용 |
| 2 | Mirror 29%는 왜 그렇게 낮은가? | 반사면은 observation slots에 잘못된 시각 증거 제공 → prior가 actual scene이 아니라 반사된 phantom geometry로 update 가능 |
| 3 | Sim 89% → Real 42% 격차의 원인? | (a) Mirror/Sink 등 실세계 광학적 도전, (b) action 정밀도 차이, (c) randomization이 sim에서만 적용 |
| 4 | Ablation에서 L_geo가 +0.8pp만 기여 — Geometric Anchor가 정말 필요한가? | Modest하지만 일관 양의 기여. Local depth + global 3D 두 level이 본질적으로 redundant할 수 있어 simplification 여지 |
| 5 | Scene update의 차원은 얼마인가? | 본문 확인 필요. Backbone hidden D=2048에 기반한 compact 표현으로만 명시 |
| 6 | Randomized에서 격차가 더 큰 이유(+2.4 vs +1.9)? | Scene prior가 환경 변동을 메모리에서 보상 — current frame에만 의존하는 baseline은 randomization에서 더 큰 손실 |
| 7 | LIBERO/CALVIN 결과는? | 본 논문 미평가 — RoboTwin 2.0과 실로봇만. Cross-benchmark는 future work |
| 8 | Joint Action-Scene Denoising의 학습 안정성? | Scene token이 action loss로부터 indirect gradient를 받음 + Scene Predictor의 direct supervision이 stability 제공 |

<!-- VERIFIED: pdf -->
