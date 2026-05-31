# LoopVLA: Learning Sufficiency in Recurrent Refinement for Vision-Language-Action Models

> **한 줄 요약**: "VLA backbone의 가장 깊은 표현이 항상 최적"이라는 통념을 정면 반박. 4 frozen Perceptual Anchor Layer 위에 3-layer shared Loop Block을 최대 8회 반복 적용하면서, 매 iteration마다 후보 action + sufficiency score를 산출. Remaining Mass Allocation(RMA)과 self-supervised distribution alignment로 "언제 멈출지"를 학습. Qwen3VL-2B에서 파라미터 -45%(2.2B→1.2B) + 처리량 최대 1.7×를 달성하면서 LIBERO 96.0% avg, LIBERO-Plus 65.8%, VLA-Arena 48.7%로 강 baseline 동급/상회.

---

## 1. 배경 및 동기

### 기존 연구의 한계
- 현재 VLA들은 vision-language backbone의 **가장 깊은 layer 표현**을 action 예측 input으로 사용 — 그러나 manipulation은 빈번한 closed-loop spatial 조정으로 구성되어, 과도한 추상화는 (a) 계산 낭비 (b) 정밀 제어용 low-level geometric cue 약화
- 기존 early-exit은 미리 정한 layer에서 멈추거나 "action consistency" 휴리스틱에 의존 → "표현이 action에 대해 실제로 충분한가?"에 직접 답하지 못함

### 핵심 질문
- **"Sufficiency"를 학습 가능한 신호로 끌어올릴 수 있는가?**
- **Layer-index에 종속되지 않고 "현재 evolving 표현 자체"의 충분성을 판단할 수 있는가?**
- **이를 통해 파라미터·throughput trade-off를 개선할 수 있는가?**

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

```
Input → [PAL 1-4 frozen] → [Loop Block 3-layer] (×N iterations) → Action / Sufficiency
                                  ↑              |
                                  └──────────────┘  (parameter sharing across iterations)
```

| 컴포넌트 | 역할 |
|---------|------|
| **PAL (Perceptual Anchor Layers)** | 첫 4개 transformer layer, frozen, 초기 표현 보존 |
| **Loop Block** | Shared 3-layer transformer; iteration 간 parameter 공유; 최대 8회 적용 |
| **Action head** | 매 iteration t의 후보 action a_t 산출 |
| **Sufficiency head** | Cross-attention(action token, loop positional encoding) → halting prob s_t |

### 2.2 Remaining Mass Allocation (RMA)

Halting score를 valid probability distribution으로 변환:
```
p(n) = s(n) · r(n),    r(n+1) = r(n) · (1 - s(n))
```
- r(n)은 step n에 도달했을 때 남아있는 mass — recursive 감소
- 모든 step의 p(n) 합이 1이 되도록 자연스럽게 정규화

### 2.3 두 단계 학습

**Stage 1 (Joint Learning)**:
- 모든 intermediate prediction을 L1 loss로 supervision
- + Entropy regularization (λ₁ = 0.001) — halting 분포가 한 step에 집중되지 않도록
- + Diversity regularization (λ₂ = 0.01) — iteration 간 action 다양성 보존

**Stage 2 (Calibration)**:
- 다른 모듈 freeze, sufficiency head만 학습
- 예측 halting distribution이 **action-quality-derived target distribution**과 KL divergence로 정렬
- Target distribution은 각 step의 action이 최종 ground-truth와 얼마나 가까운지(상대 quality)로 정의

> ❓ **예상 질문**: Layer-index 독립성이 왜 중요한가?
> **답변**: 파라미터 공유 덕분에 sufficiency가 "절대 layer 번호"가 아니라 "현재 evolving 표현 자체"에 기반. 새로운 input(같은 표현이지만 iteration이 다름)에서 일반화 가능. Layer-specific feature를 학습하면 fixed-depth와 본질적으로 같아짐.

### 2.4 추론 시 동작

- Iteration t에서 s_t가 임계 이상이면 즉시 halt → easy step은 빠르게 종료
- 어려운 step은 최대 8 iteration까지 refinement
- 결과: throughput이 step 난이도에 따라 동적으로 적응

---

## 3. 모델/효율 세부사항

### 백본 및 파라미터

| Configuration | Parameters | FLOPs | Throughput |
|---------------|-----------|-------|------------|
| Qwen3OFT (baseline) | 2.2B | 0.53T | 10.49 Hz |
| **LoopOFT*** | **1.2B** | **0.53T** | **10.93 Hz** |
| Qwen3FM (baseline) | 2.3B | 0.53T | 0.97 Hz |
| **LoopFM** | **1.3B** | **0.38T** | **2.04 Hz** |

- Backbone: Qwen3VL-2B-Instruction (SigLIP-2 visual encoder + 28-layer decoder)
- LoopFM은 FLOPs까지 28% 감소(0.53T → 0.38T) + throughput 2.1×
- 학습: AdamW + cosine, batch 64, τ=0.5

---

## 4. 실험 결과 심층 분석

### 4.1 LIBERO 4-suite

| Model | Spatial | Object | Goal | Long | Avg |
|-------|---------|--------|------|------|-----|
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| Qwen3OFT (baseline) | 95.0 | 97.0 | 97.1 | 90.5 | 94.9 |
| **LoopOFT*** | **95.0** | **100** | **97.4** | **91.0** | **96.0** |

- Object 100% — perfect score
- Long 91.0 (vs π₀ 85.2) — recurrent refinement이 long-horizon에서 특히 효과적
- 절반 파라미터로 1.1pp avg 향상

### 4.2 LIBERO-Plus (zero-shot generalization)

| Factor | LoopOFTα | 비교 |
|--------|----------|------|
| Camera | 58.3 | OpenVLA 0.8% (catastrophic) |
| Robot | 41.7 | π₀+FAST 65.1% |
| Language | 66.7 | π₀.5 65.0% |
| Light | 88.9 | – |
| Background | 88.3 | – |
| **Overall** | **65.8%** | π₀.5 급 |

- Camera 변화에서 OpenVLA가 0.8%로 무너지는 것과 대조적으로 58.3%
- Light/Background는 88%대로 매우 강건
- Robot/Language factor가 약점 → embodiment, instruction phrasing 변화에는 추가 보강 필요

### 4.3 VLA-Arena L0

| Category | LoopVLA | Qwen3OFT |
|----------|---------|----------|
| Safety | 55.6 | 54.0 |
| Distractor | 60.0 | 71.0 |
| Extrapolation | 12.0 | 13.3 |
| Long Horizon | 76.0 | 59.0 |
| **Average** | **48.7** | **47.0** |

- **Long Horizon에서 +17pp** 큰 폭의 개선 — recurrent refinement의 본질적 강점
- 그러나 **Distractor에서 -11pp** 열위 → sufficiency early-exit가 distractor presence 하에서 너무 일찍 멈추는 경향 가능

---

## 5. Ablation 분석

### 5.1 Loop 구성 (L⊗N, L×N=24 layers 등가)

| Config (L⊗N) | LIBERO Avg |
|--------------|-----------|
| 4⊗6 | 95.0 |
| **3⊗8 (default)** | **95.0** (paper main) → LoopOFT* 96.0 |
| 2⊗12 | 92.8 |
| 8⊗3 | 92.1 |

- 극단 config(매우 얕거나 매우 깊은 loop)는 degradation
- 3-layer × 8-iteration이 sweet spot

### 5.2 Sufficiency Head 필요성

| Strategy | LIBERO Avg |
|----------|-----------|
| Direct MLP halting | 94.4 |
| **Proposed sufficiency head** | **96.0 (+1.6)** |

### 5.3 Inference layer selection

| Strategy | LIBERO Avg |
|----------|-----------|
| Fixed layer | 93.8-94.9 |
| **Adaptive (sufficiency-driven)** | **96.0** |

- Adaptive selection은 monotonic improvement pattern이 아님 — 특정 step에서 step-back하는 non-monotonic refinement 경로가 가능함을 시사

---

## 6. 관련 연구 비교

| 모델 | Early-exit 전략 | Sufficiency 정의 |
|------|---------------|----------------|
| DeeR-VLA | Layer-level, action consistency | 휴리스틱 |
| FastVLA | Token pruning | 없음 |
| **LoopVLA** | **Iteration-level, RMA** | **Learned via self-supervised KL alignment** |

핵심 차이: **shared Loop Block** (parameter 공유) + **learned sufficiency**.

---

## 7. 한계 및 미해결 문제

1. **VLA-Arena Distractor에서 열위**: -11pp (60.0 vs 71.0 baseline) — sufficiency가 distractor presence에서 over-confident하게 조기 종료할 수 있음
2. **Robot/Language factor 약함 (LIBERO-Plus)**: Camera/Light/Background는 강건하나 embodiment·언어 변화에는 상대적 취약
3. **Worst-case latency 미보고**: easy step throughput은 향상되나 8 iteration 모두 사용하는 worst case의 latency는 명시되지 않음
4. **Code 미공개**: GitHub URL 부재 (StarVLA framework 사용 언급만) — 재현 부담
5. **CALVIN/SimplerEnv 평가 없음**: LIBERO 계열 + VLA-Arena에 한정 → cross-benchmark generalization 검증 부족
6. **Sufficiency calibration의 안정성**: Stage 2의 KL alignment가 collapse나 over-smoothing 없이 수렴하는 메커니즘은 self-supervised 특성상 본문에 추가 분석 필요

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| Novelty | ★★★★☆ — "Learned sufficiency"라는 신호 정의가 개념적으로 새로움 |
| Technical depth | ★★★★☆ — RMA + 2-stage training의 결합이 체계적 |
| Experimental rigor | ★★★★☆ — LIBERO + Plus + VLA-Arena 3종, ablation 풍부; 단 CALVIN/SimplerEnv 부재 |
| Practical impact | ★★★★★ — -45% 파라미터 + 1.7× throughput은 on-device 배포에 매우 매력적 |
| Writing quality | ★★★★☆ — 효율 수치가 일관 |

**강점**: VLA 분야의 암묵적 가정("깊은 표현이 최선")을 정면 반박하고, **representation sufficiency를 학습 가능한 신호로** 끌어올린 점이 conceptual 기여. 효율 이득(파라미터 -45%, throughput 1.7×) + 성능 유지의 frontier 이동이 실용적. **약점**: VLA-Arena Distractor 열위는 sufficiency의 over-confidence 문제 시사. Code 미공개로 재현 부담.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 |
|---|------|----------|
| 1 | "Sufficiency"가 결국 confidence/temperature와 다른가? | 다름 — sufficiency는 *step 간 상대 action quality*에 align됨. 표준 confidence는 absolute 분류 확률 |
| 2 | 왜 RMA가 필요한가? Sigmoid로 충분하지 않나? | RMA는 모든 step의 p(n) 합이 1이 되도록 자동 정규화 — early stopping의 valid distribution 보장 |
| 3 | VLA-Arena Distractor에서 왜 -11pp? | sufficiency가 distractor와 main object 사이의 모호함에 직면했을 때 조기 종료 가능성. Halting threshold tuning 또는 distractor-aware regularization 필요 |
| 4 | 8회 iteration의 worst-case latency는? | 미보고. easy step에서 1-2회 종료 시 1.7× 이득이나, 모든 step이 8회면 baseline보다 느려질 수도 |
| 5 | LIBERO Long 91% vs Spatial 95% — 왜 long-horizon이 더 어려운가? | Long은 multi-step manipulation chain — 각 step의 작은 오류가 누적. Recurrent refinement가 보상하나 완전 해결 못함 |
| 6 | Layer-sharing이 capacity bottleneck 아닌가? | 그렇다 — shared 3-layer는 24 unique layer 대비 capacity 적음. 그러나 ablation에서 4⊗6도 95.0%로 sweet spot은 capacity가 아님 |
| 7 | π₀와의 비교가 fair한가? π₀는 다른 학습 데이터 사용 | π₀는 OXE 사전학습 사용. LoopVLA의 학습 셋업 명확화가 필요. 하지만 동일 baseline Qwen3OFT 대비 1.1pp 향상은 의미 있음 |
| 8 | Self-supervised KL alignment가 collapse하지 않는 이유? | Entropy regularization(λ₁=0.001)이 halting 분포 spread를 강제. Diversity regularization(λ₂=0.01)이 iteration 간 action 다양성 보존 |

<!-- VERIFIED: pdf -->
