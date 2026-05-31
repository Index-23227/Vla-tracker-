# Afford-VLA: Action-Aligned Visual Planning via Internalized Affordance

> **한 줄 요약**: Qwen3-VL-4B-Instruct 백본 위에 4개의 학습 가능한 `<AFF>` query token으로 task-conditioned affordance mask를 *내부에서* 생성하고, straight-through Top-K patch pooling을 통해 affordance embedding을 GR00T 스타일 DiT-B flow-matching action head에 직결시키는 affordance-grounded VLA. LIBERO 97.4%, LIBERO-Plus 78.1%, SimplerEnv 58.1%.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 많은 VLA(OpenVLA, π0, CogACT, GR00T)들은 visual planning 단계가 *암시적*으로 VLM hidden state에 흡수됨 → 정책의 affordance reasoning이 해석/조작 불가
- 외부 affordance predictor를 붙이는 방식(예: RoboPoint, ManipLLM, RoboAfford 류)은 별도 모듈로 train되어 *action에 직접 align*되지 않음 — affordance가 단지 *시각적 힌트*에 그침
- 결과: language perturbation, layout shift 등 LIBERO-Plus 류 OOD에서 성능 급락

### 핵심 질문
- **Affordance를 VLA 외부 모듈이 아니라 *내부 explicit 인터페이스*로 만들고, 그것이 action generation을 *미분 가능하게* 조건짓도록 한다면 OOD robustness가 향상되는가?**
- **Affordance mask → action embedding 변환은 어떤 미분 가능 메커니즘이 가장 효과적인가? (hard pooling? dense soft? sparse Top-K with straight-through?)**

📌 [Figure 1] — `<AFF>` query tokens → two-way decoder → patch-level affordance mask → Top-K STE pooling → DiT-B flow-matching conditioning

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

| 컴포넌트 | 사양 | 역할 |
|---------|-----|------|
| Backbone VLM | Qwen3-VL-4B-Instruct | 시각·언어 표현 |
| `<AFF>` queries | view 당 4개 학습 가능 토큰 | task-conditioned affordance prompting |
| Two-way decoder | 2 layers, 8 heads, hidden=256 | affordance mask 디코딩 (light-weight) |
| Patch pooling | Top-K, K=16, STE + softmax (T=1.0) | mask → embedding 변환 (미분 가능) |
| Action head | GR00T-style DiT-B flow-matching | 연속 action 생성 |

전체 파라미터는 약 **4B 수준** (Qwen3-VL-4B + 소량의 affordance decoder + DiT-B head).

### 2.2 Affordance Mask 내부 생성

각 view에 대해:
1. `<AFF>` query token이 multimodal feature(VLM의 patch tokens + text instruction)에 cross-attention
2. Two-way decoder(2 layers)가 query/patch 양방향 attention으로 mask logits 산출
3. Mask는 patch grid 위의 affordance score (task-relevant interaction regions)

### 2.3 Differentiable Action-Alignment: Top-K STE Pooling

핵심 메커니즘: affordance score $s_i$로부터 *Top-K patch*를 sparse하게 선택하되, 미분 가능하도록:

$$
w_i = \text{softmax}(s_i / \tau) \quad \text{with } \tau=1.0
$$

$$
\text{mask}_i^{\text{hard}} = \mathbb{1}[i \in \text{TopK}(s, K=16)]
$$

Forward pass: hard top-K mask 사용
Backward pass: softmax 가중치로 gradient flow (straight-through estimator)

최종 affordance embedding:
$$
e_{\text{aff}} = \sum_i (\text{mask}_i^{\text{hard}} \cdot w_i / Z) \cdot \text{patch}_i
$$

이 embedding이 DiT-B flow-matching head의 conditioning에 더해짐.

> ❓ **예상 질문**: 왜 hard top-K (K=16) + STE인가? Dense soft pooling이 더 일반적이지 않은가?
> **답변**: Dense soft pooling은 비-affordance 영역의 patch까지 conditioning에 새어 들어가 학습 noise를 만든다. Hard top-K는 *공간적으로 sparse*한 affordance에 부합하지만 직접 미분 불가. STE로 forward는 hard, backward는 soft로 처리해 양쪽 장점 결합. Ablation에서 hard-only 91.3, dense soft 96.0, **Top-K + STE 97.4**로 STE가 결정적 (LIBERO).

### 2.4 Two-Stage Training Schedule

| Stage | Steps (LIBERO / SimplerEnv) | 학습 대상 |
|-------|----------------------------|---------|
| Warmup | 4,000 / 4,000 | affordance decoder만, VLM·action head frozen, **GT mask** 사용 |
| Joint | 140,000 / 200,000 | 전체 모델, predicted mask, 모든 컴포넌트 학습 |

> ❓ **예상 질문**: GT affordance mask는 어디서 오는가?
> **답변**: 본 논문은 affordance mask GT를 *interaction region heuristic*(grasp point, contact patch)으로 구성. LIBERO 같은 시뮬레이션에서 contact label은 가용. Real-world 확장 시 GT mask 확보가 bottleneck이 될 수 있다.

---

## 3. 데이터셋

- **LIBERO** (4 suites): spatial / object / goal / long, 각 10 task × 50 demos
- **LIBERO-Plus**: LIBERO 위의 7개 perturbation 카테고리 (camera, robot, language, light, background, noise, layout)
- **SimplerEnv**: WidowX bridge tasks (put spoon, put carrot, stack block, put eggplant)

학습 데이터는 LIBERO/SimplerEnv 각각의 standard fine-tune split.

---

## 4. 실험 결과

### 4.1 LIBERO 4-suite (Table)

| Suite | Afford-VLA |
|-------|----------|
| Spatial | **97.8** |
| Object | **99.6** |
| Goal | **97.6** |
| Long | **96.8** |
| **Average** | **97.4** |

GR00T-N1.6 ~94.2, π0.5 ~97.0과 비교해 경쟁력 (정확 비교는 paper Table).

### 4.2 LIBERO-Plus (Table 2) — OOD 견고성

| Perturbation | Afford-VLA |
|-------------|----------|
| Camera | 56.0 |
| Robot | 56.8 |
| Language | **91.5** |
| Light | **96.8** |
| Background | **97.0** |
| Noise | 80.9 |
| Layout | 78.9 |
| **Average** | **78.1** |

> ❓ **예상 질문**: Camera(56.0)와 Robot(56.8)이 왜 가장 낮은가?
> **답변**: Camera 변동은 patch 좌표계가 통째로 바뀌므로 affordance mask의 *공간적 위치*가 무의미해진다. Robot 변동은 end-effector 외형 변화로 affordance heatmap이 misalign. 즉 **affordance-guided pooling은 *appearance shift*(Light, Background, Language)에는 매우 강하나, *geometric/spatial shift*에는 본질적으로 약점**을 가짐.

### 4.3 SimplerEnv (Table 1, WidowX)

| Task | Success |
|------|---------|
| Put spoon | 66.6 |
| Put carrot | 54.2 |
| Stack block | 14.6 |
| Put eggplant | 96.8 |
| **Average** | **58.1** |

> ❓ **예상 질문**: Stack block 14.6%가 왜 이렇게 낮은가?
> **답변**: Stack block은 *동적 stacking*으로 affordance가 *시간적*으로 변하고(첫 block 위에 두번째 block) trajectory 동역학이 중요. Static "어디를 잡을지" 정보만 제공하는 affordance mask로는 long-horizon coordination에 도움이 적다.

---

## 5. 어블레이션

### 5.1 Pooling Strategy (LIBERO Avg)

| Strategy | Avg Success |
|---------|------------|
| Hard top-K only (forward & backward) | 91.3 |
| Dense soft pooling | 96.0 |
| **Top-K with STE softmax surrogate** | **97.4** |

→ STE의 +1.4 ~ +6.1 향상이 method의 핵심.

### 5.2 Affordance Internalization
- 외부 affordance predictor 사용 (frozen)  vs 내부 `<AFF>` token → 내부화가 우월
- 직접 affordance mask gradient가 VLM까지 propagate → joint optimization 이점

### 5.3 K 값 (Top-K patch 수)
- K=16이 최적 (논문 부록)
- K가 너무 작으면 정보 부족, 너무 크면 dense soft에 가까워짐

> ❓ **예상 질문**: Affordance가 정말 *학습되었는지* (mask가 task-relevant region을 가리키는지) 어떻게 검증?
> **답변**: 본 논문은 attention map / mask visualization을 제시 (Figure section). 정량적으로는 GT mask가 있는 warmup stage 이후에도 mask quality(IoU 등)를 측정한 표가 있다. 다만 *mask 품질 → 성공률* 직접 회귀 분석은 약한 편.

---

## 6. 한계 및 미해결 문제

1. **Geometric shift 취약**: Camera/Robot perturbation에서 56%대 → affordance가 *appearance feature*에 의존, *geometric grounding*은 약함
2. **Long-horizon coordination 약점**: SimplerEnv stack block 14.6% → static affordance로는 시간적 dependency 처리 불가
3. **GT affordance mask 의존**: Warmup 단계에 GT mask 필요 → real-world scale-up bottleneck
4. **Single-arm 한정**: dual-arm/bimanual 평가 부재
5. **Param 효율성**: 4B VLM + affordance decoder + DiT-B는 모바일/edge 배포에 큰 부담
6. **K=16 hyperparameter sensitivity** 정량 분석이 paper 본문에 압축됨

### Attribution
- 성능 향상이 (a) Top-K STE pooling, (b) Qwen3-VL-4B의 강력함, (c) DiT-B flow matching, (d) `<AFF>` token internalization 중 어느 비중인지 부분 분리됨 — STE ablation은 명확, 내부 vs 외부 affordance 비교도 명확

---

## 7. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — affordance 내부화 + STE Top-K pooling의 조합이 깔끔하고 신선 |
| **Technical depth** | ★★★★☆ — 미분 가능 sparse pooling의 체계적 설계 |
| **Experimental rigor** | ★★★★☆ — LIBERO/LIBERO-Plus/SimplerEnv 광범위, ablation 명확 |
| **Practical impact** | ★★★☆☆ — LIBERO appearance OOD에서 강하나 geometric/temporal OOD에는 한계 |
| **Writing quality** | ★★★★☆ — 메커니즘과 ablation을 명확히 분리 |

**강점**: affordance를 VLA 내부 explicit 인터페이스로 만들고 *action-aligned*하게 grounding한 첫 체계적 시도. LIBERO 97.4 + LIBERO-Plus 78.1은 견고. **약점**: Camera/Robot geometric shift와 stack block 류 long-horizon에서 affordance만으로 부족함을 노출.

---

## 8. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | LIBERO 97.4 vs 비슷한 점수의 baseline(π0.5, GR00T-N1.6) 대비 *진정한 기여*는? | 동일 점수면 LIBERO-Plus와 ablation에서 차별화. Light/Background에서 96~97%는 appearance-OOD 강건성이 method 고유 가치 |
| 2 | STE는 표준 trick인데 왜 새로운 기여로 주장? | 새 기여는 STE 자체가 아니라 *affordance mask → action conditioning*의 미분 가능 sparse pooling으로의 적용. Hard 91.3 vs STE 97.4의 gap이 contribution |
| 3 | Top-K K=16은 patch grid에 종속(예: 16x16 patches면 16/256 = 6.25%) — 다른 입력 해상도에서 일반화? | 본 논문은 LIBERO/SimplerEnv standard 해상도에서만 검증. 해상도 sensitivity 평가는 부재. 일반화 위해 K를 비율로 정의하는 변형 가능 |
| 4 | Affordance mask quality(IoU) vs success rate의 상관? | 정량 회귀는 약하나 mask visualization을 통해 "good mask → good success" 정성 제시. 강한 quantitative correlation 분석이 missing |
| 5 | Real-world 평가가 거의 없는데 LIBERO/SimplerEnv 결과가 real-world에 전이될까? | 같은 그룹의 후속 작업에서 검증 필요. Light/Background OOD 강점은 real-world에 좋은 신호. Camera/Robot OOD 약점은 real에서 더 큰 문제가 될 수 있음 |
| 6 | GR00T-N1.6 같은 baseline의 LIBERO-Plus 점수? | LIBERO-Plus는 비교적 신생 benchmark로 모든 baseline의 비교 점수가 표에 다 나오지 않음. Camera/Robot에서 baseline도 비슷하게 떨어지는지 확인 필요 |

<!-- VERIFIED: pdf -->
