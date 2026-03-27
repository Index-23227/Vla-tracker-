# UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control from Egocentric Human Videos

> **한 줄 요약**: 8종 dexterous hand(6-24 DoF)에 대한 50K+ trajectory 데이터셋(UniDex-Dataset)과 82차원 통합 action space(FAAS)를 구축하고, flow matching 기반 3D VLA로 81% task progress와 zero-shot cross-hand transfer 달성.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- Dexterous hand 연구가 **단일 hand 형태에 국한** → cross-hand generalization 불가
- 서로 다른 hand의 **action space가 상이** (DoF 6~24) → unified policy 구축 어려움
- Egocentric human video는 풍부하나 로봇 hand에 직접 활용 불가

### 핵심 질문
- **다양한 dexterous hand를 하나의 통합 action space로 표현하고, 단일 VLA로 제어할 수 있는가?**

---

## 2. 방법론 심층 분석

### 2.1 FAAS (Function-Actuator-Aligned Space)

82차원 통합 action space:
- **18-dim**: Wrist poses (9/hand × 2 hands, 6D rotation + 3D translation)
- **64-dim**: Joint commands (32/hand, 21 shared base slots + hand-specific)

이를 통해 DoF가 다른 hand들을 **동일 표현 공간**에서 학습.

### 2.2 Architecture

- **Vision**: Uni3D pointcloud encoder (ViT, 2D pretrained로 초기화)
- **Action**: **Conditional flow matching** (velocity field prediction, Euler integration, step δ=0.1)
- **Input**: Single-view colored pointcloud + language instruction + proprioception
- **Output**: Action chunk $A_t = [a_t, ..., a_{t+H-1}]$

### 2.3 Training

| 단계 | GPU | Batch | Duration |
|------|-----|-------|----------|
| **Pretrain** | 8× H800 | 128 | ~24h, 3 epochs (~30K steps) |
| **Post-train** | 2× H800 | 8 | ~4h/task, 50 epochs (~3K steps) |

- Optimizer: AdamW, cosine scheduler, initial LR 1e-4
- 50 demonstrations per task for post-training

### 2.4 UniDex-Dataset

- **9M paired image-pointcloud-action frames** (30 fps)
- **50K+ trajectories** across 8 hands
- Sources: H2O, HOI4D, HOT3D, TACO (egocentric RGB-D)

---

## 3. 실험 결과 심층 분석

### Real-World Tool-Use (5 tasks, 2 hands)

| Model | Avg Task Progress | Final SR |
|-------|------------------|----------|
| DP (Diffusion Policy) | 29.0±19.9% | 22.0±22.5% |
| DP3 | 35.0±17.1% | 30.0±18.7% |
| π₀ | 38.0±7.4% | 35.0±10.0% |
| UniDex-VLA (No Pretrain) | 32.5±18.5% | 23.0±12.0% |
| **UniDex-VLA** | **81.0±12.1%** | **76.0±17.8%** |

- π₀ 대비 **+43%p task progress** → pretraining의 결정적 기여

### Zero-shot Cross-Hand Transfer

| Target Hand | SR (%) |
|------------|--------|
| Inspire (train) | 76% |
| Oymotion (zero-shot) | 60% |
| Wuji (zero-shot) | 40% |

### Human-Robot Co-training

- Human:robot demo exchange rate ≈ 2:1
- Human demos ~5.2× faster to collect
- Robot data 필수 (human only로는 insufficient)

---

## 4. 한계 및 미해결 문제

1. **표준 벤치마크 없음**: LIBERO, CALVIN 등 미평가 → 직접 비교 어려움
2. **Single-view pointcloud 의존**: Multi-view나 RGB-only setup에서의 성능 미검증
3. **FAAS의 hand 호환성**: 8종 hand에서 검증, 더 다양한 형태 (prosthetics 등)에서는?
4. **Parameter count 미공개**

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — FAAS unified action space + 8-hand dataset |
| **Technical depth** | ★★★★★ — 9M frames, cross-hand transfer, human co-training |
| **Experimental rigor** | ★★★★☆ — Real-world 5 tasks, zero-shot transfer. 표준 벤치마크 부재 |
| **Practical impact** | ★★★★★ — Dexterous manipulation의 foundation 연구 |

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | DexVLA와의 차이는? | DexVLA는 diffusion expert on bimanual arms, UniDex는 flow matching on dexterous hands. 다른 embodiment |
| 2 | FAAS에서 hand가 21 DoF 미만이면? | Unused dimensions을 masking. FAAS의 shared 21 base slots이 다양한 DoF를 수용 |
| 3 | Pretraining 없이 (32.5%) vs 있으면 (81.0%) — 이게 data의 효과인가 architecture의 효과인가? | Data 효과가 지배적. FAAS가 data scaling을 가능하게 하는 enabler |

<!-- VERIFIED: pdf -->
