# Fast-WAM: Do World Action Models Need Test-time Future Imagination?

> **한 줄 요약**: World-Action Model에서 test-time에 미래 비디오를 실제로 "상상"할 필요가 없음을 밝히고, 학습 시에만 world model 목적함수를 활용하여 실시간 190ms latency (4배+ 가속)와 동등 성능 달성.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- WAM (GR-1, UniPi, Motus)은 **test-time에 미래 비디오를 생성** → 이를 조건으로 action 예측
- 비디오 생성의 높은 latency (~800ms+)가 real-time control 방해
- 핵심 의문: 비디오 생성이 test-time에 정말 필요한가, 아니면 **학습 시의 auxiliary signal**로만 유용한가?

### 핵심 질문
- **미래 상상(imagination)을 학습 시에만 수행하고, 추론 시에는 skip하면 어떤 성능 변화가 있는가?**
- **학습 시 world model 목적함수가 representation quality를 어떻게 향상시키는가?**

---

## 2. 방법론 심층 분석

### 2.1 Training-time Only Imagination

```
[학습 시]
Observation → [Shared Encoder] → representation
                                    ↓ (branch 1: video prediction — auxiliary)
                                    ↓ (branch 2: action prediction — primary)

[추론 시]
Observation → [Shared Encoder] → representation → action prediction only
              (video prediction branch OFF)
```

> ❓ **예상 질문**: 학습 시 video prediction이 representation에 미치는 영향의 메커니즘은?
> **답변**: Multi-task learning의 regularization 효과. Video prediction이 물리적 dynamics에 대한 inductive bias를 encoder에 주입하여, action-relevant features의 품질을 높임. 이는 auxiliary task가 주요 작업을 돕는 전통적 multi-task learning 관점.

### 2.2 Architecture Details

Shared encoder 이후 분기:
- **Video head**: Diffusion-based future frame generation
- **Action head**: Flow matching-based action prediction

두 head가 encoder를 공유하므로, video head의 학습 signal이 encoder를 풍부하게 함.

### 2.3 190ms Real-time Inference

Video generation 제거로:
- 기존 WAM: ~800ms (video generation ~600ms + action ~200ms)
- Fast-WAM: ~190ms (action only)
- **4x+ 속도 향상**, 5Hz 실시간 제어 가능

---

## 3. 실험 결과 심층 분석

| 모델 | SR (%) | Latency (ms) | 비고 |
|------|--------|-------------|------|
| Imagine-then-Execute WAM | 82.3 | ~800 | Full video gen |
| Action-only (no world model) | 75.8 | ~190 | No auxiliary |
| **Fast-WAM** | **81.5** | **~190** | Train-time only |

- Fast-WAM은 full WAM의 **98.9% 성능**을 유지하면서 **4.2x 가속**
- Action-only 대비 +5.7%p → **training-time imagination의 가치 입증**

---

## 4. Ablation 분석

| 설정 | SR (%) |
|------|--------|
| Full Fast-WAM | 81.5 |
| Train-time video + test-time video | 82.3 |
| No video at all (train & test) | 75.8 |
| Random video labels (noise) | 74.2 |
| Test-time video only (no train) | 76.1 |

- **Train-time video가 핵심** (+5.7%p), test-time video는 marginal (+0.8%p)
- Random labels는 해로움 → 의미 있는 video prediction signal이 중요

---

## 5. 관련 연구 비교

| 모델 | Train Video | Test Video | Latency | Performance |
|------|-----------|-----------|---------|------------|
| UniPi | ✓ | ✓ | Very High | Good |
| GR-1 | ✓ | ✓ | High | Good |
| Motus | ✓ | ✓ | Medium-High | Very Good |
| **Fast-WAM** | **✓** | **✗** | **Low (190ms)** | **~Same** |

---

## 6. 한계 및 미해결 문제

### 방법론적 미비점
1. **"Imagination" 범위**: Video prediction만 test-time에서 제거. 다른 형태의 imagination (latent planning)도 불필요한지 미탐구
2. **복잡한 long-horizon 태스크**: 단순 태스크에서는 test-time imagination이 불필요할 수 있으나, **plan이 중요한 long-horizon**에서는 필요할 수 있음
3. **Training cost**: 학습 시에는 여전히 video prediction을 수행하므로, 학습 비용은 기존 WAM과 동일
4. **한정된 벤치마크**: 제한된 환경에서만 테스트

### Attribution 문제
- Train-time video의 효과가 "dynamics understanding"에서 오는지 "data augmentation/regularization"에서 오는지 분리 불완전
- Random image generation (의미 없는 image prediction)도 regularization 효과가 있을 수 있는데, 이 비교가 부분적

---

## 7. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — "Test-time imagination이 불필요하다"는 도발적 질문 |
| **Technical depth** | ★★★☆☆ — 아이디어 자체는 간단 |
| **Experimental rigor** | ★★★★☆ — 체계적 ablation |
| **Practical impact** | ★★★★★ — Real-time WAM의 핵심 병목 해결 |
| **Writing quality** | ★★★★★ — 질문 설정이 날카로움 |

**강점**: "정말 test-time imagination이 필요한가?"라는 근본적 질문을 정면으로 다룸. 실험적 답변이 명쾌. 4x 가속의 실용적 가치. **약점**: Long-horizon task에서 이 결론이 유지되는지 미검증.

---

## 8. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Long-horizon task (10+ steps)에서도 test-time imagination이 불필요한가? | 이 논문의 결론이 short-horizon에서만 유효할 가능성 높음. Long-horizon에서는 explicit planning이 더 중요 |
| 2 | Latent imagination (pixel이 아닌 latent space)은 test-time에 유용할 수 있지 않나? | 매우 유망한 중간 지점. Pixel generation의 overhead 없이 latent planning의 이점. FLARE가 이 방향 |
| 3 | Video prediction head를 완전히 제거하고 contrastive learning 등으로 대체하면? | Dynamics 이해를 위한 다른 auxiliary task도 유효할 수 있음. 비교 부재 |
| 4 | 0.8%p (test-time video의 기여)가 safety-critical 상황에서 중요하지 않은가? | 특정 시나리오에서는 0.8%p가 결정적일 수 있음. Risk-sensitive 환경에서는 full WAM이 바람직 |
