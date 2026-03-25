# DepthCache: Depth-Guided Training-Free Visual Token Merging for VLA Model Inference

> **한 줄 요약**: Depth 정보를 구조적 사전지식(structural prior)으로 활용하여 VLA의 visual token을 지능적으로 병합·압축하는 training-free 추론 가속 기법으로, 3개 VLA에서 1.28x speedup과 최소한의 성능 하락 달성.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- VLA 모델의 visual token 수가 추론 latency의 주요 bottleneck (수백~수천 개 토큰)
- 기존 token pruning/merging (ToMe 등)은 **2D 공간에서 균일하게 적용** → 로봇 조작에 중요한 근거리 물체와 무관한 배경을 동등하게 취급
- Depth-aware token selection은 학습이 필요한 경우가 많아 기존 VLA에 즉시 적용 불가

### 핵심 질문
- **Depth 기반으로 "어디를 자세히 보고 어디를 대충 볼 것인가"를 결정하면 token 효율이 올라가는가?**
- **추가 학습 없이(training-free) 기존 VLA에 plug-in 가능한 가속이 있는가?**

---

## 2. 방법론 심층 분석

### 2.1 Depth-Guided Region Partitioning

Depth map을 기준으로 이미지를 영역 분할:

$$R_k = \{(u,v) : d_k^{\text{min}} \leq D(u,v) < d_k^{\text{max}}\}, \quad k = 1, ..., K$$

- **근거리 영역** (로봇 작업 공간): 낮은 merge ratio → 세밀한 token 유지
- **원거리 영역** (배경): 높은 merge ratio → 공격적 압축

> ❓ **예상 질문**: Depth 추정이 부정확한 영역에서 잘못된 merge ratio가 적용되면?
> **답변**: Depth estimation 오류가 직접 token allocation에 영향. 특히 depth discontinuity (물체 경계)에서 잘못된 분할이 발생 가능. 저자들은 GT depth와 estimated depth 모두 테스트하나 estimation 오류의 체계적 분석은 제한적.

### 2.2 Differentiated Merge Ratios

영역별 merge ratio:

$$r_k = r_{\text{base}} \cdot f(d_k), \quad f(d) = \alpha \cdot \frac{d - d_{\min}}{d_{\max} - d_{\min}}$$

근거리: $r \approx 0.1$ (10% 병합), 원거리: $r \approx 0.7$ (70% 병합).

> ❓ **예상 질문**: 모든 task에서 근거리=중요, 원거리=불중요라는 가정이 성립하는가?
> **답변**: Navigation 등에서는 원거리 정보가 중요. 본 논문은 **tabletop manipulation** 위주로 평가하여 이 가정이 성립하나, 일반적이지 않음.

### 2.3 Training-Free Token Merging

기존 ToMe (Token Merging) 알고리즘에 depth-guided ratio를 적용:
1. 각 영역 내에서 cosine similarity 기반 유사 토큰 쌍 탐색
2. 영역별 merge ratio에 따라 병합
3. 결과적으로 총 token 수 감소 → 이후 transformer layer의 self-attention 연산 절감

---

## 3. 실험 결과 심층 분석

| VLA 모델 | 원본 SR (%) | DepthCache SR (%) | Speedup |
|---------|-----------|------------------|---------|
| OpenVLA | 76.5 | 75.8 (-0.7) | 1.28x |
| RoboVLM | 82.3 | 81.5 (-0.8) | 1.25x |
| SpatialVLA | 85.1 | 84.2 (-0.9) | 1.31x |

- 3개 VLA 모두에서 **1%p 미만 성능 하락**으로 1.25-1.31x 속도 향상
- Training-free라는 점에서 즉시 적용 가능

---

## 4. Ablation 분석

| Merge 전략 | SR 변화 | Speedup |
|-----------|--------|---------|
| Uniform merge (no depth) | -2.5%p | 1.28x |
| Depth-guided (proposed) | -0.7%p | 1.28x |
| Random merge | -4.1%p | 1.28x |
| Aggressive (50% more merge) | -3.2%p | 1.45x |

- **동일 speedup에서 depth-guided가 uniform 대비 1.8%p 우수** → depth 정보의 가치 입증

---

## 5. 관련 연구 비교

| 방법 | Training-Free | Depth-Aware | VLA 검증 | Speedup |
|------|-------------|------------|---------|---------|
| ToMe | ✓ | ✗ | ✗ (ViT) | 1.2-2x |
| FastV | ✓ | ✗ | △ | 1.3x |
| DeeR-VLA | ✗ | ✗ | ✓ | 5x+ (다른 차원) |
| **DepthCache** | **✓** | **✓** | **✓** | **1.28x** |

---

## 6. 한계 및 미해결 문제

### 방법론적 미비점
1. **제한적 speedup**: 1.28x는 DeeR-VLA (5x+)나 quantization (2-4x)에 비해 modest. Token merging만으로는 dramatic한 가속 어려움
2. **Depth estimation overhead**: Depth map 계산 자체의 latency가 보고되지 않음. 만약 depth estimation에 20ms가 소요되면 순수 speedup이 줄어듦
3. **Task-specific 가정**: 근거리=중요라는 가정이 모든 로봇 시나리오에 적용되지 않음
4. **다른 가속 기법과의 결합**: Early exit, quantization 등과의 결합 효과 미검증

### Attribution 문제
- 1.28x speedup 중 "depth-guided"의 기여가 얼마인지 불명확. Uniform merge도 1.28x를 달성하되 성능이 더 낮음 → depth의 기여는 speedup이 아닌 **성능 보존**

---

## 7. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★☆☆ — ToMe + depth의 자연스러운 결합, 혁신성은 제한적 |
| **Technical depth** | ★★★☆☆ — 간결하나 깊이 부족 |
| **Experimental rigor** | ★★★★☆ — 3개 VLA에서 검증, 다양한 ablation |
| **Practical impact** | ★★★★☆ — Training-free, plug-in 가능 |
| **Writing quality** | ★★★★☆ — 깔끔 |

**강점**: Training-free로 즉시 적용 가능하다는 실용성이 최대 장점. Depth의 활용이 직관적이고 효과적. **약점**: Speedup이 modest하고 다른 가속 기법 대비 경쟁력이 제한적.

---

## 8. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | DeeR-VLA와 결합하면 multiplicative speedup이 가능한가? | 이론적으로 가능 (token merge + early exit). 실증 부재 |
| 2 | Depth 대신 attention map으로 중요 영역을 판단하면? | 가능하나 attention 계산 자체가 overhead. Depth는 사전 계산 가능하여 더 효율적 |
| 3 | 1.28x speedup이 real-time control에 의미 있는 차이를 만드는가? | 40ms → 31ms 수준. 25Hz → 32Hz. Marginal하지만 누적 효과는 있음 |
| 4 | Segmentation mask (SAM 등)를 쓰면 더 정밀한 영역 분할이 가능하지 않나? | 가능하나 SAM의 추가 latency가 문제. Depth는 이미 많은 VLA에서 입력으로 사용되므로 추가 비용 없음 |
