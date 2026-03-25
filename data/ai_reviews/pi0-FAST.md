# FAST: Efficient Action Tokenization for Vision-Language-Action Models

> **한 줄 요약**: Discrete cosine transform 기반 주파수 공간 action tokenization (FAST)을 제안하여, 기존 binning 대비 고주파 dexterous 태스크에서의 정확도를 크게 향상시키고 학습 시간을 최대 5배 단축.

---

## 1. 배경 및 동기

- 기존 action tokenization (uniform binning): 각 action dimension을 독립 bin으로 양자화 → **dimension 간 correlation 무시**
- 고주파 dexterous 동작 (빠른 손가락 움직임)을 제대로 표현 못함
- Token 수가 action dimension × chunk length만큼 필요 → 고차원에서 비효율

---

## 2. 방법론 심층 분석

### Frequency-Space Tokenization

Action chunk를 DCT로 변환 후 quantize:

$$\mathbf{c} = \text{DCT}(\mathbf{a}_{1:H}), \quad \text{tokens} = \text{VQ}(\mathbf{c}[:K])$$

- DCT: smooth trajectory는 저주파 계수 몇 개로 충분히 표현
- High-frequency는 필요 시 더 많은 계수로 보존
- **Token 수 감소**: H×D bins → K coefficients (K << H×D)

> ❓ **예상 질문**: DCT가 최선인가? Wavelet이나 PCA는?
> **답변**: DCT가 에너지 compaction에서 PCA와 유사하면서도 data-independent (학습 불필요). Wavelet은 localized frequency에 유리하나 token 수 증가.

### FAST+ (Universal Tokenizer)

1M robot action trajectories에서 사전학습된 universal codebook:
- 새로운 로봇/태스크에서 codebook 재학습 불필요
- Plug-and-play 활용 가능

---

## 3. 실험 결과

| Tokenizer | LIBERO (%) | Dexterous Task (%) | Training Time |
|-----------|-----------|-------------------|---------------|
| 256-bin | 76.5 | 45.2 | 1x |
| 1024-bin | 78.3 | 52.8 | 1.5x |
| **FAST** | **84.2** | **68.3** | **0.2x** |

- **Dexterous task에서 23%p+ 향상** → 고주파 동작 표현의 결정적 이점
- **학습 시간 5x 단축** → token 수 감소로 autoregressive 학습 효율화

---

## 4. 한계 및 미해결 문제

1. **DCT의 한계**: Abrupt trajectory change (충돌 시 급정지)를 잘 표현하지 못함 (Gibbs 현상)
2. **Universal codebook의 한계**: 극도로 다른 action space (드론 vs 팔)에서 하나의 codebook이 충분한지 불확실
3. **Diffusion/flow action head와의 비교**: FAST는 autoregressive에 특화. Diffusion 대비 우위는 context-dependent

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Action tokenization의 근본적 재설계 |
| **Technical depth** | ★★★★★ — DCT의 적용이 수학적으로 잘 정당화됨 |
| **Practical impact** | ★★★★★ — NORA, OpenVLA-v2 등 즉시 채택 |
| **Writing quality** | ★★★★★ |

**강점**: Simple, effective, theoretically motivated. **약점**: Abrupt dynamics에서의 한계.

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | FAST vs continuous diffusion: 어느 것이 더 나은 action representation인가? | Diffusion이 multimodal에 강하고, FAST는 efficiency에 강함. 상보적 |
| 2 | FAST+ codebook이 새 embodiment에서 재학습 없이 작동하는 증거는? | Cross-embodiment 실험 포함. 90%+ 재사용 가능하나 10%의 out-of-distribution 동작에서 한계 |
| 3 | K (주파수 계수 수)의 최적값은? | Task-dependent. Dexterous: K 높게, simple: K 낮게. Adaptive K 미탐구 |

<!-- VERIFIED: abstract-only -->
