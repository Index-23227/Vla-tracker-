# DeeR-VLA: Dynamic Inference of Multimodal LLMs for Efficient Robot Execution

> **한 줄 요약**: OpenFlamingo (3B/9B) 기반 MLLM에 multi-exit 구조를 추가하여, action consistency 기반 early-exit으로 LLM GFLOPs를 **3.1-5.2x 절감** (31.2→6.0 GFLOPs), 실제 latency **68.1% 감소** (55→17.5ms)하면서 CALVIN 성능 유지(4.08 vs 4.07).

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 대형 MLLM 기반 로봇 정책이 **모든 시점에서 전체 모델을 실행** → 단순 동작에도 과다 연산
- 기존 모델 압축(pruning, quantization)은 **정적** → easy/hard 구분 없음
- "파지 직전"에는 깊은 reasoning, "빈 공간 이동"에는 얕은 처리로 충분

### 핵심 질문
- **태스크 난이도에 따라 동적으로 LLM depth를 조절하면 성능 저하 없이 얼마나 절감 가능한가?**

---

## 2. 방법론 심층 분석

### 2.1 아키텍처

**Base**: OpenFlamingo (3B: 24 layers, 9B: 32 layers)
- Vision encoder: ViT + Perceiver Resampler
- Action head: **4-layer LSTM + 3-layer MLP** (경량)
- **N개 exit point**: 매 2 self-attention layer마다 exit (3B: 6 exits from 12 layers)

### 2.2 Exit Decision: Action Consistency

연속 exit의 action 예측 간 L2 distance로 수렴 판단:

$$\|\pi_\theta(\tilde{x}_t^i, h_{t-1}) - \pi_\theta(\tilde{x}_t^{i-1}, h_{t-1})\|_2 < \eta_i$$

- 두 연속 exit의 action이 유사하면 → 이미 수렴, early exit
- Feature similarity나 time-based 기준보다 **action consistency가 최우수** (Table 4)

> ❓ **예상 질문**: Overconfident early exit의 위험은?
> **답변**: Action consistency는 confidence 자체가 아닌 **action 변화의 안정성**을 측정하므로, overconfidence 문제를 부분적으로 회피. 그러나 두 exit 모두 같은 방식으로 틀릴 수 있는 경우는 포착 불가.

### 2.3 Threshold 결정

두 가지 방식:
1. **Dataset-based**: 지수분포 가정으로 target exit proportion에서 threshold 도출
2. **Online**: Bayesian Optimization으로 constraint violation penalty 최소화

### 2.4 학습 전략

| 항목 | 값 |
|------|-----|
| Sampling s₁ | Uniform random exit per timestep |
| Sampling s₂ | Segment-based (연속 temporal window에서 같은 exit) |
| Auxiliary loss | 각 exit에 independent prediction head |
| Total loss | $\mathcal{L}_{total} = \mathcal{L}^* + \mathcal{L}_{aux}$ |

---

## 3. 학습 설정

| 항목 | 값 |
|------|-----|
| Batch size | 32 (4×8 distributed) |
| LSTM window | 12 timesteps |
| Dropout | LSTM 0.3, MLP 0.4 |
| LR | MLLM 1e-4, action head 2.5e-5 |
| Epochs | 3B: 4+4 (D→D), 3+1 (ABCD→D); 9B: ~24h A100 |
| GPU | 8× V100 (3B) / 8× A100 (9B) |

---

## 4. 실험 결과 심층 분석

### CALVIN Benchmark (Table 2)

| Setting | 모델 | Avg Len | LLM GFLOPs |
|---------|------|---------|-----------|
| D→D | RoboFlamingo++ | 2.71 | 31.2 |
| D→D | **DeeR** | **2.83** | **8.6 (3.6x↓)** |
| ABCD→D | RoboFlamingo++ | 4.07 | 31.2 |
| ABCD→D | **DeeR** | **4.13** | **10.0 (3.1x↓)** |
| ABC→D | RoboFlamingo++ | 2.59 | 31.2 |
| ABC→D | **DeeR** | **2.82** | **12.5 (2.5x↓)** |

- **성능 유지/약간 향상 + 3.1-3.6x GFLOPs 절감**

### 실제 추론 효율 (Table 5, ABCD→D)

| 모델 | Inference Time | GFLOPs | Avg Len |
|------|--------------|--------|---------|
| RoboFlamingo++ | 55ms | 31.2 | 4.07 |
| **DeeR** | **17.5ms** | **6.0** | **4.08** |

- **Wall-clock 68.1% 감소** (55ms → 17.5ms)
- **5.2x GFLOPs 절감** (31.2 → 6.0)
- 성능은 4.07 → 4.08으로 **오히려 약간 향상**

### Quantization 결합 (Table 6, ABCD→D)

| Precision | Memory | Avg Len |
|----------|--------|---------|
| Float32 | 6GB | 4.13 |
| Float16 | 3GB | 4.12 |
| Int4 | **1.7GB** | 3.91 |

- FP16: 성능 거의 동일하며 메모리 절반
- Int4: -0.22 하락이지만 1.7GB로 edge device 가능

---

## 5. Ablation 분석

### Auxiliary Loss (Table 3, ABCD→D)

| 설정 | Avg Len | GFLOPs |
|------|---------|--------|
| Without auxiliary losses | 2.64 | 4.9 |
| **With auxiliary losses** | **2.71** | 10.0 |

- Auxiliary loss가 성능을 **유의미하게 향상** (2.64 → 2.71)

### Exit Criterion 비교 (Table 4)

- **Action consistency**: 최우수
- Feature similarity: 차선
- Time-based: 최하

---

## 6. 메모리 사용량

| 모델 | LLM Memory |
|------|-----------|
| OpenFlamingo 3B DeeR-S | **2GB** |
| OpenFlamingo 3B DeeR-B | 6GB |
| OpenFlamingo 9B DeeR-S | 8GB |
| OpenFlamingo 9B DeeR-B | 12GB |

---

## 7. 관련 연구 비교

| 방법 | 압축 유형 | Dynamic | 성능 보존 | Latency 감소 |
|------|---------|---------|----------|------------|
| Pruning | Static | ✗ | △ | ~2x |
| Quantization (INT4) | Static | ✗ | △ | ~2x |
| **DeeR-VLA** | **Dynamic Early Exit** | **✓** | **✓** | **3.1x (wall-clock 68.1%)** |

---

## 8. 한계 및 미해결 문제

### 방법론적 미비점
1. **OpenFlamingo backbone**: 3B/9B 규모이며 최신 VLM (7B LLaMA 등) 대비 구형. 최신 VLA에서의 적용성 미검증
2. **CALVIN only**: LIBERO, SimplerEnv 등 다른 벤치마크 결과 없음
3. **Temporal consistency**: 시점마다 다른 depth에서 exit하면 action의 시간적 일관성 문제 → 논문에서 "segment-based sampling (s₂)"으로 완화하나 정량적 분석 부족
4. **Action consistency 기준의 한계**: 두 exit이 동일하게 틀리는 경우 (같은 방향의 오류) 포착 불가
5. **Threshold tuning**: Dataset-based 또는 Bayesian Optimization — 새 환경에서 재조정 필요

### Attribution 문제
- 성능 유지가 **early exit 설계**의 효과인지, **auxiliary loss가 각 exit의 독립적 학습 품질을 보장**했기 때문인지 분리 필요
- Auxiliary loss 없으면 2.64 (vs 2.71) → auxiliary가 핵심

---

## 9. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Early exit의 VLA 적용, action consistency criterion |
| **Technical depth** | ★★★★★ — Exit mechanism, threshold 결정, quantization 결합 등 체계적 |
| **Experimental rigor** | ★★★☆☆ — CALVIN 단일 벤치마크 |
| **Practical impact** | ★★★★★ — Edge deployment의 핵심 문제 해결. 17.5ms, 2GB |
| **Writing quality** | ★★★★☆ |

**강점**: 55ms→17.5ms (68.1% 감소)의 실제 wall-clock 절감이 인상적. 성능 유지가 확실. Quantization과 결합하면 1.7GB까지 축소 가능 → Jetson 등 edge device 배포 가능. **약점**: CALVIN only, 구형 backbone, temporal consistency 분석 부족.

---

## 10. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | OpenVLA (7B)에 적용하면 얼마나 절감되는가? | 직접 검증 없음. 7B에서는 절감 폭이 더 클 것으로 예상 (deeper model = more early-exit 기회) |
| 2 | 17.5ms면 57Hz인데, 이게 action quality를 향상시키는가? | 빈번한 re-planning이 closed-loop 성능을 높일 수 있음. 다만 실험에서 이를 직접 보이지 않음 |
| 3 | Safety-critical 상황에서 early exit가 위험하지 않은가? | Action consistency가 "변화 없음 = 안전"으로 판단. 하지만 두 exit이 동일하게 위험한 action을 예측할 수 있음 |
| 4 | DepthCache (visual token merging)와 결합하면? | Multiplicative 절감 가능 — token 수 감소 + early exit = 더 빠른 추론 |
| 5 | 어떤 layer에서 주로 exit하는가? | "Easy" 상황(approach): 초기 layer, "Hard" 상황(stacking, push switch): 깊은 layer. 시각화 포함 |

<!-- VERIFIED: pdf -->
