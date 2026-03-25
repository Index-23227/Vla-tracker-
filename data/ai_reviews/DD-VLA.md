# DD-VLA: Discrete Diffusion VLA — Bringing Discrete Diffusion to Action Decoding

> **한 줄 요약**: OpenVLA의 Prismatic-7B backbone 내에서 **bidirectional discrete diffusion**을 구현하여, autoregressive 대비 2x 빠른 추론(68.8ms vs 136.2ms)과 LIBERO 96.3%, SimplerEnv-Fractal 64.1%를 달성하며, 적응적 디코딩 순서와 secondary re-masking으로 오류 교정 지원.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- **Autoregressive VLA** (OpenVLA): 왼→오른 순차 생성으로 56 NFE (neural function evaluation) 필요 → 느림, 초기 토큰 오류 전파
- **Continuous diffusion VLA** (CogACT): 별도 diffusion head 필요 → VLM과 아키텍처 단절
- DD-VLA의 핵심: **같은 cross-entropy loss**를 사용하면서 diffusion 프레임워크를 VLM 내부에 통합

### 핵심 질문
- **Discrete diffusion을 VLM backbone 내에서 통합하여, AR과 continuous diffusion 양쪽을 능가할 수 있는가?**

---

## 2. 방법론 심층 분석

### 2.1 아키텍처

**Base**: OpenVLA의 Prismatic-7B (SigLIP + DINOv2 + Llama 2)
- Causal attention mask를 **action 위치에서만 bidirectional**로 수정
- Vision, language token은 causal, **action token은 서로 bidirectional attention**
- 별도 adapter/head 없이 **동일 transformer 내에서 discrete diffusion 수행**

### 2.2 Action Tokenization

- 7-DoF action: 3 translation + 3 rotation + 1 gripper
- **256 bins** (quantile-based scheme)
- Action chunk: **H=8** (LIBERO, SimplerEnv-Fractal), **H=3** (SimplerEnv-Bridge)
- Total tokens per chunk: 7 × H

### 2.3 Masked Cross-Entropy Loss

$$\mathcal{L} = -\sum_{i \in \mathcal{M}_{\gamma_t}} \log p_\theta(a_{0,i} | \tilde{a}_t, c)$$

- $\mathcal{M}_{\gamma_t}$: 마스킹된 위치 집합 (mask ratio $\gamma_t$)
- **VLM과 동일한 cross-entropy 목적함수** 유지 → pre-trained VLM prior 보존

### 2.4 Adaptive Decoding

**12-step refinement** (cosine schedule):
1. 모든 action token을 [MASK]로 시작
2. 각 step에서 confidence ranking으로 unmask 순서 결정
3. **Confidence gap** (top-1 vs top-2 확률 차이) 기반 ranking이 최적

### 2.5 Secondary Re-masking (2-check)

1. **Threshold check**: $\text{confidence} < \eta_t^{abs} = 0.5 \times (1 - t/T)$이면 re-mask
2. **Residual-drop check**: 이전 step 대비 confidence 큰 하락 시 re-mask

> ❓ **예상 질문**: Re-masking이 수렴을 보장하는가?
> **답변**: "Bayes reverse kernel과의 alignment을 유지하면서 cross-iteration consistency를 개선한다"고 주장. 실험적으로 re-masking이 +0.4%p 기여하며 수렴 문제 미보고.

---

## 3. 학습 설정

| 항목 | 값 |
|------|-----|
| GPU | **4× NVIDIA A800** |
| Batch size | 32 |
| LIBERO-Spatial/Object | 150K steps |
| LIBERO-Goal/Long | 300K steps |
| SimplerEnv | 100K steps |

---

## 4. 실험 결과 심층 분석

### LIBERO (Table 1)

| 모델 | Spatial | Object | Goal | Long | **Avg** |
|------|---------|--------|------|------|---------|
| OpenVLA (AR) | - | - | - | - | 76.5 |
| OpenVLA-OFT (Discrete) | - | - | - | - | 95.5 |
| OpenVLA-OFT (Cont-Diffusion) | - | - | - | - | 95.6 |
| π₀ | - | - | - | - | 94.2 |
| **DD-VLA** | **97.2** | **98.6** | **97.4** | **92.0** | **96.3** |

- OpenVLA-OFT 대비 **+0.7~0.9%p** (discrete 95.5 vs DD-VLA 96.3)
- π₀ 대비 **+2.1%p**

### SimplerEnv — Fractal Google Robot (Table 2)

| 메트릭 | π₀ | π₀-FAST | OpenVLA-OFT | **DD-VLA** |
|-------|-----|---------|------------|-----------|
| Visual Matching | 58.8% | 61.9% | 63.0% | **71.2%** |
| Variant Agg. | - | - | - | **56.9%** |
| **Overall** | - | - | - | **64.1%** |

- π₀ 대비 **+12.4%p** (visual matching)

### SimplerEnv — Bridge WidowX (Table 3)

| 모델 | π₀ | π₀-FAST | GR00T-N1 | **DD-VLA** |
|------|-----|---------|---------|-----------|
| Task SR | 40.1% | 48.3% | 49.5% | **54.2%** |

### OOD Robustness (Tables 4-5)

| Augmentation | DD-VLA Drop | OpenVLA-OFT (Discrete) Drop |
|-------------|------------|---------------------------|
| Language | **-1.4%p** | -8.0%p |
| Vision (Goal) | -21.0%p | - |
| Vision (Spatial) | **-1.0%p** | - |

- **Language robustness가 DD-VLA의 큰 강점** (-1.4% vs -8.0%)

---

## 5. Ablation 분석

### Decoding Strategy (Table 6, LIBERO-Goal)

| 전략 | SR (%) |
|------|--------|
| Parallel Decoding | 95.6 |
| + Random Order | 95.8 |
| + Confidence Gap | 96.6 |
| + Max Confidence | 97.0 |
| **+ Secondary Re-mask** | **97.4** |

### Temperature Schedule (Table 7)

| Temperature | SR (%) |
|------------|--------|
| Hard (τ=0) | 96.2 |
| Fixed (τ=1) | 96.4 |
| **Linear Decay** | **97.4** |

### Denoising Steps

- T=8~12에서 대부분의 gain
- T≥16: <1% 추가 향상, throughput 감소
- **T=12 optimal**

---

## 6. 추론 효율 (Table 8)

| 방법 | Latency (ms) | Speed (Hz) | NFE |
|------|-------------|-----------|-----|
| OpenVLA (AR) | 136.2 | 7.34 | 56 |
| OpenVLA (no KVcache) | 209.5 | 4.77 | 56 |
| OpenVLA-OFT (Parallel) | **31.1** | **32.14** | **1** |
| Cont-Diffusion (50 steps) | 199.9 | 5.00 | 50 |
| Cont-Diffusion (12 steps) | 67.1 | 14.91 | 12 |
| **DD-VLA (12 steps)** | **68.8** | **14.53** | **12** |

- AR 대비 **2x 빠름** (136.2→68.8ms), **4.7x 적은 NFE** (56→12)
- OpenVLA-OFT parallel (31.1ms)보다는 느림 — 12-step denoising overhead
- Continuous diffusion 12-step (67.1ms)과 **동등한 latency**

---

## 7. 관련 연구 비교

| 모델 | Token Type | Generation | Error Correction | VLM Prior 보존 | LIBERO |
|------|-----------|-----------|-----------------|-------------|--------|
| OpenVLA | Discrete | AR | ✗ | ✓ | 76.5% |
| OpenVLA-OFT | Continuous | Parallel | ✗ | △ | 95.6% |
| **DD-VLA** | **Discrete** | **Discrete Diffusion** | **✓ (re-masking)** | **✓ (same CE loss)** | **96.3%** |

---

## 8. 한계 및 미해결 문제

### 방법론적 미비점
1. **Quantization error**: 256-bin discretization의 precision 한계. Quantile-based이므로 data distribution에 의존
2. **Vision OOD 약점**: LIBERO-Goal에서 vision augmentation -21%p → 시각 변화에 취약
3. **OpenVLA-OFT parallel (31.1ms)에 비해 2x 느림**: DD-VLA는 12 NFE 필요, parallel은 1 NFE. Latency 우선 시 parallel이 유리
4. **Real-robot 결과 부재**: LIBERO, SimplerEnv 모두 시뮬레이션
5. **LIBERO near-saturation**: 96.3%에서 추가 개선의 의미 제한

### Attribution 문제
- DD-VLA vs OpenVLA-OFT의 차이(+0.7%p)가 **discrete diffusion** 때문인지 **bidirectional attention** 때문인지 분리 필요

---

## 9. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — VLM 내부에서 discrete diffusion, 동일 CE loss 유지 |
| **Technical depth** | ★★★★★ — Adaptive decoding, re-masking, temperature schedule 등 세밀 |
| **Experimental rigor** | ★★★★☆ — LIBERO + SimplerEnv + OOD + 효율 분석. Real-world 부재 |
| **Practical impact** | ★★★★☆ — Unified architecture, language robustness |
| **Writing quality** | ★★★★★ — 매우 체계적 |

**강점**: VLM prior를 보존하면서 diffusion의 이점을 얻는 우아한 설계. Language robustness와 adaptive decoding이 차별화. **약점**: Sim-only, vision OOD 취약, parallel decoding 대비 latency 불리.

---

## 10. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | OpenVLA-OFT parallel (31.1ms)이 더 빠른데, DD-VLA의 이점은? | DD-VLA는 error correction (re-masking), language robustness (-1.4% vs -8.0%), confidence-based ordering. Latency가 아닌 **품질**에서 차별화 |
| 2 | 12 step을 4 step으로 줄이면? | 미보고. T=8에서 "대부분의 gain"이므로 T=4에서도 competitive할 가능성. Latency ~23ms 예상 |
| 3 | Vision OOD -21%p를 어떻게 개선하는가? | Data augmentation이나 robust visual encoder로 완화 가능. 현 구조의 구조적 한계보다는 학습 데이터 문제 |
| 4 | Continuous diffusion과 latency가 동일하면, discrete의 이점은? | VLM prior 보존 (same CE loss), language robustness, 별도 head 불필요. 성능도 +0.7%p |

<!-- VERIFIED: pdf -->
