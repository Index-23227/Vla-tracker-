# VLA-IAP: Training-Free Visual Token Pruning via Interaction Alignment for VLA Models

> **한 줄 요약**: Geometric prior (Sobel edge) + semantic prior (cross-modal attention) + motion prior (temporal difference)를 결합한 training-free token pruning으로, OpenVLA-OFT에서 LIBERO **97.8%** (baseline 97.1%) at **1.25× speedup**, 최대 **1.54× at 30% retention**.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- VLA의 visual token 수가 추론 bottleneck (DepthCache와 유사한 문제)
- 기존 pruning (FastV, ToMe)은 **로봇 manipulation의 interaction pattern을 고려하지 않음**
- 접근(approach) 단계와 조작(interaction) 단계에서 필요한 visual information이 다름

### 핵심 질문
- **Robot-environment interaction 패턴에 맞춰 token pruning 전략을 적응적으로 전환하면?**

---

## 2. 방법론 심층 분석

### 2.1 Three Priors

| Prior | 구현 | 역할 |
|-------|------|------|
| **Geometric** | Sobel edge detection → patch-level scores | 구조적 affordance (물체 경계) |
| **Semantic** | Visual-text cross-modal attention + spatial smoothing | Task-relevant 영역 |
| **Motion** | 2nd-order temporal difference + morphological ops + Gaussian smoothing | 움직이는 영역 |

Final score: $\text{Sem} + \text{Temp} + w_{\text{edge}} \times \text{Edge}$

### 2.2 Interaction-Aligned Dynamic Strategy

**IoU threshold ($\theta_{\text{iou}}$)** 기반으로 mode 전환:

**Conservative mode** (IoU ≤ θ): Pre-interaction (접근 단계)
- Double-weak exclusion: semantic과 motion **모두** 낮은 영역만 제거
- 중요 영역을 보수적으로 보존

**Aggressive mode** (IoU > θ): Active interaction (조작 단계)
- Semantic mask를 core region으로 수축 + motion mask와 union
- 불필요 배경을 적극 제거

### 2.3 Evaluated VLA Models

- **OpenVLA-OFT** (7B): Primary (regression/parallel decoding)
- **π₀, π₀.₅**: Flow matching action head
- **DreamVLA**: Diffusion transformer

---

## 3. 실험 결과 심층 분석

### LIBERO (OpenVLA-OFT)

| Retention | Spatial | Object | Goal | Long | **Avg** | **Speedup** |
|----------|---------|--------|------|------|---------|-----------|
| 100% (baseline) | 98.6 | 98.2 | 96.6 | 94.8 | 97.1 | 1.00× |
| **70%** | 97.6 | **99.6** | **98.4** | 95.6 | **97.8** | **1.25×** |
| 50% | 97.3 | 99.1 | 98.2 | 95.2 | 97.5 | 1.37× |
| 30% | 96.6 | 98.8 | 98.0 | 94.8 | 97.1 | **1.54×** |

- 70% retention에서 baseline **능가** (97.8 > 97.1) → pruning이 regularization 효과

### Real Robot (π₀.₅)

| Setup | Baseline (ms) | VLA-IAP (ms) | Speedup |
|-------|-------------|-------------|---------|
| Single-arm | 88.1 | 59.7 | **1.48×** |
| Dual-arm | 124.3 | 84.6 | **1.47×** |

### Ablation (OpenVLA-OFT)

| Configuration | SR (%) | Speedup |
|--------------|--------|---------|
| Base | 94.5 | 1.00× |
| +Spatiotemporal (k=0.5) | 94.8 | 1.42× |
| +IoU Switching (θ=0.05) | 97.2 | 1.32× |
| **+Edge (full)** | **97.8** | **1.25×** |

---

## 4. 한계 및 미해결 문제

1. **Framework이지 model이 아님**: VLA-IAP 자체는 action을 생성하지 않음. Base VLA에 의존
2. **IoU threshold 설정**: θ=0.05가 범용적인지 task마다 조정 필요한지 미분석
3. **DepthCache와의 직접 비교 없음**: 유사한 동기의 다른 방법과 head-to-head 미수행
4. **Sobel edge의 한계**: Textureless surface에서 edge detection 실패 가능

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Interaction-aligned adaptive pruning |
| **Technical depth** | ★★★★★ — 3 priors + mode switching + 4 VLA 검증 |
| **Experimental rigor** | ★★★★★ — LIBERO + Real + 4 VLA models + ablation |
| **Practical impact** | ★★★★★ — Training-free, 1.25-1.54× speedup |

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | DepthCache와의 차이는? | DepthCache는 depth 기반 공간 분할, VLA-IAP는 interaction pattern 기반 시간적 mode 전환. 상보적 |
| 2 | 70% retention에서 baseline을 넘는 이유는? | Token pruning이 noisy/irrelevant visual information을 제거하여 regularization 효과 |
| 3 | DeeR-VLA (early exit)와 결합하면? | Token pruning + early exit = multiplicative 가속. 미검증이나 매우 유망 |

<!-- VERIFIED: pdf -->
