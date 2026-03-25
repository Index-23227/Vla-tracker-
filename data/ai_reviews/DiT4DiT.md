# DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control

> **한 줄 요약**: Video DiT (Cosmos-2B 초기화)와 Action DiT (GR00T-N1 기반)를 cascaded coupling하되, **single forward pass의 intermediate feature를 추출**하여 video reconstruction을 bypass, LIBERO **98.6%** SOTA + RoboCasa 50.8% + 7x 빠른 수렴 달성.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- WAM(GR-1, Motus)에서 video와 action의 joint modeling이 **representation entanglement** 야기
- Test-time video generation이 latency bottleneck (Fast-WAM이 지적)
- Video reconstruction에 최적화된 feature가 action prediction에 최적이 아닐 수 있음

### 핵심 질문
- **Video backbone의 intermediate feature를 single forward pass로 추출하면, 다중 denoising step의 video generation을 완전히 bypass할 수 있는가?**

---

## 2. 방법론 심층 분석

### 2.1 Dual-DiT Framework

**Video DiT**: Cosmos-Predict2.5-2B로 초기화, causal video VAE + flow-matching
**Action DiT**: GR00T-N1 기반, AdaLN + cross-attention to video features

핵심: Video DiT의 **forward hook**으로 **layer 18**의 hidden activation을 intercept → Action DiT의 conditioning

### 2.2 Tri-Timestep Training

세 개의 독립적 diffusion timestep:
- **τ_v (video)**: Uniform [0,1] — 전체 denoising trajectory 노출
- **τ_f (feature extraction)**: **Fixed** — 안정적 visual conditioning
- **τ_a (action)**: Beta분포 — critical control phase에 편향

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{action}} + \lambda \cdot \mathcal{L}_{\text{video}}$$

### 2.3 Single Forward Pass Feature Extraction

> **핵심 발견**: Multi-step denoising보다 **single forward pass**가 더 좋은 feature 제공
> - 추가 denoising step → **pixel reconstruction에 과도하게 특화** → action feature 품질 하락
> - Layer 18이 최적 (초기 layer: poor, terminal layer: "drastic collapse")

---

## 3. 실험 결과 심층 분석

### LIBERO (Table 1)

| Suite | DiT4DiT | π₀.₅ | CogVLA (prev SOTA) |
|-------|---------|-------|-------------------|
| Spatial | 98.4 | 98.8 | - |
| Object | **99.6** | 98.8 | - |
| Goal | **98.6** | 98.0 | - |
| Long | **97.6** | - | 95.4 |
| **Average** | **98.6** | - | 97.4 |

### RoboCasa-GR1 Tabletop (Table 2, 24 tasks)

| Model | Avg SR (%) |
|-------|-----------|
| Qwen3DiT baseline | 36.2 |
| GR00T-N1.6 | 40.8 |
| GR00T-N1.5 | 41.8 |
| **DiT4DiT** | **50.8 (+9.0 vs GR00T-N1.5)** |

- 24 task 중 **16개에서 최고**
- Precision-demanding: CanToDrawerClose **74% vs 56%** (+18%p)

### Real-World Unitree G1 (Figure 5)

| Task | DiT4DiT | GR00T-N1.5 | Qwen3DiT |
|------|---------|-----------|----------|
| Arrange Flower | **75%** | 25% | 0% |
| Stack Cup | **60%** | 25% | <10% |
| Move Spoon | **40%** | 15% | <10% |
| Drawer | **90%** | - | 0% |
| Box Packing | **50%** | - | 0% |

- Qwen3DiT가 real deployment에서 **collapse** → DiT4DiT만 안정적 작동

### Generalization

| 설정 | DiT4DiT | Qwen3DiT |
|------|---------|----------|
| Unseen objects (sim) | 54.5% | 32.0% (+22.5%p) |
| Zero-shot flower (real) | 70% | 10% |
| Zero-shot cup quantity (real) | 50% | - |

---

## 4. Ablation 분석

### Feature Extraction Layer (Figure 8)
- Layer 2-8: Poor
- **Layer 18: Optimal**
- Layer 24-28: "Drastic collapse" (pixel reconstruction에 특화)
- All layers 평균: 단일 layer보다 열위

### Denoising Steps for Feature (Figure 8)
- **Single forward pass: Optimal**
- Step 증가 → 성능 단조 하락 → "pixel-level reconstruction에 over-commit"

### Joint vs Decoupled Training (Figure 8c)
- t-SNE: Joint training → 더 선명한 temporal separation
- Silhouette score: **0.09 → 0.17** (거의 2배)

---

## 5. 효율성

| Metric | DiT4DiT | GR00T-N1.5 | Qwen3DiT |
|--------|---------|-----------|----------|
| Inference freq | **6 Hz** | 13 Hz | 9 Hz |
| Convergence | **7x faster** | - | - |
| Data efficiency | **10x** | - | - |
| Params (trainable) | **2.2B** | - | - |

- **6Hz는 가장 느림** → dual DiT의 computational overhead
- 그러나 **7x 빠른 수렴 + 10x data efficiency**로 학습 비용 상쇄

---

## 6. 학습 설정

- Sim pre-training: **241,450 episodes** (GR1 simulated)
- Real fine-tuning: **200 demos/task** (Unitree G1)
- Text encoder + visual VAE: **frozen**
- 15% of GR00T-N1.5의 official training data 규모

---

## 7. 한계 및 미해결 문제

1. **6Hz inference**: GR00T-N1.5 (13Hz), Qwen3DiT (9Hz) 대비 가장 느림. Real-time reactive control에 부적합
2. **LIBERO saturation**: 98.6%에서 추가 차별화 어려움
3. **Cosmos-2B 의존**: Video DiT 초기화에 대규모 pre-trained model 필요
4. **Layer 18의 최적성**: 다른 아키텍처/데이터에서도 동일 layer가 optimal인지 불명확

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Single-pass feature extraction의 발견이 매우 가치있음 |
| **Technical depth** | ★★★★★ — Tri-timestep, layer ablation, joint vs decoupled 분석 |
| **Experimental rigor** | ★★★★★ — LIBERO + RoboCasa + Real G1 + ablation 포괄적 |
| **Practical impact** | ★★★★☆ — 6Hz가 deployment 제약 |
| **Writing quality** | ★★★★☆ |

**강점**: "Video reconstruction을 bypass하고 intermediate feature만 활용"이라는 핵심 통찰. 7x 수렴 + 10x data efficiency. **약점**: 6Hz latency, Cosmos-2B 의존.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Fast-WAM과 어떻게 다른가? | Fast-WAM은 train-time only video, DiT4DiT는 single-pass feature extraction. 둘 다 test-time generation을 skip하나 메커니즘이 다름 |
| 2 | 6Hz를 높이려면? | Video DiT를 distill하거나 feature extraction을 가속. 하지만 dual-DiT의 구조적 overhead |
| 3 | Layer 18이 다른 모델에서도 최적인가? | Cosmos-2B specific일 수 있음. 다른 video backbone에서 재검증 필요 |
| 4 | Joint training의 video loss가 실질적으로 필요한가? | λ=0이면 (action only) 어떻게 되는지 ablation이 더 명확했을 것 |

<!-- VERIFIED: pdf -->
