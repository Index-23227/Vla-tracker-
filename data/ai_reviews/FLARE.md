# FLARE: Robot Learning with Implicit World Modeling

> **한 줄 요약**: DiT에 M=32 learnable future token을 추가하고, SigLIP-2 기반 action-aware embedding의 cosine alignment loss로 latent world modeling을 수행하여, RoboCasa 24-task에서 **70.1%** (baseline 61.9%, +8.2%p), GR1 24-task에서 **55.0%** (+11%p), 추론 overhead **zero**.

---

## 1. 배경 및 동기

- **Explicit world model** (GR-1, UniPi): 미래 프레임을 pixel space에서 생성 → latency
- **No world model** (Diffusion Policy): Temporal dynamics 미활용
- **FLARE의 핵심**: Latent space에서 implicit하게 world model → **추론 시 overhead 없음**

---

## 2. 방법론 심층 분석

### 2.1 Architecture

DiT input sequence에 **M=32 learnable future token embeddings** 추가:

```
[Proprioceptive state] + [Noised action chunk] + [M future tokens]
                            ↓
                     DiT Transformer (8 layers)
                            ↓
        Action prediction + Future token activations (at layer L=6)
                            ↓
        MLP projection → Cosine alignment with target embedding
```

### 2.2 Target Embedding: Action-Aware Q-Former

- **SigLIP-2 encoder**: Images (256 tokens) + text (32 tokens)
- **4 transformer layers** for fusion
- **Q-Former cross-attention**: 32 learnable query tokens으로 compression
- 이 target embedding이 **frozen** (EMA, ρ=0.995)

$$\mathcal{L}_{\text{FLARE}} = 1 - \cos(\text{MLP}(h_L^{\text{future}}), f_{\bar{\theta}}(o_{t+1}))$$

### 2.3 Total Loss

$$\mathcal{L} = \mathcal{L}_{\text{action}} + \lambda \mathcal{L}_{\text{FLARE}}, \quad \lambda = 0.2$$

- **Inference 시 FLARE loss 미사용** → future tokens 제거 가능 → **overhead 없음**

### 2.4 학습 설정

| 단계 | GPU | Batch | Steps |
|------|-----|-------|-------|
| Embedding model pre-train | **256× H100** | 8192 | 150K |
| Policy training | **32× H100** | 1024 | 80K |

- Optimizer: AdamW (β₁=0.95, β₂=0.999)
- Data: **~2,000 hours** robotic data (GR00T N1 + 7 OXE datasets)

---

## 3. 실험 결과 심층 분석

### RoboCasa (24 tasks)

| Model | Pick/Place | Doors/Drawers | Others | **Avg** |
|-------|-----------|--------------|--------|---------|
| Diffusion Policy | 29.2 | 78.7 | 61.3 | 51.7 |
| UWM | 35.6 | 82.0 | 74.2 | 60.8 |
| GR00T N1 (scratch) | 44.1 | 80.0 | 69.6 | 60.6 |
| Policy Only (no FLARE) | 43.8 | 78.7 | 75.2 | 61.9 |
| **FLARE** | **53.2** | **88.8** | **80.0** | **70.1 (+8.2%p)** |

- **Pick/Place에서 +9.4%p** (43.8→53.2) — 가장 큰 향상. 정밀 manipulation에서 이점

### GR1 (24 tasks)

| Model | Pick/Place | Articulated | **Avg** |
|-------|-----------|-------------|---------|
| UWM | 30.1 | 38.4 | 29.5 |
| GR00T N1 | 51.8 | 42.8 | 45.1 |
| Policy Only | 46.6 | 47.4 | 44.0 |
| **FLARE** | **58.2** | **51.3** | **55.0 (+11%p)** |

### Real Robot (GR1 humanoid)

- 100 trajectories: **95.1%** (baseline 81.1%, **+14%p**)
- "Avoids knocking over fragile objects" → implicit dynamics understanding

### Human Video Learning

- 150 human egocentric demos + 10 robot demos: **80%** on novel objects
- 1 robot demo alone: 60%
- → FLARE alignment loss가 human video에서도 유용

---

## 4. Ablation 분석

### Target Embedding Model (Table 2)

| Method | SR (%) |
|--------|--------|
| No FLARE (baseline) | 43.9 |
| SigLIP-2 (256 raw tokens) | 49.6 |
| SigLIP-2 (64 pooled) | 50.9 |
| **Action-aware Q-Former** | **55.0** |

### DiT Layer Selection (Figure 8)
- **Layer 6 (of 8): Optimal**
- Layer 4: "Notable drop" — 초기 layer에서 action과 future alignment이 충돌
- Layer 8: 약간 하락 (pixel reconstruction에 특화)

### λ Coefficient (Figure 8)
- **λ=0.2: Optimal**
- Robust across range ("strong performance across setups")

### EMA Update Rate (Figure 9)
- **ρ=0.995: Best** (72.7% RoboCasa)
- ρ=0.999: 71.6%
- ρ=0.99: 70.8% (instability)
- ρ=1.0 (no EMA): 71.5% (still outperforms baseline)

---

## 5. 한계 및 미해결 문제

1. **Compute cost**: Embedding model pre-training에 **256× H100** 필요. 대규모 연구 인프라
2. **Latency 미보고**: "Minimal modifications"이라 주장하나 **구체적 inference ms 수치 없음**
3. **"26% 향상"의 맥락**: RoboCasa pick-and-place에서의 최대값. 전체 평균은 +8.2%p
4. **Single-step prediction**: 1-step latent prediction만. Multi-step prediction의 효과 미검증
5. **GR00T N1에 의존**: GR00T 아키텍처 위에서만 검증. 다른 VLA backbone에서의 일반성 미확인

### Attribution 문제
- FLARE의 효과가 **dynamics understanding**인지 **representation regularization**인지 불완전 분리. Contrastive learning (augmented current view predict)으로도 비슷할 수 있으나 이 비교 부재

---

## 6. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Latent implicit WM, Q-Former alignment |
| **Technical depth** | ★★★★★ — Layer selection, EMA, λ 포괄적 ablation |
| **Experimental rigor** | ★★★★★ — 48 tasks (RoboCasa+GR1) + Real + Human video |
| **Practical impact** | ★★★★★ — Zero inference overhead, +8-14%p consistent |
| **Writing quality** | ★★★★★ |

**강점**: Inference overhead가 zero이면서 +8-14%p 일관적 향상. Human video에서도 학습 가능. Layer/λ/EMA ablation이 매우 체계적. **약점**: 256×H100 pre-training, 정확한 latency 미보고.

---

## 7. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Contrastive loss (SimCLR)로 대체하면? | 이 비교가 핵심이나 부재. Future prediction vs current augmentation의 차이를 분리해야 함 |
| 2 | Multi-step prediction (t+2, t+3)이 더 좋은가? | 미검증. Error accumulation vs longer-horizon dynamics의 trade-off |
| 3 | 256×H100 없이 embedding model을 학습할 수 있는가? | Pre-trained SigLIP-2를 frozen으로 사용하면 pre-training 불필요하나 성능 하락 (55.0→50.9) |
| 4 | Fast-WAM과의 관계는? | 원리 동일 (학습 시에만 WM 활용). Fast-WAM은 pixel-level video, FLARE는 latent-level. FLARE가 더 효율적 |

<!-- VERIFIED: pdf -->
