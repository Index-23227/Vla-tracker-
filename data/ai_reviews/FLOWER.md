# FLOWER: Democratizing Generalist Robot Policies with Efficient VLA Flow Policies

> **한 줄 요약**: VLM의 30-50% layer를 pruning하여 intermediate fusion하고, 18-layer flow transformer(339M) + Global-AdaLN으로 action 생성하는 **947M VLA**로, CALVIN ABC **4.53 SOTA**, 10개 벤치마크에서 평균 25.1% 향상, **200 H100 GPU-hours** (48h on 4 H100s)로 학습.

---

## 1. 배경 및 동기

- 대형 VLA (7B+)의 학습 비용이 연구 접근성 제약
- VLM의 모든 layer가 action에 필요한 것은 아님 → **compute 낭비**
- Diffusion/flow action head에 capacity가 충분하지 않음

---

## 2. 방법론 심층 분석

### 2.1 VLM Pruning

| VLM Type | 전략 |
|---------|------|
| Encoder-Decoder (Florence-2-L) | **Decoder 전체 제거** (~50%) |
| Decoder-Only (SmolFlow2-Video) | **후반 30% layer 제거** |

"Intermediate fusion": 중간 layer의 hidden state를 flow transformer에 cross-attention으로 주입.

### 2.2 Architecture 세부

| Component | Params | 역할 |
|-----------|--------|------|
| ViT | 360M | Visual encoding |
| VLM (pruned) | 205M | Language + vision reasoning |
| Flow Transformer | 339M | Action generation |
| Others | 43M | Projections, etc. |
| **Total** | **947M** | |

**Flow Transformer**: 18 layers, 1024 latent dim, 16 heads, SwiGLU MLP, RMSNorm

### 2.3 Global-AdaLN-Zero

모든 layer에서 **단일 공유 modulation weight** + action-type별 unique signal:
- Standard per-layer AdaLN 대비 **20% parameter 절감**
- 성능: 4.43 (standard) vs **4.53** (global) — 오히려 약간 향상

### 2.4 Inference

| Metric | Value |
|--------|-------|
| Throughput | **311 Hz** |
| Latency | **0.052s (52ms)** |
| VRAM | **1.85 GB** |

---

## 3. 학습 설정

| 항목 | 값 |
|------|-----|
| Data | 250K trajectories (8 open datasets) |
| Mix | DROID 23.5%, BridgeV2 28.6%, Fractal 24.7% |
| Batch | 256 × 4 grad accum |
| Steps | 360K (saturates ~350K) |
| **Total compute** | **200 H100 GPU-hours** (48h on 4 H100s) |

---

## 4. 실험 결과 심층 분석

### CALVIN ABC→D (Table 10)

| Setting | Task 1 | Task 5 | **Avg Len** |
|---------|--------|--------|-------------|
| D→D | 97.4% | - | 4.35 |
| **ABC→D** | **99.4%** | **84.9%** | **4.53 ± 0.04 (SOTA)** |
| ABCD→D | 99.2% | 92.3% | 4.67 |

Baselines: VPP 4.29, Seer 4.28, MoDE 4.01, RoboDual 3.66

### 10 Benchmarks 종합

| Benchmark | FLOWER | Best Baseline | Δ |
|-----------|--------|--------------|---|
| CALVIN ABC | **90.6%** | 85.8% (OpenVLA) | +4.8 |
| LIBERO Long | **94.9%** | 85.8% (π₀) | **+9.1** |
| LIBERO 90 | 94.7% | 96.0% (MoDE) | -1.3 |
| SIMPLER Bridge | **40.0%** | 30.0% (Octo) | +10.0 |
| SIMPLER Google | 32.2% | 42.4% (RT-1X) | **-10.2** |
| Real Kitchen | **61.0%** | 30.0% (OpenVLA) | **+31.0** |
| Real Generalization | **51.0%** | 23.4% (OpenVLA) | **+27.6** |
| Aloha Insertion | 54.0% | 54.0% (ACT) | 0.0 |
| LIBERO Spatial | 97.5% | 98.0% (π₀.₅) | -0.5 |
| LIBERO Object | 99.1% | 99.1% | 0.0 |

- **Real Kitchen +31%p, Real Generalization +27.6%p** → real-world에서 강력
- **SIMPLER Google -10.2%p** → 이 벤치마크에서 약점

### OpenVLA 대비

| Metric | OpenVLA | FLOWER | 비율 |
|--------|---------|--------|------|
| Params | 7.7B | **947M** | **8.1x 작음** |
| Training | ~10K GPU-hrs | **200 GPU-hrs** | **50x 적음** |
| Real Kitchen | 31% | **61%** | **+97%** |
| Throughput | ~5 Hz | **311 Hz** | **62x 빠름** |

---

## 5. Ablation 분석

### Fusion Strategy (Table 1)

| 방법 | LIBERO-Long | CALVIN ABC |
|------|-----------|-----------|
| **Intermediate** | **93.4%** | **89.5%** |
| Late | 61.8% | 71.2% |
| Early | 33.4% | 57.1% |

### Pruning Ratio (Table 2)

| Retained Layers | CALVIN ABC | LIBERO-Long |
|----------------|-----------|-----------|
| **0.3 (70% pruned)** | **72.1%** | **70.7%** |
| 0.5 (50% pruned) | 66.4% | 62.5% |
| Full (no pruning) | 66.3% | 69.2% |

> ⚠️ **핵심 발견**: **Full VLM보다 70% pruned가 더 좋음!** (CALVIN: 72.1% vs 66.3%)

### Critical Components (Table 3, CALVIN avg len)

| Removed Component | Avg Len | Δ |
|------------------|---------|---|
| Full FLOWER | **4.53** | — |
| Frozen VLM | 2.65 | **−40%** |
| No Flow head | 3.33 | −25% |
| Small backbone | 4.26 | −4% |
| Discrete tokens | 1.12 | **−75%** |
| No Global-AdaLN | 3.33 | −25% |

- **Discrete tokenization이 가장 치명적** (4.53→1.12)

---

## 6. 한계 및 미해결 문제

1. **SIMPLER Google -10.2%p**: 이 벤치마크에서 약점. RT-1X는 Google Robot에 특화되어 있어 불공정할 수 있으나 결과는 결과
2. **Manipulation only**: Navigation, locomotion 미지원
3. **8/10 벤치마크가 simulation**: Real-world 결과(2개)는 인상적이나 규모 제한
4. **Iterative sampling**: Rectified flow 4-8 steps이 deterministic policy보다 느림 (그래도 311Hz)

---

## 7. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — "Less VLM, more action head"의 역발상, pruning이 오히려 성능 향상 |
| **Technical depth** | ★★★★★ — 10 benchmarks, 풍부한 ablation |
| **Experimental rigor** | ★★★★★ — 가장 많은 벤치마크에서 평가 |
| **Practical impact** | ★★★★★ — 200 GPU-hrs, 947M, 311Hz, open-source |
| **Writing quality** | ★★★★★ |

**강점**: 947M/200 GPU-hrs로 7.7B/10K GPU-hrs를 능가하는 효율성 혁명. Pruning이 오히려 성능 향상이라는 반직관적 결과. **약점**: SIMPLER Google 약점, sim 위주.

---

## 8. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 왜 pruning이 성능을 올리는가? | 후반 layer가 language generation에 특화 → action에 불필요한 representation 생성. 제거하면 action-relevant intermediate features가 직접 활용됨 |
| 2 | SIMPLER Google에서 왜 약한가? | RT-1X가 Google Robot에 대규모 데이터로 특화. FLOWER는 cross-embodiment이라 특정 로봇에서 불리 |
| 3 | 200 GPU-hrs를 더 늘리면? | 350K step에서 saturation. 더 긴 학습보다 **더 많은/다양한 데이터**가 효과적일 것 |
| 4 | Discrete tokens이 1.12면 치명적인데, 왜 다른 VLA는 discrete로 잘 하는가? | FLOWER의 flow transformer가 continuous action에 최적화. Discrete tokenization은 flow의 이점을 파괴 |

<!-- VERIFIED: pdf -->
