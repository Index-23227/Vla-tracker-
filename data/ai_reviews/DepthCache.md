# DepthCache: Depth-Guided Training-Free Visual Token Merging for VLA Model Inference

> **한 줄 요약**: Depth 기반 영역 분할 + progressive cross-frame token merging + dual protection mechanism으로, **3개 VLA (π0.5, OpenVLA, GR00T)에서 training-free로 1.07-1.33x speedup과 0.2-1.0%p 이내 성능 하락** 달성. Real-world에서 191ms→143ms (1.33x).

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- VLA 모델의 visual token 수가 추론 latency의 주요 bottleneck
- 기존 token pruning/merging (FastV, ToMe)은 **2D 공간에서 균일하게 적용** → 로봇 조작에 중요한 근거리 물체와 배경을 동등하게 취급
- 기존 방법들이 VLA에 적용 시 **심각한 성능 하락** (FastV: π0.5에서 -20.3%p)

### 핵심 질문
- **Depth 기반 구조적 사전지식으로 token merging을 가이드하면, 성능 보존과 속도 향상을 동시에 달성할 수 있는가?**

---

## 2. 방법론 심층 분석

### 2.1 Core Components

1. **Depth-guided Region Partitioning**: K=3 depth clusters로 영역 분할, 근거리=낮은 merge, 원거리=높은 merge
2. **Progressive Cross-frame Merging**: W=5 frame window에서 점진적으로 merge ratio 증가 (step ratio η=0.2 → ~5 frame에서 수렴)
3. **Dual Protection**: Attention-based semantic protection (τ_att = μ+σ) + depth gradient 기반 geometric edge protection
4. **Change-triggered Re-initialization**: 장면 변화 감지 시 cache 초기화

### 2.2 Auxiliary View Pipeline

Wrist camera 전용 2-state machine:
- End-effector 정지 시: aggressive compression
- 움직임 시: 덜 aggressive compression

### 2.3 Key Hyperparameters

| Parameter | Default | 역할 |
|-----------|---------|------|
| Warmup frames (N) | 5 | 초기 cache 구축 |
| Depth clusters (K) | 3 | 영역 분할 수 |
| Progressive window (W) | 5 | Merge 수렴 window |
| Max merge ratio (r_max) | 0.7 | 최대 token 제거 비율 |
| Step ratio (η) | 0.2 | Progressive step size |

**Training-free**: 사전학습된 체크포인트에 **수정 없이** 적용. NVIDIA RTX 4090 (24GB)에서 실행.

---

## 3. 실험 결과 심층 분석

### LIBERO Simulation — π0.5 (3.3B)

| Method | Spatial | Object | Goal | Long | **Avg SR** | **Speedup** | Token Ratio |
|--------|---------|--------|------|------|-----------|-----------|-------------|
| Baseline | 99.0 | 97.0 | 99.0 | 96.5 | **97.9** | 1.00x | 100% |
| + FastV | 73.4 | 89.4 | 79.1 | 68.5 | 77.6 | 1.30x | 68.0% |
| + ToSA | 79.3 | 82.0 | 71.6 | 62.4 | 73.8 | 0.94x | 57.8% |
| **+ DepthCache** | **98.9** | **97.1** | **98.3** | **96.1** | **97.6 (−0.3)** | **1.28x** | 68.2% |

- FastV: **-20.3%p** 하락 vs DepthCache: **-0.3%p** → 핵심 차별화

### LIBERO Simulation — OpenVLA (7B)

| Method | Avg SR | Speedup |
|--------|--------|---------|
| Baseline | 76.7 | 1.00x |
| + FastV | 64.0 (-12.7) | 1.39x |
| + SP-VLA | 71.9 (-4.8) | 1.50x |
| **+ DepthCache** | **75.7 (−1.0)** | **1.21x** |

### LIBERO Simulation — GR00T (2.2B)

| Method | Avg SR | Speedup |
|--------|--------|---------|
| Baseline | 93.1 | 1.00x |
| **+ DepthCache** | **92.9 (−0.2)** | **1.07x** |

### ⭐ Real-World (π0.5, 20 trials/task)

| Task | Baseline | DepthCache |
|------|---------|-----------|
| Pick & Place | 20/20 | 20/20 |
| Stack Blocks | 18/20 | 17/20 |
| Drawer & Place | 17/20 | 15/20 |
| **Aggregate** | **55/60 (91.7%)** | **52/60 (86.7%)** |

| Metric | Baseline | DepthCache |
|--------|---------|-----------|
| Inference latency | **191ms** | **143ms (1.33x)** |

### Extended Real-World

| Scenario | Baseline | DepthCache | Time Reduction |
|---------|---------|-----------|---------------|
| Multi-Object Sorting | 15/15 | 13/15 | **28.6s→22.1s (-22.7%)** |
| Perturbation Recovery | 11/15 | 12/15 | **17.4s→13.7s (-21.3%)** |

---

## 4. Ablation 분석 (π0.5, LIBERO)

| Configuration | Avg SR | Δ | Speedup |
|--------------|--------|------|---------|
| **Full DepthCache** | **97.6** | — | 1.28x |
| w/o Depth Partitioning | 79.4 | **−18.2** | 1.34x |
| w/o Progressive Merge | 81.0 | **−16.6** | 1.32x |
| w/o Dual Protection | 90.3 | **−7.3** | 1.48x |
| w/o Re-initialization | 92.8 | −4.8 | 1.25x |
| w/o Auxiliary View | 97.8 | +0.2 | 1.06x |

**Component impact ranking**: Depth Partitioning (−18.2) > Progressive Merge (−16.6) > Dual Protection (−7.3)

### Parameter Sensitivity

- **r_max ∈ [0.3, 0.7]**: Broad safe plateau with near-invariant SR
- **r_max > 0.7**: 성능 하락 시작
- **η = 1.0 (one-shot merge)**: 81.0% → progressive가 필수

---

## 5. 한계 및 미해결 문제

### 방법론적 미비점
1. **Modest speedup**: 1.07-1.33x는 DeeR-VLA (3.1-5.2x)나 quantization (2-4x)에 비해 modest. Token merging만으로는 dramatic 가속 어려움
2. **Real-world 성능 하락**: 55/60 → 52/60 (−5%) — simulation보다 real에서 하락이 더 큼
3. **Depth estimation overhead 미보고**: DepthCache는 depth map이 필요하나, depth 계산 비용이 speedup에 포함되는지 불명확 (GT depth vs estimated)
4. **GR00T에서 1.07x**: 거의 무의미한 speedup. 모델 아키텍처에 따라 효과 편차가 큼

### Attribution 문제
- Speedup은 동일(1.28-1.34x)한데 depth partitioning 없으면 성능이 18.2%p 하락 → depth의 기여는 speedup이 아닌 **성능 보존**에 있음

---

## 6. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Depth-guided token merging의 체계적 설계 |
| **Technical depth** | ★★★★★ — 5개 구성요소의 포괄적 ablation, 3개 VLA 검증 |
| **Experimental rigor** | ★★★★★ — 3 VLA × LIBERO + real-world + ablation + sensitivity |
| **Practical impact** | ★★★★☆ — Training-free plug-in. 다만 speedup modest |
| **Writing quality** | ★★★★☆ |

**강점**: 3개 VLA에서 일관적으로 성능 보존 (−0.2~−1.0%p). Training-free. 기존 방법(FastV −20.3%p) 대비 압도적. **약점**: Speedup이 modest (1.07-1.33x), real-world에서 sim보다 큰 하락.

---

## 7. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | DeeR-VLA (early exit)와 결합하면? | Token merging + early exit = multiplicative 가속 가능. 미검증 |
| 2 | Depth를 어디서 얻는가? Real-world에서 GT depth인가? | π0.5 등은 depth camera 사용. Monocular estimation 시 추가 latency 발생하여 순수 speedup 감소 가능 |
| 3 | GR00T에서 1.07x면 의미 없지 않은가? | GR00T의 vision token이 이미 적어서(compact encoder) merging 여지가 적음. 큰 vision encoder를 가진 VLA에서 효과 극대화 |
| 4 | r_max=0.7이면 70% 토큰 제거인데, 원거리에서 중요 정보 손실은? | Dual protection이 semantic/geometric edge를 보호. Ablation에서 protection 제거 시 -7.3%p |

<!-- VERIFIED: pdf -->
