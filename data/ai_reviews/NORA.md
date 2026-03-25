# NORA: A Small Open-Sourced Generalist VLA Model for Embodied Tasks

> **한 줄 요약**: Qwen-2.5-VL-3B + FAST+ tokenizer(+2048 tokens)로, 970K OXE demonstration에서 **4000 H100 GPU-hours** 학습하여, WidowX real-world **56.7%** (OpenVLA 40%), LIBERO-Long **74.6%** (NORA-Long), inference **8.3GB** GPU memory.

---

## 1. 배경 및 동기

- 7B+ VLA의 compute 비용이 연구 접근성과 실시간 제어를 제약
- **더 작은 모델(3B)로 7B 성능에 match하면서 효율적인 배포** 목표

---

## 2. 방법론

### Architecture

- **VLM**: Qwen-2.5-VL-3B
- **Action tokenizer**: FAST+ (DCT + BPE, +2048 tokens to vocabulary)
- **Action chunk**: 1 step (NORA), **5 steps (NORA-Long)**
- **Single visual frame** per timestep

### 학습

| 항목 | 값 |
|------|-----|
| Data | OXE (970K real-world demos) |
| GPU | **8× H100, ~3 weeks** |
| Total compute | **~4000 H100 GPU-hours** |
| Batch size | 256 |
| Gradient updates | 1.1M |
| Peak LR | 5e-5 (warmup + cosine decay) |
| Precision | bf16 + FlashAttention |
| Inference memory | **8.3 GB** |

---

## 3. 실험 결과 심층 분석

### WidowX Real Robot

| Task Category | RT-1 | OpenVLA | SpatialVLA | **NORA** |
|-------------|------|---------|-----------|---------|
| OOD Objects | 0.3% | 56.7% | 6.7% | **83.3%** |
| Spatial Tasks | 0% | 46.7% | 26.7% | **53.3%** |
| Multi-Object | 0% | 16.7% | 0% | **33.3%** |
| **Overall** | 4.4% | 40% | 11.1% | **56.7%** |

- **OOD objects에서 OpenVLA (56.7%) 대비 NORA (83.3%)** → +26.6%p
- 개별 task 최대 **90%** (carrot/banana in pot)

### LIBERO (Simulation)

| Model | Spatial | Object | Goal | Long | **Avg** |
|-------|---------|--------|------|------|---------|
| OpenVLA (FT) | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| SpatialVLA + AC | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| NORA + AC | 85.6 | 89.4 | 80.0 | 63.0 | **79.5** |
| **NORA-Long** | **92.2** | **95.4** | **89.4** | **74.6** | **87.9** |

- NORA-Long (action chunk 5): **87.9%** — OpenVLA 대비 +11.4%p
- LIBERO-Long에서 74.6% (OpenVLA 53.7% 대비 +20.9%p)

### Action Chunking Trade-off

- **Simulation**: NORA-Long (87.9%) >> NORA (73.9%) → chunking이 큰 도움
- **Real-world**: NORA-Long이 WidowX에서 worse — "accumulated actions resulted in excessively large movements"
- → **Chunking은 high control frequency (20Hz+)에서 효과적, low freq에서 해로울 수 있음**

---

## 4. 한계 및 미해결 문제

1. **Real-world chunking 실패**: NORA-Long이 실제 로봇에서 overshoot → control frequency 의존성
2. **4000 H100-hrs**: 3B 모델치고 상당한 학습 비용. FLOWER (200 GPU-hrs)보다 20x
3. **Multi-object 33.3%**: 여러 물체 처리에서 아직 부족
4. **LIBERO fine-tuned 기준**: Zero-shot LIBERO 결과가 낮음 (73.9% without chunking)

### Attribution 문제
- 성능이 **FAST+ tokenizer** 때문인지, **Qwen-2.5-VL backbone** 때문인지 분리 필요

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Qwen-2.5-VL + FAST+ 결합 |
| **Technical depth** | ★★★★☆ — Chunking trade-off 분석 |
| **Experimental rigor** | ★★★★☆ — Real + Sim, 다양한 OOD 평가 |
| **Practical impact** | ★★★★★ — 3B, 8.3GB, open-source |
| **Writing quality** | ★★★★☆ |

**강점**: OOD objects에서 83.3%, open-source 3B 모델로 7B OpenVLA 능가. **약점**: Real-world chunking 실패, 4000 GPU-hrs, multi-object 약점.

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | FLOWER (950M)와 비교하면? | FLOWER: CALVIN 4.53 SOTA, 200 GPU-hrs. NORA: real-world OOD 강점, 4000 GPU-hrs. 효율성은 FLOWER, 일반화는 NORA |
| 2 | Chunking이 real에서 실패하는 근본 이유는? | WidowX의 낮은 control freq (5Hz)에서 5-step chunk = 1초 open-loop. 환경 변화에 대응 불가 |
| 3 | FAST+ 없이 standard 256-bin이면? | FAST+가 DCT compression으로 token efficiency 향상. Standard binning은 더 많은 token 필요 → 더 느림 |

<!-- VERIFIED: pdf -->
