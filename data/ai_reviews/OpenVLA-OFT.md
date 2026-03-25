# OpenVLA-OFT: Optimizing Speed and Success in VLA Fine-Tuning

> **한 줄 요약**: VLA fine-tuning의 3대 설계 선택 (parallel decoding + L1 regression + action chunking)을 체계적으로 분석하여, OpenVLA에서 LIBERO **76.5%→97.1%** (+20.6%p), **26x throughput** (LIBERO 8-step chunk), **43x** (ALOHA 25-step), real bimanual에서 기존 대비 **+15%p** 달성.

---

## 1. 배경 및 동기

- OpenVLA의 성능이 fine-tuning 전략에 극도로 의존하나, **최적 recipe가 체계적으로 연구되지 않음**
- Autoregressive → parallel, discrete → continuous, token prediction → regression 등 다양한 선택지의 trade-off 미탐구

---

## 2. 3대 Fine-tuning Recipe

| 요소 | 기존 OpenVLA | **OFT** |
|------|------------|---------|
| Decoding | Autoregressive (K×D passes) | **Parallel (single pass)** |
| Representation | 256-bin discrete | **Continuous** |
| Loss | Cross-entropy | **L1 regression** |
| Chunking | Single step | **H=8 (LIBERO), H=25 (ALOHA)** |

### Parallel Decoding

Autoregressive: K×D forward passes → **single forward pass** with bidirectional attention on actions.

### FiLM Language Grounding (OFT+)

Feature-wise Linear Modulation으로 language conditioning 강화:
- FiLM 없으면 language following이 **33% (random chance)** 로 하락
- "Overfitting to spurious visual correlations" 방지

---

## 3. 실험 결과

### LIBERO

| Model | SR (%) | Throughput |
|-------|--------|-----------|
| OpenVLA (original) | 76.5 | 1x |
| OpenVLA (LoRA FT) | ~82 | 1x |
| **OpenVLA-OFT** | **97.1** | **26x** |

### Real-World ALOHA (Bimanual)

- OFT+: π₀, RDT-1B, Diffusion Policy, ACT 대비 **+15%p absolute**
- 43x throughput (25-step chunks)
- Cloth folding, ingredient scooping 등 dexterous task

### FiLM Ablation

| Setting | Language Following (%) |
|---------|----------------------|
| OFT without FiLM | **33%** (random) |
| **OFT+ with FiLM** | **>80%** |

---

## 4. 한계

1. **L1 regression의 multimodality 한계**: Unimodal → bimodal action에서 averaging
2. **Fine-tuning specific**: Pre-training recipe에는 미적용
3. **Architecture 혁신 아닌 training recipe 혁신**

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Systematic fine-tuning analysis |
| **Technical depth** | ★★★★★ — 포괄적 실험 설계 |
| **Experimental rigor** | ★★★★★ — LIBERO + real bimanual |
| **Practical impact** | ★★★★★ — 즉시 적용 가능 |
| **Writing quality** | ★★★★★ |

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | L1 대신 diffusion이면? | CogACT가 이를 수행. Diffusion이 multimodal에 강하나 latency trade-off |
| 2 | FiLM이 없으면 33%인 게 놀라운데? | Language에 condition되지 않으면 모델이 visual shortcut만 학습. FiLM이 language를 explicit하게 주입 |
| 3 | 97.1%는 saturation 아닌가? | 맞음. LIBERO-Long 등 더 challenging한 metric이 필요 |

<!-- VERIFIED: pdf -->
