# OpenVLA-OFT: Optimizing Speed and Success in VLA Fine-Tuning

> **한 줄 요약**: VLA fine-tuning 전략을 체계적으로 분석하여, parallel decoding + action chunking + L1 regression의 조합이 LIBERO 97.1% (76.5%→97.1%), 26x throughput 향상, real bimanual 태스크에서 SOTA를 달성함을 실증.

---

## 1. 배경 및 동기

- OpenVLA의 성능이 fine-tuning 전략에 극도로 의존하나, **최적 fine-tuning recipe가 체계적으로 연구되지 않음**
- Autoregressive → parallel decoding, token prediction → regression 등 다양한 선택지의 trade-off 미탐구

---

## 2. 방법론

### 체계적 Fine-tuning 분석

| 요소 | 선택지 | 최적 |
|------|-------|------|
| Decoding | Autoregressive vs **Parallel** | Parallel |
| Loss | Cross-entropy vs **L1 regression** | L1 |
| Chunking | Single vs **Chunk (H=4)** | Chunk |
| Representation | Discrete tokens vs **Continuous** | Continuous |

### OpenVLA-OFT Recipe

기존 OpenVLA의 discrete autoregressive를:
1. **Parallel decoding**: 모든 action dimension 동시 예측 (26x 속도)
2. **L1 regression**: Continuous output, discretization 제거
3. **Action chunking**: H=4 future actions 동시 예측

---

## 3. 실험 결과

| 모델 | LIBERO (%) | Throughput | Real Bimanual |
|------|-----------|-----------|---------------|
| OpenVLA (original) | 76.5 | 1x | 35% |
| OpenVLA (LoRA FT) | 82.3 | 1x | 42% |
| **OpenVLA-OFT** | **97.1** | **26x** | **78%** |

- **76.5% → 97.1%**: 동일 모델에서 fine-tuning만으로 20%p+ 향상
- **26x 추론 가속**: Parallel decoding 덕분

---

## 4. 한계 및 미해결 문제

1. **Continuous regression의 multimodality 한계**: L1 loss는 unimodal → bimodal action에서 averaging
2. **Fine-tuning specific**: 사전학습 시에도 이 recipe가 유효한지 미검증
3. **모델 자체는 변경 없음**: Architecture 혁신이 아닌 training recipe 혁신

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Systematic fine-tuning analysis |
| **Technical depth** | ★★★★★ — 포괄적 실험 설계 |
| **Experimental rigor** | ★★★★★ — LIBERO + real bimanual |
| **Practical impact** | ★★★★★ — 즉시 적용 가능한 recipe |
| **Writing quality** | ★★★★★ |

**강점**: "OpenVLA가 나쁜 게 아니라 fine-tuning이 나빴다"는 통찰. 즉시 적용 가능한 실용적 기여. **약점**: L1 regression의 multimodality 한계.

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | L1 대신 diffusion loss를 쓰면? (→ CogACT) | CogACT가 이를 수행. Diffusion이 multimodal에 더 강하나 latency trade-off |
| 2 | 이 recipe를 다른 VLA (GR00T, pi0)에 적용하면? | 일반적으로 유효할 가능성 높으나 미검증 |
| 3 | 97.1%는 LIBERO saturation이 아닌가? | 맞음. 더 challenging한 벤치마크에서의 차별화 필요 |
