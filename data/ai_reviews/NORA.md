# NORA: A Small Open-Sourced Generalist VLA Model for Embodied Tasks

> **한 줄 요약**: 3B 파라미터의 compact VLA로 Qwen-2.5-VL backbone과 FAST+ tokenizer를 결합하여, 대형 VLA를 능가하는 성능을 훨씬 적은 연산 비용으로 달성.

---

## 1. 배경 및 동기

- 7B+ VLA의 compute 비용이 실시간 제어와 연구 접근성을 제약
- **더 작은 모델로 더 나은 성능**이 가능함을 보이는 것이 목표
- Qwen-2.5-VL의 강력한 vision-language 능력을 활용

---

## 2. 방법론

### Architecture

- **VLM**: Qwen-2.5-VL (3B variant)
- **Action tokenizer**: FAST+ (frequency-space tokenization)
- **Training**: 970K real-world demonstrations (Open X-Embodiment)

### FAST+ Tokenizer

Discrete cosine transform 기반 action compression:
$$\mathbf{c} = \text{DCT}(\mathbf{a}_{1:H}), \quad \text{tokens} = \text{Quantize}(\mathbf{c})$$

> ❓ **예상 질문**: FAST+ tokenizer가 standard binning 대비 왜 나은가?
> **답변**: Frequency domain에서 compression하므로 smooth trajectory의 정보를 효율적으로 보존. High-frequency noise는 자연스럽게 제거. 특히 dexterous task에서 큰 이점.

---

## 3. 실험 결과

| 모델 | Params | LIBERO Avg (%) | Real Robot SR (%) |
|------|--------|---------------|------------------|
| OpenVLA | 7B | 76.5 | 52.3 |
| Octo | 93M | 58.7 | 35.2 |
| **NORA** | **3B** | **84.2** | **62.5** |

- **3B로 7B OpenVLA를 큰 폭으로 능가** → backbone 선택과 tokenizer의 중요성
- FAST+ tokenizer의 기여가 핵심

---

## 4. 한계 및 미해결 문제

1. **FAST+ tokenizer의 한계**: DCT는 periodic pattern에 유리하지만, abrupt action change에는 부적합
2. **3B의 reasoning 한계**: 복잡한 multi-step planning에서 7B+ 대비 약할 수 있음
3. **Cross-embodiment**: 특정 로봇에서의 fine-tuning 결과 위주

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Qwen-2.5-VL + FAST+ 결합 |
| **Technical depth** | ★★★★☆ |
| **Experimental rigor** | ★★★★☆ |
| **Practical impact** | ★★★★★ — Small + open + performant |
| **Writing quality** | ★★★★☆ |

**강점**: 작고 빠르면서 성능도 좋은 practical VLA. Open-source. **약점**: Reasoning 능력의 upper bound, FAST+ tokenizer의 edge cases.

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | FLOWER (950M)와 비교하면? | FLOWER가 더 작고 CALVIN에서 SOTA. NORA는 FAST+ 기반 real-robot 결과에 강점 |
| 2 | Qwen-2.5-VL 대신 다른 VLM이면? | Backbone 선택이 성능에 큰 영향. Qwen의 vision encoder 품질이 핵심 |
| 3 | Long-horizon task에서 3B의 한계는? | 4+ step task에서 7B 대비 planning 능력 저하 예상. 미검증 |
