# LLaVA-VLA: Rethinking the Practicality of VLA — A Comprehensive Benchmark and Improved Baseline

> **한 줄 요약**: VLA의 실용성(practicality)을 체계적으로 재평가하기 위한 CEBench 벤치마크를 제안하고, consumer-grade GPU에서 구동 가능한 경량 LLaVA-VLA 모델을 설계.

---

## 1. 배경 및 동기

- 기존 VLA 평가가 **실용성을 간과**: 추론 속도, GPU 메모리, fine-tuning 비용 등
- 대부분의 VLA가 A100/H100 필수 → consumer GPU (RTX 3090/4090)에서 실행 불가
- Navigation과 manipulation을 동시에 다루는 VLA 부재

---

## 2. 방법론

### CEBench (Comprehensive Evaluation Benchmark)

| 평가 축 | 메트릭 |
|--------|--------|
| Task performance | Success rate |
| Efficiency | Inference FPS, GPU memory |
| Practicality | Fine-tuning cost, deployment ease |

### LLaVA-VLA

- LLaVA backbone (경량)
- Navigation + manipulation 동시 지원
- **Consumer GPU (RTX 4090)에서 실행 가능**

---

## 3. 실험 결과

| 모델 | SR (%) | FPS | GPU Memory |
|------|--------|-----|-----------|
| OpenVLA (7B) | 76.5 | 5.2 | 28GB |
| **LLaVA-VLA (3B)** | **72.8** | **12.5** | **12GB** |

- 3.7%p 성능 하락이지만 **2.4x 빠르고 절반의 메모리** 사용

---

## 4. 한계 및 미해결 문제

1. **성능 격차**: 대형 모델 대비 여전히 성능 하락
2. **CEBench의 포괄성**: 포함된 task diversity가 실제 deployment 시나리오를 충분히 반영하는지 불명확
3. **Navigation 성능**: Manipulation 대비 navigation 평가가 불충분

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Practicality-focused 평가 프레임워크 |
| **Practical impact** | ★★★★★ — Consumer GPU deployment |

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Quantization (INT4)을 적용하면 7B도 RTX 4090에서 되지 않는가? | 가능하나 quantization 성능 하락과 LLaVA-VLA의 native 경량이 더 나을 수 있음 |
| 2 | FLOWER (950M)와 비교하면? | FLOWER가 더 작으면서 성능 우수. LLaVA-VLA는 navigation까지 포함하는 점이 차별화 |
