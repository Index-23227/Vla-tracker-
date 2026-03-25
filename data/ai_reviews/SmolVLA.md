# SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics

> **한 줄 요약**: 기존 VLA 대비 10배 작은 compact 모델로 single-GPU 학습과 consumer hardware 배포를 목표로 설계, asynchronous inference로 높은 제어 주파수 달성.

---

## 1. 배경 및 동기

- 대형 VLA의 compute 비용이 연구 접근성과 실제 배포를 제약
- **Single GPU로 학습 + 배포 가능한** 모델 필요

---

## 2. 방법론

### Compact Architecture (~700M-1B)

- 경량 VLM backbone
- Efficient action head
- **Asynchronous inference**: Action execution과 model inference를 pipeline하여 effective control rate 증가

> ❓ **예상 질문**: Async inference에서 stale observation 문제는?
> **답변**: Model inference 동안 이전 action chunk를 실행하므로, 1-2 step의 delay 발생. 빠른 환경 변화에서 문제될 수 있으나, action chunking이 이를 부분적으로 완화.

---

## 3. 실험 결과

| 모델 | Params | SR (%) | GPU |
|------|--------|--------|-----|
| OpenVLA (7B) | 7B | 76.5 | A100 |
| FLOWER (950M) | 950M | 68.5 | A100 |
| **SmolVLA** | **~700M** | **~65** | **RTX 3090** |

- 성능은 약간 낮으나 **consumer GPU에서 실행 가능**

---

## 4. 한계 및 미해결 문제

1. **성능 격차**: 대형 모델 대비 ~10%p 하락
2. **Async latency**: Pipeline delay가 reactive task에서 문제
3. **Complex task**: 경량 모델의 reasoning 한계

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★☆☆ — 경량화 자체는 incremental |
| **Practical impact** | ★★★★★ — Democratization |

**강점**: Consumer GPU에서의 VLA 실현. **약점**: 성능 trade-off.

---

## 6. 🔥 예상 날카로운 질문

| # | 질문 | 핵심 답변 |
|---|------|----------|
| 1 | FLOWER vs SmolVLA: 어느 것이 더 practical한가? | FLOWER가 성능 우수, SmolVLA가 배포 용이. Task 요구에 따라 선택 |
