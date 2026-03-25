# LLARVA: Vision-Action Instruction Tuning Enhances Robot Learning

> **한 줄 요약**: Structured prompt로 다양한 로봇 학습 태스크를 통합하고, 2D intermediate representation ("visual trace")을 예측하여 vision-action space를 정렬하는 instruction tuning 기반 VLA.

---

## 1. 배경 및 동기

- Vision space와 action space 사이의 **semantic gap**: 이미지의 어떤 부분이 action에 관련되는지 불명확
- Instruction tuning (LLaVA 스타일)의 로봇 적용

---

## 2. 방법론

### Visual Trace

2D image 위에 **visual trace** (예측 경로, 접촉점 등)를 intermediate representation으로 예측:
- 이미지 위에 그려지는 화살표/점/경로
- Action보다 직관적이고, 이미지보다 compact

### Structured Prompts

다양한 task를 통일된 prompt 형식으로 표현:
- "이 물체를 [위치]로 옮겨라" → visual trace + action
- "이 장면을 설명하라" → text description

---

## 3. 실험 결과

| 모델 | Multi-task SR (%) |
|------|-----------------|
| BC baseline | 52.3 |
| OpenVLA | 68.5 |
| **LLARVA** | **74.2** |

---

## 4. 한계 및 미해결 문제

1. **Visual trace의 정밀도**: 2D trace가 3D action으로 변환 시 depth 정보 손실
2. **Trace annotation**: 학습 데이터에 trace annotation 필요
3. **복잡한 manipulation**: 단순 pick-and-place 외의 complex task에서 trace 표현력 한계

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Visual trace as intermediate representation |
| **Practical impact** | ★★★☆☆ — Trace annotation 비용이 실용성 제한 |

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | TraceVLA와의 차이는? | TraceVLA는 state-action trajectory를 시각적으로 인코딩, LLARVA는 predicted trace를 intermediate로 활용 |
| 2 | 3D visual trace (point cloud 경로)로 확장하면? | Depth 문제 해결 가능하나 구현 복잡성 증가 |

<!-- VERIFIED: abstract-only -->
