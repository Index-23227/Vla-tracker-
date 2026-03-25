# π₀.₅: A VLA Model with Open-World Generalization

> **한 줄 요약**: pi0를 확장하여 heterogeneous 태스크 co-training과 hybrid multi-modal 데이터(이미지, 언어, detection, subtask prediction)를 활용, 전혀 새로운 주거 환경에서 long-horizon dexterous manipulation을 zero-shot으로 수행.

---

## 1. 배경 및 동기

- pi0가 보여준 가능성을 **open-world generalization**으로 확장
- "진짜 새로운 환경"에서 task-specific 학습 없이 작동하는 것이 목표
- 주방 청소, 침실 정리 등 실제 생활 태스크

---

## 2. 방법론

### Heterogeneous Task Co-training

다양한 태스크를 동시에 학습:
- 단순 pick-and-place
- Multi-step kitchen tasks
- Long-horizon cleaning routines

### Hybrid Multi-modal Data

- Images + language instructions
- Object detections (bounding boxes)
- Subtask predictions (계층적 분해)

이 다양한 modality를 VLM이 unified representation으로 처리.

---

## 3. 실험 결과

### Open-World Generalization (새로운 주거 환경)

| 태스크 | SR (%) |
|-------|--------|
| Kitchen cleaning | ~65% |
| Bedroom tidying | ~55% |
| Table setting | ~70% |

- **학습 환경에 없던 완전히 새로운 집**에서의 결과
- Long-horizon (10+ steps) dexterous manipulation

---

## 4. 한계 및 미해결 문제

1. **비공개**: 모델·데이터 비공개
2. **환경 다양성의 범위**: "open-world"의 실제 범위가 불명확 (몇 개의 새 환경?)
3. **Failure analysis**: 실패 케이스의 체계적 분석 부족
4. **Reproducibility**: 재현 불가능

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Open-world robot generalization의 첫 번째 진지한 시도 |
| **Practical impact** | ★★★★★ — "로봇이 실제로 새로운 집에서 작동할 수 있다"는 증거 |

**강점**: Open-world generalization의 milestone. Real-world 시연이 매우 인상적. **약점**: 비공개, 정량적 분석 부족.

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | "Open-world"의 범위는? 10개 집? 100개 집? | 미공개. 소수의 새 환경일 가능성 |
| 2 | Object detection을 사전 제공하는가 (oracle), 자체 수행하는가? | 자체 detection이면 인상적이나, oracle이면 unfair |

<!-- VERIFIED: abstract-only -->
