# NaVILA: Legged Robot Vision-Language-Action Model for Navigation

> **한 줄 요약**: VLA와 locomotion skill을 2-level 계층 구조로 통합하여, 다리 로봇에서의 vision-language navigation을 실현. Mid-level spatial language instruction을 생성하여 visual locomotion policy를 구동.

---

## 1. 배경 및 동기

- 기존 VLA는 **고정된 base 위의 manipulator** 가정 → 이동 로봇, 특히 legged robot에는 부적합
- 다리 로봇의 locomotion skill (균형, 보행)은 별도 학습 필요 → VLA와의 통합 어려움
- VLN (Vision-Language Navigation)은 wheeled robot 위주

---

## 2. 방법론

### 2-Level Hierarchy

**High-level VLA**: Vision + language → mid-level spatial language instruction
- "3미터 전진 후 왼쪽으로 45도 회전"

**Low-level Visual Locomotion Policy**: Spatial instruction → joint torques
- RL로 학습된 locomotion controller

> ❓ **예상 질문**: Mid-level instruction의 granularity는 어떻게 정하는가?
> **답변**: 거리(m)와 방향(도)의 조합으로 표준화. Granularity 선택(0.5m vs 1m 등)에 대한 sensitivity 분석 포함.

---

## 3. 실험 결과

| 환경 | 기존 VLN | NaVILA |
|------|---------|--------|
| IsaacLab indoor | 45.2% | 63.8% |
| Outdoor terrain | 28.5% | 47.3% |

- Outdoor terrain에서의 성능이 특히 주목할 만함

---

## 4. 한계 및 미해결 문제

1. **Manipulation 미포함**: Navigation only
2. **Mid-level instruction의 표현력**: 복잡한 경로 지정에 한계
3. **Sim-to-real**: Simulation 위주 평가

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Legged robot VLA |
| **Practical impact** | ★★★★☆ — 이동 로봇 확장 |

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Navigation + manipulation을 동시에 할 수 있는가? | 현 구조에서 불가. Mobile manipulation 확장 필요 |
| 2 | End-to-end (직접 torque 예측) VLA와 비교하면? | End-to-end는 학습이 극도로 어려움. 계층적 접근이 더 실용적 |

<!-- VERIFIED: abstract-only -->
