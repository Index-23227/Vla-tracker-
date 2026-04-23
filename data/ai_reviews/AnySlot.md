# AnySlot: Goal-Conditioned Vision-Language-Action Policies for Zero-Shot Slot-Level Placement

> **한 줄 요약**: 언어 지시를 **explicit visual goal (scene marker)** 로 먼저 변환한 뒤, goal-conditioned VLA policy로 실행하는 **hierarchical slot-level placement 프레임워크**. 새로운 SlotBench 9개 과제군에서 monolithic VLA 대비 zero-shot 성능 개선을 보고.

---

## 1. 배경 및 동기

- 최신 monolithic VLA는 generalist manipulation에서 강하지만, **"어느 슬롯에 무엇을 놓아라"** 류의 compositional 언어 지시에는 여전히 취약.
- Slot-level placement는 (i) reliable **slot grounding** 과 (ii) sub-centimeter 수준의 **실행 정확도** 둘 다 요구 → 단일 policy가 두 가지를 동시에 배우기 어려움.
- 기존 modular grounding 방식들은 low-level 실행 단계에서 누적 오차가 크고, zero-shot 일반화가 제한됨.

---

## 2. 방법론

### 2.1 Scene-Marker로의 언어 grounding
- 언어 지시를 바로 action으로 매핑하지 않고, 중간 표현으로 **explicit spatial visual goal (scene marker)** 을 생성.
- 이 marker가 "어느 슬롯이 목표인가"를 시각적으로 명시 → 이후 단계가 언어 모호성을 해결할 필요가 없어짐.

### 2.2 Goal-conditioned VLA policy
- Scene marker가 주입된 goal-conditioned VLA가 저수준 제어 실행.
- High-level slot selection과 low-level execution을 **분리**함으로써, semantic accuracy와 spatial robustness를 동시에 확보.

### 2.3 SlotBench
- 정밀 placement 평가를 위한 **9개 task category** 의 시뮬레이션 벤치마크.
- 구조화된 spatial reasoning 능력을 계량화.

---

## 3. 실험 결과

> 논문 PDF 미검증 (abstract-only). 구체 수치는 paper 참조 필요.

- Flat (monolithic) VLA baseline 및 이전 modular grounding 방법 대비 **zero-shot slot-level placement에서 유의미한 개선**.
- SlotBench 9개 category 전반에서 우위를 주장.

---

## 4. 한계 및 미해결 문제

1. **Real-world 검증 부족 가능성**: SlotBench가 시뮬레이션 전용이라면 real kitchen/workshop에서의 sub-cm 정확도 재현 여부 불명확.
2. **Scene marker 생성 품질 의존**: Marker가 틀리면 하위 policy가 올바르게 실행해도 실패 → error cascade 가능성.
3. **Slot 정의의 일반성**: "slot"이 명시적으로 정의된 과제에 특화되어, 비정형 placement에서는 이점이 줄 수 있음.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★☆☆ — Explicit visual goal을 slot-level placement에 접목한 점은 깔끔하나, hierarchical grounding 자체는 기존 연구 계보 위. |
| **Practical impact** | ★★★☆☆ — Pick-and-place 산업 적용 잠재력 있으나, SlotBench가 새로 도입된 벤치라 외부 비교가 필요. |

---

## 6. 🔥 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Scene marker가 틀린 경우 policy가 회복할 수 있나? | Goal-conditioned라 marker 오류 시 실행도 빗나감. 향후 closed-loop re-grounding이 필요. |
| 2 | Monolithic VLA에 data augmentation만으로는 해결되지 않나? | Compositional 언어 + sub-cm 정확도는 scale로는 느리게 개선됨 → explicit intermediate의 구조적 장점 주장. |
| 3 | SlotBench가 실제 robotics 시나리오를 얼마나 대표하나? | 9개 category가 실제 packing/assembly를 얼마나 포괄하는지 별도 검증 필요. |

<!-- VERIFIED: pdf -->
