# StableIDM: Stabilizing Inverse Dynamics Model against Manipulator Truncation via Spatio-Temporal Refinement

> **한 줄 요약**: **manipulator truncation** (로봇 팔이 시야에서 잘림) 상황에서 IDM이 붕괴하는 문제를, **robot-centric masking + Directional Feature Aggregation (DFA) + Temporal Dynamics Refinement (TDR)** 의 **공간-시간 정제**로 안정화.

---

## 1. 배경 및 동기

- IDM(Inverse Dynamics Model)은 visual obs → low-level action을 매핑, **데이터 라벨링**과 **policy 실행**의 핵심.
- 그러나 manipulator가 카메라 프레임 밖으로 나가는 **truncation** 상황에서 상태 추정이 ill-posed → 제어 불안정.
- 기존 IDM은 이런 partial observability를 명시적으로 다루지 않음.

---

## 2. 방법론

### 2.1 Auxiliary robot-centric masking
- 배경 잡음을 억제하는 보조 마스킹. "어디에 로봇이 있는가/있었는가"에 특징을 집중.
- Masking을 보조 loss/입력으로 사용해 feature가 robot 주변 영역에 민감해지도록 유도.

### 2.2 Directional Feature Aggregation (DFA)
- **Geometry-aware spatial reasoning** 모듈.
- 보이는 팔의 방향을 추정하고, 그 방향을 따라 **anisotropic feature** 를 집계 → 잘린 부분의 위치를 방향 단서로 외삽.

### 2.3 Temporal Dynamics Refinement (TDR)
- **motion continuity** 를 이용해 시간적으로 smooth/correct.
- 순간적 truncation으로 인한 스파이크형 오차를 trajectory 수준에서 교정.

---

## 3. 실험 결과

> 논문 PDF 미검증 (abstract-only). 구체 수치는 paper 참조 필요.

- Abstract는 "multiple benchmarks and real-robot experiments"에서 개선을 보고한다고 언급.
- 구체 성능 수치(벤치마크, 오차 감소율 등)는 PDF 확인 필요.

---

## 4. 한계 및 미해결 문제

1. **Truncation 외 failure mode**: Full occlusion, severe motion blur 등은 DFA의 "visible arm direction" 가정이 깨지면 효과가 떨어짐.
2. **Cross-embodiment 일반화**: DFA가 특정 팔 형태에 과적합될 가능성 → 다른 모폴로지로의 전이는 별도 검증 필요.
3. **IDM이 정책의 병목은 아닐 수 있음**: 상류 policy/perception 에러가 크면 IDM 정제의 이익이 제한적.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★☆☆ — Truncation이라는 구체 failure mode를 타겟한 spatial+temporal 정제. 구성 요소는 기존 아이디어의 재조합. |
| **Practical impact** | ★★★★☆ — 데이터 라벨링과 policy 실행 양쪽에서 IDM은 폭넓게 쓰이며, truncation은 실제 시설에서 흔함 → 실용적 가치 높음. |

---

## 6. 🔥 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | DFA는 왜 isotropic feature보다 나은가? | Truncated 영역의 정보는 본질적으로 **방향성** 있음 (팔은 선형 구조). Anisotropic aggregation이 해당 방향으로 정보를 전파하기 적합. |
| 2 | TDR이 진짜 motion continuity를 가정해도 되나? | 대부분 매니퓰레이션은 저가속도 연속 운동이라 합리적. 충격·접촉 순간에는 이 가정이 일시적으로 깨질 수 있음. |
| 3 | 멀티뷰 카메라로 해결하면 되지 않나? | 실제 배포에서 멀티뷰 setup이 항상 가능한 건 아님. StableIDM은 single-view truncation robustness 자체를 목표로 함. |

<!-- VERIFIED: abstract-only -->
