# Goal2Skill: Long-Horizon Manipulation with Adaptive Planning and Reflection

> **한 줄 요약**: VLM 기반 high-level planner(메모리·검증·반성)와 diffusion VLA low-level executor를 **closed-loop**로 결합해 long-horizon manipulation에서 baseline 대비 3배 이상의 성공률 향상을 보인 dual-system 프레임워크.

---

## 1. 배경 및 동기

- 기존 VLA: 제한된 observation window + end-to-end action prediction → long-horizon, memory-dependent task에 취약.
- Partial observability, occlusion, multi-stage dependency가 있는 태스크는 **지속적 memory**, **adaptive decomposition**, **실패 복구**가 필수.
- 단일 네트워크로 reasoning과 motor control을 모두 처리하면 둘 다 어중간 → 분리하자.

---

## 2. 방법론

### High-Level Planner (VLM-based agent)
- Structured task memory 유지.
- Goal decomposition, outcome verification, error-driven correction을 담당.
- Sub-task를 low-level executor에 순차 또는 적응적으로 하달.

### Low-Level Executor (VLA + diffusion)
- Geometry-preserving filtered observation을 입력으로 받는 diffusion-based action generator.
- 각 sub-task를 visuomotor control로 실행.

### Closed Loop
- Executor 결과 → Planner가 verify → 실패 시 reflection & replan → 재실행.
- 구조화된 memory가 단순 history가 아닌, planning-relevant state로 축적.

---

## 3. 실험 결과

> 논문 PDF 미검증 (abstract-only). 구체 수치는 paper 참조 필요.

- **RMBench**의 대표 long-horizon 태스크에서 **평균 성공률 32.4%** 기록, 가장 강한 baseline(9.8%) 대비 3배 이상 향상.
- Ablation: structured memory와 closed-loop recovery 모두 필수 요소로 확인.

---

## 4. 한계 및 미해결 문제

1. **VLM 호출 비용·latency**: High-level planner가 매 step마다 호출되면 실시간성 저해.
2. **Verification의 신뢰성**: Outcome verification을 VLM이 담당 → 잘못된 verify면 무한 루프 또는 premature 종료.
3. **RMBench 특화 가능성**: 32.4%라는 절대 수치는 여전히 낮아, 일반 long-horizon 환경으로 확장성 검증 필요.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★☆☆ — Dual-system + reflection 자체는 최근 흐름, 조합·검증이 기여 |
| **Practical impact** | ★★★★☆ — Baseline 대비 3배 이상 개선은 실질적 |

**강점**: Structured memory와 closed-loop recovery의 ablation 기여가 명확. **약점**: 여전히 32.4% 성공률 → long-horizon의 본질적 난제는 남아있음.

---

## 6. 🔥 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 단일 VLA에 CoT 붙이는 것과 뭐가 다른가? | Memory의 구조화와 verify→recover loop이 핵심 차별점 |
| 2 | Planner-Executor 호출 빈도는? | 일반적으로 sub-task 단위 (abstract에 구체 수치 없음) |
| 3 | RMBench만으로 충분한가? | LIBERO-Long, VIMA-Bench 등 추가 검증 권장 |
| 4 | Executor의 diffusion은 왜 선택? | Geometry-preserving filtered observation과 결합 시 정밀 제어 유리 |

<!-- VERIFIED: abstract-only -->
