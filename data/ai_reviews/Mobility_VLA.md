# Mobility VLA: Multimodal Instruction Navigation with Long-Context VLMs and Topological Graphs

> **한 줄 요약**: Long-context VLM과 topological graph 기반 제어를 결합한 계층적 navigation 시스템으로, 자연어+이미지 multimodal 지시를 이해하여 실제 환경에서 로봇 navigation 수행.

---

## 1. 배경 및 동기

- Robot navigation에서 multimodal instruction (텍스트 + 참조 이미지) 처리 필요
- 기존 VLN (Vision-Language Navigation)은 discrete graph 기반 → 실제 로봇 제어에 부적합
- Long-context VLM의 등장으로 긴 시퀀스의 관측 이력 처리 가능

---

## 2. 방법론

### Hierarchical System

**High-level**: Long-context VLM이 multimodal instruction을 해석하고 topological graph 위에서 경로 계획
**Low-level**: Local navigation policy로 waypoint 간 이동

> ❓ **예상 질문**: End-to-end VLA 대비 hierarchical의 이점은?
> **답변**: Navigation의 long-range planning은 end-to-end로 학습하기 어려움. Topological graph가 global structure를 제공하고 VLM이 semantic reasoning을 수행하여 각각의 강점을 활용.

---

## 3. 실험 결과

| 설정 | 기존 VLN | Mobility VLA |
|------|---------|-------------|
| Seen env | 68% | 78% |
| Unseen env | 42% | 58% |
| Multimodal instruction | N/A | 65% |

---

## 4. 한계 및 미해결 문제

1. **Manipulation 미포함**: Navigation only → 범용 VLA가 아님
2. **Topological graph 의존**: 사전 구축된 map 필요
3. **Dynamic environments**: 정적 환경 가정

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Long-context VLM + topological nav |
| **Practical impact** | ★★★☆☆ — Map 의존이 실용성 제한 |

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Map 없이 작동할 수 있는가? | 현 구조에서 불가. Online mapping과의 결합 필요 |
| 2 | NaVILA와의 차이는? | NaVILA는 legged robot locomotion 포함, Mobility VLA는 wheeled robot navigation 특화 |
