# ABot-Claw: A Foundation for Persistent, Cooperative, and Self-Evolving Robotic Agents

> **한 줄 요약**: OpenClaw 위에 embodied 제어 레이어를 덧붙여, **heterogeneous 로봇 스케줄링 + cross-embodiment multimodal memory + critic-based closed-loop 피드백**을 통합한 **self-evolving multi-robot agent foundation**.

---

## 1. 배경 및 동기

- VLA는 강력한 perception/intuitive response를 주지만 **open-loop** 특성으로 long-horizon 과제에서 실패.
- System-2 agent들은 planning은 좋지만 **closed sandbox + predefined toolkit** 한계.
- OpenClaw는 전체 시스템 권한을 가진 local runtime이지만, 장시간·다중 로봇 embodied 제어 아키텍처가 없음.
- 따라서 "언어 의도 → 물리 행동"을 닫힌 루프로 잇고 **진화 가능한** 에이전트 기반이 필요.

---

## 2. 방법론

### 2.1 Unified embodiment interface (Capability-driven scheduling)
- 이질적 로봇(모바일 베이스, 매니퓰레이터, 다리형 등)을 capability 단위로 추상화.
- 과제 요구 capability에 따라 적합한 embodiment에 동적으로 할당.

### 2.2 Visual-centric cross-embodiment multimodal memory
- 비전 중심의 **persistent context retention** 을 통해, 여러 로봇/세션에 걸쳐 grounded retrieval 가능.
- Long-horizon 과제에서 상태/이력을 참조 가능.

### 2.3 Critic-based closed-loop feedback
- **Generalist reward model** 이 online으로 진행도를 평가.
- 실패/탈선 시 local correction 또는 re-planning을 트리거 → 진정한 closed-loop.

### 2.4 Layered 아키텍처
- **OpenClaw layer** (시스템 권한 런타임) + **shared service layer** (memory, scheduler, critic) + **robot embodiment layer** (하드웨어 추상).
- 레이어 분리로 새로운 로봇/서비스 추가가 용이.

---

## 3. 실험 결과

> 논문 PDF 미검증 (abstract-only). 구체 수치는 paper 참조 필요.

- 정량 지표는 abstract에 제시되지 않음.
- 주장: 자연어 의도 → 물리 행동까지 **closed-loop**, open/dynamic 환경에서 **점진적 self-evolving**이 가능.

---

## 4. 한계 및 미해결 문제

1. **Reward model의 일반성**: "generalist reward model"이 실제로 모든 task에 calibration되어 있는지는 별도 검증 필요.
2. **자원/지연 비용**: Persistent memory + critic + planner의 조합은 latency와 스토리지를 크게 늘림.
3. **평가 프로토콜**: Self-evolving을 정량화할 표준 벤치마크 부재 → 논문 자체 시나리오에 의존할 가능성.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★☆☆ — System-2 agent + OpenClaw 런타임을 embodied 제어에 접목한 통합 시도. 각 구성요소는 기존 아이디어의 조합에 가까움. |
| **Practical impact** | ★★★★☆ — Long-horizon·multi-robot 실환경에 바로 배포 가능한 foundation이라면 영향력 큼. 실제 재현성이 관건. |

---

## 6. 🔥 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | "Self-evolving"의 정의와 측정 방법? | Abstract만으로는 불명확. Memory 누적 + critic feedback 기반 online adaptation으로 추정되며, 정량 지표는 PDF 확인 필요. |
| 2 | Critic이 틀리면 어떻게 되나? | Closed-loop의 classic 문제. Reward hacking / false positive 시 교정 방향 자체가 왜곡됨. Generalist reward의 robustness가 핵심. |
| 3 | OpenClaw의 "전체 시스템 권한"은 보안 관점에서 괜찮나? | Local runtime이므로 격리 가능하지만, 공유 service layer가 여러 로봇을 제어한다면 공격 표면 확대. |

<!-- VERIFIED: abstract-only -->
