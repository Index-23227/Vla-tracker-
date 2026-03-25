# π*₀.₆: A VLA That Learns From Experience

> **한 줄 요약**: RECAP (RL with Experience and Corrections via Advantage-conditioned Policies)을 통해 VLA가 실제 배포 경험에서 자기개선하여, 태스크 throughput 2배 이상 증가 및 실패율 절반 감소 달성.

---

## 1. 배경 및 동기

- BC 기반 VLA는 demonstration data에만 의존 → 배포 후 새로운 상황에서 개선 불가
- **실제 배포 경험**(성공/실패/교정)을 학습에 활용하는 self-improvement 필요
- 기존 RL의 VLA 적용은 시뮬레이터 의존 → 실제 환경에서의 RL이 핵심

---

## 2. 방법론 심층 분석

### RECAP Framework

세 가지 데이터 소스를 통합:
1. **Demonstrations**: 기존 텔레오퍼레이션 데이터
2. **On-policy rollouts**: 현재 policy로 수행한 실제 작업
3. **Interventions**: 사람이 실패를 교정한 데이터

Advantage-conditioned policy:
$$\pi(\mathbf{a} | \mathbf{o}, A) = \begin{cases} \text{high-advantage expert} & \text{if } A > \tau \\ \text{filtered behavior} & \text{otherwise} \end{cases}$$

> ❓ **예상 질문**: Advantage function을 실제 환경에서 어떻게 추정하는가?
> **답변**: Success/failure binary reward + 사람 intervention을 reward signal로 활용. Dense reward는 불필요하지만, binary reward의 정보량이 제한적이라 학습 효율이 낮을 수 있음.

---

## 3. 실험 결과

### Real-world Complex Tasks

| 메트릭 | pi0 (BC) | **pi*0.6 (RECAP)** |
|-------|---------|-------------------|
| Task throughput | 1x | **2x+** |
| Failure rate | 1x | **~0.5x** |
| Laundry folding | 62% | **~80%** |
| Espresso making | 45% | **~70%** |

---

## 4. 한계 및 미해결 문제

1. **사람 intervention 비용**: RECAP에 사람 교정 데이터가 필요 → 완전 자율 자기개선이 아님
2. **비공개**: 상세 알고리즘, 데이터 규모 비공개
3. **Safety during exploration**: On-policy rollout에서 위험한 행동 가능성
4. **Scalability**: 태스크 수 증가 시 intervention 비용 선형 증가

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Real-world VLA self-improvement |
| **Practical impact** | ★★★★★ — 배포 후 지속 개선 |

**강점**: "배포 후 학습"의 실용적 실현. Complex task에서의 큰 개선. **약점**: 사람 의존, 비공개.

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Intervention 없이 (자동 보상만으로) 자기개선이 가능한가? | 성능 하락 예상. Intervention의 정보가 매우 높은 가치 |
| 2 | AcceRL과의 차이는? | AcceRL은 시뮬레이션 비동기 RL, RECAP은 실제 환경 경험 기반 |

<!-- VERIFIED: abstract-only -->
