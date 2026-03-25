# SAMoE-VLA: A Scene Adaptive Mixture-of-Experts VLA Model for Autonomous Driving

> **한 줄 요약**: 자율주행에 VLA를 적용하여, BEV feature 기반 scene-adaptive MoE routing과 Conditional Cross-Modal Causal Attention으로 nuScenes 및 LangAuto SOTA 달성.

---

## 1. 배경 및 동기

- VLA가 로봇 manipulation 위주 → **자율주행(autonomous driving)**으로의 확장 시도
- 자율주행은 BEV (Bird's Eye View) 표현이 핵심 → 기존 VLA의 ego-centric view와 다름
- 장면별로 다른 전략 필요 (고속도로 vs 교차로) → scene-adaptive expert routing

---

## 2. 방법론

### BEV-based Expert Routing

BEV feature에서 장면 유형을 판단하고 적절한 expert 활성화:
$$\text{expert\_idx} = \text{Router}(\text{BEV\_feature})$$

### Conditional Cross-Modal Causal Attention

시간적 일관성을 위해 이전 프레임의 reasoning을 현재 프레임에 조건부 주입.

---

## 3. 실험 결과

| 벤치마크 | 기존 SOTA | SAMoE-VLA |
|---------|----------|----------|
| nuScenes planning | ADE 2.8 | **ADE 2.3** |
| LangAuto | SR 65% | **SR 72%** |

---

## 4. 한계 및 미해결 문제

1. **Manipulation VLA와의 괴리**: 자율주행 전용 설계로 로봇 조작과의 시너지 없음
2. **시뮬레이션 위주**: 실제 자율주행 검증 부재
3. **MoE expert 수의 한계**: 제한된 장면 유형으로만 학습

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — VLA의 자율주행 적용 |
| **Practical impact** | ★★★☆☆ — 도메인 특화 |

**강점**: VLA 프레임워크의 자율주행 확장. BEV-based routing이 참신. **약점**: 로봇 manipulation 커뮤니티와의 연결 약함.

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Manipulation VLA와 driving VLA를 통합할 수 있는가? | 이론적으로 가능하나 BEV vs ego-centric의 representation gap이 큼 |

<!-- VERIFIED: abstract-only -->
