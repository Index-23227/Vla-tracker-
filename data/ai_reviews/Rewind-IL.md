# Rewind-IL: Online Failure Detection and State Respawning for Imitation Learning

> **한 줄 요약**: Action-chunked generative imitation policy 위에 붙는 training-free online safeguard — TIDE로 failure를 감지하고 VLM이 지정한 safe checkpoint로 "rewind"하여 실행을 재개.

---

## 1. 배경 및 동기

- Long-horizon action-chunked IL policy는 demonstration manifold를 벗어나면 locally plausible하지만 복구 불능한 action을 계속 생성
- 기존 runtime monitor: (a) failure data가 필요하거나, (b) benign drift에 과도하게 trigger하거나, (c) 감지만 하고 recovery가 없음
- 추가 학습 없이 deploy 시점에 붙일 수 있는 detect + recover 파이프라인이 필요

---

## 2. 방법론

### TIDE: Temporal Inter-chunk Discrepancy Estimate
Overlapping action chunk들 간의 self-consistency(겹치는 구간이 서로 얼마나 일관되는지)를 zero-shot으로 측정. Split conformal prediction으로 threshold를 calibrate하여 false-positive rate를 제어.

### State Respawning with VLM Checkpoints
오프라인에서 VLM이 demonstration의 "semantically safe intermediate state"를 checkpoint로 지정하고, frozen policy encoder로 feature database 구축. 온라인에서 failure 감지 시 가장 가까운 verified checkpoint로 로봇을 되돌린 뒤 policy state를 초기화하고 재실행.

---

## 3. 실험 결과

> 논문 PDF 미검증 (abstract-only). 구체 수치는 paper 참조 필요.

- Training-free 주장 → 다양한 generative action-chunked policy에 적용 가능
- Project page: https://sjay05.github.io/rewind-il

---

## 4. 한계 및 미해결 문제

1. **물리적 rewind 실현성**: "State respawning"이 실제 환경에서 물체를 원상복구해야 한다면 reset 메커니즘 또는 reversible task만 지원 가능.
2. **VLM checkpoint 품질**: Recovery point 선정은 VLM 성능에 민감, 복잡한 장면에서 잘못된 safe state 지정 시 오히려 악영향.
3. **TIDE의 coverage 한계**: Inter-chunk self-consistency는 policy가 "확신하며 틀리는" mode collapse failure에 둔감할 수 있음.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Detection + recovery를 training-free로 통합한 구성, VLM-guided checkpoint는 참신 |
| **Practical impact** | ★★★★☆ — Long-horizon chunked policy(예: Diffusion Policy, ACT, π0)에 범용적으로 얹을 수 있어 유용성 높음 |

---

## 6. 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 왜 action chunk 간 self-consistency가 failure signal인가? | Chunk overlap 구간의 불일치는 policy가 latent state를 일관되게 유지하지 못한다는 증거 → drift 초기 신호. |
| 2 | Real-world에서 "rewind"가 가능한 task는? | 가역적 manipulation(pick-and-place의 초기 state 복귀 등). Irreversible task(예: 용접, 붓기)는 제한적. |
| 3 | Split conformal calibration set은 어떻게 구성? | Nominal(성공) demonstration의 inter-chunk discrepancy 분포를 hold-out으로 사용하여 사용자 지정 FPR 달성. |

<!-- VERIFIED: abstract-only -->
