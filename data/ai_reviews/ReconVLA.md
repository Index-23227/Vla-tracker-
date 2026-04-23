# ReconVLA: An Uncertainty-Guided and Failure-Aware Vision-Language-Action Framework for Robotic Control

> **한 줄 요약**: Pretrained VLA를 재학습 없이 wrapping하여 conformal prediction으로 action token과 robot state에 calibrated uncertainty를 부여, failure를 사전 감지하는 경량 framework.

---

## 1. 배경 및 동기

- VLA가 action을 뱉을 때 "얼마나 확신하는지"에 대한 calibrated signal이 없어 실세계 신뢰성이 낮음
- Dropout/ensemble 기반 uncertainty는 retraining이 필요하고 calibration 보장이 약함
- Conformal prediction은 분포 가정 없이 marginal coverage를 보장 → pretrained policy에 black-box로 얹을 수 있음

---

## 2. 방법론

### Action-token Conformal Prediction
Pretrained VLA의 action token output에 split conformal prediction을 적용하여, 각 action에 대해 calibrated confidence set을 생성. 이 신뢰도는 execution quality 및 task success와 상관관계를 가진다고 주장.

### State-space Conformal Outlier Detection
Robot state 자체에도 conformal prediction을 적용하여 OOD/unsafe state를 failure 이전에 감지. Action-level uncertainty와 complementary하게 작동하는 dual-signal failure detector 구성.

---

## 3. 실험 결과

> 논문 PDF 미검증 (abstract-only). 구체 수치는 paper 참조 필요.

- Simulation + real robot manipulation에서 평가
- Abstract 주장: failure anticipation 향상, catastrophic error 감소, retraining 불필요

---

## 4. 한계 및 미해결 문제

1. **Calibration set dependency**: Split conformal은 i.i.d. calibration set을 요구, robot deployment 환경에서 distribution shift 시 coverage 보장이 약화.
2. **Recovery 부재**: Failure를 "감지"할 뿐 recovery mechanism은 제공하지 않음 (Rewind-IL과 대비).
3. **Action token granularity**: Continuous control에서 token-level uncertainty가 end-effector trajectory 수준 uncertainty와 어떻게 매핑되는지 명확치 않음.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★☆☆ — Conformal prediction을 VLA에 적용한 초기 시도로서 의미, 기법 자체는 기존 통계 기법 |
| **Practical impact** | ★★★★☆ — Training-free, any-VLA compatible → 실제 deployment 직후 safety layer로 유용 |

---

## 6. 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Conformal prediction이 VLA에서 왜 적합한가? | Pretrained 가중치 수정 없이 calibrated coverage를 marginally 보장하므로 closed-model deployment scenario에 적합. |
| 2 | Action-level vs state-level signal은 어떻게 결합? | 두 신호를 OR/threshold fusion하는 것이 자연스러움. Detailed ensemble 규칙은 PDF 확인 필요. |
| 3 | Failure 감지 후 무엇을 하나? | Abstract는 anticipation만 다룸. Downstream policy는 stop/human-handoff/recovery를 별도로 구현해야 함. |

<!-- VERIFIED: abstract-only -->
