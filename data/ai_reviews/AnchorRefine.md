# AnchorRefine: Synergy-Manipulation Based on Trajectory Anchor and Residual Refinement for Vision-Language-Action Models

> **한 줄 요약**: 사람의 coarse-to-fine reaching에서 착안해 VLA action 생성을 "trajectory anchor + residual refinement"로 분해하고, gripper를 decision-aware MSE로 별도 보정하여 LIBERO-Long 97.4%, CALVIN ABC→D Avg.Len 4.40을 달성한 backbone-agnostic framework.

---

## 1. 배경 및 동기

- 기존 VLA(π0, OpenVLA, X-VLA 등)는 전체 action chunk를 하나의 monolithic target으로 예측하여 **대진폭 transport**가 학습을 지배하고 작지만 성공을 결정하는 **국소 correction**이 과소 모델링됨(Sec. 1).
- LIBERO failure 분석에서 **88% 이상이 gripper 관련**(Fig. 1c): gripper가 너무 일찍/늦게 닫히는 등 decision-boundary error가 주원인.
- 사람의 goal-directed reaching은 coarse transport → intermediate correction → fine adjustment의 구조(Fig. 1a, Elliott et al.). 저자들은 이 구조적 분리를 VLA에 적용.
- 중요한 관찰(Fig. 1b): 학습된 anchor output에 대한 residual은 **mean norm 6.7 → 0.4, covariance trace 1.2 → 0.15**로 압축되어 학습하기 쉬운 target space가 된다.

---

## 2. 방법론

### Trajectory Anchor Planner (Sec. 3.3)
Phase 1에서 shared multimodal context Z_t와 anchor latent token ξ_a로부터 arm action chunk를 직접 예측하고 full ground-truth로 reconstruction loss(L_arm) 학습(Eq. 3-4). 자동으로 globally coherent motion structure를 포착.

### Residual Refine Module (Sec. 3.4)
Phase 2에서 anchor를 동결하고 **detached anchor prediction에 대한 residual을 target**으로 학습(Eq. 5): R*_t,arm = A_t,arm − sg(Â^anc_t,arm). 같은 context에서 ξ_r로부터 R̂_t,arm을 예측, residual space에서 L_refine supervise(Eq. 7). 최종 arm action = anchor + residual.

### Decision-Aware Gripper Refinement (Sec. 3.5)
Gripper는 continuous regression이 아닌 decision-boundary 문제로 취급. 방향 신호 s_t,h ∈ {-1,0,1}와 anchor-phase 확률 q^anc을 결합한 boundary-aware target r_t,h^grip,* = s_t,h·(|q^anc - 0.5| + ε) (Eq. 8-9). MSE로 최적화(Eq. 10). 추론 시 anchor 확률에 residual을 더해 threshold 적용(Eq. 11).

### Architecture
- Backbone: GR-1 (0.22B, regression) 또는 X-VLA (1.1B, diffusion)
- Loss: L_phase2 = L_refine + λL_grip, λ=0.01
- Training: 2-phase sequential (anchor 200k step → refine 80k step)

---

## 3. 실험 결과

### CALVIN ABC→D & LIBERO-LONG (Table 1)

| Method | CALVIN 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | Avg.Len | LIBERO-LONG |
|---|---|---|---|---|---|---|---|
| GR-1* (0.2B) | 91.0 | 79.2 | 68.5 | 60.3 | 52.0 | 3.51 | 74.5 |
| AnchorRefine(GR-1) | 91.9 | 81.6 | 71.9 | 63.4 | 55.3 | **3.64** | **82.3** (+7.8) |
| X-VLA* (0.9B) | 97.0 | 91.6 | 86.9 | 81.3 | 74.6 | 4.31 | 95.8 |
| AnchorRefine(X-VLA) | **97.3** | **93.1** | **89.1** | **83.7** | **76.5** | **4.40** | **97.4** (+1.6) |

저자들의 주장대로 longer horizon일수록 이득이 커짐(CALVIN 5/5에서 GR-1 +3.3%p).

### Ablation (Table 2, on LIBERO-Long)

| Variant | SR (%) | Δ |
|---|---|---|
| Full AnchorRefine | 82.3 | 0.0 |
| w/o gripper refinement | 78.5 | -3.8 |
| Naive gripper MSE | 72.6 | -9.7 |
| w/o detached anchor (stop-gradient) | 75.7 | -6.6 |
| Explicit anchor concatenation | 79.6 | -2.7 |
| Anchor phase only (0.22B, deeper) | 76.4 | -5.9 |
| Refine(direct action, no residual) | 73.6 | -8.7 |

Capacity 증가만으로는 설명 불가(76.4 vs 82.3), residual space 학습이 핵심.

### Mechanistic Analysis (Sec. 4.3)
- 10 LIBERO-Long tasks 200 rollout에서 **Fail→Success 전환이 Success→Fail보다 빈번**, gripper 관련 오류 약 26% 감소.
- Refinement objective의 training loss가 baseline 대비 더 빠르게 수렴하고 더 낮은 plateau 도달(Fig. 3b).

### Real-world (LeRobot SO101, Fig. 4)
4개 task 각 20 trial 평가. Avg SR **59%** (ACT 39, π0 51, X-VLA 41 대비). Stack Blocks, Open Drawer 같은 terminal-precision-critical task에서 이득이 두드러짐.

---

## 4. 한계 및 미해결 문제

1. **Non-adaptive refinement (저자들이 Sec. 5에서 명시적으로 인정)**: 현재는 고정된 2-phase refinement; task difficulty에 따라 refinement 강도를 조절하는 adaptive 메커니즘이 부재.
2. **Open-world generalization 제한 (Sec. 5 self-acknowledged)**: LIBERO/CALVIN의 제한된 환경에서만 평가, 완전한 unseen task 일반화는 미검증.
3. **2-phase training 추가 비용**: anchor 200k + refine 80k step으로 총 학습 시간이 backbone 단독 대비 증가.
4. **Gripper 외 discrete decision 일반화**: Decision-aware 보정이 gripper에만 특화됨; 다른 discrete action(툴 switching 등)으로의 확장은 미제시.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★☆☆ — Residual policy learning / coarse-to-fine 아이디어는 새롭지 않지만, VLA action chunk 내부에서 anchor-relative residualization + decision-aware gripper를 결합한 조합은 깔끔한 재해석 |
| **Practical impact** | ★★★★☆ — regression/diffusion backbone 양쪽에서 일관된 이득, GR-1 기준 +7.8% LIBERO-LONG이 큼. 학습만 다시 하면 되므로 upstream 호환성 좋음 |

AnchorRefine의 핵심 메시지는 단순하다. "큰 움직임과 작은 교정을 같은 loss로 학습하지 말라." Fig. 1b의 residual statistics(norm 94% 감소, covariance 87.5% 감소)는 이 주장의 **empirical 증명**이며 Table 2의 ablation이 이를 뒷받침한다. 특히 detached anchor와 implicit guidance의 조합이 설계상 우아하며, gripper-specific MSE는 간단하지만 LIBERO failure mode에 정확히 대응하는 처방이다.

---

## 6. 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 왜 anchor에서 stop-gradient가 필수인가? | Stop-gradient 없으면 anchor가 downstream refinement objective에 의해 계속 끌려가서 "global scaffold" 역할이 붕괴. Table 2: detachment 제거 시 82.3 → 75.7로 6.6%p 하락. |
| 2 | Explicit anchor concat 대신 implicit(target-level) coupling을 택한 이유? | Feature-level 결합은 anchor bias를 refinement에 전파; target 수준에서만 연결하면 refinement가 "남은 것"에만 집중 가능. Table 2: 79.6 vs 82.3. |
| 3 | Gripper MSE를 BCE 대신 쓴 이유? | Target이 binary label이 아니라 **anchor-probability 기준 boundary-crossing magnitude**(Eq. 9). 방향과 크기 정보를 모두 보존해야 해서 continuous loss가 적합. Naive MSE는 72.6로 오히려 악화되므로 decision-aware formulation이 핵심. |

<!-- VERIFIED: pdf -->
