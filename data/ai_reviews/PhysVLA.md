# PhysVLA: Towards Physically-Grounded VLA for Embodied Robotic Manipulation

> **한 줄 요약**: 어떠한 frozen VLA backbone에도 retraining 없이 wrap 가능한 inference-time physics corrector. Phase-aware FSM + selective Euler-Lagrange gate로 LIBERO-Spatial에서 OpenVLA를 36% → 53%로, OpenVLA-OFT를 92% → 95%로 끌어올리고 Robosuite Lift에서 ~10x trajectory-jerk robustness를 달성한다.

---

## 1. 배경 및 동기

- 현재 VLA 모델(RT-2, π0, OpenVLA, CogACT, TinyVLA, SmolVLA 등)은 demonstration data만으로 학습되어 rigid-body dynamics나 contact constraint 같은 물리 법칙을 명시적으로 강제하지 않는다.
- 이로 인한 "physics gap"이 존재: single-step OpenVLA는 LIBERO-Spatial에서 36%, chunked OpenVLA-OFT는 92%지만 contact-rich task에서 여전히 실패한다.
- 기존 inference-time 보완책인 uniform EMA(temporal smoothing)는 contact phase의 responsive burst를 죽여 성공률을 오히려 떨어뜨린다(OpenVLA 36% → 28%).
- 저자들의 핵심 아이디어: 물리를 학습 단계에 통합(PINN, DeLaN, HNN 등)하는 대신 **inference 시점에 조건부로** 적용하자.

---

## 2. 방법론

### Branch A: Phase-Aware Finite-State Machine (Sec. 3.3)
시뮬레이터 상태 s_t = (p_eef, q_t, p_obj, c_t)로부터 manipulation phase φ_t를 기하학적 predicate로 판정한다(Eq. 3):
- **Approach** (d_xy^bowl ≥ 6 cm): premature-grasp veto, g ← 0
- **Grasp** (d_xy^bowl < 6 cm, bowl unlifted): grasp waypoint p*로 β=0.5 blending
- **Transport** (bowl lifted, d_xy^plate ≥ 6 cm): +2cm vertical lift bias + transport-only EMA(α=0.92)
- **Placement** (bowl lifted, d_xy^plate < 6 cm): deceleration ramp scaling(Eq. 4)

### Branch B: Selective Euler-Lagrange Gate (Sec. 3.3)
Franka Panda 7-DoF에서 r_EL = M(q)q̈ + C(q,q̇)q̇ + G(q) - τ (Eq. 5)를 MuJoCo dynamics oracle로 추출. ε = 0.05 N·m(clean trajectory residual 평균보다 한 자릿수 작음)을 초과할 때만 inertia-weighted blending(Eq. 6-7) 적용. Softmax(diag(M(q))^(-1))로 저관성 joint에 더 큰 correction.

### Capped Blender (Eq. 2)
a_t = (1 - c)·a_VLA + c·a_phys, **c = 0.05**. 95%는 VLA의 자체 prediction, 5%만 physics correction. |g| > 1.5인 의도적 grasp는 hard override로 보호. 단일 hyperparameter, 모든 backbone/task 공통.

### Overhead
RTX 4090 기준 per-step ≈ 0.6 ms (Branch A 0.2 ms + Branch B 0.4 ms). 20 Hz 제어 주기(50 ms) 안에 충분히 들어가며 VLA forward pass(30-90 ms) 대비 무시 가능.

---

## 3. 실험 결과

### LIBERO-Spatial Aggregate (Table 3, 50 episodes per cell)

| Backbone | Baseline Succ | Temporal Succ | **PhysVLA Succ** | Baseline Stab | **PhysVLA Stab** |
|----------|--------------:|--------------:|-----------------:|--------------:|-----------------:|
| OpenVLA (single-step) | 36% | 28% | **53%** (+17pp) | 20.1% | **36.8%** (+16.7pp) |
| OpenVLA-OFT (chunked) | 92% | 92% | **95%** (+3pp) | 86.1% | **88.9%** (+2.8pp) |
| Force-VLA | 40% | 36% | **53%** (+13pp) | 20.0% | **38.2%** (+18.2pp) |
| Generalist-VLA (flow-matching) | 36% | 26% | **50%** (+14pp) | 29.9% | **49.2%** (+19.3pp) |

- Temporal smoothing은 single-step backbone에서 일관되게 성공률을 깎으나, PhysVLA는 4개 backbone 모두에서 zero per-task regression으로 향상.
- 구조적 한계: T4(cabinet drawer), T9(wooden cabinet) 같은 occluded geometry task는 single-step에서 0% 유지.

### Robosuite Lift Cross-Simulator Sweep (Sec. 4.2.1)
σ ∈ {0, 0.05, ..., 0.40} XY noise sweep에서 PhysVLA mean jerk 0.064 → 0.075(+17%), Baseline 0.064 → 0.176(+175%). **~10x jerk robustness ratio**. σ=0.40에서 Δjerk = -0.102 (58% 감소).

### Real-world Agilex Piper Pick-and-Place (Sec. 4.2.3, n=20)
- Baseline 45% → **PhysVLA 95%** (절대 +50pp)
- Mean jerk ≈ 0.05 → ≈ 0.005 (~10x smoother)
- 동일한 OpenVLA backbone과 시뮬레이션과 동일한 hyperparameter; retraining 없음.

---

## 4. 한계 및 미해결 문제

1. **URDF/관성 파라미터 의존성**: Euler-Lagrange gate는 정확한 kinematic/inertial calibration을 가정한다. 부정확한 URDF로는 closed-form residual 대신 data-driven dynamics approximator로 fallback 필요.
2. **5% cap의 구조적 한계**: T5 같은 sub-centimetre precision task에서 cap이 너무 작아 fundamental gap을 해소하지 못함(저자들이 명시).
3. **Phase predicate의 task-specific 설계**: approach/grasp/transport/place 4-phase FSM은 pick-and-place에 맞춰져 있으며, deformable이나 articulated manipulation에서는 phase 정의 자체가 새로 필요하다.
4. **Simulator state 의존**: 평가는 MuJoCo state vector를 readout하는 setting으로, 진정한 partial-observability 환경(pure visual feedback)에서는 phase detector가 perception 오류에 취약할 수 있다.
5. **Cross-backbone sample size**: 각 (backbone × task)당 n=5 trial(aggregate 50 episodes)로 통계적 power가 낮다.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★☆☆ — physics-informed control 자체는 오래된 분야지만, "frozen VLA를 건드리지 않고 inference 시점에서 조건부로 EL residual을 fire"하는 framing은 깔끔한 합성 |
| **Practical impact** | ★★★★☆ — 5% cap, 단일 hyperparameter, <1 ms overhead, 4개 backbone 일관 향상, real-world transfer 검증. 누구나 OpenVLA 위에 즉시 wrap 가능하다는 점이 매력적 |

PhysVLA는 새 모델을 학습하지 않고 **runtime composability**라는 관점에서 VLA를 개선한다는 점이 가장 인상적이다. CBF-QP나 MPPI 같은 always-on safety filter와 달리 선택적 gating으로 latency budget을 지키며, temporal smoothing의 well-known 단점(contact phase에서의 responsiveness 손실)을 phase-aware FSM으로 정확히 우회한다. 단점은 (a) LIBERO-Spatial이라는 좁은 평가 범위, (b) URDF/관성 의존성, (c) chunked backbone(OFT)에서의 marginal gain(+3pp)이지만, "프리트레인된 VLA에 한 줄로 끼워넣는" 형태의 baseline으로는 매우 유용하다.

---

## 6. 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Branch A 없이 Branch B만 쓰면? | EL gate는 step-level kinodynamic inconsistency만 잡고 phase-level 오류(예: approach 도중 premature grasp)는 못 잡는다. Branch A는 geometric predicate로 phase를 명시하여 contact phase에서의 responsiveness를 명시적으로 보존하는 역할이라 분리 불가. |
| 2 | Capped blender의 c=0.05는 어떻게 정했나? | 모든 backbone과 task에 공통 적용되는 단일 값. 5%만 physics가 개입하므로 단일 step의 오류는 작게 흡수되고, FSM의 phase prior가 강할 때만 누적되어 큰 영향. Per-backbone tuning이 없다는 점이 composability의 핵심. |
| 3 | OFT에서 +3pp만 향상되는 이유? | Chunked decoding이 이미 short-horizon temporal coherence를 확보해 step-level 불일치가 적기 때문. 저자들은 OFT에서 EL gate를 끄고 Branch A만 보고하며, OFT 92% → 95%는 phase-level rule이 챙기는 잔여 contact-rich 실패. |
| 4 | Real-world Agilex Piper에서 45% → 95%가 너무 큰 것 아닌가? | 절대 trial 수 n=20이라 통계적 power는 제한적. 또한 hardware는 6-DoF이지만 PhysVLA Branch A는 cm-level geometric predicate로 작동하므로 embodiment-agnostic. Mean jerk 10x 개선이 motion quality 차이를 뒷받침. |
| 5 | T4/T9가 여전히 0%인 이유? | 두 task는 cabinet/drawer occluded geometry로, 시야 외 영역에서 target pose를 추정해야 한다. PhysVLA의 phase detector는 d_xy distance에 의존하므로 visible geometry가 없으면 phase 자체를 잘못 잡는다. 저자들은 이를 "post-hoc injection의 구조적 한계"로 명시. |

---

## 7. 다른 연구와의 관계

- **PINN / DeLaN / HNN / SymODEN [10-14]**: physics를 training-time loss/architecture로 통합. PhysVLA는 inference-time만 사용.
- **CBF / Neural Lyapunov [15, 50, 51]**: always-on certificate로 매 step QP를 풀어야 함. PhysVLA는 closed-form residual + gating으로 <1 ms.
- **MPPI / DMP / RMP / OSC [52-55]**: sampling-based refinement. PhysVLA는 sample 없이 단일 evaluation.
- **ForceVLA / FD-VLA**: training-time force-residual head 필요. PhysVLA는 backbone 무관, weight 접근 없음.
- **Temporal ensembling (ACT / Octo [29, 70])**: uniform EMA로 jerk는 줄지만 contact responsiveness 손실. PhysVLA는 phase-conditional smoothing.

---

## 8. 재현성 및 코드

- 코드 공개 명시 없음(논문 v1 기준 code_url 미공개).
- MuJoCo state vector, Franka Panda URDF, OpenVLA/OpenVLA-OFT/Force-VLA/Generalist-VLA weight, Robosuite Lift, Agilex Piper hardware 필요.
- 단일 hyperparameter(ε = 0.05 N·m, c = 0.05, β = 0.5, α = 0.92, δ_grasp = 6 cm)가 명시되어 있어 reimplementation 자체는 가능.

---

## 9. 비판적 분석

- 저자들의 "backbone-agnostic" 주장은 OpenVLA-7B 기반 4 variant에서만 검증되었다. 진정한 cross-backbone(예: SmolVLA, CogACT, π0)으로의 확장 실험은 없다.
- LIBERO-Spatial 외 LIBERO suite(Object/Goal/Long)는 평가 안 됨. Spatial은 phase predicate가 가장 잘 맞는 setting이므로 generalization claim이 약하다.
- Robosuite Lift 결과는 jerk만 강조되고 reward 차이는 11.06 vs 10.89로 marginal하다.
- Real-world 결과는 매우 좋지만 단일 task(sponge-on-plate)에 n=20에 불과.

---

## 10. 미래 연구 방향

저자들의 명시:
1. Deformable manipulation을 위해 phase predicate를 learned visual cue로 확장.
2. EL residual gating을 training-time soft policy prior로 통합.
3. On-board sensor(IMU, force/torque)에서 dynamics를 직접 소싱하여 simulator 의존성 제거.

추가 가능한 방향:
- Phase predicate의 자동 학습(예: VLM이 phase를 예측).
- Bimanual / mobile manipulation에서의 multi-effector phase 동기화.
- Sim-to-real에서 URDF 오차에 대한 robustness 정량 평가.

---

## 11. 한국어 용어 정리

| 영어 | 한국어 |
|------|--------|
| Vision-Language-Action (VLA) | 비전-언어-행동 모델 |
| Action head | 행동 헤드 |
| Phase-aware FSM | 단계 인식 유한 상태 기계 |
| Euler-Lagrange residual | 오일러-라그랑주 잔차 |
| Inertia-weighted blending | 관성 가중 혼합 |
| Kinodynamic consistency | 운동-동역학 일관성 |
| Capped blender | 상한 혼합기 |
| Training-free | 학습 불필요 |
| Plug-and-play | 즉시 적용형 |
| Frozen backbone | 동결 백본 |

---

## 12. 결론

PhysVLA는 "이미 학습된 VLA를 어떻게 더 잘 쓸 것인가"라는 실용적 질문에 단순하지만 효과적인 답을 준다. 4개 backbone 모두에서 zero regression으로 +13~17pp 절대 향상, ~10x jerk robustness, real-world transfer 모두 입증되어 LIBERO-Spatial / pick-and-place setting의 baseline corrector로서 충분한 가치를 가진다. 한계는 URDF/관성 calibration 의존성, 5% cap의 구조적 상한, phase predicate의 task-specific 설계지만, "physics를 inference-time에 조건부로 켠다"는 framing 자체가 향후 inference-time intervention 연구에 좋은 reference point가 될 것이다.

<!-- VERIFIED: pdf -->
