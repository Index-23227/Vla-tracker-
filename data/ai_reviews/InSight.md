# InSight: Self-Guided Skill Acquisition via Steerable VLAs

> **한 줄 요약**: π0.5 VLA를 **primitive-level로 steerable하게 LoRA fine-tune**한 뒤, **VLM(Gemini 3 Flash)이 novel task에서 부족한 primitive를 식별·실행·평가·재학습**시키는 데이터 플라이휠로 **사람의 추가 데모 없이** 새 manipulation skill을 자율 획득. xArm에서 twist 92%, pour 96%, 14-primitive twist-then-pour composition 80% 성공, base pick-and-place 100% 유지.

---

## 1. 배경 및 동기

### 문제 정의: VLA의 skill 한계와 데이터 비용
- OpenVLA, π0.5 등 VLA는 학습 데이터에 포함된 skill의 경계를 넘기 어려움 — 새 skill마다 사람의 teleop 데모 + fine-tuning이 필요.
- 실세계 RL은 sample complexity와 안전성 문제로 제약, simulation RL도 보통 수천 trial 요구.
- 모티베이션 시나리오: 화성에서 돌을 **scoop**하도록 학습된 로봇이 dust storm 후 패널을 **sweep**해야 할 때, sweeping 데모가 없으면 실패.

### Key Insight — manipulation은 본질적으로 compositional
- Sweeping과 scooping은 approach·lowering primitive를 공유하고 lateral push만 다름.
- Block flipping은 grasp-and-lift를 pick-and-place와 공유하고 rotate primitive만 추가.
- 기존 VLA는 이런 primitive를 내부적으로 인코딩하지만 **task instruction에 얽혀 있어** 개별 호출이 불가능.

### 본 논문의 차별점
- LLM/VLM을 **test-time planner**로 쓰는 SayCan·VoxPoser·Code-as-Policies·Hi Robot과 달리, **학습된 policy 자체를 확장**.
- Stellar VLA·SkillsCrafter처럼 expert routing·subspace 분리가 아닌 **단일 VLA에 새 primitive를 흡수**시키는 data-centric continual learning.

---

## 2. 방법론 심층 분석 — 2단계 파이프라인

### 2.1 Stage 1: 자동 primitive segmentation으로 steerable VLA 만들기 (§3.1)

**Primitive의 정의 (§3.1.1)**:
- TAMP precondition 형식주의를 따라 각 primitive는 (precondition, effect) 쌍.
- **단일 dominant motion mode** (translation/rotation 한 축 또는 gripper transition) + natural-language label.
- 정책 vocabulary $V$ = VLA가 학습한 primitive label 집합. Plan $P=(p_1,\dots,p_n)$ 중 $p_i \notin V$가 **primitive gap**.

**자동 segmentation 절차 (§3.1.2, Fig. 2a)**:
1. **Task → primitive plan**: VLM에 task description과 예시 primitive set("close gripper", "lift upward" 등)을 주어 ordered primitive sequence 생성.
2. **Gripper transition으로 boundary 후보 찾기**: gripper command velocity의 open/close 전환점.
3. **Per-frame motion caption** 생성: `EE x,y,z,rx,ry,rz`, `dxy`, `dz`, `|drxy|`, `|drz|`, dominant-axis 태그 (`xy`/`z`/`rxy`/`rz`).
4. **VLM이 각 frame을 plan primitive에 매칭** → segment boundary 반환.
5. **Localized refinement**: EE delta change-point와 가장 빠른 시각적으로 명확한 frame을 정렬.

**VLA fine-tuning**:
- π0.5 backbone (Gemma-2B VLM + Gemma-300M action expert) — 다른 weight는 frozen, **LoRA** adapter만 학습.
- 각 segmented primitive = 1 training episode, primitive label이 language prompt.
- Action space에 **progress channel** $\in [0,1)$ 추가, 각 primitive 내 normalized timestep으로 supervise.
- Termination: progress > 0.95, EE motion이 auto-advance threshold 아래로, 혹은 OOD `move to` primitive의 경우 **VLM completion check** fire.

### 2.2 Stage 2: VLM-guided primitive 획득 루프 (§3.2, Fig. 2b)

**(1) Primitive gap 식별**:
- 새 task가 주어지면 VLM planner가 vocabulary $V$를 보고 primitive sequence + skill_gaps를 반환 (JSON).
- 제약: 각 gap은 **single-axis motion** — 다축이 필요한 task(예: tilt forward 후 tilt back)는 multiple gaps.

**(2) Gap parameterization**:
- 사전 실행 VLM call이 현재 scene 이미지를 보고 axis (translation `{dx,dy,dz}` 또는 rotation `{drx,dry,drz}`)와 signed magnitude (cm 또는 deg) 제안.
- Chained gap(예: pour-forward → tilt-back)은 직전 gap의 parameter를 prior로 전달해 paired motion의 axis/magnitude 일관성 보장.

**(3) Rollout**:
- 알려진 primitive는 steerable VLA가, gap primitive는 low-level controller가 VLM parameter로 실행.

**(4) Success 평가 및 데이터셋 통합**:
- Post-plan VLM oracle이 before/after scene 이미지 비교로 task success 판정.
- 성공한 새 primitive rollout만 VLA training set에 추가 → $V \cup \{p\}$로 retrain.
- 재학습 후에는 VLM이 high-level planner로만 동작, primitive 실행은 다시 VLA가 담당.

---

## 3. 시뮬레이션 실험 — LIBERO

### 3.1 Block Flipping from Pick-and-Place Only (§4.1, Fig. 3)
- 150 human teleop pick-and-place 데모 (block on side) → ~700 primitive episodes, 7 primitive types.
- 새 task: block을 peg가 위로 향하도록 flip — `rotate-block` primitive가 vocabulary에 없음 → primitive gap.

| Total rollouts | Acquired rotate primitives | Flip success |
|---|---|---|
| ~100 | 59 | **19%** |
| ~250 | 153 | **52%** |
| **479** | **246** | **75%** (upper bound 80%) |
| RL SAC (same budget) | — | **0%** (reach 23%, grasp 10%, never flip) |

→ Primitive bootstrapping이 동일 compute budget의 RL보다 **현실적으로 sample-efficient**.

### 3.2 Drawer Closing from Drawer Opening Only — OOD initial state (§4.2, Fig. 4)
- 50 drawer-opening 데모 → 3 primitives: (move-to-handle, close gripper, pull open).
- 새 task: 이미 열린 drawer를 **닫기** → base policy의 approach primitive는 `closed drawer` 초기 상태에서만 학습됨.
- **VLM completion check**가 진행 중인 primitive 완료를 주기적으로 평가 → OOD에서 approach가 일찍 끝나도록 트리거하고 새 `push drawer closed` primitive로 전환.
- 82 episodes에서 70 successful close-drawer primitives 수집 → unified VLA 재학습.
- **최종 25 trials 평가에서 100% 성공**, 동시에 open 능력 retain. (incorrect axis selection이 가장 흔한 실패 mode)

---

## 4. 실세계 실험 — xArm (§4.3)

### 4.1 Setup
- 6-DoF UFactory xArm, 224×224 scene + wrist RGB, EE pose + gripper state.
- Base: 50 human pick-and-place 데모 (top-grasp + side-grasp 병에 대한).
- Baselines: **CaP-X** (Code-as-Policies식 zero-shot VLM composition) + **π0.5** (동일 50 데모로 LoRA fine-tune, 새 primitive 없음).

### 4.2 Twist Cap Open / Pour Beans / Composition (Fig. 6, 25 trials/method)

| Task | π0.5 baseline | CaP-X | **InSight** |
|---|---|---|---|
| Twist cap open (end-to-end) | 0% | 32% | **92%** |
| Pour beans into bowl (end-to-end) | 0% | 16% | **96%** |
| Twist-then-pour composition (14 primitives) | 0% | 4% | **80%** |

- π0.5 baseline은 twist/pour 데모가 없어 완전 실패.
- CaP-X는 per-primitive reliability가 낮아 long-horizon에서 compounding failure로 4%까지 붕괴.
- InSight는 per-primitive 성공률이 일관되게 높아 14 primitives 연쇄에서도 80% 달성.
- 각 새 skill은 **20 acquired primitives**만으로 충분 (twist 23 trials, pour 31 trials, Table 1).

### 4.3 Base Skill Retention (Fig. 8, N=15)
- Twist + pour primitive 추가 후에도 **Top pick-and-place 100% / Side pick-and-place 100%** 유지.
- 흥미롭게도 π0.5 baseline은 side pick-and-place에서 86%로 떨어짐 (특정 측면 그립의 fine-tuning 영향).

### 4.4 효율성 (Fig. 7, Table 1)
- **Per-trial wall-clock**: InSight twist 55 s vs CaP-X 109 s, pour 124 s vs 207 s.
- Twist 한 skill 자율 획득에 robot 23.8 min + VLM 8.4 min = **40 min wall-clock**.

---

## 5. 실세계 실험 — Sweeping from Scooping (§4.4, Fig. 9)
- Mars 시나리오: scoop 데모만 있는 로봇이 sweep 수행.
- Scoop과 sweep은 approach·lower primitive를 공유 → 차이는 **마지막 contact motion** (sweep은 lateral push, scoop은 bin 아래로 descent).
- VLM이 lateral push를 missing primitive로 식별 → InSight 자율 획득 → **5/5 trial 성공**.
- Contact-rich, non-prehensile motion에도 framework가 작동함을 입증.

---

## 6. Implementation Details (Appendix A, B)

- **VLA**: π0.5; Gemma-2B VLM + Gemma-300M action expert. 두 weight만 LoRA로 학습, 나머지 frozen.
- **입력**: 2개 224×224 RGB view (scene + wrist) + EE pose + gripper state.
- **출력**: EE delta + absolute gripper command + progress channel ∈ [0,1).
- **Termination**: progress > 0.95 / EE motion < auto-advance / VLM completion check (OOD only).
- **VLM**: Gemini 3 Flash, strict JSON 응답 강제. 4가지 역할:
  1. Offline 데모 segmentation (frame-by-frame plan assignment).
  2. Planning + primitive gap flagging (`primitive_sequence`, `step_notes`, `skill_gaps`).
  3. Primitive-gap proposal (single-axis motion + signed magnitude).
  4. Oracle check (before/after image로 task & primitive 완료 판정).
- **PLAN_TASK system prompt 핵심 규칙**: (a) gap은 novel part만, (b) 동일 결과를 낼 수 있는 기존 primitive 우선, (c) 각 gap은 하나의 single-axis motion.

---

## 7. 강점 (Strengths)

1. **Zero target-skill human demos** — pick-and-place 데모만으로 twist 92%, pour 96%, sweep 100% 달성.
2. **Compositional generalization** — 별개로 획득한 twist + pour를 14-primitive composition으로 80% 성공.
3. **Base skill 100% 유지** — joint retraining (V ∪ {p}) 덕분에 catastrophic forgetting 없음.
4. **VLA-agnostic 설계** (저자 주장) — π0.5 외 다른 VLA로 교체 가능한 framework.
5. **현실적 효율성** — 새 skill 1개 획득 비용 40~85분 wall-clock, RL 대비 sample 효율 압도.
6. **OOD 일반화 메커니즘** — VLM completion check가 imperfect progress channel을 brigde, OOD initial state에서도 동작.

---

## 8. 한계 (Weaknesses & Limitations) — 저자 명시 + 비평

### 저자 명시
1. **Single-axis 제약** — primitive gap이 single translation/rotation으로 제한 → 복잡한 trajectory primitive 획득 불가.
2. **Success-only filtering** — 실패 분석이나 VLM feedback 미활용, 더 sample-efficient한 학습 여지.
3. **Human reset 필요** — 각 rollout마다 사람이 환경 reset.

### 비평
4. **VLM oracle 의존성** — Gemini 3 Flash의 success 판정 오류가 dataset에 직접 주입됨. False positive가 누적되면 vocabulary 품질 저하 가능.
5. **VLA 일반성 검증 부족** — "VLA-agnostic"이라 명시했지만 실험은 모두 π0.5에 한정. OpenVLA, RT-2 등 다른 backbone에서의 작동은 미검증.
6. **Primitive granularity 휴리스틱** — primitive 입도(granularity)는 VLM의 prompt-driven 결정, 자동 평가 지표 없음.
7. **LIBERO standard suite 미사용** — block flip / drawer close는 LIBERO 환경을 *활용*했지만 LIBERO-10/Spatial/Object/Goal benchmark에 비교 가능한 score 없음.

---

## 9. 다른 연구와의 위치 (Related Work Positioning)

| 연구 | VLM 역할 | 새 skill 학습 | Primitive 확장 |
|---|---|---|---|
| SayCan [12] | LLM planner + value function | ✗ | ✗ (fixed skill set) |
| VoxPoser [13] | LLM/VLM이 3D value map 생성 | ✗ | ✗ |
| Code-as-Policies [11], CaP-X [14] | LLM이 program 합성 | ✗ (test-time만) | ✗ |
| Hi Robot [24] | hierarchical VLA로 instruction 해석 | ✗ | ✗ |
| STEER [10], Steerable Policies [9] | dense language relabeling | ✗ | fixed primitive set |
| VLS [16] | VLM이 reward 합성 | △ (test-time steering) | ✗ |
| Stellar VLA [33] | expert routing in evolving knowledge space | ○ | continual but separated |
| SkillsCrafter [34] | semantic skill subspace 분리 | ○ | separated |
| **InSight (본 연구)** | **VLM이 gap 식별·실행·oracle·retraining 트리거** | **○** | **단일 VLA에 흡수, joint retrain** |

→ **차별점**: steerability를 control interface가 아닌 **skill acquisition foundation**으로 재정의.

---

## 10. 실용성 평가 (Practitioner's Perspective)

### 곧바로 활용 가능한 인사이트
- **Primitive label을 language prompt로 conditioning + progress channel** 추가 — 다른 VLA에도 1주일 내 적용 가능한 패턴.
- **Single-axis gap parameterization**: VLM이 7-DoF 전체 trajectory를 만들기 어려울 때 우회 전략으로 일반화 가능.
- **VLM completion check**: progress channel의 OOD failure를 보완하는 lightweight 메커니즘.

### 채택 시 고려사항
- π0.5 inference cost + Gemini 3 Flash 4가지 prompt → API 비용/latency 적지 않음 (twist trial 평균 21.9 s VLM call).
- Manual reset 자동화(real-to-sim-to-real, world model)가 production화의 핵심.

### 적용이 어려운 영역
- Force-control이나 precision insertion처럼 single-axis 가정이 깨지는 task.
- VLM oracle이 시각만으로 success를 판단하기 어려운 task (예: 정확한 토크/접촉력).

---

## 11. 향후 연구 방향 (Future Work)

- **Richer primitive acquisition**: VLM-generated waypoints, trajectory optimization, online RL로 multi-axis primitive 확장.
- **Failure analysis 통합**: 실패 rollout에서 VLM feedback → axis 재제안 → sample efficiency 향상.
- **자동 reset**: real-to-sim-to-real 또는 learned world model로 candidate rollout 사전 검증.
- **고차원 embodiment**: mobile manipulator, humanoid (full upper body 또는 dual-arm)로 확장.
- **VLA 다양화 검증**: OpenVLA, GR00T N1, π0.7 등 다른 backbone에서 framework 작동 확인.
- **Primitive vocabulary 품질 metric**: 인간 평가 없이 acquired primitive의 reusability 측정.

---

## 12. 종합 평가 및 결론

**점수: 8.5 / 10**

### 채점 근거
- **Novelty (9/10)**: VLM을 test-time planner가 아닌 **acquisition loop의 능동 agent**로 재정의한 발상이 강력.
- **Technical depth (8/10)**: 자동 segmentation 파이프라인 + progress channel + VLM completion check + single-axis gap 제약의 시스템 설계가 정교. 다만 LoRA + π0.5라는 익숙한 building block 위에 구축.
- **Experimental rigor (8.5/10)**: 시뮬레이션 2 + 실세계 4 task, 25-trial baseline 비교 (CaP-X, π0.5, RL SAC), per-primitive breakdown, time/cost breakdown, retention 측정까지 균형 잡힌 평가. 다만 VLA backbone 일반성은 미검증.
- **Practical impact (9/10)**: Mars sweeping 시나리오에서 5/5 성공, 14-primitive composition 80% 성공은 production 후보로 강력. Human demo 없이 새 skill 획득 가능하다는 약속이 lifelong robotics에 직접 적용.
- **Clarity (9/10)**: Stage 1/2 구분, 4가지 VLM 역할의 prompt 공개, ablation 명확.

### 한 줄 결론
> **InSight는 "VLA가 무엇을 못하는지"를 VLM이 식별하고 "그것만" 자율 학습시키는 가장 깔끔한 framework 중 하나**다. Steerability를 control interface에서 **skill acquisition의 기반**으로 격상시킨 관점 전환이 본 논문의 핵심 기여이며, single-axis 제약과 VLA-agnostic 일반성 검증이 후속 연구의 자연스러운 다음 스텝.

<!-- VERIFIED: pdf -->
