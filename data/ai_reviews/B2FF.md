# B2FF: Back to the Familiar Future — Failure Recovery for VLA Policies via Pre-Imagined Milestone Selection

> **한 줄 요약**: 동결(frozen)된 foresight-driven VLA 정책에 대해, 실행 전에 깨끗한 초기 관측으로부터 미래 이미지 milestone 뱅크를 미리 상상해 두고, 실패가 발생하면 recoverability-aware selector가 이 뱅크에서 milestone을 골라 action-only denoising의 고정된 visual subgoal로 사용하는 inference-time 복구 프레임워크.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- VLA 정책은 nominal trajectory에서 벗어나는 off-trajectory 편차가 발생하면, **실패 상태(off-trajectory observation)에서 직접 re-planning**하는 방식으로 복구를 시도 → 정책이 학습 시 경험하지 못한 unfamiliar state space에서 action을 생성해야 하므로 불안정
- 기존 복구 기법(FailSafe, RACER, Phoenix, DCDP 등)은 주로 **action-generation level**에서 개입 → 정책 가중치 업데이트, fine-tuning, 또는 실행 파이프라인 우회가 필요
- 작업 자체가 물리적으로 여전히 실행 가능(physically feasible)함에도, 정책이 unfamiliar observation에서 action을 잘못 생성하는 문제가 핵심

### 핵심 질문
- **저수준 action generator를 전혀 수정하지 않고**, 단지 visual condition만 바꿔서 frozen VLA가 안정적으로 복구하도록 만들 수 있는가?
- Foresight-driven VLA의 **intermediate future representation**을 복구 인터페이스로 활용할 수 있는가?

📌 [Figure 1 삽입] — B2FF 개요: 실패 상태에서 future를 re-predict하지 않고, 미리 상상해 둔 familiar future bank에서 milestone을 선택해 action-only denoising의 고정된 visual anchor로 사용.

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

B2FF는 frozen foresight-driven VLA(논문에서는 UD-VLA[20] 사용)의 두 가지 inference mode를 활용:

| Mode | 수식 | 사용 시점 |
|------|------|-----------|
| Joint subgoal–action generation | π_θ(v_t, a_t \| I, o_t) | Nominal 실행 |
| Action-only denoising with fixed image subgoal | π_θ(a_t \| I, o_t; v_t ← v*) | Recovery 실행 |

복구 시에는 future-image 토큰을 고정된 v*로 설정하여, frozen VLA가 unfamiliar observation에서 출발하더라도 familiar future anchor를 향한 action chunk를 생성하도록 강제.

### 2.2 Familiar Future Bank

- 에피소드 시작 시 깨끗한 초기 관측 o_0로부터 frozen VLA의 future-image marginal을 **재귀적으로 쿼리**:
  - v_0 = o_0, v_m ~ π_θ(v | I, v_{m-1}), m = 1, ..., M
- 이 과정에서 **action은 실행되지 않음** → 뱅크는 실패 상태가 아니라 깨끗한 초기 상태로부터 상상된 "familiar" visual anchor들로 구성
- 논문 설정: M = 12, deterministic 생성

### 2.3 Recovery-Mode Entry와 Candidate Set

- Controlled 평가에서는 주입된 perturbation 시점에 맞춰 recovery index f가 정해지고, online-triggered 변형은 proprioceptive history로부터 검출기 D를 학습해 임계값 τ_D를 넘으면 트리거: f = min{t | D(p_{1:t}) > τ_D}
- 고정된 offset 집합 Δ = {−1, 0, +1, +2, +4} (rollback, retry, skip-ahead 가설을 모두 포함)으로 local candidate set 구성:
  - C_f = {v_{f+δ} | δ ∈ Δ, 1 ≤ f+δ ≤ M} ⊂ B

### 2.4 Recoverability-Aware Milestone Selector

선택기 F_ϕ는 현재 실패 컨텍스트 (o_f, H_f, C_f)에서 각 후보에 점수를 부여:
- v* = arg max_{v ∈ C_f} F_ϕ(v | o_f, H_f, C_f)
- 구조: frozen visual tokenizer → 경량 projector → **Perceiver-style attention** [26]으로 candidate-context feature 요약 → MLP scoring head

---

## 3. Selector Training (3-Stage)

### Stage 1: Proxy Temporal Initialization
- TCN[27] 스타일 triplet 손실 L_TCN: VLA가 예측한 future image를 시간적으로 가까운 trajectory frame과 가깝게, 먼 frame과는 멀게
- L_gap: pair representation에서 discretized temporal-gap bin을 예측하는 cross-entropy
- 통합 목적: L_proxy = L_TCN + β_gap L_gap

### Stage 2: Supervised Warm-Start (Counterfactual Labels)
- 각 후보 c에 대해 v* = c로 frozen VLA를 rollout → 이진 성공 라벨 y_c 수집
- Candidate-wise BCE:
  - L_BCE = −(1/|C_f|) Σ_c [y_c log σ(s_c) + (1−y_c) log(1−σ(s_c))]
- Grouped ranking loss (Y_f > 0일 때): 같은 실패 컨텍스트 내 성공 후보가 실패 후보보다 높은 normalized score를 받도록
- 가중 합: L_sup = λ_BCE L_BCE + λ_group L_group

### Stage 3: One-Step Actor-Critic Fine-Tuning
- Maximum-entropy actor-critic[28,29] 영감
- Critic: 두 개의 head를 Huber regression[30]으로 라벨 y_c에 직접 회귀
  - L_Q = E[(w_f/|C_f|) Σ_c Σ_k Huber(Q_k − y_c)²]
- Actor: 후보를 discrete recovery choice로 보고 softmax(s)를 min_k Q_k 방향으로 시프트, lower-entropy α_low 사용
- Behavior-cloning 정규화로 grouped positive-candidate objective 재사용:
  - L_ft = L_Q + L_actor + λ_BC L_group

---

## 4. 실험 결과

### 4.1 Failure-Injected LIBERO (Table 1)

| Method | Avg | Object | Spatial | Goal | Long |
|--------|------|--------|---------|------|------|
| DP | 26.9 | 37.5 | 24.2 | 42.5 | 3.3 |
| DCDP | 28.1 | 37.5 | 20.8 | 47.5 | 6.7 |
| MolmoAct | 48.3 | 43.3 | 38.3 | 60.8 | 50.8 |
| SPR-VLA | 50.6 | 44.2 | 50.8 | 56.7 | 50.8 |
| UD-VLA (baseline) | 56.3 | 52.5 | 58.3 | 58.3 | 55.8 |
| **B2FF** | **74.0** | **69.3** | **66.0** | **73.3** | **87.3** |
| B2FF (online trigger) | 64.5 | 62.0 | 60.0 | 72.7 | 63.3 |

→ UD-VLA 대비 **+17.7 percentage point** 평균 향상. 다른 baseline family의 개선(DCDP +1.2, SPR-VLA +2.3)을 압도.

### 4.2 Standard LIBERO (Table 1)

| Method | Avg | Object | Spatial | Goal | Long |
|--------|------|--------|---------|------|------|
| UD-VLA | 91.3 | 93.2 | 94.2 | 88.8 | 88.8 |
| **B2FF** | **93.7** | 94.8 | 95.8 | 91.6 | 92.6 |

→ 외부 실패가 주입되지 않은 상황에서도 모든 suite에서 baseline을 능가 → 복구 모듈이 nominal 실행을 해치지 않고 오히려 자연 발생적 deviation도 완화함을 시사.

### 4.3 Selector Ablation (Table 2, LIBERO-Object)

**Training objective:**
| Variant | Grip. | Obj. | Lay. | All |
|---------|-------|------|------|-----|
| Scratch sup. | 62.5 | 67.5 | 60.0 | 63.3 |
| Proxy sup. | 67.5 | 67.5 | 60.0 | 65.0 |
| Weak RL | 67.5 | 70.0 | 60.0 | 65.8 |
| RL + Pos. BC | 67.5 | 70.0 | 60.0 | 65.8 |
| **B2FF (Huber+LE+BC)** | 67.5 | 72.5 | 67.5 | **69.3** |

**Selector input:**
| Variant | History | All |
|---------|---------|-----|
| Obs. only | 0 | 50.8 |
| + Cand. | 0 | 61.7 |
| + 1 hist. | 1 | 64.2 |
| + 3 hist. | 3 | 66.7 |
| **B2FF** | 4 | **69.3** |

→ Candidate-relative context와 충분한 history(4-step)가 핵심. Observation-only 점수는 18.5pp 낮음.

### 4.4 Real-World Experiment

- 600개 시연 trajectory로 학습한 실세계 VLA 정책 + LIBERO에서 초기화한 selector를 35개 실세계 recovery group으로 fine-tune
- 3개 작업(object stacking, pick-and-place, drawer closing with placement) × 2종 실패(gripper shift, target-object shift) × 총 **90 trials**
- B2FF가 **61.1% overall success**로 base UD-VLA 및 fixed-anchor baseline을 일관되게 능가

---

## 5. 핵심 기여

1. **Frozen VLA를 전혀 수정하지 않는 inference-time 복구**: action generator를 손대지 않고, visual condition selection만으로 복구를 달성.
2. **Pre-imagined familiar future bank**: 실패 상태에서 future를 re-predict하는 대신, 깨끗한 초기 상태에서 미리 상상해 둔 milestone들을 사용 → unfamiliar state에서의 불안정한 future 예측 회피.
3. **Recoverability-aware selector**: TCN proxy 초기화 + BCE/grouped 라벨 + Huber critic + lower-entropy actor + positive BC를 결합한 3-stage 학습 파이프라인.
4. **실증적 검증**: failure-injected LIBERO에서 +17.7pp, standard LIBERO에서도 일관된 향상, 실세계 transfer까지 입증.

---

## 6. 강점

- **Plug-and-play**: 어떤 foresight-driven VLA에도 weight update 없이 부착 가능. UD-VLA에 대해 검증.
- **해석 가능성**: 선택된 milestone이 사람이 보아도 의미 있는 "familiar future anchor"로 작용 → 정성 분석에서 명확한 복구 경로를 보여줌.
- **Nominal 실행 무해성**: 복구 모듈을 끼워도 standard LIBERO 성능이 오히려 +2.4pp 개선 → 자연 발생 편차에도 도움.
- **Selector 학습 신호 검증**: Figure 3b에서 selector 점수 quintile과 복구 성공률이 양의 상관 → 학습된 점수가 실제로 recoverability를 반영함.

---

## 7. 한계 및 약점

- **Recoverable deviation에 한정**: 작업 자체가 물리적으로 여전히 실행 가능한 경우만 다룸. Semantic 오류, irreversible 실패(workspace exit, severe occlusion, missing object, 잘못된 task understanding), VLA에 없는 skill이 필요한 경우는 처리 불가.
- **Foresight-driven VLA 요구**: 설정 가능한 future-image subgoal 인터페이스가 있는 VLA에만 적용됨. 순수 end-to-end action policy(예: standard DP, OpenVLA 등)에는 직접 적용 불가.
- **단일 pre-execution bank**: 한 번 만든 뱅크에서 한 milestone을 fixed recovery window 동안 사용 → adaptive bank 재구성, closed-loop re-selection, 반복 트리거 처리는 future work.
- **Online trigger 성능 갭**: Perturbation-aligned 시나리오(74.0%)와 online trigger 변형(64.5%) 사이에 약 10pp 차이 → 트리거 추정의 신뢰성이 병목.
- **카운터팩추얼 데이터 수집 비용**: 각 학습 실패마다 후보 milestone 전체에 대해 5-chunk rollout이 필요 → 학습 단계에서 추가 시뮬레이션 비용 발생.

---

## 8. 다른 연구와의 비교

| 측면 | DCDP[13] | RACER[10] | FailSafe[9] | Phoenix[11] | SPR-VLA[25] | **B2FF** |
|------|----------|-----------|-------------|-------------|--------------|----------|
| 개입 지점 | Action correction | Recovery policy | Reasoning+recovery | Self-reflection | Plan & rewind | **Visual condition selection** |
| VLA 업데이트 | No (training-free) | Yes | Yes | Yes | Yes (progress-aware) | **No (frozen)** |
| 실패 데이터 합성 | No | Yes | Yes | – | – | Counterfactual rollout만 (selector용) |
| Foresight 필요 | No | No | No | No | Yes (progress) | **Yes (future-image)** |
| LIBERO 실패 주입 평균 | 28.1 | – | – | – | 50.6 | **74.0** |

→ B2FF는 "action generator를 건드리지 않고 visual condition만 바꾼다"는 점에서 유일하며, 같은 frozen VLA(UD-VLA) 기반에서 가장 큰 개선폭(+17.7pp)을 확보.

---

## 9. 평가 조건 정리

- **Failure-injected LIBERO**: 4개 suite × 150 episode, 3종 실패 유형(gripper shift, object shift, object laydown)에 균형 분배. End-effector 정렬 오류, 물체 변위, 의도치 않은 자세/방향을 포함하되 workspace exit, severe occlusion, 물리적 파손, 실행 불가능한 지시는 제외.
- **Standard LIBERO**: 외부 실패 주입 없이 자연 발생 정책 실패만 평가.
- **Recovery protocol**: M = 12, Δ = {−1, 0, +1, +2, +4}, controlled에서는 perturbation 시점과 정렬, online 변형에서는 proprioceptive 검출기로 추정.
- **Selector test-time inference**: 단일 forward pass(후보 별 trial rollout 불필요).

---

## 10. 코드 및 재현성

- 코드: 본문 기준으로 공개 저장소 정보 명시되지 않음(논문 본문에서 GitHub URL 제공 없음)
- 핵심 하이퍼파라미터: M = 12, |Δ| = 5, recovery window W, decision horizon T, λ_BCE, λ_group, α_low (구체 수치는 Appendix 참조)
- Frozen 백본: UD-VLA[20] (arXiv 2511.01718), MOVQ visual tokenizer + Emu3
- Selector 구성: frozen visual tokenizer → 경량 projector → Perceiver-style attention → MLP scoring head

---

## 11. 향후 연구 방향

1. **Adaptive bank construction**: 실행 중 새로운 milestone을 추가하거나, 실패 컨텍스트에 맞춰 뱅크를 동적으로 재생성.
2. **Closed-loop re-selection**: 복구 window 내에서도 selector를 반복 호출해 milestone을 교체.
3. **Repeated-trigger handling**: 한 에피소드 내 여러 트리거가 발생할 때의 안정적인 동작.
4. **Stronger online trigger estimation**: Proprioceptive history 외에 시각/언어 신호를 결합해 perturbation-aligned 성능과의 갭(약 10pp) 축소.
5. **다양한 foresight-driven VLA로 일반화**: DreamVLA, CoT-VLA, FlowVLA 등으로 확장.
6. **Semantic 실패까지 확장**: 잘못된 grounding이나 task understanding 오류를 다루기 위한 high-level reasoning 모듈과의 결합.

---

## 12. 결론

B2FF는 "복구"를 **action을 다시 생성하는 문제가 아니라 visual condition을 선택하는 문제**로 재정의한다. Frozen foresight-driven VLA가 갖는 intermediate future representation을 인터페이스로 삼아, 깨끗한 초기 상태에서 상상한 familiar future bank에서 recoverability-aware selector가 milestone을 골라 action-only denoising의 고정 anchor로 사용한다. 그 결과 가중치 업데이트 없이도 failure-injected LIBERO에서 +17.7pp, standard LIBERO에서 +2.4pp 향상을 달성했고, 실세계 30개 episode에서도 61.1% 복구 성공률로 이전됨을 보였다. 이는 foresight-driven VLA의 시대에 **"무엇을 보고 다음 행동을 할 것인가"의 선택**이 그 자체로 강력한 복구 메커니즘이 될 수 있음을 시사한다.

<!-- VERIFIED: pdf -->
