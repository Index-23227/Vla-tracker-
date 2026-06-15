# ProbeAct: Probe-Guided Training-Free Failure Recovery in Vision-Language-Action Models

> **한 줄 요약**: OpenVLA-OFT를 frozen black-box로 두고, **(i) layer-8 hidden state에서 PCA + 4-layer MLP가 K개 객체의 3D 좌표를 추출(Hungarian matching) → (ii) gripper width / end-effector pose / probe 좌표만으로 6-phase kinematic state machine이 empty/soft grasp · mid-transport drop · placement를 phase-별로 판별 → (iii) 반복 실패 좌표에 hierarchical CBF safe set h(x;c,r)=‖x-c‖²-r²을 동적으로 instantiate하고 closed-form QP projection으로 nominal action을 minimally 보정**하는 training-free 3-tier 개입 framework. LIBERO-plus 7-perturbation 총점 OpenVLA-OFT 69.6% → 74.1% (+4.5p), action endpoint 드리프트 34.9 cm vs probe 10.4 cm로 "feature는 살아있는데 action head만 무너지는" memory trap의 실증 증거 제시.

---

## 1. 배경 및 동기

### VLA의 OOD 취약성
- VLA(OpenVLA, OpenVLA-OFT, π0, RT-2, Octo)는 behavioral cloning 기반으로, lighting/camera viewpoint/initial state 변화에 success rate가 급락 [LIBERO-plus, LIBERO-pro].
- 핵심 진단(recent analyses): "Memory trap" — policy가 visual context로 dynamic하게 grounding하지 못하고 nominal trajectory를 blind reproduce.

### 핵심 가설: 표현은 살아있다, action head만 무너진다
- 저자 관찰(Section 4.3, Table 3): OpenVLA-OFT의 forward pass에서 두 quantity 측정 — hidden-state probe 좌표 vs gripper-close 시점 end-effector 좌표.
- 성공 episode: probe 3.4 cm / endpoint 7.8 cm (둘 다 정확)
- 실패 episode: probe **10.4 cm** / endpoint **34.9 cm** — **action endpoint만 ~25 cm drift**.
- 결론: 공간 정보는 intermediate representation에 보존되어 있다. failure는 feature → motor projection에서만 발생. ⇒ retraining action head 대신 **여전히 informative한 hidden state를 읽어서 minimal correction**으로 해결.

### 기존 inference-time recovery와의 차이
- FailSafe, Visual-Symbol Diagnosis, CycleVLA, "Affordances at Inference-time" 등 [8,12,13,14]: external VLM, additional visual symbols, 3D reprojection 등 **외부 인프라/sensing** 의존.
- ProbeAct: 모든 geometric reference를 VLA **자체** hidden state에서 추출 → 외부 센서 redundant.

---

## 2. 방법론 심층 분석

### 2.1 시스템 개요
Frozen VLA 옆에 inference-time loop 3-module:
1. **Perception**: Internal probe → 안정적 3D object track
2. **Logic**: Kinematic state machine → 물리 실행 실패 감지 (relative robot-object 동기화)
3. **Control**: Hierarchical CBF filter → nominal action a_t → 안전 action a_t^cbf로 minimal closed-form projection

### 2.2 Multi-Target Hidden-State Probe

**Feature 추출**:
- VLA layer 8의 16×16 spatial token → 4×4 grid로 average pooling → 각 cell 4096-dim
- PCA로 d_pca=1024 차원 축소

**Architecture**:
- 4-layer MLP φ : ℝ^1024 → ℝ^{K×3} × ℝ^K
- Hidden dims [2048, 1024, 512, 256], LayerNorm + ReLU + dropout(0.1)
- 출력: K개 (3D position p̂_k, sigmoid confidence c_k); c_k < 0.5는 discard

**Hungarian matching loss** (Eq. 1):
$$L = \frac{1}{N}\sum_{n=1}^{N} \min_{\sigma \in S_K} \left[\sum_{i=1}^{M^{(n)}} \|\hat{p}^{(n)}_{\sigma(i)} - p^{(n)}_i\|_2^2 + \beta \sum_{i=1}^K \text{BCE}(c^{(n)}_{\sigma(i)}, m^{(n)}_i)\right]$$

- M^{(n)}: 프레임 n의 GT object 수, K-slot 패딩 + mask m_i ∈ {0,1}
- 결과적으로 **permutation invariant** — slot k가 시간 t와 t+1에서 같은 객체를 의미하지 않음

**Inference-time identity tracking**:
- Frame-별 permutation invariance 문제 해결: **temporal axis online Hungarian matching**
- 각 timestep: 새 prediction을 기존 track에 pairwise Euclidean displacement 최소화로 assign
- Spatial gating: max plausible inter-frame velocity로 ID switching noise 차단
- Gating 초과 prediction → 새 track spawn

### 2.3 Object-Agnostic Kinematic State Machine

**6 phases**: Approach, Monitor, Grasping, PostGrasp + Placed event
**Signals**: gripper width q, end-effector pose e=(x_e,y_e,z_e), probe-tracked object positions
**No task-specific tuning**: relative state change + kinematic synchronization, threshold는 hardware tolerance만

#### PreGrasp → Grasping
- ‖e - p_obj‖ ≤ ρ_enter ⇒ 시작 상태 (target identity, p_enter, z_enter) 기록

#### Grasp 검증 (kinematic synchronization)
- Rolling window에서: Δz_e = z_e - z_enter, Δz_obj = z_obj - z_obj,enter
- **성공 조건**: q > ε_limit (gripper width 안정) AND Δz_e > τ_lift AND Δz_obj > τ_track
- 즉 **객체도 실제로 함께 올라갔는지** 검사 → empty-air grasp 차단

#### Grasping 실패 모드
1. **Hard empty grasp**: q ≤ ε_limit (gripper가 완전 닫힘 — 손가락 사이 객체 無)
2. **Soft empty grasp**: Δz_e > τ_lift이지만 Δz_obj ≤ τ_noise (kinematic decoupling — 객체 그대로)

#### Mid-transport drop (PostGrasp)
- q ≤ ε_limit 급변 + end-effector active motion 중 → drop 위치 = CBF zone 후보, 원본 target으로 recovery

#### Placement 검증 (3 동시 조건)
(a) q > q_hold + ε_release (열림), (b) ‖e_t - p_obj‖ > ‖e_{t-1} - p_obj‖ (diverging retreat), (c) 객체 좌표가 target 위치 tolerance 내 안정

### 2.4 Hierarchical Control Barrier Function Filter

**핵심 통찰**: 모든 minor error에 CBF zone을 영구적으로 추가하면 robot이 과도하게 제약되어 baseline SR 자체가 떨어진다 ⇒ **two-tier hierarchy**:

- **1차 실패**: stateless **push-back** 만 적용 → VLA가 self-correction 시도하도록 자유 허용
- **같은 region 반복 실패** ⇒ memory trap으로 진단 → **persistent CBF zone** instantiate

**Barrier function**:
$$h(x; c, r_{safe}) = \|x - c\|^2 - r_{safe}^2, \quad H = \{x : h(x) \ge 0\}$$

**1차 CBF 제약**: ∇h(x)^T u ≥ -γh(x)

**Minimal-intervention QP closed-form** (Eq. 2):
$$u_{filtered} = u_{vla} + \max\left(0, \frac{-\gamma h(x) - \nabla h(x)^T u_{vla}}{\|\nabla h(x)\|_2^2}\right) \nabla h(x)$$

여기 ∇h(x) = 2(x - c).

- VLA action이 이미 safe boundary 존중 ⇒ projection = 0 ⇒ **exact identity mapping** (baseline 보존)
- Zone은 (i) end-effector가 true target proximity ρ_clear 진입 시 또는 (ii) Placed event 발생 시 dynamic flush
- **Grasping/Transporting phase에서는 filter bypass** — 객체와 물리적 접촉 방해 금지

> ❓ **예상 질문**: 왜 PreGrasp에서만 filter on?
> **답변**: Grasping 중 CBF 활성화 시 손-객체 접촉이 safe-set violation으로 잘못 분류되어 grasp 자체가 막힘. 물리 접촉이 본질적으로 필요한 phase는 명시적으로 bypass.

### 2.5 Multi-Step Task Support
- "put both moka pots on the stove" 같은 multi-pick: language에서 (both/two/and) keyword parsing → N (pick, place) 분해
- Placed 후: (a) blacklist에 방금 놓인 객체 추가, (b) probe tracker reset, (c) 모든 CBF zone clear → 다음 pick은 fresh

---

## 3. 데이터 전략

- **추가 robot demonstration 無**, **VLA fine-tune 無**
- Probe 학습 데이터: 50,000 (hidden-state, object-positions) pair, baseline VLA rollout으로 자동 수집 (simulator의 obj_of_interest list로 GT 라벨)
- 200 epoch, AdamW, batch 512, cosine LR

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|---|---|
| Base VLA | OpenVLA-OFT (7B, LLaMA-7B backbone), frozen |
| Probe input | layer 8, 4×4 spatial avg pool, PCA d=1024 |
| Probe hidden | [2048, 1024, 512, 256], LN+ReLU+Dropout 0.1 |
| Confidence threshold | 0.5 |
| Probe training | 50k pairs, 200 epochs, AdamW, batch 512, cosine LR |
| ID tracking | Online Hungarian + spatial gating |
| State machine phases | Approach / Monitor / Grasping / PostGrasp / Placed |
| CBF tier 1 | Stateless push-back |
| CBF tier 2 | Persistent h(x;c,r)=‖x-c‖²-r² with QP projection |
| Compute | 2× NVIDIA RTX PRO 6000 (Blackwell) GPU |
| Step overhead | +6 steps (~5%) on tasks baseline already succeeds |

---

## 5. 실험 설계 및 평가 프로토콜

**Benchmark**: LIBERO-plus [5] — 7 perturbation categories
1. Camera Viewpoints
2. Robot Initial States
3. Language Instructions
4. Lighting
5. Background Textures
6. Sensor Noise
7. Object Layout

**Metric**: LIBERO 내장 goal predicate 기반 success rate

**Comparison**: 7개 SOTA VLA (OpenVLA, NORA, WorldVLA, UniVLA, π0, π0-Fast, RIPT-VLA, OpenVLA-OFT) vs ProbeAct (= OpenVLA-OFT + framework)

---

## 6. 실험 결과 심층 분석

### 6.1 LIBERO-plus Main Results (Table 1, SR %)

| Model | Camera | Robot | Language | Light | Background | Noise | Layout | **Total** |
|---|---|---|---|---|---|---|---|---|
| OpenVLA | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| NORA | 2.2 | 37.0 | 65.1 | 45.7 | 58.6 | 12.8 | 62.1 | 39.0 |
| WorldVLA | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| UniVLA | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 43.9 |
| π0 | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| π0-Fast | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| RIPT-VLA | 55.2 | 31.2 | 77.6 | 88.4 | 91.6 | 73.5 | 74.2 | 68.4 |
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | **69.6** |
| **ProbeAct (ours)** | **63.8** | **40.3** | **82.0** | **93.6** | **93.5** | **76.8** | **80.9** | **74.1** |

**핵심 관찰**:
- 전체 1위 (+4.5p over OpenVLA-OFT)
- **Spatial shift 카테고리(Camera, Robot, Layout)에서 가장 큰 게인**: Camera +7.4p, Robot +8.4p, Layout +6.7p
- 저자 해석: "Phantom Grasp" pathology — 정확한 target identification은 되지만 motor는 memorized trajectory로 collapse. ProbeAct는 절대 좌표가 아닌 **relative kinematic synchronization** 평가하므로 spatial offset에 정확하게 개입.

### 6.2 Fine-tuned VLA Generalization (Table 2, Robot Initial States, SR %)

LIBERO-plus 저자들이 공개한 OpenVLA-OFT-mixdata (LIBERO + LIBERO-plus 혼합 fine-tune)에 ProbeAct 추가:

| Method | Spatial | Object | Goal | Long-10 | **Avg** |
|---|---|---|---|---|---|
| OpenVLA-OFT-m | 30.6 | 23.9 | 20.8 | 36.6 | 28.0 |
| **+ ProbeAct** | **32.6** | **30.7** | **25.7** | **39.7** | **32.2** |
| Δ | +2.0 | +6.8 | +4.9 | +3.1 | **+4.2** |

**함의**:
1. Mixdata-style 학습은 memorization 문제를 **완화하지만 제거하지 못함** — perturbed initial state로 학습해도 여전히 off-target grasp 발생
2. ProbeAct는 data-side remedy와 **직교** — fine-tune 위에 누적 가능

### 6.3 Action-Output Drift Analysis (Table 3, cm, 300 LIBERO-plus episodes)

| Subset | Probe error | Endpoint error |
|---|---|---|
| All | 6.9 cm | 23.6 cm |
| Success | 3.4 cm | 7.8 cm |
| **Failure** | **10.4 cm** | **34.9 cm** |

**이 논문의 thesis 직접 증명**: 실패 시 hidden state는 여전히 ~10 cm 정확도를 유지하는데 action endpoint는 35 cm 떨어진 곳으로 commit. ⇒ feature는 살아있다, action head만 무너진다 ⇒ retraining action head보다 **read still-informative hidden state + minimal correction**이 옳은 처방.

### 6.4 Probe Layer & Pooling Ablation (Table 4, R²)

| Layer | img-spatial | img-mean | last | lang-mean |
|---|---|---|---|---|
| **8** | **0.968** | 0.926 | 0.815 | 0.869 |
| 12 | 0.958 | 0.919 | 0.856 | 0.879 |
| 16 | 0.947 | 0.919 | 0.877 | 0.894 |
| 20 | 0.945 | 0.918 | 0.886 | 0.921 |
| 24 | 0.938 | 0.912 | 0.875 | 0.910 |
| 28 | 0.934 | 0.912 | 0.861 | 0.914 |

- **img-spatial이 모든 layer에서 dominant** — 2D layout 보존이 본질적
- Layer 8 (shallow-mid)이 최고 — geometric detail이 deeper layer로 갈수록 semantic/action 표현으로 abstract되어 사라짐
- 깊은 layer일수록 lang-mean이 상대적으로 개선 — task-conditioned representation으로 shift

### 6.5 Step Efficiency (Table 5)

| Subset | # Tasks | Baseline | ProbeAct | Δ |
|---|---|---|---|---|
| Both succeed | 1,643 | 114 | 120 | +6 |
| ProbeAct rescued from failure | 151 | 600 (timeout) | **197** | **-403** |
| Both fail (timeout) | 724 | 600 | 600 | 0 |
| **All (avg)** | 2,591 | 275 | **255** | **-20** |

- **+5% overhead** on already-succeeding tasks → hierarchical activation policy의 효과 입증
- **rescue 시에는 -403 step** (timeout → 197) → 개입이 targeted, exhaustive search 아님
- 전체 평균: ProbeAct가 baseline보다 **net 20 step 적게 사용** (rescue에서 절약된 시간 > overhead)

---

## 7. Ablation 분석

본문이 명시한 ablation은 Layer/Pooling (§6.4)뿐이지만, design choice들의 implicit ablation은 다음과 같다:

| Design Choice | 제거 시 예상 결과 |
|---|---|
| Hierarchical 2-tier CBF | tier-1 skip 시 baseline-restrictive (모든 zone 영구화 → baseline SR 손실), tier-2 skip 시 반복 실패 회피 불가 |
| PreGrasp만 filter on | Grasping/Transport에서 활성화 시 물리 접촉 자체가 violation → grasp 실패 |
| Online Hungarian tracking | Permutation invariance 직접 사용 시 ID switching → 잘못된 target에 zone instantiate |
| Spatial gating | False positive prediction이 새 track spawn → 다수 track으로 시스템 confused |
| Object-agnostic (no task-specific threshold) | Task-specific tuning 필요 시 generalization 손실 |

저자가 **명시적 ablation table은 한정적** — 이는 한계로 지적할 수 있음.

---

## 8. 관련 연구 비교

| 방법 | 외부 sensing | VLA 가중치 수정 | Failure 감지 | Recovery 메커니즘 |
|---|---|---|---|---|
| FailSafe [13] | Reasoning module | ✗ | LLM-based | LLM-prescribed |
| Visual-Symbol Diagnosis [8] | External VLM + visual symbols | ✗ | Symbolic | Visual-symbol guided |
| CycleVLA [14] | ✗ | ✓ (학습) | Subtask backtrack | MBR decoding |
| Affordances-at-Inference [12] | ✗ | ✗ | Affordance grounding | Affordance constraint |
| Recovery RL [29] | Sim | ✓ (학습) | RL signal | Learned recovery policy |
| TAMP [27] | Domain model | ✗ | Plan validation | Replan |
| **ProbeAct** | **無** | **無 (frozen)** | **Object-agnostic kinematic FSM** | **Closed-form CBF QP** |

**유일성**:
- Probe-based latent perception을 **active runtime sensor**로 재해석 (기존 probing은 retrospective analytical tool)
- CBF의 safe set을 a priori specification이 아닌 **dynamic online observation으로 construct**
- Classical CBF safety theory와 modern end-to-end foundation model의 첫 결합

---

## 9. 한계 및 미해결 문제

### 저자 명시
1. **Simulation only** (LIBERO-plus). Real-world sensor noise, soft-body deformation, contact friction 미검증
2. **Multi-step pick은 keyword parsing** (both/two/and) — 일반 instruction parsing 부재

### 추가 비판점
3. **OpenVLA-OFT 단일 backbone**만 평가 — π0 (flow-matching), CogACT, RDT 등에서 probe 효과 미검증. Layer 8 + 4×4 spatial pooling이 다른 아키텍처에서 적용 가능한지 불명.
4. **Probe 학습이 simulator-only** — Real-world에서는 obj_of_interest GT 위치 unavailable. Sim-to-real transfer를 위한 self-supervised probe training 부재.
5. **CBF radius r_safe**의 hyperparameter 민감도, sphere 형태 safe set의 일반화 (장애물이 비대칭일 때) 미평가.
6. **Multi-object scene (K)의 상한** 명시 없음 — K가 크면 Hungarian 비용 O(K³). 5+ objects 환경 미검증.
7. **"Same region failure"의 정의 threshold** (몇 cm? 몇 번 반복?) 본문에 정확한 수치 부재.
8. **Probe training cost** (50k rollout + 200 epoch) — 다른 task suite로 옮길 때 재학습 필요? Cross-task transfer 미평가.
9. **Action endpoint 정의의 한계** — "policy가 처음 gripper 닫는 순간"의 end-effector 위치. 그 직전 motion 변화는 분석 안 됨.
10. **CBF projection이 translational subspace u ∈ ℝ³** 만 다룸 — orientation correction은 부재. Pick orientation 오류는 복구 불가.

---

## 10. 총평

| 항목 | 평가 |
|---|---|
| **Novelty** | ★★★★☆ — Probe-as-runtime-sensor + dynamic-online CBF 결합은 ProbeAct가 처음. Classical safety theory와 foundation model의 깔끔한 연결. |
| **Technical depth** | ★★★★☆ — Hungarian matching, CBF, kinematic FSM 모두 잘 알려진 기법이지만 통합 설계와 hierarchical 2-tier 전략은 thoughtful. Eq. 2 closed-form projection 깔끔. |
| **Experimental rigor** | ★★★☆☆ — 7-perturbation LIBERO-plus + mixdata generalization + drift analysis + layer ablation으로 thesis는 확실히 뒷받침. 단 backbone 단일, simulation only, sphere safe set 가정 등 ablation/sensitivity는 미흡. |
| **Practical impact** | ★★★★☆ — Training-free, 외부 sensor 無, +5% step overhead로 OOD robustness 4.5p 개선. Plug-and-play wrapper 형식이라 즉시 배포 가능. |
| **Writing quality** | ★★★★☆ — "Action endpoint drift" 분석으로 thesis를 empirical하게 prove하는 방식이 매우 설득력 있음. 6-phase FSM은 figure로 보강 필요. |

**강점**:
1. **Memory trap의 empirical decomposition** (probe 10.4 cm vs endpoint 34.9 cm) — "feature는 살아있다"를 정량 입증한 첫 사례
2. **Training-free + frozen-weight + no external sensor** 3박자
3. **Hierarchical CBF의 baseline 보존 보장** (projection = identity when nominal action is already safe)
4. **Mixdata fine-tune 위에도 누적** (직교성)
5. **Step efficiency net positive** (-20 step) — overhead가 rescue 이득보다 작음

**약점**:
1. Simulation-only, single backbone, single safe-set geometry
2. Probe training이 simulator GT에 의존 (sim-to-real gap)
3. Translational correction만 — orientation은 미해결
4. Ablation table이 layer/pooling 1개만, 2-tier hierarchy / spatial gating / phase-bypass 등 핵심 design choice의 정량 ablation 부재

---

## 11. 예상 날카로운 세미나 질문

| # | 질문 | 핵심 답변 요점 |
|---|---|---|
| 1 | "Probe가 layer 8에서 R²=0.968 인데 왜 baseline VLA는 그 정보를 활용하지 못하는가?" | 핵심 thesis. Layer 8까지는 spatial info 보존 (정확히 표현됨), 그러나 layer 8 → action head로 가는 mapping이 BC trajectory에 overfit. ProbeAct는 layer 8을 **bypass route**로 사용. |
| 2 | "CBF radius r_safe와 tier 전환 threshold('same region failure')의 정확한 값은?" | 본문 본체에는 명시 없음. Appendix 또는 코드 필요. Robustness 분석 부재 — 큰 한계. |
| 3 | "Layer 8이 OpenVLA-OFT(LLaMA-7B 32 layer)에 특화된 것 아닌가? π0의 flow-matching에서는?" | 정확한 비판. 다른 backbone에서는 'spatial peak layer'를 재탐색해야 함. Table 4의 layer sweep이 다른 모델에서도 필요. |
| 4 | "Probe 학습에 simulator obj_of_interest GT를 쓰는데 real robot에서는?" | 저자 limitation으로 명시. Future work에서 self-supervised probe / DINO-based pseudo-label 필요. |
| 5 | "soft empty grasp 감지 threshold (τ_lift, τ_track, τ_noise)는 hardware-specific 인데 generalization은?" | "Embodiment-specific hardware tolerance"로 정당화 — 한 번 robot 측정으로 모든 task에 적용 가능 주장. 단 robot 변경 시 재측정 필요. |
| 6 | "Multi-step pick의 keyword parsing (both/two/and)는 robust한가? 'place the two red blocks next to the green one'에서?" | 본문 핵심 한계. NLP 기반 task decomposition 부재 — LLM/parser 통합 future work. |
| 7 | "CBF가 PreGrasp에서만 active. 만약 grasp 도중 객체가 굴러가서 다른 위치에 떨어지면?" | Mid-transport drop은 별도 감지 (gripper width 급변 + EE motion 중). Drop 위치가 CBF zone으로 instantiate되어 다음 사이클 회피. 단 정확한 fall trajectory 예측은 안 함. |
| 8 | "Phantom Grasp pathology의 근본 원인이 BC training distribution shift라면 ProbeAct가 fundamental fix인가, symptom relief인가?" | Symptom relief 자체로 인정. 저자도 'plug-and-play orthogonal to training pipelines'로 위치 잡음. Pre-training methodology 개선과 stack 가능 (Table 2가 증거). |
| 9 | "Sphere safe set h=‖x-c‖²-r²이 너무 단순. 비대칭 장애물 (예: 좁은 틈)에서는?" | 본문은 'simple memory trap escape'만 목표. Implicit shape (NeRF/SDF-based CBF) 같은 후속 work 필요. |
| 10 | "Step overhead +6 step (5%)가 진짜 negligible? 50 Hz inference면 +0.12s/episode인데, real robot에서 timing-critical task는?" | LIBERO 같은 slow-task에서는 negligible. 빠른 dynamic task (catching, throwing)에서는 검증 필요. |

---

## 12. 결론

ProbeAct는 **"VLA의 OOD failure는 perception이 아니라 action head의 memory trap"** 이라는 핵심 진단을, hidden-state probe vs action endpoint의 정량 비교(10.4 cm vs 34.9 cm)로 empirically prove하고, 이를 직접적으로 해결하는 **training-free 3-tier intervention loop** (probe → kinematic FSM → hierarchical CBF QP)을 제안한다. LIBERO-plus 7-perturbation 총점 OpenVLA-OFT 69.6% → 74.1% (+4.5p), 특히 spatial-shift 카테고리(Camera +7.4p, Robot +8.4p, Layout +6.7p)에서 큰 개선을 보이며, mixdata fine-tune 위에도 +4.2p 누적된다. Step overhead는 ~5%에 그치고, rescue task에서는 600-step timeout을 197-step으로 줄여 net step efficiency가 positive(-20 step). 핵심 기여는 (1) probe 출력을 retrospective analytical tool이 아닌 **active runtime sensor**로 재해석, (2) CBF safe set을 a priori가 아닌 **dynamic online failure observation**으로 instantiate하며, (3) classical safety theory와 modern foundation model의 첫 결합. 한계는 simulation-only, single backbone (OpenVLA-OFT), sphere-only safe set, orientation correction 부재, probe training의 simulator GT 의존성. 그러나 plug-and-play orthogonality 덕분에 향후 pre-training 개선·real-robot deployment·multi-modal safe set과 stack될 여지가 크며, "hidden state는 살아있고 action head만 무너진다"는 진단은 후속 inference-time intervention research의 기준점이 될 것이다.

<!-- VERIFIED: pdf -->
