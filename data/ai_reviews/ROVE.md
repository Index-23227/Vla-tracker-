# ROVE: Unlocking Human Interventions for Humanoid Manipulation via Reinforcement Learning

> **한 줄 요약**: Humanoid VLA post-training에서 인간 개입(intervention) 데이터가 expert가 아닌 **mixed-quality**라는 점을 정면으로 다룬 RL 프레임워크. Qwen3-VL-4B + flow-matching DiT actor를 동결한 채, **takeover adaptation 구간을 음수 reward 경계로 분리**하고, 로봇 rollout + 인간 개입 + cross-embodiment human ego-video를 함께 학습하는 **state-value critic V(s)** 를 expectile regression(OVE)으로 최적화한다. 실 IRON-R01-1.11 humanoid에서 Erase the whiteboard 45.0% → 80.0% (iter 3), Put the bread into the toaster 56.7% → 86.7%.

---

## 1. 배경 및 동기

### 1.1 문제: Humanoid manipulation을 위한 VLA post-training의 새로운 어려움
- VLA 모델은 grounding된 로봇 정책 학습의 유망한 토대이지만, 지금까지는 주로 **parallel-jaw 그리퍼를 가진 로봇 팔**에서 입증되어 왔음 (π0, π0.5, GR00T N1 등).
- Humanoid + dexterous hand로 확장 시: whole-body kinematics, 손 자유도 폭증, 누적 오차 취약성, contact-rich 동역학 → **offline demonstration만으로는 충분치 않음** → deployment 단계의 인간 개입이 본질적으로 필요.

### 1.2 기존 human-in-the-loop 가정의 붕괴
- HIL-SERL, RECAP, HG-DAgger 등은 모두 **"인간 개입 = 최적 교정"** 이라는 강한 가정에 의존.
- 로봇 팔 + 3D mouse / leader-follower 환경에서는 이 가정이 어느 정도 성립.
- Humanoid + dexterous hand teleoperation은 **retargeting 부정확성, 힘/촉각 피드백 결여, joint limit, 조작자 숙련도**가 동시에 작용해 **takeover 직후의 hesitation, retraction, 적응 동작**이 trajectory에 상당량 섞임 → 이를 그대로 expert supervision으로 사용하면 정책이 망설이는 행동을 모방하게 됨 (논문에서 HG-DAgger가 baseline 아래로 떨어진 결과로 입증).

### 1.3 ROVE의 입장
- 개입 데이터는 본질적으로 **mixed-quality**라고 인정하고, suboptimal 구간과 회복(recovery) 구간을 분리한 뒤, **state-value critic**으로 회복 가능성이 높은 행동을 우선 부각시키는 RL post-training 파이프라인 설계.

---

## 2. 방법론 심층 분석

### 2.1 전체 구조 (Fig. 3)

| 단계 | 입력 | 출력 | 학습 여부 |
|---|---|---|---|
| (a) Human-in-the-loop 데이터 수집 | Autonomous rollout + supervisor 트리거된 takeover | (rollout, adaptation, recovery) 3-stage trajectory | 데이터 수집 |
| (b) Critic 사전학습 | 대규모 robot + EgoDex 인간 영상 | V_ϕ(s_t) | MC return regression (Eq. 3) |
| (c) Critic fine-tuning | 작업별 robot + 인간 개입 + 인간 ego-video | V_ϕ(s_t) | OVE: H-step TD bootstrap + expectile τ=0.7 (Eq. 4-5) |
| (d) Actor fine-tuning | (c)의 advantage 라벨 | π_θ(a_{t:t+H-1} | s_t, I_t) | Advantage-conditioned CFGRL behavior cloning |
| (e) 반복 | 새 πθ를 배포해 (a)부터 재수집 | K=3 iterations | Iterative offline RL |

### 2.2 3단계 episode 분해 (핵심 설계)

각 episode는 다음과 같이 잘림:
1. **Autonomous VLA rollout** — VLA가 끝까지 가거나 supervisor가 near-failure로 판단해 멈춤.
2. **Intervention-adaptation** — VR + mocap operator가 humanoid 현재 자세에 자기 몸을 맞추는 transient 구간. **여기서 hesitation / retraction / 재정렬이 발생**.
3. **Intervention-recovery & completion** — operator가 본격적으로 task를 회복시키고 완료.

### 2.3 Reward 설계 (Eq. 2)

```
r_t = 0              if t = T 그리고 episode 성공
    = C_fail         if t = T 그리고 autonomous rollout 실패
    = C_fail         if t = t_r (adaptation stage의 종료)
    = -1             otherwise
```

- **결정적 차이**: 페널티 C_fail을 intervention 시작 시점이 아니라 **adaptation 종료 시점 t_r**에 배치 → adaptation 구간의 망설이는 행동을 recovery label에 섞지 않음.
- γ = 1로 설정 → cumulative return이 **time-to-completion**을 자연스럽게 encode (짧게 성공할수록 return 큼). 라벨은 [0, 1]로 정규화.
- C_fail = -500 (fine-tuning) / -평균 episode 길이 (pretraining).

### 2.4 Optimistic Value Estimation (OVE)

Mixed-quality 데이터의 평균을 그대로 fitting하면 critic이 과도하게 보수적이 됨. ROVE는 **state value V(s)** 만 학습 (Q(s,a)가 아님 — cross-embodiment human video는 robot과 action space를 공유하지 않음).

**TD target (H-step bootstrap, H=16)**:

```
V̂_t = Σ_{i=t}^{t+H-1} γ^{i-t} r_i + γ^H V_ϕ̄(s_{t+H})
```

**Expectile loss (τ = 0.7)**:

```
L_OVE(ϕ) = E[ |τ - 𝟙{V̂_t - V_ϕ(s_t) < 0}| · (V̂_t - V_ϕ(s_t))^2 ]
```

- τ = 0.5이면 평균 회귀(MC), τ → 1로 갈수록 **in-distribution optimistic 통계량**을 학습.
- OOD action을 query하지 않기 때문에 offline RL의 overestimation 위험 회피.

### 2.5 Cross-embodiment human experience video
- Task별 **180개 ego-centric 인간 영상** (head-mounted capture, 성공/실패 절반씩) 추가.
- **Critic에만 사용** (D_k^critic 에만 합쳐짐). Actor는 D_k^actor (로봇 데이터만)로 학습.
- 효과: partial-progress / near-failure / recovery 상태에 대한 supervision 보강. Fig. 5에서 인간 비디오 없는 critic은 부분 erase 상태를 과대평가하지만, 인간 비디오 합치면 task progress를 더 정확히 반영.

### 2.6 Advantage-conditioned policy extraction (Algorithm 1)

```
A_ϕ(s_t, a_{t:t+H-1}) = Σ γ^{i-t} r_i + γ^H V_ϕ(s_{t+H}) - V_ϕ(s_t)
I_t = 𝟙{A_ϕ > η_k}        # η_k = advantage 분포의 70th percentile
max_θ E[log π_θ(a_{t:t+H-1} | s_t, I_t)]
```

CFGRL classifier-free guidance: positive/negative condition으로 두 번 디코딩 후
`v_cfg = v_neg + β(v_pos - v_neg)` 로 결합 → high-advantage 행동 쪽으로 sampling 편향.

---

## 3. 아키텍처 및 학습 세부

### 3.1 Actor (VLA policy)
- **Backbone**: Qwen3-VL-4B-Instruct (frozen). Final-layer hidden dim 2560.
- **Action head**: Flow-matching **DiT** decoder, 연속 action chunk 출력.
- **Action space**: 50-dim, IRON-R01-1.11 humanoid의 body joint + dexterous hand 결합.
- **Action chunk horizon**: 학습 시 50, 추론 / critic TD / advantage 라벨 시 **H = 16** (16 스텝 실행 후 replan).
- **State regularization** (학습 only):
  - Action head dropout: full-state dropout p=0.3.
  - Gaussian noise: p=0.4, base σ=0.01 (head/waist 작게, arm/hand/EE 단위 스케일).
  - 시각 토큰엔 적용 X — proprio overfitting 방지 목적.

### 3.2 Critic
- VLM backbone: **VLAC checkpoint**의 transformer **layer 23** intermediate representation (hidden dim 2048).
- Value head: lightweight transformer + scalar 회귀.
- VLM은 동결, value head만 학습.

### 3.3 학습 설정
- **Critic**: 8000 steps, 8 GPUs (140GB), per-device batch 64, LR 1e-4 (iter 0) / 1e-5 (iter ≥ 1). TD target은 [-1, 0]으로 정규화 후 expectile τ=0.7.
- **Actor**: 8000 steps, 8 GPUs, per-device batch 16, LR 1e-4 / 1e-5.
- **K = 3 iterations** of rollout-intervention.

### 3.4 데이터 규모 (Table 2)

| Task | Demo episodes | Iter 1 | Iter 2 | Iter 3 | Intervention fraction |
|---|---|---|---|---|---|
| Erase the whiteboard | 225 | 82 | 71 | 79 | 25.50% |
| Put the bread into the toaster | 220 | 97 | 69 | 104 | 4.53% |

- 추가로 **task당 180 ego-centric human video** (critic 전용).

---

## 4. 실험 결과

### 4.1 실 humanoid (IRON-R01-1.11) 성능 (Fig. 4)

| 설정 | Erase the whiteboard | Put the bread into the toaster |
|---|---|---|
| SFT (demo-only / iter 0) | 45.0% | 56.7% |
| ROVE iter 3 | **80.0%** | **86.7%** |
| HG-DAgger | SFT 이하 (망설임 동작 모방) | SFT 이하 |
| Filtered BC | ROVE 이하 | ROVE 이하 |
| RECAP | ROVE 이하 | ROVE 이하 |
| 더 많은 demo로 SFT scaling | 부분적 향상 | 부분적 향상 (그러나 failure-recovery 동작은 거의 안 나옴) |

### 4.2 Iteration별 monotonic 향상
- 3회 iteration 동안 두 task 모두에서 성공률이 단조 증가. **Closed-loop improvement**: 더 나은 policy → 더 정보적인 experience → critic의 advantage 신호가 더 sharp해짐.

### 4.3 Ablation: Value-label 구성 (Table 1, Erase the whiteboard)

| 설정 | 성공률 | Δ |
|---|---|---|
| ROVE (default, H=16, t_r = adaptation 종료) | 80% | — |
| Critic horizon H=50 | 65% | **-15%** |
| t_r = intervention 시작 (adaptation 포함) | 50% | **-30%** |

→ **t_r의 위치가 critic horizon보다 더 큰 영향**. Adaptation 구간을 recovery 라벨에 섞으면 OVE의 in-distribution optimism이 noisy한 takeover 행동을 **misleadingly optimistic**한 라벨로 증폭시킴.

### 4.4 Critic 품질 분석
- **Fig. 5**: 인간 비디오 없이 학습한 critic은 erasing이 절반만 끝난 상태에 과대평가. 인간 비디오 추가 시 task progress를 더 정확히 반영.
- **Fig. 6**: OVE는 MC critic 대비 실패-회복 경계에서 더 명확한 negative-advantage 영역을 부여 → harmful 행동과 회복 가능한 행동을 더 잘 구분.

### 4.5 비-humanoid sanity check: D4RL AntMaze (Appendix C, Fig. 7)
- OVE-CFGRL vs IQL-CFGRL: antmaze-medium/large × diverse/play 4종 모두에서 경쟁력 있는 normalized return.
- 메시지: V(s) + expectile 조합이 Q(s,a) + expectile (IQL)의 실용적 대안이 될 수 있음을 단일 embodiment 데이터셋에서도 확인.

---

## 5. 강점

1. **문제 정의가 sharp**: Humanoid teleoperation의 takeover 부적합성을 **adaptation stage**로 명시화하고, 이를 reward boundary로 격리한 것 자체가 본 논문의 가장 큰 기여.
2. **State value 선택의 정합성**: Cross-embodiment human ego-video까지 critic에 끌어들이려면 Q(s,a)는 정의 자체가 어색해짐. V(s)로 일관적인 OVE objective 설계 → 데이터 소스가 진짜로 통합됨.
3. **Iterative closed loop**: 단발 RL이 아닌 3-iteration loop로 실제 성능이 단조 증가 → deployment 가능성을 보여줌.
4. **HG-DAgger가 무너지는 것을 baseline으로 명시**: 기존 가정이 humanoid에서 작동하지 않음을 실험으로 입증.
5. **Ablation으로 t_r의 중요성을 정량화**(-30%) → 단순히 OVE만이 아니라 reward labeling이 알고리즘의 본체임을 보여줌.

---

## 6. 약점 및 의문점

1. **Action head는 표준 flow-matching DiT 그대로**. 본 논문의 본질적 기여는 actor 아키텍처가 아니라 **post-training pipeline + critic objective**. VLA-tracker 관점에서 "새 VLA 아키텍처"라기보다 "기존 VLA를 위한 post-training 레시피"에 가까움.
2. **Task 다양성 부족**: Erase the whiteboard, Put the bread into the toaster 두 가지만. 두 task 모두 tabletop 범주에 들어가며 loco-manipulation은 다루지 않음 (저자도 한계로 언급).
3. **End-effector sensing 부재**: wrist camera, tactile 미사용 → 더 정밀한 조작 task에서의 일반화는 미지수.
4. **Human video 효과의 인과 분해 부족**: 180개라는 절대량이 critic에 들어갔을 때 얼마만큼이 quantity 효과인지 vs cross-embodiment supervision의 본질적 효과인지 명확히 분리되지 않음.
5. **70th percentile threshold**: η_k 선택의 민감도(예: 50/80/90 percentile) 비교가 본문에 없음. CFGRL의 β도 fixed value로 보고됨.
6. **Offline-iterative 한정**: 진정한 online RL은 아니며, deployment-time exploration 안전성, 최신 critic의 online 업데이트는 future work.

---

## 7. 관련 연구와의 위치

- **RECAP / π0.6-style experience learning [3]**: 동일하게 advantage-conditioned BC지만 critic objective가 다르고, intervention을 adaptation 이후로 **무조건 positive**로 가정 → ROVE는 이를 OVE로 완화.
- **HG-DAgger [5]**: 인간 개입을 모두 expert로 간주 → humanoid에서 SFT 이하로 떨어짐.
- **HIL-SERL [6]**: 그리퍼 + space mouse 환경에서 동작. Humanoid teleoperation gap은 다루지 않음.
- **HOIST [arxiv 2606.00252]**: 비슷한 시기의 humanoid VLA RL refinement. HOIST는 **suspended payload + flow-matching noise steering**에 특화, ROVE는 **dexterous hand task + state-value critic**에 특화 → 상보적 두 축.
- **IQL [12]**: 동일한 expectile principle이지만 Q(s,a) 학습 → ROVE는 V(s) 학습으로 cross-embodiment 데이터 활용 가능.
- **VLAC [39]**: ROVE의 critic 초기화에 사용된 VLA-critic 모델 — 정확히 layer 23 hidden dim 2048에서 feature 추출.

---

## 8. 재현성 평가

| 요소 | 공개 여부 |
|---|---|
| 코드 | **공개 안 됨** (paper code_url 없음) |
| Critic / actor checkpoint | 공개 안 됨 |
| IRON-R01-1.11 humanoid | XPENG 자체 플랫폼 |
| Human ego-video | 자체 수집 (180/task) |
| Demo 데이터 | 자체 수집 (~220/task) |
| 학습 하이퍼파라미터 | 본문 + Appendix G에 상당히 구체적 (스텝, LR, batch, dropout 확률, τ=0.7, β, percentile 70th 등) |
| Reward 설계 | 명시적 수식 |

→ 코드와 humanoid 플랫폼 없이는 재현이 사실상 어려움. 그러나 **알고리즘 자체는 RECAP 류 파이프라인 위에 OVE objective + t_r boundary를 얹는 형태로 비교적 명확히 기술**되어 있어 algorithmic 재현은 가능.

---

## 9. 핵심 인사이트 (왜 이 논문이 중요한가)

1. **Humanoid VLA post-training의 첫 번째 정직한 진술**: 인간 개입을 expert로 다루지 말라.
2. **Reward labeling은 critic objective만큼 중요**: t_r를 잘못 두면 OVE조차 망가진다.
3. **State value의 재발견**: V(s)는 cross-embodiment heterogeneous 데이터 통합에 Q(s,a)보다 우월할 수 있다.
4. **Iterative offline RL은 실 humanoid에서 closed-loop로 작동**: 세 번 돌려서 80%/86.7%까지.

---

## 10. VLA-tracker 분류 (action_head_category)

- 본 논문의 actor는 **flow-matching DiT** action decoder 사용 → category: **flow_matching**.
- 다만 알고리즘 본체는 actor-agnostic post-training 레시피이며, autoregressive / discrete diffusion 등 다른 head로의 일반화 가능성도 열려 있음 (논문 본문에서 명시적으로 다루지는 않음).

---

## 11. 한계 (저자 진술 + 본 리뷰 추가)

저자 명시:
1. Human experience는 critic 학습에만 쓰이고 actor 직접 학습엔 미사용 → representation-level supervision으로 확장 가능.
2. wrist camera / tactile 부재.
3. Loco-manipulation 미적용.
4. Offline / iterative-offline 단계 — 진정한 online RL이 아님.

본 리뷰 추가:
5. Action head 아키텍처가 ROVE 고유 기여가 아니므로 "새 VLA"라기보다 "VLA post-training method".
6. 두 task 평가는 통계적 결론을 강하게 뒷받침하기엔 좁다 (n=20, n=30 trial).
7. Threshold(η_k)와 guidance scale β의 민감도 분석 부재.

---

## 12. 종합 평가

**Score: 8.0 / 10**

- **+** Humanoid VLA post-training이라는 **2026년 현재 가장 hot한 missing piece**를 정면으로 다룸.
- **+** OVE + t_r boundary + state-value + cross-embodiment human video를 일관된 framework로 묶음.
- **+** 두 실 task에서 단조 증가하는 closed-loop improvement를 입증.
- **+** HG-DAgger의 실패 모드를 정확히 진단.
- **−** Task 다양성과 actor 아키텍처 차별성은 제한적.
- **−** 코드 / checkpoint 미공개로 외부 재현이 사실상 불가.
- **−** Ablation은 핵심 두 축(H, t_r)만, threshold·β·human video 양은 미탐구.

VLA-tracker 등재 권고: **ACCEPTED**. Humanoid VLA RL post-training의 **HOIST 자매 작품**으로 (HOIST = noise-steering / payload, ROVE = state-value / dexterous), 향후 humanoid VLA RL 비교에서 반드시 인용될 baseline.

<!-- VERIFIED: pdf -->
