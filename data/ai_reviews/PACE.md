# PACE: Phase-Progress-Aware Credit for Long-Horizon Embodied Manipulation

> **한 줄 요약**: 장기 조작에서 sparse terminal reward만으로는 어떤 step이 과제를 전진·정체·후퇴시켰는지 알 수 없다는 문제를, (1) 지역 시간창과 motion 차분으로 phase·intra-phase progress를 추정해 분포형 remaining-cost critic의 기댓값을 보정하는 **GLC-Critic**과 (2) 커버리지에 따라 positive-only CFG(PR-CFG)와 positive/negative 조건화(FACD)를 전환하는 **Progressive Policy Distillation**으로 해결. π0.5 기반으로 LIBERO-Long 83.3%(ReCAP 73.8%), 실제 양팔 로봇 81.8%(ReCAP 66.5%).

- **arXiv**: 2608.15026v1 (2026-08-15, cs.RO)
- **소속**: Dalian University of Technology, Jilin University
- **정책 백본**: π0.5-Base / **Critic**: SigLIP-SO400M + Gemma-3-270M
- **코드**: 공개 정보 없음

---

## 1. 배경 및 동기

VLA 사후학습은 전문가 데모와 정책 상호작용 궤적에 의존한다. 그러나 장기 과제에서는 한 에피소드가 수백 step과 여러 phase를 포함하는데 성공/실패는 끝에서야 드러난다. π*0.6의 ReCAP, ARM 같은 critic 기반 방법이 있지만 신뢰할 만한 가치 신호를 얻기는 세 가지 이유로 어렵다.
1. sparse terminal reward로는 step별 기여를 분리하기 어렵다.
2. phase가 달라도 시각 관측이 비슷해 단일 프레임 critic은 phase 전이·접촉 상태 변화를 식별하지 못한다.
3. 가치 추정 오류가 advantage로 전파되어 정책을 오도한다.

또한 초기 상호작용 데이터에는 성공 행동이 드물어 negative credit을 공격적으로 쓰면 드문 유용 행동까지 억제할 수 있다.

## 2. 문제 정의: Survival-cost 기반 remaining cost

각 과제를 per-step cost c_t = 1인 MDP로 보고, 성공 종료는 0, 실패는 C_fail의 terminal cost. 시점 t의 목표는 할인된 누적 cost + 할인된 terminal cost이며 V = −C. 시뮬레이션 성공 판정은 LIBERO predicate, 실제 로봇은 사람 평가자이며 Qwen3-VL은 성공/실패 판정에 절대 쓰이지 않는다.

## 3. GLC-Critic 구조

- **지역 시간 증거**: 학습 가능한 visual encoder가 f_t를 뽑고, 중심 창 [f_{t−r},…,f_{t+r}]을 Transformer로 집계(h_temp). 별도 MLP가 [f_t; f_t−f_{t−1}; f_{t+1}−f_t]로 motion 차분 인코딩(h_motion).
- **Phase / progress 헤드**: 과제별 헤드가 phase z_t(K_ph ∈ [3,5])와 progress p_t ∈ [0,1] 예측.
- **Remaining-cost 보정**: B개 bin의 분포형 base critic 기댓값 Ĉ_raw에 과제별 경량 MLP의 스칼라 보정 λΔ_t를 더함. softmax에 공통 logit shift는 무의미하고 bin별 보정은 차원 부담이 크므로 **기댓값 스칼라 보정**을 택했다.
- 학습: L_cost + α_z L_phase + α_p L_progress 후, phase/progress를 고정하고 보정 MLP를 MSE로 학습.
- 중심 창은 비인과적이며 **오프라인 라벨러 전용**이다. 배포 정책에는 미래 관측이 들어가지 않는다.

Phase 경계는 시뮬레이션에서 privileged predicate, 실제 로봇에서는 수동 주석. 실패 궤적의 마지막 미완 phase 내 progress만 Qwen3-VL로 추정 후 수동 검토.

## 4. Step-level credit

s_t = Ĉ_t − [Σ_{i<H} γ^i c_{t+i} + γ^H Ĉ_{t+H}]. 양수면 H-step 행동이 조건부 기대보다 좋았다는 뜻. 과제별 quantile 순위화에 쓰이며, 상태 독립 가치 이동에는 라벨이 불변이다. 저자들은 이것이 편향 없는 policy-gradient advantage라고 주장하지 않는다.

## 5. Progressive Policy Distillation

두 모드가 같은 flow-matching 목적(조건 g ∈ {∅, pos, neg})을 공유한다.
- **PR-CFG** (평균 rollout SR ≈ 50% 이하): 과제별 상위 quantile을 positive로, 무조건 + positive 조건 FM 손실. 추론 시 v_unc + η(v_pos − v_unc).
- **FACD** (50% 초과): positive/negative 조건 모두 사용, 조건 drop 확률 0.3. 추론 시 positive로 끌어당기고 negative에서 밀어냄.
- 실패 에피소드 positive 비율 상한, 데모는 항상 positive.
- 반복 루프: critic은 누적 버퍼, 정책 증류는 새 라운드 데이터만. **VLM 백본은 고정**, action expert가 갱신된다.

## 6. 실험 설정

- LIBERO-Long 10개 과제, 과제당 전문가 데모 10개에서 시작, 4 라운드 수집-업데이트. 과제당 20개 고정 초기 상태, 3개 독립 학습 seed.
- 실제 로봇: AgileX PiPER 양팔, Table Wiping / Towel Folding / Mug Placement and Knob Turning, 과제당 텔레오퍼레이션 데모 100개 + human-in-the-loop 수정, 과제당 50 trial.
- 비교: OpenVLA, Xiaomi-Robotics-0(XR-0), π0.5 SFT, ReCAP (ReCAP과 PACE는 같은 base·데모·rollout·업데이트 예산 공유).

## 7. 주요 결과

**LIBERO-Long (Table 1)**

| 방법 | SR (%) | AvgT (steps) | Held-out MAE | Boundary MAE |
|---|---|---|---|---|
| SFT (OpenVLA) | 52.7±1.6 | 310.8 | — | — |
| SFT (XR-0) | 57.2±1.3 | 298.7 | — | — |
| SFT (π0.5) | 63.3±1.3 | 276.9 | — | — |
| ReCAP | 73.8±1.8 | 254.26 | 133.03 | 136.02 |
| **PACE** | **83.3±2.5** | **252.55** | **97.78** | **77.01** |

Wilcoxon 검정: 8개 과제 개선, 2개 동률, 악화 0 (W=0, exact p=0.0078). Boundary MAE 43.4% 감소, boundary return-rank 상관 0.745 → 0.871.

**실제 로봇 (Table 4)**

| 방법 | Wiping | Towel | Mug+Knob |
|---|---|---|---|
| SFT (π0.5) | 54.7 | 51.3 | 47.3 |
| ReCAP | 62.7 | 70.7 | 66.0 |
| **PACE** | **84.7** | **83.3** | **77.3** |

과제 macro SR 81.8% (ReCAP 대비 +15.3), 성공 trial 평균 시간 76.3 s → 58.7 s. critic Held-out MAE 68.17 → 18.89, Boundary MAE 58.73 → 4.12.

## 8. Ablation

**Critic (Table 2)**: Raw Critic 73.8 → w/o temporal aggregation 81.6, w/o motion diff 80.9, w/o phase head 81.2, w/o progress head 82.7, Full 83.3. 모든 보정 변형이 Raw보다 크게 좋고, 각 구성요소는 작지만 일관된 기여.

**PPD 커리큘럼 (Table 3)**: 같은 초기화·데이터에서
- 시뮬 라운드 1(입력 SR 44.2): PR-CFG 54.5 vs FACD 49.0
- 시뮬 라운드 2: PR-CFG 61.0 vs FACD 64.5
- 실제 라운드 1(16.4): 32.9 vs 22.2 / 라운드 4(62.0): 67.8 vs 81.8

저커버리지에서는 positive-only가, 고커버리지에서는 양방향 credit이 유리하다는 역전이 router 설계를 정당화한다.

## 9. 강점

- critic 개선이 phase 경계 근처에 집중된다는 가설을 Boundary MAE라는 맞춤 지표로 검증.
- 통계 검정(Wilcoxon, Fisher exact, Holm 보정)을 꼼꼼히 제시.
- credit 추정과 credit 활용(커리큘럼)을 분리해 각각 ablation.
- 실제 양팔 로봇에서 SR과 실행 시간 모두 개선.

## 10. 약점 및 한계

- **LIBERO-Long 한 suite만**, 그것도 과제당 10 데모의 저데이터 설정이라 표준 LIBERO 리더보드와 직접 비교 불가.
- phase 경계가 시뮬 privileged predicate 또는 수동 주석에 의존 → 비구조적 과제 확장성 제한 (저자도 인정).
- 실패 궤적 progress의 수동 검토 비용.
- critic 자체가 별도 SigLIP + Gemma 모델로 추가 학습 비용.
- 50% router 임계값은 경험적.

## 11. 재현 및 확장 아이디어

- phase를 비지도 발견(예: 변화점 검출, VLM 자동 분할)으로 대체.
- 다른 LIBERO suite, RoboTwin, CALVIN 같은 장기 체인 벤치마크로 확장.
- FACD 스타일 양방향 guidance를 online RL(PPO)과 결합해 비교.

## 12. 총평

PACE는 "좋은 critic이 곧 좋은 사후학습"이라는 직관을 phase-progress라는 구조적 사전지식으로 구체화했다. π0.5의 action expert를 credit-조건 flow matching으로 실제 재학습한 자체 정책이며, LIBERO-Long과 실제 로봇에서 정량 개선을 보고하므로 트래커 대상 정책이다. 다만 평가 범위가 좁고 주석 의존성이 높아, 일반 장기 과제로의 확장은 후속 과제다.

**한 문장 요약**: 가치 오차는 phase 경계에서 폭발한다 — 그 경계를 알려주는 critic과, 데이터가 쌓일 때까지 negative credit을 아끼는 커리큘럼이 π0.5를 ReCAP보다 9.5점 더 끌어올린다.

<!-- VERIFIED: pdf -->
