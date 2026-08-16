# Cross-View Action-Flow Consistency: Camera-Robust VLA Policies (arXiv:2608.06965)

> **한 줄 요약**: 같은 MuJoCo 상태를 서로 다른 scene camera로 재렌더링한 "action-equivalent 뷰 쌍"을 만들고, flow-based VLA(pi0.5)의 **action-flow velocity field 자체**를 두 뷰 간에 일치시키는 학습-전용 정규화(L_CV)를 추가하여, 추론 인터페이스(단일 scene RGB + 언어 + proprioception)를 전혀 바꾸지 않고 LIBERO-Plus 카메라 교란 트랙 87.2%(FM-only 대비 +7.4pp)를 달성.

---

## 1. 배경 및 동기

고정된 scene camera로 수집한 데모로 fine-tuning된 VLA는 카메라가 부딪히거나, 재장착되거나, 거리·높이·방향이 바뀌면 **태스크·물체·언어·로봇 상태가 모두 동일해도** 실패한다. 이 논문의 nominal-only baseline이 그 증거다: ID(정상 카메라) 85.8%인 정책이 카메라 교란 트랙에서 16.8%로 붕괴하고, C1(거리/스케일)에서는 1.1%까지 떨어진다.

기존 대응책 대부분은 **추론 계약(inference contract)을 바꾼다**. depth, point cloud, 캘리브레이션된 multi-view, camera ray/label, novel-view rendering, geometry-aware 아키텍처가 그것이다. 저자들의 질문은 정반대다:

> flow-based VLA가 depth·point cloud·camera label·기하 입력 없이도 scene-camera 이동에 강건해질 수 있는가?

---

## 2. 핵심 아이디어 — 불변성을 어디에 걸 것인가

논문의 주장은 "원하는 불변성은 **action-level**"이라는 것이다. 서로 다른 이미지가 같은 물리 상태를 관측하고 같은 지시·같은 로봇 상태를 공유한다면, 정책은 **같은 행동을 의도해야 한다**.

flow-based VLA에서 행동은 학습된 velocity field `v_theta(x_t, t | o, l, q)`를 noisy action chunk 위로 적분해 생성된다. 따라서 저자들은 중간 feature가 아니라 **행동을 직접 만들어내는 대상**, 즉 특정 flow 좌표에서의 local velocity 예측을 정규화한다.

이것이 view-invariant representation learning과 결정적으로 다른 지점이다: 중간 feature가 두 뷰에서 일치한다고 해서 action decoder의 출력 분포가 일치한다는 보장은 없다. (Appendix G의 실패한 feature-level 시도들이 이를 경험적으로 뒷받침한다.)

---

## 3. 방법론 — Action-equivalent pair와 L_CV

**Pair 정의.** 물리 상태 `s`, 카메라 설정 `c`, 렌더링 관측 `o = h(s,c)`에 대해 학습 쌍은

```
(o_0, o_p, l, q, a),   o_0 = h(s, c_0),  o_p = h(s, c_p)
```

두 이미지는 **오직 scene camera만 다르고** 태스크 상태·언어·proprioception·행동 타깃은 공유한다.

**Flow 규약.** `t=1`이 noise, `t=0`이 시연 행동이며

```
x_t = t*eps + (1-t)*a,   u_t = eps - a
```

**손실.** K개의 (t_k, eps_k) 샘플에 대해 두 뷰 모두 표준 flow matching으로 시연 행동에 anchor되고,

```
L_FM^k = 0.5 * ( ||v_0^k - u_tk||^2 + ||v_p^k - u_tk||^2 )
L_CV^k = || v_p^k[:, :A_act] - v_0^k[:, :A_act] ||^2
L      = mean_k L_FM^k + lambda_CV * mean_k L_CV^k
```

같은 `(t_k, eps_k)`가 쌍 내 두 뷰에 공유되며, gradient는 **양방향(bilateral)** 으로 흐른다. 활성 행동 차원은 `A_act = 7`(LIBERO 7-DoF), padding 차원은 FM 항만 받는다.

📌 [Figure 2 삽입] — weight-shared 이중 forward + cross-view loss 구조

---

## 4. Mean–residual 재정식화 (왜 L_CV가 FM과 중복이 아닌가)

`v_bar = (v_0 + v_p)/2`, `delta = (v_0 - v_p)/2`로 두면 pair당 목적함수는 정확히

```
L^k = ||v_bar^k - u_tk||^2 + (1 + 4*lambda_CV) * ||delta^k||^2
```

로 분해된다. 시연 행동은 **평균 예측 v_bar를 통해서만** 들어오고, cross-view 항은 **뷰 불일치 잔차 delta에만** 작용한다. 두 가지 귀결:

1. L_CV는 어느 쪽 뷰도 시연 행동에서 멀어지게 밀지 않는다 — 순수하게 뷰별 잔차 페널티를 `4*lambda_CV`만큼 키운다.
2. 두 뷰가 이미 같은 타깃에 anchor되어 있어도 L_CV는 flow matching과 **중복이 아니다**. 그 효과는 velocity field 중 scene-camera pose에 민감한 성분에 집중된다.

> ❓ **예상 질문**: 그렇다면 velocity를 맞추는 것이 최종 action chunk를 맞추는 것을 보장하는가?
> **답변**: Proposition 1(Appendix E)이 uniform Euler grid + L-Lipschitz 가정 하에 `||a_0* - a_p*||^2 <= C * E[||v(x_t^(0)|o_0) - v(x_t^(0)|o_p)||^2]` 형태의 상한을 준다. 다만 bound는 nominal 적분 궤적 위에서 평가되고 L_CV는 학습시 선형 보간점 `x_tk`에서 평가되므로 엄밀히는 **국소 surrogate**다(최적 CFM 해에서 직선 궤적일 때 두 집합이 일치). 저자들 스스로 "정성적 동기이지 tight한 정량 추정은 아니다"라고 명시한다.

---

## 5. Shuffled-pair — 메커니즘 검증 장치

배치 내 derangement `pi`로 **L_CV 안에서만** 짝을 깨고(FM 타깃은 각 행마다 정확히 유지) 학습하면:

```
E_{s,s'} ||v_p(s') - v_0(s)||^2 = E_s ||v_p(s) - v_0(s)||^2 + 2*Tr( Cov_s(v_p(s), v_0(s)) )
```

두 번째 항은 뷰 불일치가 아니라 **state-marginal 공분산**이다. 의미 있는 state-dependent 정책에서 이 항은 보통 양수이므로, shuffled 목적함수를 최소화하는 것은 per-state velocity 예측을 state-marginal 평균 쪽으로 끌어당기며 **supervised FM 신호와 능동적으로 대립**한다. 즉 이 control은 "generic smoothing인가, 진짜 action-equivalence가 필요한가"를 가르는 리트머스지다.

---

## 6. 실험 설정 (중요: wrist 마스킹)

- **학습 데이터**: LIBERO 데모를 저장된 MuJoCo state로 reset 후 `sim.forward()` → nominal/perturbed 두 뷰 렌더링. libero_spatial/object/goal/100에서 **338,575 same-state pair** 생성. 쌍은 simulator state, action chunk, 언어, proprioception이 모두 같아야 유효.
- **초기화/스케줄**: 전 정책 pi0.5 체크포인트에서 10,000 step fine-tuning. 제안 설정은 K=2, lambda_CV=0.10, t~Beta(2,3).
- **평가**: 공식 LIBERO-Plus 카메라 교란 벤치마크 1,599 인스턴스(C1 n=313, C2 n=992, C3 n=294), seed당 4,797 rollout(각 인스턴스 3회). ID는 표준 LIBERO 4 suite x 30 trial.
- **⚠ wrist 스트림은 학습·평가 전 구간 마스킹**. LIBERO-Plus는 scene camera만 교란하고 wrist는 안정적이므로, wrist를 남기면 "교란되지 않은 시각적 지름길"이 생겨 scene-camera 귀속이 오염된다. **그 대가로 절대 수치는 wrist를 유지하는 표준 LIBERO-Plus 수치와 비교 불가**이며, 모든 비교는 동일 프로토콜 내부에서만 이루어진다.
- **분리 보장**: 평가 카메라 pose·초기 상태·행동 라벨 중 어느 것도 학습에 들어가지 않는다.

---

## 7. 주 시뮬레이션 결과 (Table 1/3, seeds 42–44)

| Method | 학습 데이터 / 목적함수 | ID | Camera | C1 | C2 | C3 |
|---|---|---|---|---|---|---|
| Nominal-only baseline | nominal LIBERO, FM only | 85.8 | 16.8 | 1.1 | 13.2 | 45.7 |
| Naive mixed-camera SFT | LIBERO-Plus 카메라 데이터, FM only | 69.2 | 74.7 | 68.4 | 75.7 | 78.2 |
| FM-only on same-state pairs | 재렌더링 쌍, FM only (lambda_CV=0) | 95.0 ± 4.3 | 79.8 ± 0.8 | 72.6 ± 0.2 | 79.9 ± 1.5 | 87.2 ± 0.7 |
| **Proposed (bilateral K=2)** | 재렌더링 쌍 + L_CV | **95.0 ± 0.8** | **87.2 ± 0.4** | **81.8 ± 1.0** | **87.9 ± 0.5** | **90.6 ± 0.9** |

핵심 관전 포인트는 3행 vs 4행이다. **동일 데이터·동일 아키텍처·동일 optimizer·동일 10k step, 오직 lambda_CV만 다르다.** 그럼에도 +7.4pp. 즉 이득은 "카메라 다양성 데이터 노출"이 아니라 **cross-view 항 자체**에서 온다. naive mixed-camera SFT 대비로는 +12.5pp이면서 ID는 69.2 → 95.0으로 오히려 크게 높다.

seed별(Table 5)로도 제안 기법의 최저 seed(86.9)가 FM-only 최고 seed(80.7)를 +6.2pp 상회해 분포가 겹치지 않는다. 카테고리별 이득은 C1 +9.2pp, C2 +8.0pp, C3 +3.4pp.

부수적으로 ID 평균은 둘 다 95.0이지만 seed 분산이 ±4.3 → ±0.8로 줄었다(FM-only ID가 92.5~100.0으로 요동). 저자들은 ID rollout 수가 적다는 이유로 이를 **부차적 관찰**로만 취급한다.

---

## 8. Ablation과 메커니즘 분석 (Table 2/4, Fig. 3)

| 질문 | 설정 | ID | Camera |
|---|---|---|---|
| 데이터 노출만으로 되나? | FM-only on same-state pairs | 95.0 ± 4.3 | 79.8 ± 0.8 |
| 한쪽만 coupling하면? | Stop-gradient K=2, Beta(1.5,1) | 96.7 | 84.3 |
| 짝이 틀리면? | Shuffled single-sample CV | 80.8 | 50.4 |
| 짝이 틀리면? | Shuffled bilateral K=2 + Beta(2,3) | 50.8 | 25.8 |
| 완성 레시피 | Proposed bilateral K=2 + Beta(2,3) | 95.0 ± 0.8 | 87.2 ± 0.4 |

- **Shuffled 붕괴가 이 논문의 백미다.** K=1에서 84.9 → 50.4, K=2에서 87.2 → **25.8**. 매칭이 맞을 때 가장 강한 레시피가 틀렸을 때 가장 심하게 무너진다(bilateral + multi-sample이 옳은 타깃과 틀린 타깃을 똑같이 증폭). "velocity field의 일반적 smoothing"이라는 대안 설명을 정면으로 배제한다.
- **Stop-gradient teacher–student**는 ID 최고(96.7)지만 Camera는 84.3으로 bilateral 대비 -2.9pp. 다만 저자 스스로 이 행은 gradient 구조와 flow-time 분포를 동시에 바꾸므로 요인 분리가 안 된다고 명시한다.
- **lambda_CV 민감도**(탐색적 single-sample 체제): 0.05 → 81.1, 0.10 → 80.9(신뢰구간 중첩), 0.20 → 73.6, 0.50 → 68.1. 즉 [0.05, 0.10]에서는 둔감하지만 0.2 이상에서 급격히 나빠진다.
- **Pair-consistent augmentation이 필수**: 독립 spatial augmentation 80.9 vs pair-consistent 84.9. 두 슬롯에 독립 crop/affine을 걸면 카메라 pose와 무관한 인공 픽셀 불일치가 생겨 L_CV가 augmentation artifact를 먹어버린다. photometric은 독립이어도 무방(기하 관계를 바꾸지 않으므로).
- **t ~ Beta(2,3)** 선택 근거: mode가 t=1/3로, velocity 예측이 무의미한 pure-noise 경계(t=1)와 이미 수렴한 경계(t=0) 양쪽을 피한다.

---

## 9. 실로봇 held-out 카메라 평가 (Table 6/7/8)

RealMan RM-75 7-DoF, delta EE pose 15 Hz, RealSense D435i 640x480@15fps. 태스크 3종(Battery box, Headphone stand, Laptop lid), 태스크당 70 데모, 동기화된 scene camera 3대(C0/C1/C2)로 **같은 timestamp의 같은 물리 상태**를 쌍으로 사용. 독립 수집 궤적은 action-equivalent가 아니므로 쌍으로 쓰지 않는다. 배포 시에는 단일 RGB만 받고 wrist는 마스킹.

| Method | Camera | Battery box | Headphone stand | Laptop lid | Aggregate |
|---|---|---|---|---|---|
| FM (multi-view mixed) | Seen | 83.3 (25/30) | 83.3 (25/30) | 100 (30/30) | 88.9 (80/90) |
| FM (multi-view mixed) | Held-out | 56.7 (17/30) | 16.7 (5/30) | 86.7 (26/30) | 53.3 (48/90) |
| **Proposed** | Seen | 86.7 (26/30) | 80.0 (24/30) | 96.7 (29/30) | 87.8 (79/90) |
| **Proposed** | Held-out | 76.7 (23/30) | 46.7 (14/30) | 100 (30/30) | **74.4 (67/90)** |

- **seen 카메라에서는 두 방법이 사실상 동률**(80/90 vs 79/90)이고, **held-out에서만 갈린다**(48/90 vs 67/90, two-proportion test p < 0.005). 이는 "이득이 데이터가 아니라 학습 신호에서 온다"는 시뮬레이션 결론과 정확히 같은 구조다.
- **기하학적으로 공격적인 배치일수록 이득이 커진다**: H0(더 멀리) 20/30 → 24/30, H1(방위각 회전) 17/30 → 23/30, H2(더 높고 top-down) 11/30 → 20/30.
- 가장 극단적 셀: Headphone stand @ H2에서 FM-only 0/10 → 제안 4/10.
- 그럼에도 Headphone stand는 여전히 16/30 실패. 실패는 접촉 직전 gripper–헤드폰 밴드 정렬에 몰려 있고, held-out 뷰가 얇은 밴드를 가리거나 depth 단서를 압축할 때 작은 접근각 오차가 mis-grasp로 이어진다.

---

## 10. 폐기된 feature-level 대안들 (Appendix G) — 음성 결과의 가치

본 방법에 도달하기 전 시도된 것들:

1. **Canonical-token injection** (VGGT hidden layer에서 학습한 view-stable 토큰을 cross-attention으로 주입): Camera 15.3%로 nominal-only 바닥 수준. matched/shuffled/constant canonical 토큰 control이 모두 유사 → 정책이 sample-specific canonical 내용을 행동 관련 신호로 **쓰지 않았다**.
2. **Canonical residual action anchor** (canonical 토큰으로 residual action-flow 보정, 학습 스칼라 gate): gate가 거의 닫힌 채 유지, 강제로 열면 nominal rollout이 붕괴 → 얼려둔 view-stable 표현은 그 자체로 **실행 가능한 행동 표현이 아니다**.
3. **Hard action-path bottleneck** (dense image token 열을 action suffix에서 마스킹): same-data attribution 실패 — 동일 30% subset에서 표준 FM 71.4 vs 정규화 bottleneck 67.4.

반복된 병목은 "정책이 의도한 view-stable 채널을 **우회하는 고용량 경로**를 쓴다"는 것이었고, 이것이 배포되는 정책의 action-flow 출력을 직접 정규화하는 최종 설계의 동기가 되었다.

---

## 11. 강점과 한계

**강점**
- **추론 계약 불변**: Appendix H의 비교표가 보여주듯, camera conditioning(extrinsics/Plücker ray), camera-space action(캘리브레이션 필요), test-time adaptation(view synthesis 모듈), 3D 입력(depth/point cloud), multi-view representation과 달리 배포 시 **아무것도 추가하지 않는다**. 따라서 이들과 직교하며 결합 가능.
- **인과적 실험 설계**: same-data FM-only control(λ만 다름) + shuffled-pair control + 3 seed는 VLA 논문 평균보다 훨씬 엄격하다. wrist를 일부러 마스킹해 귀속을 깨끗이 한 선택도 정직하다.
- **이론과 control이 서로를 지지**: mean-residual 분해가 "왜 중복이 아닌가"를, shuffled 분해가 "왜 매칭이 필요한가"를 설명하고 두 예측 모두 실험에서 확인된다.

**한계 (저자 명시)**
- **진짜 action-equivalent 쌍이 필수**. 시뮬에서는 MuJoCo state reset, 하드웨어에서는 동기화 카메라가 필요하다. 서로 다른 rollout에서 온 근사 쌍은 안전한 지름길이 아니다 — shuffled 분해대로 state-marginal 평균으로 끌려가며(25.8% 붕괴) 기존 대부분의 단일 카메라 로봇 데이터셋은 이 방법에 쓸 수 없다.
- **단일 RGB의 관측 가능성 한계**: 입력에 시각적으로 존재하지 않는 정보는 복원하지 못한다. held-out 뷰가 태스크 핵심 기하를 가리면 정규화가 anchor할 대상 자체가 없다.
- **평가 범위**: LIBERO-Plus C1/C2/C3와 정적 외부 카메라 3+3 배치, 태스크당 셀 10 rollout에 한정. 움직이는 카메라, 조명 변화, clutter, 투명/변형 물체, mobile-base는 미검증.
- **backbone 범위**: pi0.5-style flow head 한 종류에서만 검증. discrete-token VLA로의 이식(action-token logits 정규화)은 future work.
- **부차적**: 별도 코드/프로젝트 페이지 URL이 PDF 본문에 노출되지 않아(“project page에 링크” 라고만 기술) 재현성 확인은 공개 저장소 확인이 필요하다.

---

## 12. 종합 평가

이 논문은 **새 아키텍처가 아니라 새 학습 신호**를 제안한다. 그리고 그 점을 스스로 잘 알고 있어서, 논문 전체가 "이득이 데이터가 아니라 목적함수에서 온다"를 증명하는 데 설계되어 있다: 동일 데이터 FM-only control(+7.4pp), 3 seed 비중첩, shuffled 붕괴(87.2 → 25.8), 실로봇 seen 동률/held-out 격차.

VLA-Tracker 관점에서 두 가지를 강조해 둘 필요가 있다. 첫째, **95.0(ID)과 87.2(camera)는 wrist 마스킹 프로토콜의 수치이므로 다른 모델의 LIBERO / LIBERO-Plus 값과 직접 비교하면 안 된다** — 논문이 명시적으로 경고한다. 둘째, 이 방법은 리더보드 상단을 노리는 모델이라기보다, **flow-matching action head를 가진 임의의 VLA에 얹을 수 있는 fine-tuning 레시피**로 읽는 것이 맞다. 실용적 채택 장벽은 성능이 아니라 데이터다: 동기화 멀티카메라(또는 시뮬 state reset)로 action-equivalent 쌍을 만들 수 있는가에 전부가 달려 있고, 저자들도 "작은 동기화 데이터셋으로 짧은 적응 단계를 돌리는" data-efficient 버전을 최우선 future work로 꼽는다.

<!-- VERIFIED: pdf -->
