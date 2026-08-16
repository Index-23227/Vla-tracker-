# SiMDex: Mining Similar Egocentric Videos for Cross-Embodiment Dexterous Manipulation

> **한 줄 요약**: "얼마나 많이"가 아니라 **"어떤 human 데이터를"** 쓸 것인가. ~32M egocentric human 샘플 풀에서 robot demonstration마다 유사 샘플을 recall-ranking-re-ranking 3단 캐스케이드로 마이닝(~1.49M, 풀의 5% 미만)해 VLA post-training에 투입. 아키텍처·훈련 절차 변경 0. 실기 dexterous 3태스크에서 동일 크기 random sampling 베이스라인 대비 **47.7% → 61.1% (+13.4)**.

- arXiv: 2608.04196 (2026-08-04) / Date: August 6, 2026
- 소속: The University of Tokyo, ByteDance Seed, HKU, SJTU, Tsinghua
- Project: https://lin-nie.github.io/SiMDex/

---

## 1. 배경 및 동기

- VLA는 데이터 규모에 따라 성능이 예측 가능하게 스케일하지만, **robot 데이터 수집 자체가 병목**(하드웨어 비용, 이동성, teleoperation 숙련도).
- 대안으로 egocentric human video가 부상. Ego4D / Ego-Exo4D / EgoDex 등 수천~수백만 시간 규모.
- 그러나 egocentric pool은 **극도로 이질적**(요리, 사교, 스포츠 …). 특정 manipulation 태스크에 관련된 건 극히 일부.
  - 무차별 학습 → 노이즈 주입, task-relevant supervision 희석.
  - 전부 버림 → pre-training에서 얻은 cross-embodiment 지식 낭비.
- 핵심 질문: pre-training이 끝난 뒤 **post-training 때 어떤 human 데이터를 다시 방문할 것인가**. "how much"보다 "which"가 중요.

---

## 2. 선행 연구와의 차별점

| 계열 | 대표 | 한계 |
|---|---|---|
| Robot 데이터 내 curation | CUPID [2], Re-Mix [18], behavior retrieval [12] | robot 데이터 안에서만 동작, 외부 human pool 미채굴 |
| Human video retrieval | [37, 53] | 소규모 video bank, 검색 결과를 auxiliary input / in-context exemplar로 주입 → **정책 아키텍처 변경 필요** |
| Egocentric scaling | Zheng et al. [51] | human-robot aligned play를 수동으로 맞춰야 함 (data-task correspondence가 전제 조건) |

SiMDex는 **수천만 규모 cross-embodiment pool에 대해 per-demonstration, fine-grained action-level 유사도 기반 검색**을 수행하고, 그 결과를 아키텍처 수정 없이 VLA 훈련에 직접 투입하는 최초 접근이라고 주장.

---

## 3. 방법론 (1) — Morphology-Agnostic 표현

Cross-embodiment 마이닝의 전제 조건은 **로봇 손과 인간 손의 형태 차이를 추상화한 표현**.

### Wrist-local retargeting
- wrist pose `T_t ∈ R^{4x4}`, world 좌표 fingertip `q_t ∈ R^3`
- wrist-local 변환: `q_loc_t = T_t^{-1} q_t ∈ R^3` → workspace 위치를 제거하고 **intrinsic grasp geometry만 분리**
- 손당 5개 fingertip → `p_t ∈ R^15`
- wrist action `d_t ∈ R^6`: translation `R_t^T (o_{t+1} - o_t)`, rotation은 `R_t^T R_{t+1}`에서

### Shared action
- Bimanual: `a_t = (d^L_t, d^R_t, p^L_t, p^R_t) ∈ R^42`
- fingertip이 wrist-relative + Cartesian이므로 **workspace 위치에 불변**하고 kinematic 구조 매칭이 불필요.

### Sources
- Robot: bimanual teleoperation → forward kinematics로 wrist/fingertip 복원
- Human: **EgoDex [19]** (tracked wrist/fingertip transform 제공) → body-centered frame 변환으로 ego-motion 제거, hand visibility·velocity 필터, temporal smoothing 후 동일 retargeting
- 각 에피소드는 frame-level 샘플로 슬라이스, 각 샘플 = 현재 관측 + language instruction + **30-step future target**

---

## 4. 방법론 (2) — 3단 마이닝 캐스케이드

Anchor `D_r = {τ_n}^N`, target pool `D_h = {τ_m}^M`, `M >> N`. Exhaustive 비교가 불가능하므로 **비용·정밀도가 점증하는 3단 캐스케이드**(산업 추천 시스템의 recall-ranking-re-ranking을 차용).

### Stage I: Recall (경량·확장성)
- **Pose**: 초기 fingertip state를 L2 정규화한 descriptor `p̂ ∈ R^15`, Euclidean 최근접 → 유사 초기 grasp 자세
- **Language**: instruction sentence embedding `e ∈ R^384`, cosine 유사도 → task semantics
- 두 랭킹을 **rank fusion**으로 결합

### Stage II: Ranking (fine-grained motion)
미래 action 시퀀스에 대해 4개 상보적 성분:
1. wrist **translation waveform** (속도 프로파일, motion rhythm)
2. wrist **rotation waveform**
3. **finger trajectory** `F_fg ∈ R^{30x15}`
4. **wrist trajectory** `F_ee ∈ R^{31x3}`

융합 점수 `r = r_tr + r_rot + r_fg + r_ee`. 이후 **source trajectory 단위 dedup**(최상위 1개만 유지). → **이 결과가 실제 훈련에 쓰이는 mined subset `D*_h`**.

### Stage III: Re-ranking (embodiment-agnostic 검증)
- Dense **optical flow**를 clip-level descriptor로 집계해 재점수.
- Shared action space가 아닌 **픽셀 수준 motion**을 보므로 손 형태·retargeting 정확도와 독립적인 검증 신호.

---

## 5. 방법론 (3) — VLA 훈련

- **Base model**: GR-Dexter [47] 계열, **π0-like flow-matching VLA**. VL backbone이 `I_t`, `l`을 인코딩하고 action decoder가 `a_{t:t+H} ∈ R^{H x 88}`, `H = 30` 예측.
- **88차원 action space** = shared 42 + robot-specific 46 (arm/hand joint).
  - Robot 샘플 → 88차원 전부 supervise
  - Human 샘플 → **shared 42차원만** supervise, 나머지는 placeholder + binary mask `m ∈ {0,1}^88`로 제외
- **Flow matching loss (masked MSE)**:
  `L = Σ_{h,d} m_{h,d} (û_{τ,h,d} - u_{τ,h,d})^2 / Σ_{h,d} m_{h,d}`
- 이 마스킹 덕분에 **아키텍처 수정 없이** human/robot 단일 모델 공동 학습.
- Mined subset과 robot demo는 **독립 가중치의 두 데이터 스트림**으로 로드, **1:1 mixture, 40K steps**.

> 저자 강조: SiMDex는 VLA가 **무엇을** 배우는지(shaping what)를 바꿀 뿐 **어떻게**(how) 배우는지는 건드리지 않는다.

---

## 6. 실험 설정

### 데이터
- Robot: **~1.35M frame-level 샘플 (~12.4시간)**, bimanual teleoperation
- Human pool: **32,034,551 샘플 (~32M)**, EgoDex 기반 — ~300시간 30fps, **164,959 에피소드**, ~1초/30-step 슬라이딩 윈도우
- SiMDex 마이닝 결과: **~1.49M 샘플 (풀의 5% 미만)**. 베이스라인은 동일 개수를 random sampling.
- 하이퍼파라미터 전부 동일. **유일한 차이는 human 데이터의 출처.**

### 태스크 (모두 실기, 산업 조립 시나리오)
| Task | Max | 구성 | 평가 포인트 |
|---|---|---|---|
| I. Drill | 3 | grasp → assemble(fixture) → press trigger | tool use, multi-step coordination |
| II. Flick Wheel | 3 | grasp → 두 손가락 twist → 한 손가락 flick | **fine-grained finger dexterity (최난이도)** |
| III. Pick & Place | 4 | 상이한 기하의 4개 물체를 지정 위치에 | 물체 형태 일반화 |

- 모든 태스크가 **엄격한 순차 의존**(중간 실패 시 이후 단계 차단).
- 각 sub-task는 완료 진행도로 **0-1 연속 점수**, 태스크당 **10 trials (2 rounds x 5)**, mean±std.

---

## 7. 주요 결과 (RQ1, Table 1)

| Task | Method | Sub-task 점수 | Total | SR(%) |
|---|---|---|---|---|
| **Drill** (max 3) | | Grasp / Assem. / Drill | | |
| | GR-Dexter | 0.80±.00 / 0.80±.00 / 0.33±.00 | 1.93±.00 | **64.5** |
| | SiMDex | 0.73±.10 / 0.63±.33 / 0.27±.28 | 1.63±.71 | 54.5 |
| | Δ | -0.07 / -0.17 / -0.07 | -0.30 | **-10.0** |
| **Flick Wheel** (max 3) | | Grasp / Twist / Flick | | |
| | GR-Dexter | 0.60±.14 / 0.13±.19 / 0.00±.00 | 0.73±.33 | 24.5 |
| | SiMDex | 0.80±.09 / 0.47±.09 / 0.10±.00 | 1.37±.19 | **45.5** |
| | Δ | +0.20 / +0.34 / +0.10 | +0.64 | **+21.0** |
| **Pick & Place** (max 4) | | Obj.1 / 2 / 3 / 4 | | |
| | GR-Dexter | 0.30±.05 / 0.67±.09 / 0.53±.09 / 0.67±.09 | 2.16±.05 | 54.0 |
| | SiMDex | 0.67±.19 / 0.87±.00 / 0.97±.05 / 0.83±.05 | 3.33±.19 | **83.4** |
| | Δ | +0.37 / +0.20 / +0.44 / +0.16 | +1.17 | **+29.4** |
| **Overall** | | | | **47.7 → 61.1 (+13.4)** |

해석:
- **Flick Wheel**: 성공률 거의 2배(24.5 → 45.5). 베이스라인은 Twist(0.13)·Flick(0.00)에서 사실상 전멸 → **마이닝된 human 데이터가 fine-grained finger skill을 전달**한다는 직접 증거.
- **Pick & Place**: 4개 물체 전부에서 일관된 향상 → 기하 일반화 강화.
- **Drill**: 유일하게 하락(64.5 → 54.5). 단, 분산이 매우 큼(Total std ±.71)이고 Sec 4.4에서 **부호가 robot-data 예산에 따라 뒤집힘**을 보임.

---

## 8. 데이터 스케일링 ablation (RQ2, Fig. 3)

Robot 데이터를 0.25x(~3.1h) ~ 2x(~24.9h)로 스케일하고 mined human 데이터는 고정.

- **Overall Δ**: +6.8 (0.25x), **+17.2 (0.5x)**, +12.8 (1x), +6.1 (2x) — **모든 스케일에서 베이스라인 상회**.
- **안정적 성능 바닥(floor)**: SiMDex는 0.5x~2x 구간에서 **~57-58%로 거의 일정**. 반면 베이스라인은 **50.9% → 40.9%**로 급락.
- **Drill의 부호 반전**: 희소 robot 데이터에서는 SiMDex 우세(+8.9 @0.25x, +13.4 @0.5x), robot 데이터가 충분해지면(1x, 2x) 열세. 저자 해석 — drilling에 해당하는 고품질 human demo가 풀에 희귀해서, robot 데이터가 부족할 때만 완전히 착취되고, 충분해지면 소수의 마이닝 샘플이 신호보다 **분산**을 더한다.
- **실용적 함의**: **~6시간(0.5x) robot 데이터가 ~25시간(2x) 베이스라인과 동급** → robot 데이터 수집량 **4배 절감**.

> 주의: Fig. 3은 single round 평가라 1x 컬럼이 Table 1(2-round 평균)과 미세하게 다름.

---

## 9. 정성 분석 (RQ3, Fig. 4)

3개 anchor에 대해 각 stage 최상위 검색 결과를 시각화:

- **Recall**: 거친 단서만 매칭. 유사 초기 자세나 language 관련 물체("wheel", "basket")는 잡지만, 실제 동작은 tool use / twist가 아니라 단순 planar grasping인 경우가 많음.
- **Ranking**: pinching·twisting 패턴, finger curvature, 회전 방향이 anchor에 근접. 그러나 **visual grounding이 없어서** kinematic 값만 우연히 맞는 먼 손 위치 샘플이 상위에 올 수 있음.
- **Re-ranking**: optical flow로 이를 해소 — Drill은 왼손 고정 + 오른손 tool 정렬, Flick Wheel은 stabilize-and-twist 협응, Pick & Place는 작은 물체 순차 정밀 파지. 로봇의 bimanual 행동을 밀접하게 반영.

각 stage가 **상보적 신호**를 기여함을 확인.

---

## 10. 한계 및 미해결 문제

저자 스스로 명시한 3가지:

1. **단일 시나리오**: robot 데이터가 산업 조립 1개 시나리오, 3개 태스크, ~12.4시간뿐. 장면/태스크/embodiment 전반의 일반성 미검증.
2. **Human pool coverage 의존성**: Drill 사례처럼 풀에 목표 스킬과 유사한 고품질 demo가 없으면 검색이 착취할 신호가 없고, robot 데이터가 충분할 땐 **분산만 주입**할 수 있음.
3. **순수 kinematic 유사도**: wrist 6D pose + fingertip 위치만 봄 → **contact force, object state, interaction semantics 미포착**. Ranking 단계가 "손이 거의 움직이지 않는 drilling 클립"을 검색하는 mismatch가 발생하고, re-ranking(optical flow)이 이를 교정하지만 **비용이 만만치 않음**.

추가로 리뷰어 관점의 한계:
- 마이닝 파이프라인 자체의 하이퍼파라미터(recall 후보 수, dedup 임계값 등)에 대한 민감도 분석 부재.
- Stage III(re-ranking)가 최종 훈련 subset을 만드는 데 쓰이지 않음(**Stage II 출력이 훈련용**) — re-ranking은 검증/시각화 역할. 즉 3단 중 마지막 단계가 실제 성능 수치에 직접 기여하는지 불명확.
- 베이스라인이 "동일 크기 random human 데이터" 하나뿐. "human 데이터 없이 robot only" 또는 다른 curation 기법(CUPID, Re-Mix 류)과의 비교 부재.

---

## 11. VLA-Tracker 맥락에서의 위치

- **정책 산출물 있음**: GR-Dexter 기반 π0-like flow-matching VLA를 자체 학습(40K steps, 1:1 mixture)하고 실기 성공률을 보고 → 단순 데이터 파이프라인 논문이 아님.
- **벤치마크**: LIBERO / CALVIN / SimplerEnv 등 시뮬 벤치마크 **전무**. 실기 3태스크 성공률만 존재 → `benchmarks.real_world`.
- **동일 계열**: GR-Dexter(2512.24210, ByteDance Seed)가 base model이자 베이스라인. 두 논문을 함께 읽을 것.
- **분류**: `action_head_category: flow_matching`, model_type은 VLA이지만 실질 기여는 **data curation for post-training**.

---

## 12. 세미나 Q&A 예상

| # | 질문 | 답변 |
|---|---|---|
| 1 | 결국 데이터 선택 논문인데 왜 VLA로 분류하나? | 자체 학습한 π0-like flow-matching 정책(88-dim action, H=30)을 산출하고 실기 성공률(61.1%)로 평가하기 때문. 마이닝은 훈련 데이터 소스만 바꾸고 아키텍처는 그대로. |
| 2 | 왜 recommendation 비유인가? | 3천만 규모 target pool에 대한 exhaustive 비교가 불가능. 산업 추천 시스템이 billion-scale catalog에서 쓰는 recall(경량 후보 대량 회수) → ranking(정밀 점수) → re-ranking(최종 검증) 캐스케이드 구조가 그대로 대응. |
| 3 | Morphology gap을 어떻게 넘나? | 학습된 alignment 모듈 대신 **fingertip-centric 기하 표현**. wrist-local 변환으로 workspace 위치를 제거하면 인간 손과 로봇 손이 동일한 R^42 서술로 수렴. 저자는 이 스케일에서는 무거운 학습 모듈보다 빠르고 신뢰할 수 있는 검색이 실용적이라고 명시. |
| 4 | Human 데이터가 robot-specific 차원을 오염시키지 않나? | 이진 마스크 m ∈ {0,1}^88으로 human 샘플의 gradient를 shared 42차원으로만 제한. 나머지 46차원은 placeholder. |
| 5 | Drill이 왜 나빠졌나? | 풀에 고품질 drilling human demo가 희귀. robot 데이터가 충분하면 소수의 마이닝 샘플이 신호 대신 분산을 더함. 실제로 0.25x/0.5x에서는 +8.9/+13.4로 부호가 뒤집힘. 논문의 중심 주장(robot 데이터가 병목일 때 선택적 마이닝이 가장 유효)과 일치. |
| 6 | 왜 5%만 써도 되나? | egocentric pool의 대부분은 요리·사교·스포츠 등 무관 활동. 무차별 혼합은 노이즈 주입 + task-relevant supervision 희석. 동일 크기 random 대비 +13.4는 "양이 아니라 정밀도"라는 주장의 직접 증거. |
| 7 | Stage III는 성능에 기여하나? | 논문 기준으로 훈련용 subset은 **Stage II의 dedup 출력**. Stage III는 embodiment-agnostic 검증 신호이자 정성 분석용. 훈련 subset 생성에 넣지 않은 이유는 비용(dense optical flow)으로 보이며, 정량 기여도는 미보고. |
| 8 | 재현 가능성은? | EgoDex는 공개, GR-Dexter base와 robot teleoperation 데이터(~12.4h, ByteDexter 하드웨어)는 비공개. 마이닝 파이프라인 자체는 개념적으로 재현 가능하나 실기 수치 재현은 하드웨어 종속. |

<!-- VERIFIED: pdf -->
