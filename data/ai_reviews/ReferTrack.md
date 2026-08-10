# ReferTrack: Referring Then Tracking for Embodied Visual Tracking

> **한 줄 요약**: Embodied visual tracking에서 "누구를 따라갈 것인가"를 추상적 spatial latent(TrackVLA++의 polar token)로 추론하는 대신, **탐지기가 뽑아준 image-space bbox 목록 중 하나를 고르는 객관식 문제**로 바꾼 VLA. 단일 Refer-CoT 토큰 `<ped_i>`(또는 `<NO_EXIST>`)이 타깃을 지목하고, 그 bbox가 FIFO 큐를 통해 TVBI 토큰으로 visual history에 주입된 뒤 waypoint가 디코딩된다. EVT-Bench 단일 전방 카메라 설정에서 4B 백본 + SFT만으로 STT 89.4 / DT 73.3 / AT 74.1 SR을 기록해, 7B single-view SOTA인 TrackVLA++를 AT에서 +22.9 SR로 앞지르고 일부 multi-camera 베이스라인까지 넘어선다.

- arXiv: 2607.20061v1 (2026-07-22)
- 소속: RCV Laboratory (SUSTech) / Tencent Robotics X / Peking University / Futian Laboratory
- 코드: https://github.com/MedlarTea/referTrack

---

## 1. 배경 및 동기

Embodied visual tracking(EVT)은 온보드 비전만으로 자연어로 지정된 특정 사람을 계속 따라가는 태스크다. 성공은 두 능력의 결합에 달려 있다.

1. **Target identification** — 지시문에 맞는 보행자가 누구인가
2. **Trajectory planning** — 충돌 없이 적정 추종 거리(1–3 m)를 유지하는 궤적 생성

초기 파이프라인은 이 둘을 분리했다. SAM/GroundingDINO 같은 visual foundation model로 식별하고, 그 결과를 학습 기반 플래너에 넘긴다. 해석 가능하지만 **인식 오류가 플래닝 단으로 그대로 전파**된다.

최근 VLA는 둘을 하나의 정책으로 통합한다. 논문은 이 계열을 두 갈래로 정리한다.

- **범용 co-training 계열** (NavFoM, Uni-NaVid, VLingNav, ABot-N0): VLN, ObjectNav, PointNav 등과 함께 학습해 데이터 물량으로 EVT 기본기를 얻는다.
- **특화 계열** (TrackVLA, TrackVLA++): EVT와 식별용 QA 데이터셋에만 집중해, 훨씬 적은 데이터로 동등하거나 더 나은 성능을 낸다.

TrackVLA++는 여기에 CoT를 도입했다 — waypoint 예측 전에 **spatial-aware polar-coordinate token**을 뱉는다. 저자들의 비판은 여기서 시작된다: 추상적 spatial latent 공간에서의 추론은 (a) **감독하기 어렵고**, (b) **image-space detection과의 정렬이 약해서** 복잡하고 붐비는 장면에서 정밀한 visual grounding에 실패한다.

---

## 2. 핵심 아이디어: Referring Then Tracking

저자들의 제안은 보완적 패러다임이다. 타깃 식별을 **egocentric view에서 탐지된 indexed bbox 집합 중 하나를 고르는 제약된 객관식 문제**로 정식화한다.

이 프레이밍의 매력은 세 가지다.

1. **감독이 쉽다** — 정답이 이산 인덱스 하나다. Cross-entropy 한 줄.
2. **VLM의 grounding 메커니즘과 정렬된다** — 멀티모달 LLM은 이미 region proposal 중 선택하거나 bbox를 토큰으로 출력하도록 설계되어 왔다(RexSeek 계열).
3. **감독 데이터가 확장 가능하다** — 이게 가장 중요한 논점이다. 값비싼 closed-loop navigation 시연에만 의존하지 않고, 웹의 referring/grounding 데이터, 로보틱스 데이터셋(JRDB, TPT-Bench), 자율주행 데이터셋(KITTI, SiT)의 2D 보행자 어노테이션이 모두 candidate catalog를 채울 수 있다.

> ❓ **예상 질문**: 그러면 왜 이 논문은 웹 데이터를 안 쓰는가?
> **답변**: 선행 EVT 연구(TrackVLA, TrackVLA++)와의 공정 비교를 위해 **의도적으로 동일한 SYNTH-PEDES ReID 데이터셋만** 사용했다고 명시한다. 확장 경로는 열어두되 이번 논문에서는 밟지 않는다. 정직한 선택이지만, 논문이 주장하는 "scalable path"의 실증은 미래 과제로 남는다.

---

## 3. 방법론 심층 분석

### 3.1 문제 정식화

타임스텝 T에서, 타깃 외형에 대한 지시문 L과 전방 RGB 관측 스트림 O_{1:T}가 주어지면 에이전트는 연속 궤적 W_T = {w_1, w_2, ...}를 예측한다. 각 waypoint w_i = (x, y, θ) ∈ R³는 지면 위 egocentric 변위와 방향 변화다.

전체 추론은 **LLM을 두 번 순차적으로 통과**한다:

```
E_refer_T = LLM(L, C_T, E_V_{1:T})
E_A_T     = LLM(L, C_T, E_V_{1:T}, E_refer_T)
W_T       = ActionHead(E_A_T)
```

여기서 E_refer_T ∈ {⟨ped_1⟩, ..., ⟨ped_K⟩, ⟨NO_EXIST⟩}.

### 3.2 Observation Encoding — TVBI 토큰

Dual encoder(SigLIP + DINOv2) 특징을 concat한 뒤 grid pooling으로:

- **fine tokens** V_fine ∈ R^{64×C} — 현재 관측의 세밀한 디테일
- **coarse tokens** V_coarse ∈ R^{4×C} — 과거 프레임의 넓은 맥락

최근 H 프레임 슬라이딩 윈도우로 `V_T = {V_coarse_{T-H}, ..., V_coarse_{T-1}, V_fine_T}`를 구성한다. 장기 맥락과 추론 지연 사이의 절충이다.

핵심은 프레임 그룹 사이에 끼워넣는 **TVBI(temporal-viewpoint-bbox indicator) 토큰**이다. NavFoM의 TVI 토큰을 확장해:

```
E_TVBI(t) = E_TVI(t) + P_bbox(b_t)
```

- b_t는 정규화된 bbox ∈ [0,1]⁴
- P_bbox는 catalog 인코딩에서도 **공유되는** 2-layer MLP
- 타깃이 안 보이는 과거 프레임은 b_t = [0,0,0,0]이라는 **결정론적 부재 sentinel**로 채운다

📌 [Figure 2 삽입] — 전체 파이프라인. RGB → vision encoder → TVBI로 조직된 visual token, detector → candidate generator → indexed catalog, 두 스트림 + language token이 LLM으로 들어가 `<ped_i>` → action head → trajectory.

**가장 미묘한 설계 결정**: 과거 TVBI 스트림에만 bbox 큐가 조건화되고, **현재 프레임 fine token은 TVI-only**로 남긴다. 현재 관측에서 명시적 bbox 주입을 박탈함으로써, 모델은 과거 TVBI 단서와 raw visual feature만으로 타깃을 공간적·시간적으로 grounding하도록 강제된다. 이것이 downstream referring을 위한 견고한 표현을 만든다는 논리다.

### 3.3 Candidate Catalog

타임스텝 T에 실시간 탐지기(YOLO11)를 현재 전방 이미지에 돌린다. 탐지된 보행자를 인덱스 카탈로그 C_T = {⟨ped_1⟩, ..., ⟨ped_K⟩, ⟨NO_EXIST⟩}로 정렬한다.

- K명을 초과하면 **bbox 면적 기준 top-K** 우선
- ⟨NO_EXIST⟩는 **항상 포함**되는 고정 가상 인덱스 — 지시된 타깃이 시야에 완전히 부재한 경우 처리

각 후보는 식별자 토큰 ⟨ped_k⟩ 뒤에 bbox 토큰 E_bbox = P_bbox(b_T^{(k)})가 붙는 형태로 표현된다. TVBI와 **동일한 P_bbox 아키텍처를 재사용**한다는 점이 설계의 경제성을 보여준다.

### 3.4 Refer-CoT와 궤적 예측

첫 번째 LLM forward pass에서 등록된 special token 이산 어휘 위의 분류 스텝으로 **단일 Refer-CoT 토큰**을 생성한다. 타깃 선택은 엄격히 **one-token decision**으로 유지되어, 추론 스텝이 compact하고 직접 감독된다. 그 다음 E_refer_T를 conditioning prefix로 삼아 action token E_A_T를 생성하고, 전용 MLP head가 이를 M개 waypoint로 디코딩한다.

### 3.5 Referred-Target Bbox Queue

E_refer_T가 결정되면 해당 bbox를 용량 H−1의 FIFO 큐 Q에 push한다. 스텝 T에서 현재 관측 자체는 TVI-only이고, historical TVBI 스트림은 직전 큐 Q_hist_T = {b_{T-H}, ..., b_{T-1}}에 조건화된다.

- **학습 시**: Q를 ground-truth 추적 어노테이션으로 채우되, **가끔 틀린 타깃 인덱스(다른 보행자 또는 ⟨NO_EXIST⟩)를 주입하는 랜덤 노이즈**로 과거 추적 오류를 시뮬레이션한다. Autoregressive 큐가 자기 오류에 갇히는 것을 막는 실용적 장치다.
- **추론 시**: 큐가 각 스텝의 Refer-CoT 선택으로 autoregressive하게 갱신된다.

> **추론 루프의 닫힘**: Refer-CoT가 *누구*를 따라갈지 결정하고, 큐가 그 결정을 geometric history로 전파해 이후 모든 플래닝을 anchoring한다.

### 3.6 학습 목적함수

```
L = α · L_traj + L_refer + L_text,   α = 10
```

- **L_traj**: 예측 waypoint와 전문가 waypoint 사이 MSE 합
- **L_refer**: ground-truth 타깃 인덱스에 대한 cross-entropy
- **L_text**: Refer-QA 샘플의 텍스트 토큰 cross-entropy — 이 손실은 **의도적으로 action head를 우회**해서, gradient를 전적으로 visual grounding과 language alignment에 집중시킨다

Full fine-tuning 전략으로 LLM 전체 파라미터를 보조 vision/action 모듈과 함께 갱신한다(vision encoder는 Stage 2에서 frozen).

---

## 4. 데이터

### 4.1 Navigation 데이터 (1.3M)

Habitat 3.0의 EVT-Bench training split에서 전문가 추적 궤적을 뽑는다. 기존 전문가 궤적 큐레이션 파이프라인이 비공개이므로, **로봇과 지시된 휴머노이드 타깃의 시뮬레이터 상태에 접근하는 커스텀 oracle controller**를 직접 구현했다:

- 매 스텝 Habitat의 geodesic shortest-path planner를 로봇→타깃으로 질의, 경로를 densify, **선호 추종 거리 1.2 m** 근처를 유지하는 local lookahead waypoint 선택
- 큰 회전 시에는 중간 경로점 추격으로 전환
- 타깃이 접근하거나 로봇이 너무 가까워지면 **로봇 뒤쪽 backward goal**을 계산해 거리 회복
- Local goal은 가속도 스무딩이 있는 PD 속도 컨트롤러가 추종, 차동 속도 명령 (v_x, 0, ω)이 waypoint/action 감독이 됨
- Habitat following metric이 충돌·장기 타깃 상실 없이 성공한 에피소드만 전문가 시연으로 채택

📌 [Figure 4 삽입] — "빨강-노랑 슈퍼히어로 슈트를 입은 사람을 추적하라" 지시 하의 다양한 전문가 행동: 타깃 접근 시 후진, 큰 회전 시 중간 waypoint 추격, 통상적 후측방 추종, 타깃 정지 시 정지.

구성: STT 330K + AT 330K + **DT 640K (전량 유지)** = 1.3M. DT를 다운샘플하지 않은 이유는 방해 상황에서의 타깃 식별 능력을 키우기 위해서다.

### 4.2 Refer-QA 데이터 (1.3M)

SYNTH-PEDES에서 합성한다.

- 배경 이미지 하단 영역을 crop해 384×384로 resize
- 2–3개 보행자 crop을 [0.75, 1.5] 랜덤 스케일로 paste, 겹치는 레이아웃은 reject
- 각 보행자에 0–19 랜덤 카탈로그 ID 부여, image-space bbox와 caption 저장
- **이미지에 존재하지 않는 negative caption 하나를 추가**, all-zero bbox로 표현
- 프롬프트: "Please find ⟨caption⟩ in the video. Answer with object indexes."
- 감독: 정답 bbox-index 토큰, 또는 부재 caption이면 ⟨NO_EXIST⟩

📌 [Figure 5 삽입] — Refer-QA 합성 예시.

**설계의 핵심**: navigation과 Refer-QA가 지시문 → 이산 카탈로그 → 조직화된 visual token을 **동일한 순서로 배열**하므로, static QA 감독에서 배운 referring 능력이 dynamic online tracking으로 매끄럽게 전이된다.

---

## 5. 학습 설정

TrackVLA 레시피를 따르는 2-stage SFT:

| Stage | 학습 대상 | 데이터 | 설정 |
|---|---|---|---|
| 1 | vision projector P_vision만 (LLM·vision encoder frozen) | 일반 멀티모달 QA (LLaVA 계열, MVBench) | 1 epoch, LR 1e-4 |
| 2 | vision encoder 제외 전 파라미터 | navigation : Refer-QA = 1:1 | 20K steps, global batch 256 |

Stage 2 옵티마이저는 AdamW + cosine decay + linear warm-up. **LLM은 LR 2e-5, 나머지 학습 가능 모듈(projector, action head)은 1e-4**.

아키텍처 구현은 오픈소스 **OpenTrackVLA** 코드베이스 위에 올렸고, LLM 백본은 **Qwen3-4B**, 카탈로그는 **YOLO11 + ByteTrack** 탐지로 구성한다.

---

## 6. 실험 결과

### 6.1 EVT-Bench 메인 테이블

Habitat 3.0의 EVT-Bench 표준 스위트. 지표는 SR(Success Rate, ↑) / TR(Tracking Rate, ↑) / CR(Collision Rate, ↓). 태스크는 STT(Single-Target Tracking), DT(Distracted Tracking), AT(Ambiguity Tracking).

**Single-view (전방 카메라만)**

| Method | Size | RL | STT | DT | AT |
|---|---|---|---|---|---|
| IBVS | – | – | 42.9 / 56.2 / 3.8 | 10.6 / 28.4 / 6.1 | 15.2 / 39.5 / 4.9 |
| PoliFormer | – | ✓ | 4.7 / 15.5 / 40.1 | 2.6 / 13.2 / 44.5 | 3.0 / 15.4 / 41.5 |
| EVT | – | ✓ | 24.4 / 39.1 / 42.5 | 3.2 / 11.2 / 47.9 | 17.4 / 21.1 / 45.6 |
| EVT (SoM+GPT-4o) | – | ✓ | 32.5 / 49.9 / 40.5 | 15.7 / 35.7 / 53.3 | 18.3 / 21.0 / 44.9 |
| Uni-NaVid | 7B | – | 53.3 / 67.2 / 12.6 | 31.9 / 50.1 / 21.3 | 15.8 / 41.5 / 26.5 |
| NavFoM | 7B | – | 85.0 / 80.5 / – | 61.4 / 68.2 / – | – |
| VLingNav | 7B | ✓ | 88.4 / 81.2 / 2.1 | 67.7 / 73.5 / 5.5 | – |
| Qwen-RobotNav | 4B | – | 77.4 / 90.0 / 6.40 | – | – |
| Qwen-RobotNav | 8B | – | 78.6 / 89.7 / 5.70 | – | – |
| TrackVLA | 7B | – | 85.1 / 78.6 / 1.7 | 57.6 / 63.2 / 5.8 | 50.2 / 63.7 / 17.1 |
| TrackVLA++ | 7B | – | 86.0 / 81.0 / 2.10 | 66.5 / 68.8 / **4.71** | 51.2 / 63.4 / 15.9 |
| **ReferTrack** | **4B** | – | **89.4 / 92.5 / 1.6** | **73.3 / 81.8** / 7.6 | **74.1 / 85.7** / 7.7 |

**Multi-camera 참조 (3–4 카메라, 순위 비교 대상 아님)**

| Method | Size | RL | STT | DT | AT |
|---|---|---|---|---|---|
| ABot-N0 | 4B | – | 86.9 / 87.6 / 8.54 | 66.7 / 75.4 / 11.6 | 67.3 / 79.5 / 7.05 |
| NavFoM | 7B | – | 88.4 / 80.7 / – | 62.0 / 67.9 / – | – |
| CoMaTrack | 3B | ✓ | 92.1 / 90.3 / 0.9 | 74.2 / 80.5 / 2.1 | 57.5 / 73.4 / 12.0 |
| TrackVLA++ | 7B | – | 90.9 / 82.7 / 1.50 | 74.0 / 73.7 / 3.51 | 55.9 / 63.8 / 15.1 |

**읽어야 할 것들:**

1. **4B로 7B를 이긴다.** 1.3M navigation 샘플, RL 없이 SFT만으로 single-view 최강.
2. **식별이 병목인 split에서 격차가 폭발한다.** 최강 single-view 베이스라인 TrackVLA++ 대비 DT에서 +6.8 SR / +13.0 TR, AT에서 **+22.9 SR / +22.3 TR**. AT는 여러 후보가 지시문과 부분적으로 맞아 모호한 상황 — 정확히 indexed-bbox 객관식이 유리한 곳이다.
3. **Multi-camera를 넘어선다.** AT에서 ReferTrack의 74.1 SR은 multi-cam CoMaTrack(57.5, RL 포함)과 multi-cam TrackVLA++(55.9)를 크게 상회한다. 저자들의 해석: **타깃 disambiguation이 주 병목일 때는 명시적 image-space referring이 더 넓은 카메라 커버리지보다 효과적**이다.
4. **STT에서도 손해가 없다.** Single-view SR/TR/CR 모두 최고 — referring 인터페이스가 표준 추적 안정성을 희생시키지 않는다.
5. **CR은 최고가 아니다.** DT 7.6, AT 7.7로 TrackVLA++의 4.71(DT)보다 나쁘다. 논문이 강조하지 않는 부분이지만, 정확한 타깃을 더 적극적으로 쫓는 대가로 충돌이 늘었을 가능성이 있다.

### 6.2 Ablation (DT split, single forward-view)

| Variant | SR ↑ | TR ↑ | CR ↓ |
|---|---|---|---|
| ReferTrack (YOLO11-X) | 73.3 | 81.8 | 7.6 |
| TVBI w/ GT bbox † | 81.5 (+8.2) | 84.7 (+2.9) | 3.6 (−4.0) |
| w/o Refer-CoT & TVBI | 55.7 (−17.6) | 71.4 (−10.4) | 9.4 (+1.8) |
| w/o TVBI | 70.4 (−2.9) | 80.8 (−1.0) | 7.5 (−0.1) |

† Refer-CoT를 우회하고 ground-truth 타깃 bbox로 TVBI를 구성 — 완벽한 식별 하의 모션 플래닝 상한.

**해석:**

- **식별이 병목이다.** Oracle 변형이 81.5 SR, 전체 시뮬레이터 상태에 접근하는 **전문가 정책이 85.1 SR**. Oracle과 전문가 사이 간극(3.6)은 작고, oracle과 full model 사이 간극(8.2)은 크다 → 방해 상황에서 남은 손실의 대부분은 플래닝이 아니라 **식별 오류**에서 온다.
- **Refer-CoT가 견고성의 주된 원천.** 둘 다 제거하면 55.7 SR로 −17.6 급락. TVBI만 제거하면 −2.9에 그친다.
- **TVBI는 안정화 장치.** 시간적 bbox geometry가 추종 성공률을 올리지만, Refer-CoT 단독으로도 강한 식별 신호를 제공한다. TVBI는 선택된 타깃을 시간에 걸쳐 안정화하는 역할.

이 ablation은 또한 **TVBI가 이산 bbox geometry를 연속 추적 정책에 주입하는 유효한 인터페이스**임을 검증한다. 저자들은 이 타깃별 visual guidance가 강력한 tracking/ReID 모듈로도 공급될 수 있다고 지적한다 — 모듈 교체 가능성을 열어둔 셈이다.

### 6.3 실세계 배포

두 플랫폼, 각각 **단일 전방 카메라(Intel RealSense D455)** + 포터블 Wi-Fi:

- **Unitree Go2 (사족보행)**: 어수선한 장애물 사이로 타깃 추종. 보행자가 장애물을 우회할 때도 안정적이고, **좁은 FoV가 타깃의 하반신만 포착하는 상황에서도** 유지.
- **Unitree G1 (휴머노이드)**: 다수 인원 간섭 하에서 올바른 타깃 참조 및 추종 성공.

📌 [Figure 3 삽입] — 실세계 정성 결과. 좌: Go2, "검정 티셔츠와 긴 회색 바지 남성을 따라가라". 우: G1, "회색 티셔츠와 긴 검정 바지 남성을 따라가라".

**추론 파이프라인** (Appendix B): 클라우드 GPU 서버에서 WebSocket 서비스로 구동. 로봇이 JPEG 압축 RGB 프레임 + 지시문을 스트리밍하면, 서버가 디코딩 → 온라인 detector/tracker 갱신 → indexed bbox catalog 구성 → 예측 궤적 + 선택된 타깃 슬롯 반환. 네트워크 지터 하의 stale 명령을 피하려 **추론 중에는 최신 pending 프레임만 유지하고 선행 요청은 폐기**한다.

지연 최적화 3종:
1. 체크포인트는 서버 시작 시 한 번만 로드, 로봇 연결마다 독립 스트리밍 세션
2. `torch.compile`로 LLM 컴파일 + warm-up 스텝으로 steady state 도달
3. **DINOv2와 SigLIP 특징 추출을 별도 Python 스레드 / 별도 CUDA 스트림에서 병렬화**, concat 직전에만 동기화

결과: 전체 perception-and-control 루프 **평균 10.6 Hz**, 타깃 탐지는 스텝당 **12 ms**.

---

## 7. 관련 연구와의 위치

**vs. TrackVLA++** — 가장 직접적인 비교 대상. 둘 다 reasoning-then-action 패러다임이지만, **추론 어휘가 다르다**. TrackVLA++는 polar-coordinate 토큰이라는 추상 좌표를, ReferTrack은 indexed image-space bbox를 선택한다. 저자 주장: bbox-centric 표현은 VLM이 자연스럽게 소비하고 온보드 탐지기가 자연스럽게 생산하는 것이므로, 추론이 직접적 visual evidence에 grounding되고 **abstraction bottleneck이 최소화**된다.

**vs. NavFoM** — TVI 토큰으로 visual history를 구조화하는 계열을 따르되, referred target의 geometric feature를 주입하는 **TVBI로 확장**했다. 일반적 spatiotemporal indexing이 **target-conditioned memory**로 변환된다.

**vs. LOVON** — 계층적 LLM 플래너 + 저수준 컨트롤러. ReferTrack은 end-to-end 단일 정책.

**vs. RexSeek 계열 referring** — "이산 후보 중 선택"이 VLM에게 자연스러운 인터페이스라는 통찰을 EVT로 가져왔다.

---

## 8. 강점

1. **문제 재정식화가 깔끔하다.** 추상 latent 추론 → 제약된 객관식. 감독 신호가 명확해지고(cross-entropy 한 줄), 학습 데이터 확보 경로가 넓어진다.
2. **효율성 증거가 설득력 있다.** 4B / SFT-only / 1.3M 샘플로 7B, RL-refined, multi-camera 방법들을 이긴다. "스케일업·카메라 추가·RL 미세조정"이라는 통상적 처방을 아키텍처 설계로 대체할 수 있음을 보인다.
3. **Ablation이 주장과 정확히 대응한다.** Oracle bbox 변형 + 전문가 정책 상한을 함께 제시해, 남은 성능 손실이 어디에 있는지(식별)를 정량적으로 특정한다. 이런 상한 제시는 흔치 않다.
4. **⟨NO_EXIST⟩ 슬롯의 설계.** 타깃 부재를 별도 heuristic이 아니라 어휘 내 1급 시민으로 처리한다. 학습 시 [0,0,0,0] sentinel과 일관되게 맞물린다.
5. **큐 노이즈 주입.** Autoregressive 메모리가 자기 오류에 갇히는 문제를 학습 시 명시적으로 시뮬레이션한다. 실무적이고, 실세계 10.6 Hz 안정 동작에 기여했을 것이다.
6. **실세계 엔지니어링을 숨기지 않는다.** WebSocket 서비스, stale frame 폐기, dual CUDA stream 병렬 특징 추출까지 부록에 기술 — 재현성 측면에서 좋다.

---

## 9. 한계와 비판

1. **탐지기에 대한 근본적 의존.** 카탈로그가 YOLO11 + ByteTrack 출력이므로, 탐지기가 타깃을 놓치면 정답이 후보 집합에 아예 없다. 이 경우 모델은 ⟨NO_EXIST⟩를 내거나 오답을 고를 수밖에 없다. Oracle bbox 변형의 +8.2 SR 격차 중 얼마가 **탐지 실패**이고 얼마가 **선택 오류**인지는 분해되지 않았다.
2. **Top-K 면적 우선 정렬의 임의성.** K를 넘는 인원이 있으면 bbox 면적 기준 top-K만 남긴다. 지시된 타깃이 멀리 있어 작게 보이는 붐비는 장면에서는 정답이 잘려나갈 수 있다. K 값 자체가 본문에 명시되지 않았다.
3. **CR 회귀.** DT 7.6 / AT 7.7의 충돌률은 single-view TrackVLA++(4.71 / 15.9 중 DT)보다 DT에서 나쁘다. 논문은 이를 논의하지 않는다.
4. **실세계 평가가 정성적이다.** "Real-World Qualitative Evaluation"이라는 절 제목 그대로, **정량 성공률·시행 횟수·베이스라인 비교가 없다**. Figure 3의 롤아웃 2건이 전부다. 시뮬레이션 결과의 강도에 비해 sim-to-real 주장의 근거는 얇다.
5. **클라우드 서버 의존.** 10.6 Hz는 원격 고성능 GPU 서버 + Wi-Fi 기준이다. 온보드 추론 지연은 보고되지 않았다. 실배포 관점에서는 중요한 공백.
6. **확장성 주장의 미실증.** 웹 grounding 데이터, JRDB/TPT-Bench, KITTI/SiT로 카탈로그를 채울 수 있다는 것이 논문의 개념적 셀링 포인트인데, 실제로는 선행 연구와 동일한 SYNTH-PEDES만 썼다. 공정 비교라는 이유는 타당하나, 가장 흥미로운 가설이 검증되지 않은 채 남는다.
7. **단일 벤치마크.** EVT-Bench 하나에만 평가한다. TPT-Bench 같은 실로봇 egocentric 추적 데이터셋(저자 그룹 자신의 선행 연구)에서의 결과가 없다.
8. **Refer-QA 합성의 도메인 격차.** 384×384 배경에 2–3개 보행자 crop을 붙여넣은 합성 이미지와 Habitat 렌더링 / 실세계 RealSense 프레임 사이의 격차가 얼마나 전이를 제한하는지 측정되지 않았다.

---

## 10. 재현성 체크리스트

| 항목 | 상태 |
|---|---|
| 코드 공개 | ✅ https://github.com/MedlarTea/referTrack (논문에 명시) |
| 베이스 코드베이스 | ✅ OpenTrackVLA |
| LLM 백본 | ✅ Qwen3-4B |
| Vision encoder | ✅ SigLIP + DINOv2, grid pooling (64 fine / 4 coarse) |
| Detector | ✅ YOLO11 + ByteTrack |
| 학습 데이터 출처 | ✅ EVT-Bench train split (Habitat 3.0) + SYNTH-PEDES |
| 데이터 큐레이션 절차 | ✅ oracle controller 상세 기술 (Appendix A.1) |
| 옵티마이저 하이퍼파라미터 | ✅ AdamW, cosine + warm-up, LR 2e-5 / 1e-4, 20K steps, batch 256 |
| 손실 가중치 | ✅ α = 10 |
| GPU 시간 / 하드웨어 | ❌ 미보고 |
| 슬라이딩 윈도우 H | ❌ 구체적 값 미명시 |
| 카탈로그 크기 K | ❌ 구체적 값 미명시 |
| Waypoint 개수 M | ❌ 구체적 값 미명시 |
| 실세계 정량 결과 | ❌ 정성적 롤아웃만 |

---

## 11. 세미나 토론 질문

1. **Oracle bbox 변형(81.5 SR)과 full model(73.3 SR) 사이 8.2점 격차 중, 탐지기가 타깃을 놓친 경우와 LLM이 잘못 골랐던 경우의 비율은? 전자가 지배적이라면 이 논문의 기여는 "더 좋은 탐지기를 쓰라"로 축소되는가?**
2. **Refer-CoT를 단일 토큰으로 제한한 것은 compact함을 얻지만 표현력을 잃는다. 여러 후보가 동등하게 그럴듯한 진짜 모호 상황에서, 확률 분포 전체를 downstream에 넘기는 soft selection이 더 낫지 않을까? 그 경우 TVBI 큐에는 무엇을 넣는가?**
3. **현재 프레임을 TVI-only로 유지하는 설계는 "모델이 스스로 grounding하도록 강제"한다는 논리인데, 이것이 ablation으로 검증되었나? (현재 프레임에도 bbox를 주입한 변형이 표에 없다.)**
4. **AT split에서 multi-camera 방법을 크게 앞선 결과는, ReferTrack이 잘한 것인가 아니면 기존 multi-camera 방법들이 AT에서 유독 약한 것인가? AT의 모호성은 카메라 각도로 해소되는 종류가 아니라면, 공정한 대조인가?**
5. **큐 노이즈 주입 비율과 형태(다른 보행자 vs ⟨NO_EXIST⟩)에 대한 민감도는? 이 하이퍼파라미터가 실세계 견고성의 실질적 결정 요인일 가능성.**
6. **CR이 TrackVLA++보다 나쁜 이유는? 더 확신에 찬 추종이 더 공격적인 궤적을 낳는가, 아니면 waypoint MSE 손실에 충돌 항이 없기 때문인가?**
7. **논문이 제시한 "웹/자율주행 데이터로 referring 능력을 확장" 경로가 실제로 작동한다면, EVT 성능은 어디까지 갈까? 반대로 Habitat 렌더링과 실사 데이터의 도메인 격차가 이 전이를 막는다면?**
8. **10.6 Hz 원격 추론은 실제 사람 추종(보행 속도 ~1.4 m/s)에 충분한가? 네트워크 왕복 지연이 포함된 end-to-end latency는 보고되지 않았다.**

---

## 12. 총평

**기여의 성격**: 새로운 아키텍처 부품의 발명이라기보다 **추론 인터페이스의 재정식화**다. "타깃을 추상 좌표로 상상하지 말고, 탐지기가 이미 만들어준 상자 중에 고르라"는 단순한 전환이, 감독 신호를 명확하게 만들고 VLM의 기존 grounding 능력에 정확히 올라탄다. TVBI는 이 선택을 시간축으로 전파하는 부수적이지만 필수적인 배관이다.

**설득력**: 강하다. 4B / SFT-only / single-camera가 7B / RL / multi-camera를 식별 중심 split에서 압도한다는 결과는, 문제 정식화가 스케일보다 중요할 수 있다는 반복되는 교훈의 좋은 사례다. 특히 oracle 상한과 전문가 정책 상한을 함께 제시해 **"남은 손실이 어디 있는지"를 정직하게 특정**한 점은 배울 만하다.

**약점**: 실세계 검증이 정성적 수준에 머문다는 것, 탐지기 실패 모드가 분해되지 않았다는 것, 그리고 논문의 가장 매력적인 주장(웹 규모 referring 데이터로의 확장 가능성)이 실증되지 않았다는 것. 세 번째는 후속 연구의 명백한 다음 단계다.

**영향**: EVT를 넘어, "VLA의 CoT는 어떤 어휘로 이루어져야 하는가"라는 더 넓은 질문에 하나의 답을 제시한다 — **downstream 모듈이 이미 생산·소비하는 표현을 추론 어휘로 삼으라**. 조작(manipulation) VLA에서도 grasp proposal, affordance region, object mask index 등을 같은 방식으로 쓸 수 있는지가 자연스러운 확장 방향이다.

<!-- VERIFIED: pdf -->
