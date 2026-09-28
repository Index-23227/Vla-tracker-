# UCAG-P — One Policy, Many Embodiments: Unified Camera-Centric Action Geometry Pre-training for Heterogeneous Embodied Manipulation

- arXiv: 2608.26058 (2026-08-26)
- 소속: Xiaomi Embodied Intelligence Team × University of Macau
- 프로젝트: https://public-bots.github.io/UCAG-P

## 1. 한 줄 요약

UCAG-P는 로봇 팔, 휴머노이드, 사람 손의 행동을 모두 **카메라 좌표계에서 관측되는 앵커 움직임**(손목·EE p0, 파지 중심 p1)으로 통일해 하나의 Qwen3-VL-4B 기반 정책을 사전학습한다. 이 공통 움직임은 기하 조건부 변환기가 각 로봇의 명령으로 바꾼다. 벤치마크별 미세조정 없이 체크포인트 하나로 LIBERO 98.25%, RoboTwin 2.0 Clean 88.66% / Randomized 89.20%, LIBERO-Plus zero-shot 82.0%, RoboCasa GR-1 62.0%를 낸다.

## 2. 문제 설정

로봇 학습 데이터는 형태, 카메라 배치, 제어 주파수, 행동 공간이 제각각이다. 같은 동작이 EE 델타, 관절 명령, 사람 손 움직임 등으로 다르게 기록된다. 기존 방법은 세 부류로 나뉜다. 임베디먼트별 헤드나 벤치마크별 미세조정으로 데이터를 분리하거나, Qwen-RobotManip처럼 카메라 좌표계 EE 행동으로 맞추지만 여전히 로봇 EE 중심이거나, 사람 영상을 리타게팅·비디오 합성으로 변환한 뒤에야 쓸 수 있다. 저자들은 추상적이면서도 실행 가능한 명령으로 되돌릴 수 있는 공통 행동 표현이 비어 있다고 본다.

## 3. 핵심 아이디어

여러 임베디먼트에 공통으로 남는 구조는 저수준 제어기가 아니라 **카메라에 보이는 조작 기하**다. 손목·EE와 파지 중심의 움직임을 카메라 좌표계로 표현하면 사람 손도 "또 하나의 임베디먼트"가 된다. 그래서 EgoDex, EgoVerse, VITRA 영상을 리타게팅 없이 같은 목표로 지도학습할 수 있다. 실행 가능성은 별도의 **기하 조건부 변환기**가 카메라→베이스 변환, Jacobian, 로봇 상태를 받아 맡는다.

## 4. 아키텍처

- **백본**: Qwen3-VL-4B-Instruct. 멀티뷰 RGB, 언어, 선택적 고유감각 입력에 학습 가능한 action-query 토큰을 덧붙인다.
- **Motion head**: 2560차원 action-token 특징을 5120차원으로 투영하고 residual MLP 블록 2개를 거쳐, 30스텝 × 30차원 카메라 중심 움직임을 예측한다. 30차원은 좌/우 매니퓰레이터(p0 XYZ, p1 XYZ, 회전 sin/cos, 그리퍼, 패딩) 각 10차원과 카메라 이동(평행이동 3 + 6D 회전 + 패딩) 10차원이다.
- **기하 조건부 변환기(action head)**: 움직임, 카메라 자세, Jacobian, 학습 가능한 쿼리 8개로 attention-pool한 VLM 은닉 상태를 각각 정규화해 2560차원으로 투영한다. 이를 2층 8-head Transformer 인코더로 융합해 30스텝 × 80차원 희소 qpos 명령 청크를 낸다.
- **80차원 희소 명령 레이아웃**: 좌팔, 좌EE, 좌손, 우팔, 우EE, 우손, 허리, 모바일 베이스에 10차원씩 배정한다. 임베디먼트별 마스크로 활성 슬롯만 손실 계산과 제어기에 쓴다. LIBERO는 0–9와 20–29, RoboTwin은 양팔과 그리퍼, GR-1은 허리까지 활성화한다.
- 두 헤드 모두 masked L1 회귀로 학습한다(L = L_geo + λ_cmd·L_cmd). 따라서 action_head_category는 regression으로 분류했다.

## 5. 데이터

체화 코퍼스는 1,020,672 에피소드, 6,373.586시간(Table 1)이다.
- 실로봇 266.3h: RoboChallenge, RoboCoin-Piper, DROID
- 시뮬레이션 3,767.5h: RoboCasa GR-1, LIBERO, RoboTwin2 ALOHA, InternData-MultiRobot(3,649h, 전체의 57.26%), InternData-Franka
- 사람 손 2,339.7h: VITRA, EgoDex, EgoVerse. MediaPipe 등으로 손목과 파지 중심 앵커를 검출해 pseudo-action으로 쓴다.
- Stage 1에는 ShareRobot, RefSpatial-v2, RoboVQA, RoboAfford 같은 비전-언어 데이터도 섞는다.

## 6. 3단계 학습 (Table A.1)

1. **Camera-centric specialization**: VLM과 motion head를 카메라 중심 라벨이 있는 모든 샘플로 학습한다. H20 128장, 200K 스텝, 글로벌 배치 1,536.
2. **Geometry-conditioned translation**: 정답 궤적, 상태, 캘리브레이션, Jacobian으로 변환기만 학습한다(이미지 없음). H20 8장, 10K 스텝.
3. **Joint robot-human training**: motion head와 변환기를 로봇·사람 혼합 데이터로 함께 최적화한다. 변환기는 예측 궤적을 입력으로 받는다. H20 64장, 10K 스텝. 일반 VL 데이터는 뺀다.

옵티마이저는 AdamW이고 학습률은 백본 1e-5, 행동 모듈 1e-4, 코사인 감쇠를 쓴다.

## 7. 주요 결과 — 통합 체크포인트 (Table 3, B.3, B.5)

| 방법 | 유형 | LIBERO | RoboTwin Easy | RoboTwin Hard | RoboCasa GR-1 |
|---|---|---|---|---|---|
| π0.5 | Specialist | 97.6 | 82.7 | 76.8 | 37.0 |
| ABot-M0 | Specialist | 98.6 | 86.0 | 85.0 | 58.3 |
| Being-H0.7 | Specialist | 99.2 | 90.2 | 89.6 | – |
| ZR-0 | Specialist | 97.8 | 88.7 | 87.9 | 69.3 |
| JoyAI-RA | Specialist | – | – | – | 63.2 |
| Qwen-VLA-Instruct | Generalist | 97.9 | 86.1 | 87.2 | 56.7 |
| **UCAG-P** | Generalist | **98.3** | **88.7** | **89.2** | **62.0** |

- LIBERO 스위트별(Table B.3)은 Spatial 98.8, Object 98.6, Goal 99.2(표 내 최고), Long 96.4, 평균 98.25다.
- RoboTwin 2.0 50과제(Table B.5)는 Clean 88.66, Randomized 89.20으로, Randomized가 표 내 최고다. OpenMicrowave(11/13)와 TurnSwitch(54/53)는 눈에 띄게 약하다.
- RoboCasa GR-1 24과제(Table B.4)의 평균은 62.0으로 JoyAI-RA(63.2)에 이어 2위이고 6개 과제에서 최고다.

## 8. 강건성과 전이

- **LIBERO-Plus (Table 4, zero-shot)**: Camera 51.2, Robot 92.8, Language 83.5, Light 98.9, Background 98.0, Noise 75.2, Layout 74.3, 평균 82.0이다. ABot-M0(80.5)보다 높다. 로봇 초기 상태 교란에 매우 강하지만 카메라 교란에는 약하다(ABot-M0 60.4 대비 51.2).
- **새 임베디먼트 (Table B.1)**: RoboTwin에서 ALOHA를 ARX로 바꾸면 zero-shot 35.0%다. 원래 임베디먼트와 격차가 크다.
- **Ablation (Table B.2)**: 변환기에 pooled VLM 특징을 넣으면 RoboCasa GR-1이 58.3에서 62.0으로 오른다.
- **실세계 Piper (본문, Figure 7)**: 과제당 100 시연으로 SFT하고 20회 시도했다. 빵 집기 60%(π0.5 20%, 사람 손 시연으로 학습), 서랍 열기 90%(85%), 양팔 그릇 쌓기 75%(65%)다.

## 9. 관련 연구와의 위치

π0, GR00T, X-VLA 계열은 임베디먼트 프롬프트나 전용 헤드로 이질성을 다루지만, 예측하는 행동은 여전히 로봇 고유 공간에 있다. Qwen-RobotManip은 카메라 좌표계 EE 행동으로 한 걸음 나아갔지만 사람 손을 직접 포괄하지 못한다. UCAG-P는 공통 목표를 "카메라에서 보이는 앵커 움직임"으로 한 단계 더 추상화하고, 실행 가능성은 Jacobian 기반 변환기로 되돌린다. 사람 영상을 행동 지도로 바로 쓰는 계열(Being-H, EgoVLA 등)과 크로스 임베디먼트 범용 정책 계열의 교차점에 있다.

## 10. 강점

- 벤치마크별 미세조정 없는 **단일 체크포인트**로 단일팔, 양팔, 휴머노이드를 모두 경쟁력 있게 다룬다. 공정성 면에서 specialist 수치보다 의미가 크다.
- 사람 손 2,340시간을 리타게팅이나 비디오 합성 없이 같은 손실로 학습한다.
- 슬롯, 마스크, 학습 스케줄, 과제별 결과까지 부록에 상세히 공개해 재현성이 높다.
- RoboTwin Randomized 점수가 Clean보다 높을 정도로 도메인 랜덤화에 강건하다.

## 11. 한계

- 카메라 캘리브레이션, 깊이 추정, 기구학, MediaPipe 키포인트 오차가 목표와 변환기로 그대로 전파된다.
- 크로스 임베디먼트 zero-shot은 35.0%에 그친다. 카메라 중심 움직임만으로는 형태·기구학 불일치를 해결하지 못한다.
- 실세계 평가는 Piper 3과제 × 20회로 규모가 작고, 그 결과도 SFT 이후의 수치다.
- 사람 손 데이터가 성능에 얼마나 기여하는지(사람 데이터 유무 ablation)는 벤치마크 수준에서 분리해 보여주지 않는다.
- 카메라 교란(LIBERO-Plus Camera 51.2)과 관절형 물체(OpenMicrowave)에 약하다.

## 12. VLA-Tracker 관점 평가

Qwen3-VL-4B 백본, motion head, 변환기를 3단계로 직접 학습한 자체 정책이므로 수록한다. 표준 LIBERO 4 스위트 평균(98.25)을 논문이 직접 보고하므로 libero_avg로 선언했다. LIBERO-Plus는 규칙에 따라 libero 블록 안의 libero_plus_* 키로 넣었다. RoboTwin 2.0은 50과제 Clean/Randomized를 기록했지만, 논문이 두 값의 평균을 보고하지 않아 robotwin_v2_avg는 선언하지 않았다. RoboCasa는 GR-1 24과제 평균 62.0이다. 미세조정 없는 generalist라는 점을 고려하면 리더보드 상위권 specialist들과 나란히 놓아도 설득력이 있다.

<!-- VERIFIED: pdf -->
