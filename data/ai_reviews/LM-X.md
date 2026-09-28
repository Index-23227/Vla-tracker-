# LM-X — Explainable Vision-Language-Action Modeling via Progress, Event, and Uncertainty Prediction

- arXiv: 2608.25757 (v1 2026-08-26, v4 2026-09-09)
- 소속: Humanoid Robot (Shanghai) Co., Ltd.; E-surfing Digital Life Technology Co., Ltd., China Telecom

## 1. 한 줄 요약

LM-X는 Cosmos-Reason2-2B 위에 **RTG(return-to-go, 진행도·상태 품질)**, **ETG(event-to-go, 다음 의미적 이벤트까지의 행동 청크)**, **이분산(heteroscedastic) action-flow 분산(국소 명령 신뢰도)** 세 신호를 행동 생성과 함께 사전학습한 약 6B 규모의 "설명 가능한" VLA다. 실패 롤아웃 1,000시간 이상을 포함한 20,000시간 이상의 실로봇 데이터로 학습해 RoboTwin2.0 50과제(randomized-hard) 74.1%, 실세계 7과제 73.5%를 기록하며 GR00T N1.7(55.4%, 50.7%)을 앞선다.

## 2. 문제 설정

대형 VLA는 관측→행동의 블랙박스로, 정책이 과제가 진척되고 있다고 보는지, 어떤 중간 전이를 노리는지, 지금 명령이 믿을 만한지를 드러내지 않는다. 진행도·이벤트 구조나 불확실성은 보통 행동 사전학습 뒤에 덧붙이거나 추출한다. 저자들은 이 세 설명 신호를 사전학습 단계에서부터 제어와 함께 학습하고 제어 경로 안에서 사용하는 기반 모델이 없다는 점을 문제로 삼는다.

## 3. 핵심 아이디어

생물의 감각운동 조직(결과 민감 신호, 사건 분절, 확률적 예측)을 공학적 사전지식으로 삼아 예측 순서를 **진행도 → 이벤트 → 행동**으로 둔다. RTG가 ETG를 조건화하고, 둘 다 행동 expert를 조건화하며, 불확실성은 행동 expert 내부에서 추정된다. 따라서 설명은 사후 서술이 아니라 제어 계산의 일부다.

## 4. 아키텍처

- 백본: Cosmos-Reason2-2B(GR00T N1.7과 같은 계열). 16번째 층 hidden token이 공유 표현 z_t다. 고유감각은 VLM을 거치지 않고 이벤트·행동 expert에만 들어간다.
- RTG expert: z_t만 보는 2층 Transformer. 128개 순서 bin의 기대값으로 스칼라 RTG를 복원하고 smooth-ℓ1로 학습한다. 보상은 성공 종료 0, 실패 종료 −C, 그 외 −1이다.
- Event expert: 32층 DiT, flow matching. s_t와 RTG를 받아 다음 검증된 이벤트까지의 행동 시퀀스를 60 step 길이로 예측한다(이벤트 도달 후에는 마지막 행동을 반복해 패딩). 학습 샘플의 86% 이상에서 다음 이벤트를 포함한다.
- Action expert: 32층 DiT, H=30, flow matching. z_t, s_t, RTG, 이벤트 hidden state를 조건으로 평균 속도와 log-분산을 함께 예측하고 가우시안 NLL로 학습한다. 분산은 Euler 적분을 따라 대각 근사와 Hutchinson 추정으로 행동 공간까지 전파한다.
- 전체 약 6B 파라미터.

## 5. 학습 데이터와 레시피

15종 이상 embodiment(휴머노이드 10종 이상)의 20,000시간 이상 실로봇 궤적이며, ACT·Diffusion Policy·π 계열·GR00T 기반 에이전트의 실패 롤아웃이 1,000시간 이상 포함된다. OXE와 AgiBot World를 10% 확률로 섞는다. 이벤트는 과제군별 제안 신호(그리퍼 에지, VR 정렬 마커, 속도 극소, 접촉 회로)로 후보를 만든 뒤 사람이 검증한다. 손실은 λ_R=0.1, λ_E=1, λ_A=1이며, 64장의 B200에서 전역 배치 3,072로 2 epoch(약 700k step, 약 20일) 학습한다. 실패 에피소드는 100 step마다 RTG 분기만 갱신한다. 사전학습 중 이벤트 임베딩을 20% 확률로 드롭한다. 다운스트림에서는 성공 에피소드만으로 전체 모델을 후학습한다.

## 6. 사전학습 전 구성요소 검증 게이트 (Table 1, 성공률 %)

RoboTwin2.0 45과제로 사전학습하고 5개 보류 과제로 후학습·평가했다.

| 과제 | Backbone | LM-RTG | LM-Event | LM-U | Post-hoc | LM-X |
|---|---|---|---|---|---|---|
| Handover microphone | 76 | 86 | 92 | 82 | 82 | 88 |
| Lift pot | 84 | 90 | 86 | 94 | 86 | 92 |
| Open microwave | 54 | 68 | 16 | 54 | 62 | 62 |
| Rank RGB blocks | 48 | 38 | 66 | 44 | 74 | 90 |
| Hit block with hammer | 56 | 62 | 60 | 52 | 60 | 66 |
| 평균 | 63.6 | 68.8 | 64.0 | 65.2 | 72.8 | **79.6** |

단일 헤드의 효과는 과제마다 다르고(이벤트만 쓰면 microwave 54→16), 세 헤드를 모두 사전학습에 포함한 LM-X가 5과제 모두 backbone을 넘는다. 같은 구조를 후학습에서만 붙인 Post-hoc(72.8)보다 6.8pp 높다는 점이 "공동 사전학습"의 핵심 근거다. 부록 Table 10에서는 후학습 시 이벤트 조건화가 rank blocks(90 vs 78)와 hammer(66 vs 60)에는 도움이 되지만 lift pot(80 vs 92)과 open microwave(2 vs 62)에는 해가 되어, 과제별 하이퍼파라미터로 선택한다.

## 7. RoboTwin2.0 50과제 (Table 2 / Table 11)

Aloha-AgileX, randomized-hard 데모 50개/과제로 후학습하고 과제당 100회 평가했다. LM-X 사전학습에는 시뮬레이션 데이터가 전혀 없다.

| 과제(대표) | GR00T N1.7 | LM-X |
|---|---|---|
| blocks_ranking_rgb | 60 | 93 |
| blocks_ranking_size | 52 | 82 |
| stack_blocks_two | 72 | 97 |
| stamp_seal | 22 | 74 |
| put_bottles_dustbin | 1 | 64 |
| rotate_qrcode | 23 | 93 |
| beat_block_hammer | 40 | 25 |
| move_playingcard_away | 48 | 19 |
| **50과제 평균** | 55.4 | **74.1** |

+18.7pp 향상이며, 다단계 순서가 필요한 과제에서 이득이 크고 beat_block_hammer 등 일부에서는 뒤진다.

## 8. 실세계 7과제 (Table 3, 성공률 %, 과제당 20회)

| 과제 | GR00T N1.7 | π0.5 | LM-X |
|---|---|---|---|
| Tape-roll pick-and-place (AgileX-Aloha) | 75 | 45 | 65 |
| Precision part insertion (Astribot S1) | 20 | 0 | 80 |
| Tableware organization (Astribot S1) | 20 | 15 | 90 |
| Cloth folding (Astribot S1) | 60 | 30 | 80 |
| Battery insertion (Tianji-Marvin) | 35 | 50 | 40 |
| Water-bottle placement, dual-arm (Loong S1) | 75 | 40 | 90 |
| Water-bottle placement, single-arm (Loong S1) | 70 | 75 | 70 |
| **평균** | 50.7 | 36.4 | **73.5** |

Loong S1은 사전학습에 전혀 없던 embodiment다. tableware organization(+70pp)을 빼도 67.5% vs 55.8%로 앞선다.

## 9. 설명 신호의 품질 (Tables 4–6, 실세계 7과제 평균)

같은 구조를 후학습 데이터만으로 처음부터 학습한 경우와 사전학습 체크포인트에서 시작한 경우를 비교한다.

| 지표 | w/o 사전학습 | w/ 사전학습 |
|---|---|---|
| RTG Spearman ρ ↑ | 0.71 | 0.91 |
| RTG MSE ↓ | 0.100 | 0.050 |
| ETG next-event MSE ↓ | 0.884 | 0.520 |
| 고오차 청크 탐지 AUPRC ↑ | 0.49 | 0.59 |
| 실패 경고 lead time (s) ↑ | 0.93 | 1.54 |

정성 분석에서 RTG는 잘못된 파지 후 목표를 바꾸는 순간 즉시 떨어지고, 정밀 삽입에서는 파지·팁 삽입·나사 정렬 이벤트마다 계단식으로 오른다. 분산의 시간 기울기는 조준·삽입 직전의 망설임 구간에서 경고를 낸다(10% 오경보율 기준).

## 10. 강점

- 세 설명 신호를 각각 직접 감독되는 목표로 정의하고 제어 경로에 연결해, 설명이 "사후 해석"이 아니라 정책 계산의 일부가 된다.
- 비싼 20일 사전학습 전에 post-hoc 헤드 대조군까지 포함한 게이트 실험으로 설계를 정당화한 점이 방법론적으로 성실하다.
- 실패 롤아웃 1,000시간 이상을 활용해 진행도와 퇴행을 구분하는 RTG를 학습한다.
- 사전학습에 없던 Loong S1과 시뮬레이션 도메인(RoboTwin2.0)으로의 전이를 보여 준다.

## 11. 한계

- 저자 스스로 밝히듯 실세계는 과제당 20회, 방법당 체크포인트 하나, 신뢰구간 없음이며, 시뮬레이션도 단일 학습 런이다.
- 대규모 결과는 GR00T N1.7·π0.5와의 end-system 비교로, 사전학습 데이터와 파라미터 규모가 달라 구성요소 기여를 분리하지 못한다. 구성요소 기여는 5과제 게이트 규모에서만 검증된다.
- 이벤트 조건화를 과제별로 켜고 끄는 선택이 필요하며, 일부 과제에서는 크게 해롭다(open microwave 2%).
- 신호는 예측으로만 평가되었고, 복구 행동 선택에 실제로 쓰였을 때의 효과나 인과적 충실성은 검증되지 않았다. 불확실성은 인식적(epistemic) 불확실성이나 OOD 안전을 보장하지 않는다.
- RoboTwin2.0 결과는 randomized-hard 단일 분할이라 Easy+Hard 평균을 보고하는 다른 모델과 직접 비교할 수 없다.

## 12. VLA-Tracker 관점 평가

Cosmos-Reason2-2B에서 시작해 20,000시간 이상 데이터로 **자체 사전학습**하고 과제별로 후학습한 약 6B 규모 VLA이며 정량 결과가 풍부하므로 추적 대상에 포함한다. 행동·이벤트 expert가 모두 flow-matching DiT이므로 `action_head_category: flow_matching`. RoboTwin2.0 결과는 randomized-hard 분할만의 평균이므로 `robotwin_v2_hard: 74.1`로 기록하고 Easy+Hard 전체 평균과 혼동되지 않도록 `robotwin_v2_avg`는 선언하지 않았다. 50과제 세부, GR00T 기준선, 게이트 어블레이션, 이벤트 조건화 어블레이션, 실세계 기준선 2종, 신호 품질 지표는 모두 형제 블록으로 분리했다.

<!-- VERIFIED: pdf -->
