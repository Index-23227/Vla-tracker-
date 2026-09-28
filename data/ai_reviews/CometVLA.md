# CometVLA — Co-Training on an Embodied Data Pyramid towards Physical Understanding

- arXiv: 2608.30289 (v1 2026-08-31)
- 소속: JD Explore Academy, CUHK-Shenzhen, Shenzhen Institute of AI and Robotics for Society, South China University of Technology, Sun Yat-sen University (교신저자 Xiaoqiang Ji)

## 1. 한 줄 요약

CometVLA는 로봇 행동 데이터와 **같은 도메인·같은 embodiment**에서 자동 생성한 물리 상식 VQA(CometData, 약 100만 쌍)를 embodied data pyramid(VQA·egocentric 영상·시뮬레이션·원격조작) 전체와 함께 co-training하고, VLM과 행동 전문가 사이에 단일 Global Action Prior(GAP) 토큰 병목을 둔 Qwen3-VL-4B 기반 VLA다. RoboTwin 2.0 전체 50과제에서 Easy 89.24%, Hard 88.38%를 기록했다.

## 2. 문제 설정

기존 물리 VQA 데이터는 웹 이미지나 비로봇 시뮬레이션에서 나와 로봇 행동 데이터 분포와 어긋나고, egocentric 영상은 보조 사전학습 정도로만 쓰인다. 또한 "VLM의 물리 이해가 좋아지면 실제 행동 생성도 좋아지는가"는 여러 요소를 동시에 바꾸거나(MIMO-Embodied 등) 행동 전문가를 과도하게 단순화한(VLM4VLA) 연구로는 제대로 답해지지 않았다.

## 3. 핵심 아이디어

데이터·모델·학습의 세 축에서 접근한다. (i) 데이터: 정책 학습에 쓰는 바로 그 로봇 데이터 레시피로 물리 VQA를 합성해 in-domain 물리 이해를 강화한다. (ii) 모델: 행동 헤드가 물리 상식을 소비하고 샘플 간 공유되는 운동 규칙성을 모으도록 GAP 토큰을 유일한 통로로 둔다. (iii) 학습: knowledge insulation 기반 단일 co-pre-training 단계로 모든 층위의 데이터를 결합한다.

## 4. 아키텍처

Qwen3-VL-4B 백본이 다시점 이미지, 언어, FAST 이산 행동 토큰을 자기회귀 목적(L_AR + L_fast)으로 모델링한다. DiT-B flow-matching 행동 전문가는 VLM 임베딩, GAP 토큰, 상태 토큰을 조건으로 연속 행동을 생성한다(L_fm). Stop-gradient 장벽 때문에 L_fm은 행동 헤드와 GAP 파라미터만 갱신한다. GAP 토큰(1024-d)은 백본 토큰열에 직접 삽입되어 [CLS]처럼 의미·시간 문맥을 모으고, attention 상으로만 장벽을 넘어 행동 전문가에 cross-attention으로 전달된다(비대칭 attention 위상). 모든 로봇 데이터는 양팔 형식, 32차원 패딩 행동 공간으로 리타깃한다.

## 5. 학습과 추론

CometData 생성: physics middleware가 고유감각·명령 신호에서 Hamiltonian jump, gripper step, power anomaly로 keyframe을 점수화해 top-K를 고르고, 5개 도메인 16개 생성기가 템플릿과 물리 상태 서술로 teacher VLM에 질의해 QA를 만든다(인간 검수 포함). CometBench는 그중 균형 잡힌 2,000문항 held-out 세트로 LLM judge가 100점 척도로 채점한다. 사전학습: AgiBot-World-Beta, InternData-A1, EgoLive, EgoDex, 일반 VQA, CometData 혼합, 100k 스텝, lr 3e-5, VLA:VLM 손실 가중 1.0:0.1, H200 32장. 후학습: RoboTwin 또는 실로봇 데이터로 전 파라미터 파인튜닝. 실로봇 추론은 원격 H200 서버에서 WebSocket으로 제공한다.

## 6. 주요 결과 — RoboTwin 2.0 (Table 2)

| 모델 | Easy | Hard |
|---|---|---|
| π0 | 46.42 | 16.34 |
| π0.5 | 62.86 | 60.30 |
| RDT | 34.50 | 13.72 |
| X-VLA | 70.00 | 39.00 |
| Motus | 88.66 | 87.02 |
| JEPA-VLA | 73.50 | 17.70 |
| HALO | 80.50 | 26.40 |
| LingBot-VLA | 86.50 | 85.34 |
| **CometVLA** | **89.24** | **88.38** |

부록 Table A3의 50과제 과제별 성공률 평균이 정확히 89.24/88.38로 일치한다. 낮은 과제는 hanging_mug(53/54), move_can_pot(Easy 55), handover_block(Hard 52), place_can_basket(Hard 51) 등이다. Motus 대비 차이는 Easy +0.58, Hard +1.36으로 크지 않다.

## 7. 어블레이션 (Table 2)

| 구성 | Easy | Hard |
|---|---|---|
| w/o phys. VQA (같은 양의 일반 VQA로 대체) | 83.54 | 83.46 |
| w/o GAP | 86.38 | 85.12 |
| CometVLA | 89.24 | 88.38 |

물리 VQA 제거가 가장 큰 하락(약 −5pt)을 보여, in-domain 물리 이해 데이터가 행동 성능에 기여한다는 주장의 핵심 근거가 된다. GAP 제거도 약 −3pt.

## 8. 실로봇 결과 (Table 1)

AgiBot G1 양팔 로봇, 화장대 정리 시나리오 5과제, 과제당 16회: BB cream 75.00, liquid foundation 31.25, lotion 37.50, makeup sponge 93.75, serum 43.75(%). 물체 크기·질감에 따라 편차가 크다. 베이스라인 비교는 없다. 실패 예로 lotion 과제에서 그리퍼를 너무 일찍 열어 병을 넘어뜨리는 사례를 제시한다.

## 9. VLM–VLA 상관 및 GAP 분석

같은 백본·행동 전문가 구성으로 데이터 혼합·학습 설정만 달리한 여러 체크포인트에서 CometBench 점수와 RoboTwin Easy 성공률 간 도메인별 Pearson r: Spatial Reasoning 0.721, Physics & Dynamics 0.645, Error & Tool 0.624, Task Understanding 0.586, Affordance 0.552. GAP 토큰 linear probing(5-fold)에서는 평균 풀링 VLM 표현 대비 gripper state 49.8 vs 42.8, motion magnitude 54.3 vs 50.4, motion direction 32.9 vs 32.2로, 2.5배 작은 차원에서도 제어 관련 정보를 더 잘 담는다.

## 10. 강점

- 물리 VQA를 정책 학습 데이터와 같은 도메인·embodiment에서 생성해 "물리 이해 → 행동" 경로의 도메인 격차를 직접 줄였다.
- 같은 양의 일반 VQA로 대체하는 통제 어블레이션으로 데이터 내용 자체의 효과를 분리했다.
- 전체 50과제 RoboTwin 2.0에서 Easy/Hard 모두 최상위이며 과제별 수치를 전부 공개했다. 특히 Hard(randomized)에서의 강건성이 두드러진다.
- VLM 벤치마크 점수와 VLA 성공률 사이의 도메인별 상관을 정량화했다.

## 11. 한계와 의문점

- 저자들도 인정하듯 상관 분석의 VLA 성공률이 85% 부근에 몰려 있어 통계적 근거가 약하고, 데이터 포인트 수나 체크포인트 목록이 명확하지 않다.
- 실로봇은 5개 유사 pick-place 과제·16회뿐이고 베이스라인이 없어 비교 의미가 제한적이다.
- RoboTwin 베이스라인 수치는 각 논문 발표값을 모은 것으로 보이며, 학습 데이터 규모(대규모 AgiBot/InternData 사전학습 포함)가 서로 다르다.
- LIBERO, CALVIN, SimplerEnv 등 다른 표준 벤치마크 결과가 없다.
- CometBench 채점이 LLM judge에 의존한다.

## 12. VLA-Tracker 관점 평가

Qwen3-VL-4B 백본, FAST 토큰, DiT flow-matching 전문가를 co-pre-training하고 RoboTwin/실로봇에 전 파라미터 후학습한 자체 정책이므로 수록 대상이다. 전체 50과제 결과이므로 표준 `robotwin_v2` 블록에 Easy 89.24 / Hard 88.38을 기록했고, 논문이 결합 평균을 보고하지 않아 `robotwin_v2_avg`는 선언하지 않았다(리더보드는 두 설정의 평균으로 순위화). 베이스라인, 어블레이션, 과제별 표(Easy/Hard), 실로봇은 별도 블록으로 분리했다. RoboTwin 2.0 Hard 설정에서 최상위권 기준점으로 쓸 만하다.

<!-- VERIFIED: pdf -->
