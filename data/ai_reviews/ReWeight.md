# ReWeight — Leveraging Human Data for VLA Post-Training via Demonstration Retrieval and Sample Weighting

- arXiv: 2609.13851 (v1, 2026-09-12)
- 소속: The University of Hong Kong(기계공학과), TeleAI(China Telecom), Northwestern Polytechnical University 선전 연구원
- 학회: Not stated in the paper / 프로젝트 페이지: reweight-vla.github.io (코드 공개 여부는 논문에 명시되지 않음)

## 1. 한 줄 요약

ReWeight는 대규모 1인칭 인간 시연(EgoDex)을 π0.5 사후학습에 섞을 때, 교차-신체 visuomotor 표현과 최적수송(OT)으로 **어떤 인간 시연을 가져올지(검색)**와 **각 샘플을 얼마나 반영할지(가중치)**를 함께 결정해, RoboTwin 2.0 8과제 평균을 0.39(로봇 데이터만)/0.44(무작위 혼합) → 0.57로, 실세계 4과제를 40.0% → 68.8%로 끌어올린다.

## 2. 문제 설정

로봇 시연 수집은 비싸고, 인간 영상은 풍부하지만 신체 차이 때문에 무분별하게 섞으면 오히려 성능이 떨어질 수 있다. 기존 검색 기반 방법은 "무엇을 가져올지"만 다루고, 검색된 샘플 간의 이전 가능성 차이는 무시한다.

## 3. 핵심 아이디어

인간·로봇 시연을 같은 잠재 공간에서 비교할 수 있게 한 뒤, (i) 시연 단위 entropic OT(Sinkhorn) 불일치로 로봇 데이터와 같은 수의 인간 시연을 검색(1:1)하고, (ii) 프레임 단위 OT 불일치로 샘플별 연속 가중치를 부여해 사후학습 손실에 곱한다.

## 4. 아키텍처

- 인간 행동 정제: LeVR로 손 궤적을 로봇 행동공간으로 리타게팅(엄지-손가락 거리로 그리퍼 상태 추정) 후, PPO residual RL 정책(게이트된 보정)으로 속도·가속 제약 위반을 보정.
- 시각 인코더: ResNet-18 + temporal Transformer(8프레임). warm-up(SwAV + TCN + TCC) 후 도메인 적대 학습 + OT 대응 손실(시간 불일치 벌점 Q 포함)로 정렬.
- visuomotor 인코더: InfoVAE 구조로 시각 특징과 20스텝 미래 행동 청크를 융합, 단계(stage) 분류 헤드와 동일 단계 교차-도메인 정렬 보조 손실.
- 정책: π0.5(구조 변경 없음), 외부 카메라 뷰만 사용.

## 5. 학습과 추론

검색 점수는 각 인간 시연과 로봇 시연들 간 불일치 중 가장 작은 K=5개의 평균. 가중치 w = α + (1−α)·(정규화된 유사도), α=0.5. 손실은 로봇 샘플 손실 + 가중 인간 샘플 손실을 |D_mix|로 나눈 값. π0.5 사후학습: AdamW, LR 2.5e-5, 배치 64, H100 2장, 시뮬 20k/실세계 40k 스텝. 과제 그룹(적층 4과제 / 나머지 4과제)별로 인코더와 π0.5를 따로 학습한다. 추론은 일반 π0.5와 같다.

## 6. 주요 결과 — 시뮬레이션 (Table I, 과제당 100회)

| 설정 | Robot-Only | All1.0 | Random1.0 | Retrieval0.4 | Retrieval0.7 | Retrieval1.0 | ReWeight |
|---|---|---|---|---|---|---|---|
| Clean 8과제 평균 | 0.39 | 0.43 | 0.44 | 0.45 | 0.48 | 0.52 | **0.57** |
| Randomized 4과제 평균 | 0.34 | 0.29 | 0.32 | 0.38 | 0.38 | 0.40 | **0.42** |

무분별한 인간 데이터 혼합은 clean에서는 소폭 이득이지만 randomized에서는 오히려 손해(0.34 → 0.29/0.32)라는 점이 핵심 관찰이다. ReWeight 과제별(clean): Stack Bowls Two 0.96, Three 0.78, Stack Blocks Two 0.74, Three 0.30, Place Bread Basket 0.31, Can Basket 0.37, Object Basket 0.51, Put Object Cabinet 0.55.

## 7. 주요 결과 — 실세계 (DoBot 양팔)

Clean(과제당 20회): ReWeight 평균 68.8%, Robot-Only 40.0%, Naive Mixing 대비 +13.8pt(과제별 값은 Fig. 7 그림으로만 제시). 교란(Table IV, 조명·방해물 각 10회): ReWeight 48/80(60.0%) vs Naive Mixing 33/80(41.3%) vs Robot-Only 23/80(28.8%). 로봇 데이터에 없고 EgoDex에만 있는 바나나·포도도 집어 객체 수준 일반화를 보였다.

## 8. 어블레이션

인코더(Table II, 적층 4과제): 제안 인코더 Stage Acc 0.93 / KF Dist 4.1, 행동 정보 제거 0.83 / 10.0, 시간 벌점 제거 0.86 / 8.0 → 미래 행동 정보가 교차-신체 대응에 결정적. α 민감도(Table III, 3과제): α=0.1 0.65, 0.3 0.65, **0.5 0.68**, 0.7 0.64, 1.0(무가중) 0.62, Robot Only 0.40.

## 9. 분석

균일 가중 검색 변형 중에서도 가중치를 0.4 → 1.0으로 올리면 성능이 오르지만(인간 데이터 자체는 유용), 샘플별 차등 가중이 추가로 +5pt(clean)/+2pt(randomized)를 준다. 즉 "검색"과 "가중"이 서로 보완적이다. t-SNE(Fig. 5)는 정렬 후 인간/로봇 특징이 섞임을 보인다.

## 10. 강점

- "어떤 데이터를/얼마나"를 분리해 단계별 기여(All → Random → Retrieval(균일) → ReWeight)를 통제 비교했다.
- 무분별한 혼합이 randomized 조건에서 해롭다는 음성 결과까지 보고했다.
- 시뮬과 실세계, clean과 교란 조건을 모두 평가했다.

## 11. 한계

- RoboTwin 2.0 표준 50과제가 아니라 자체 선택 8과제(randomized는 4과제)라 전체 순위와 비교 불가.
- 과제 그룹마다 인코더·정책을 따로 학습해 확장성이 불명확하며, 파이프라인(리타게팅 + residual RL + 3단계 인코더 학습)이 무겁다.
- 단일 시드 결과이며 분산 보고가 없다.
- 실세계 과제별 수치가 그림으로만 제시된다.

## 12. VLA-Tracker 관점 평가

π0.5를 검색·가중된 인간+로봇 데이터로 실제 사후학습해 성능을 보고하므로 수록 대상이다. RoboTwin 결과는 비표준 부분집합이라 순위용 `robotwin_v2` 대신 `robotwin_v2_8task_clean` / `robotwin_v2_4task_randomized` 별도 블록(논문 표기대로 비율 단위)에 두었고, 실세계는 본문 수치(68.8%, 교란 60.0%)만 기록했다. 인간 영상 활용 VLA 사후학습의 "데이터 선택 + 샘플 가중" 계열 참고 사례로 유용하다.

<!-- VERIFIED: pdf -->
