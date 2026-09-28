# VLAct — Beyond Data Scaling: Representation-Centric Continued Pre-training for Vision-Language-Action Models

- arXiv: 2608.27550 (v1 2026-08-27)
- 소속: StarVLA 팀 (자문: Jiaya Jia, Bei Yu, Hengshuang Zhao, Zhuotao Tian, Shu Liu, Pengguang Chen). 논문 본문에 기관명이 명시되지 않음

## 1. 한 줄 요약

VLAct는 Qwen3-VL-4B를 공개 로봇 데이터로 **연속 사전학습(continued pre-training)**하면서 (1) 얕은 층 보호와 캡션 혼합으로 VLM 사전지식을 보존하고, (2) OFT·PI·GR00T 세 가지 연속 행동 헤드를 동시에 감독해 헤드 특화 표현 붕괴를 막고, (3) 부분 통합 교차-embodiment 행동 공간을 쓰는 "표현 중심" 레시피다. 다운스트림에서는 헤드를 새로 초기화해 미세조정하며 LIBERO-Plus 82.6%, RoboTwin 2.0(Data Scaling) Clean 92.5% / Random 90.8%를 기록한다.

## 2. 문제 설정

로봇 궤적은 웹 데이터처럼 긁어모을 수 없고 물리 세계를 희소하게만 덮는다. 따라서 같은 데이터 예산에서 "얼마나 전이 가능한 시각-행동 표현을 백본에 증류하느냐"가 핵심 병목이라는 것이 저자들의 문제의식이다. 파일럿 실험(Fig. 2)에서 이들은 단순 VLA 사전학습의 세 가지 실패 양식을 제시한다: 좁은 로봇 데이터에 의한 VLM 특징 침식, 단일 행동 헤드 감독으로 인한 헤드 특화 붕괴(예: OFT 사전학습 백본이 OFT 헤드에서는 75.8로 +14.1이지만 GR00T 헤드에서는 28.9로 −22.3), embodiment별 출력 공간으로 인한 공유 약화.

## 3. 핵심 아이디어

"사전학습 헤드는 백본을 모양 짓는 도구일 뿐 재사용 대상이 아니다." 사전학습 단계에서만 다중 헤드·캡션 스트림·통합 행동 레이아웃을 쓰고, 미세조정 시에는 이들을 버리고 새 태스크 헤드를 붙인다. 모든 비교에서 바뀌는 것은 VLM 백본 가중치뿐이므로(헤드·데이터·옵티마이저·예산 동일) 7.6–21.4pt의 개선이 백본에 귀속된다고 주장한다.

## 4. 아키텍처

- 백본: Qwen3-VL-4B (StarVLA 코드베이스).
- 사전학습 헤드: OFT(병렬 회귀), PI(flow matching), GR00T(diffusion 계열) 세 헤드가 같은 잠재 z를 받아 같은 행동 청크를 예측, L = L_OFT + L_PI + L_GR00T.
- 행동 공간: 그리퍼 차원은 embodiment 간 공유, 팔 차원은 embodiment별로 두고 비활성 차원은 마스킹. 주기적 관절각에는 360° 모듈로 잔차를 쓰는 wrap-aware loss.
- 다운스트림: 기본은 OFT 헤드(표에서 VLAct = VLAct-OFT), GR00T·PI 헤드도 평가.

## 5. 학습과 추론

사전학습 데이터는 DROID, InternA1, RoboCoin, MolmoAct(Franka 단일팔 + AgileX 양팔)와 캡션 데이터이며 16 GPU로 학습한다. 사전학습 중에는 비전 인코더와 LLM 하위 절반을 고정하고 상위 층과 헤드만 갱신하며, 미세조정 시 전체를 푼다. 실로봇은 단일팔용·양팔용 모델을 각각 8×H800에서 50k 스텝 학습한다.

## 6. 주요 결과 — LIBERO-Plus (Table 1)

| 방법 | Camera | Robot | Lang. | Light | Bg. | Noise | Layout | Total |
|---|---|---|---|---|---|---|---|---|
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| Abot-M0 | 60.4 | 67.9 | 86.4 | 96.2 | 91.6 | 86.4 | 82.6 | 80.5 |
| Qwen3VL-OFT (사내 베이스라인) | 47.0 | 60.1 | 87.0 | 96.3 | 95.3 | 73.1 | 79.2 | 75.0 |
| **VLAct** | **73.9** | **68.4** | 81.5 | **96.7** | **96.7** | 86.0 | **83.3** | **82.6** |

같은 백본 계열·OFT 헤드의 Qwen3VL-OFT 대비 +7.6pt이며, Camera(+26.9)·Noise·Layout에서 격차가 크다. Language 축은 오히려 베이스라인보다 낮다(81.5 vs 87.0).

## 7. RoboTwin 2.0 (Table 2)

| 방법 | Base Clean | Base Random | Scaling Clean | Scaling Random |
|---|---|---|---|---|
| π0.5 | 60.2 | – | 82.7 | 76.8 |
| X-VLA | 70.0 | 39.0 | 72.8 | 72.8 |
| Fast-WAM | – | – | 91.9 | 91.8 |
| HoloBrain-0-QW | – | – | 91.9 | 92.3 |
| Qwen3VL-OFT | 61.7 | 10.5 | 88.2 | 88.3 |
| **VLAct (OFT)** | **80.5** | **41.5** | 92.5 | 90.8 |
| VLAct (GR00T) | 76.0 | 22.9 | 89.6 | 87.4 |
| VLAct (PI) | 77.0 | 23.7 | **93.0** | 88.8 |

Base 설정(과제당 clean 50개)에서 가장 강하고, Data Scaling 설정(clean 50 + randomized 500)에서는 Random 기준으로 HoloBrain-0·Fast-WAM보다 약간 낮다. 저자들도 "절대 SOTA 주장은 피한다"고 명시한다. 트래커 robotwin_v2 블록에는 Data Scaling의 Clean/Random을 기록했다.

## 8. 추가 벤치마크와 교차-embodiment 전이

- VLA-Arena(부록 Table 4): 평균 54.8 (π0.5 44.3, Qwen3-VL-OFT 33.4). Long-Horizon 50.0, Safety 63.2.
- DOMINO(부록 Table 5, 35개 동적 과제): SR 18.50 / MS 34.20 (Qwen3VL-OFT 10.86 / 30.49).
- RoboDojo(Table 3, ARX X5, 사전학습에 없던 embodiment): 평균 점수 10.66, 성공률 7.60%로 35개 정책 중 점수 8위·성공률 6위. 지정된 WAM 항목 중 최고인 X-WAM(7.69 / 3.83)을 두 지표 모두 앞선다. 다만 Memory 축은 0.66 / 0.56으로 매우 약하다.
- RoboCasa-GR1(휴머노이드, 사전학습 미포함): 데이터 20%에서 49.5%, 50%에서 51.0%, 100%에서 54.0%. GR00T-N1.6 전체 데이터 47.6%, Qwen3VL-OFT 48.8%, π0.5 37.0%.
- 실로봇(Franka Research 3, 과제당 10회): 단일팔 단기 ID 평균 92.5 vs 77.5, 양팔 평균 72.0 vs 44.0, OOD 장기 과제 82.5/83.3 vs 47.5/46.6.

## 9. 어블레이션

- 얕은 층 보호(부록 Table 6): 백본 전체 갱신 78.9/77.1 → 비전만 고정 81.3/79.3 → 비전 + LLM 하위 절반 고정 82.6/80.5 (LIBERO-Plus / RoboTwin 2.0).
- 캡션 혼합, 다중 헤드 사전학습의 같은 헤드 적응 효과, wrap-aware loss, UMI 데이터 추가 등은 부록 D–G에서 분석한다.
- 다운스트림 헤드를 바꿔도 Data Scaling Clean이 89.6–93.0으로 3.4pt 이내에 머물러 "전이되는 것은 헤드가 아니라 백본 표현"이라는 주장을 뒷받침한다.

## 10. 강점

- 모든 비교에서 백본만 바꾸는 통제 설계로 사전학습 레시피의 기여를 분리했다.
- 완전 공개 데이터와 16 GPU라는 작은 예산으로 산업계 시스템(ABot-M0, LingBot-VLA 등)과 경쟁한다.
- LIBERO-Plus, RoboTwin 2.0, VLA-Arena, DOMINO, RoboDojo, RoboCasa-GR1, 실로봇까지 평가 범위가 넓고, 보지 못한 embodiment 두 종류(GR-1, ARX X5)로 전이를 확인했다.
- 파일럿 실험으로 "헤드 특화 표현 붕괴"라는 현상을 명확히 제시한 점이 설득력 있다.

## 11. 한계

- 표준 LIBERO 4-suite 결과가 없어 가장 널리 쓰이는 순위표와 직접 비교할 수 없다.
- RoboTwin Data Scaling 비교에서는 학습 연산량·스케줄·체크포인트 선택이 방법마다 다를 수 있다고 저자 스스로 밝힌다.
- 실로봇은 과제당 10회로 표본이 작고, 일부 수치는 그림에만 있다.
- RoboDojo Memory 축과 LIBERO-Plus Language 축에서는 약점이 드러난다.
- 본문에 기관 표기가 없어 소속 확인이 어렵다.

## 12. VLA-Tracker 관점 평가

사전학습된 VLM에서 출발해 자체 로봇 데이터로 백본을 연속 사전학습하고 다운스트림 미세조정까지 수행한 자체 학습 정책이므로 수록 대상이다. 트래커에는 LIBERO-Plus 7개 축과 Total(82.6)을 `benchmarks.libero` 안의 libero_plus_* 키로 넣었고, 표준 LIBERO 점수가 없으므로 libero_avg는 선언하지 않았다. RoboTwin 2.0은 다른 최신 시스템과 같은 프로토콜인 Data Scaling 설정의 Clean 92.5 / Random 90.8을 헤드라인으로 두고, Base 설정·헤드 변형·베이스라인은 별도 블록으로 분리했다. RoboCasa-GR1은 표준 RoboCasa와 다르므로 robocasa_gr1로, VLA-Arena·DOMINO·RoboDojo도 각자 블록에 기록했다.

<!-- VERIFIED: pdf -->
