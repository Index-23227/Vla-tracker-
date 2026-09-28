# GIFT: Guided Intermediate Feature Training via Action-Oriented Structural Supervision for Robotic Manipulation

> **한 줄 요약**: VLA와 World-Action Model(WAM)의 중간 시각 토큰에 기하(VGGT 정렬)·어포던스(객체 중심 20-D 상호작용 타깃)·목표 영역(마스크) 세 가지 학습 시점 보조 감독을 걸어 "action-sufficiency gap"을 줄이는 아키텍처 무관 프레임워크. 대표 변형 GIFT-WAM-IDM은 LIBERO 98.5%, LIBERO-Plus 제로샷 87.8%, RoboCasa GR1 82.3%, 실물 4과제 평균 87.5%.

- **arXiv**: 2609.04193v1 (2026-09-03, cs.RO)
- **소속**: 중국과학원 자동화연구소, UCAS, 칭화대, 푸단대, NUS
- **프로젝트**: https://openphoenix-team.github.io/GIFT-pages
- **트래커 표기**: 동명 논문(2609.07006 "GIFT: Goal-Injected Fine-Tuning")과 구분하기 위해 **GIFT-Feature**로 등록. 헤드라인 수치는 GIFT-WAM-IDM 변형.

---

## 1. 배경 및 동기

VLA는 비전-언어 사전학습으로, WAM은 비디오 예측으로 풍부한 시각 특징을 얻지만, 행동 감독이나 픽셀 재구성 목표는 제어에 필요한 구조(메트릭 기하, 객체-엔드이펙터 관계, 지시 대상 영역)를 보존한다는 보장이 없고 배경 텍스처 같은 지름길을 허용한다. 저자들은 이 "시각적 풍부함과 제어 유용성의 불일치"를 **action-sufficiency gap**이라 부른다. 기존 DreamVLA, GuidedVLA, Flex-π, World Guidance 등은 특정 요인·특정 정책 계열에 국한되어 있어, 공통 원리가 이질적 정책 구조에 이전되는지가 불분명했다.

## 2. 문제 정의

정책은 Â_t = A_φ(Z_t, C_t), Z_t = E_θ(O_t, l, s_t) 꼴. 목표는 행동 생성기 A_φ의 형태(직접 회귀 / 행동 diffusion / 역동역학)를 바꾸지 않고, 중간 특징 Z_t 자체가 제어에 필요한 구조를 담도록 학습 제약을 주는 것. 기본은 **no-injection**: 보조 예측을 행동 조건 C_t에 넣지 않고 역전파 손실로만 특징을 형성.

## 3. 방법

- **Geometry**: 동결 VGGT의 패치 특징을 학생 토큰 격자로 리샘플링하고, 얕은 층(r=6) 토큰에서 방향(코사인, λ_ang=0.2)과 로그 크기(λ_scale=0.05)를 분리 예측.
- **Affordance**: 슬롯 k마다 20-D 타깃 = 역할 ID + 앵커 좌표계의 엔티티 포즈(9-D) + 엔티티 좌표계의 엔드이펙터 포즈(9-D) + 닫힘 상태. K개 학습 쿼리 디코더가 최종층 토큰에서 Smooth-L1로 회귀.
- **Goal**: 지시 조건 저해상도 마스크를 BCE + soft Dice로 예측.
- **전체 손실**: L_native + 1.0·L_geo + 0.5·L_aff + 1.0·L_goal.
- **세 가지 인스턴스**: GIFT-VLA(StarVLA-OFT + Qwen3-VL-4B, BridgeAttention 헤드, L1 회귀), GIFT-WAM-Fast(Fast-WAM, 현재 프레임 특징으로 직접 행동 denoising), GIFT-WAM-IDM(미래 비디오를 먼저 denoise한 뒤 역동역학으로 행동 생성).
- **Injection 변형**: 어포던스·목표 디코더 특징을 행동 헤드에 추가 조건으로 넣는 대조군(기하는 항상 정렬 전용).

## 4. 데이터

- 시뮬레이션: LIBERO 4 suite, LIBERO-Plus(7개 섭동, 제로샷), RoboCasa GR1 Tabletop 24과제(과제당 1,000 데모). 기하 타깃은 VGGT, 어포던스 타깃은 시뮬레이터 특권 포즈, 목표 마스크는 인스턴스 분할에서 생성.
- 실물: xArm7(과제 1–2)과 ARX X5 듀얼암(과제 3–4), 과제당 100 텔레오퍼레이션 데모(총 400). 녹색 장갑을 절반은 비디오 인페인팅으로 제거. Grounding DINO + SAM2 + Orient Anything V2로 마스크·6-DoF 포즈 라벨 구축.

## 5. 구현 세부

- GIFT-VLA: Qwen3-VL-4B-Instruct, 224×224, 32-step chunk, 32 GPU × batch 8, 60k step, AdamW.
- GIFT-WAM: lr 1e-4, weight decay 1e-2, cosine + 5% warm-up, 32-step chunk; LIBERO는 64 GPU × batch 8 × 50 epoch, RoboCasa는 64 GPU × batch 16 × 최대 60k step.
- 추론 denoising 2 step(IDM은 비디오 2 + 행동 2). LIBERO/LIBERO-Plus는 10 step마다, RoboCasa는 12 step마다 재계획.
- 실물: Zhenwu 810E PPU 64개, 50k iteration, 상대 관절 명령.

## 6. 실험 설정

- 정합 비교 3쌍: GIFT-VLA vs StarVLA-OFT, GIFT-WAM-Fast vs Fast-WAM, GIFT-WAM-IDM vs Fast-WAM-IDM.
- LIBERO는 과제당 50 rollout, LIBERO-Plus는 전체 평가셋 각 1회, RoboCasa는 과제당 50 rollout.
- 절제: 단일 감독(Table 4), injection 유무(Table 5), denoising step(Table 6), 행동 헤드 설계(Table 7).
- 실물: 과제당 10회, 과제 2·4에 2단계 섭동(Table 9).

## 7. 주요 결과

**Table 1 – LIBERO (%)**

| 모델 | Spatial | Object | Goal | Long | Total |
|---|---|---|---|---|---|
| StarVLA-OFT | 97.8 | 98.6 | 96.2 | 93.8 | 96.6 |
| GIFT-VLA | 99.0 | 99.2 | 98.4 | 94.8 | 97.9 |
| Fast-WAM | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| GIFT-WAM-Fast | 97.8 | 99.6 | 98.6 | 94.8 | 97.7 |
| Fast-WAM-IDM | 98.8 | 97.8 | 97.8 | 97.6 | 98.0 |
| **GIFT-WAM-IDM** | 99.0 | 100.0 | 98.6 | 96.4 | **98.5** |

**Table 2 – LIBERO-Plus 제로샷 Total**: GIFT-WAM-IDM **87.8** (Camera 78.7 / Robot 80.5 / Language 90.2 / Light 99.0 / Background 89.2 / Noise 97.2 / Layout 83.3), ACoT-VLA 86.6, GAM 85.5, Fast-WAM-IDM 82.6. GIFT-VLA 79.6(vs StarVLA-OFT 75.0), GIFT-WAM-Fast 72.6(vs Fast-WAM 60.0).

**Table 3 – RoboCasa GR1 (Art. / P&P / Avg)**: GIFT-WAM-Fast 83.3/83.7/**83.6**, GIFT-WAM-IDM 84.3/81.7/82.3, GIFT-VLA 60.0/61.9/61.4; Fast-WAM 74.6, Fast-WAM-IDM 73.9, DIAL 70.2. 관절 물체 과제에서 +21.3(Fast), +24.6(IDM).

**Table 5 – Injection 절제 (LIBERO-Plus / RoboCasa)**: no-injection이 세 변형 모두에서 더 좋음 — GIFT-VLA 76.7→79.6 / 58.6→61.4, WAM 변형은 0.2–0.3pt 차.

**Table 7**: 행동 헤드 변경 자체의 효과는 75.0→75.7(+0.7)이고, 구조 감독이 추가로 +3.9pt.

**Table 8 – 실물(원 설정)**: GIFT-WAM-IDM 9/8/10/8 → **87.5%** vs Fast-WAM-IDM 52.5%; GIFT-VLA 57.5% vs StarVLA-OFT 35.0%. **Table 9 섭동**: GIFT-WAM-IDM 과제2 7/10·7/10, 과제4 8/10·5/10(합계 67.5% vs 15.0%).

## 8. Related Work 상의 위치

- 기하 중심(SpatialVLA, GeoVLA, Spatial Forcing), 어포던스/목표 중심(CoA-VLA, ReconVLA), 미래 예측 인터페이스(DreamVLA, World Guidance, DIAL), 디코더 특화(GuidedVLA)와 비교해, GIFT는 **공유 중간 특징에 대한 학습 제약**으로 세 신호를 통합하고 추론 경로에서 배제한다.
- 저자들의 선행 PokeVLA(기하 + 목표)를 어포던스로 확장하고 VLA·WAM 양 계열로 일반화.

## 9. 강점

1. 세 정책 계열에서 정합 백본·동일 학습 설정으로 비교해 "감독 원리"의 이전성을 직접 보여 줌.
2. no-injection이 더 낫다는 결과와 Table 7의 헤드 대조군으로 성능 향상을 표현 형성에 귀속시킨 통제 설계.
3. 관절 물체(RoboCasa)와 고정밀 양팔 삽입(실물 과제4)에서 큰 향상, 섭동 하 실물 강건성.
4. 추론 시 교사·보조 헤드가 필요 없어 배포 비용 증가가 없음.

## 10. 약점 및 한계

1. 어포던스 타깃이 시뮬레이터 특권 포즈 또는 사람 검수 파이프라인(Grounding DINO + SAM2 + 수동 교정)에 의존해, 대규모 데이터로 확장할 때 라벨 비용이 크다.
2. 표준 LIBERO 향상은 0.1–1.3pt로 포화 영역이며 시드 반복·분산 보고가 없다.
3. LIBERO-Plus에서 세 변형 간 편차가 크고(72.6–87.8), GIFT-WAM-Fast는 Camera 37.4로 여전히 취약. 변형 선택에 따라 결론이 달라진다.
4. 실물 평가는 과제당 10회로 표본이 작다.
5. 벤치마크별 post-training만 수행해 대규모 사전학습 단계에서의 효과는 검증되지 않았다.

## 11. 재현 및 확장 아이디어

- 라벨 비용 절감을 위해 어포던스 타깃을 자동 추정(foundation pose estimator)으로 대체했을 때의 손실 측정.
- 사전학습 단계(크로스 임바디먼트 데이터)에 GIFT 목표를 적용해 스케일링 효과 확인.
- LIBERO-Plus 섭동 유형별로 감독 가중치를 조정하는 적응형 스케줄.
- 다중 시드 및 RoboTwin 2.0 같은 양팔 벤치마크로 확장.

## 12. 총평

GIFT는 "중간 특징이 무엇을 보존해야 하는가"를 기하·어포던스·목표 세 축으로 명시하고, 이를 행동 생성 인터페이스와 분리된 학습 제약으로 구현해 VLA와 두 WAM 모두에서 일관된 이득을 보인 논문이다. 특히 injection 없이도 더 좋다는 점과 헤드 대조군은 기여를 깔끔하게 분리한다. 다만 라벨 구축 비용과 포화된 LIBERO에서의 작은 차이, 제한된 실물 표본은 감안해야 한다.

**한 문장 요약**: 동결 VGGT 정렬 + 객체 중심 어포던스 + 목표 마스크로 중간 토큰을 감독해 GIFT-WAM-IDM이 LIBERO 98.5%, LIBERO-Plus 제로샷 87.8%, RoboCasa GR1 82.3%, 실물 87.5%를 달성한 아키텍처 무관 표현 학습 프레임워크.

<!-- VERIFIED: pdf -->
