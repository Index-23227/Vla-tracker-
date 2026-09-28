# GaussianDream++ — Efficient 3D Gaussian World Modeling for Robotic Manipulation

- arXiv: 2608.25659 (2026-08-26)
- 소속: Tuojing Intelligence, 중국과학원대학(UCAS), 중국과학원 자동화연구소, 칭화대, USTC, HKUST(광저우), NTU, 베이항대, CMU, 홍콩대
- 코드: https://github.com/TuojingAI/GaussianDream
- 선행작: GaussianDream (arXiv 2605.20752, 트래커에 별도 등록)

## 1. 한 줄 요약

GaussianDream++는 선행작 GaussianDream의 "학습 시에만 3D Gaussian 현재 재구성·미래 예측으로 감독하고 배포 시 헤드를 버린다"는 원칙을 유지하면서, VGGT/TGE로 만든 1024토큰 dense prefix를 **PaliGemma 안에 직접 삽입한 20개 world token(State 16 + Prediction 4)**으로 대체한 π0.5 기반 정책이다. LIBERO 98.6%, LIBERO-Plus zero-shot Overall 87.8%, 실로봇 평균 52.5%(재현 π0.5 29.2%)를 보고한다.

## 2. 문제 설정

행동 모방 목표만으로는 메트릭 3D 구조와 단기 물리 변화가 약하게만 감독된다. 기하 강화 VLA는 현재 장면 grounding만 개선하고, 예측형/월드모델 정책은 RGB나 잠재 공간에서 미래를 모델링해 물리적 일관성이 보장되지 않으며 배포 비용이 크다. GaussianDream은 Gaussian 감독이 유효함을 보였지만, 추론 시에도 VGGT와 Temporal Gaussian Evolution(TGE) 경로로 1024토큰 prefix를 만들어야 했고, 그 하나의 prefix가 현재 상태·미래 동역학·행동 조건을 모두 떠안았다.

## 3. 핵심 아이디어

(1) world 표현을 외부 경로가 아니라 **VLA 백본 내부 토큰**으로 옮겨, Gaussian 감독으로 형성된 표현이 곧 Action Expert가 attention으로 읽는 표현이 되게 한다. (2) 현재 상태(World State Tokens)와 단기 변화(World Prediction Tokens)의 **역할을 분리**한다. (3) 미래는 현재 Gaussian을 템플릿으로 공유하고 중심 변위만 예측하며, **static-dynamic 분해**로 정지 영역의 가짜 움직임을 억제한다.

## 4. 아키텍처

π0.5(PaliGemma + flow-matching Action Expert)의 prefix에 N_s=16 State 토큰(4×4 공간 배치, 보정된 공간·ray 임베딩)과 N_p=4 Prediction 토큰(단기 예측 슬롯)을 추가한다. PaliGemma가 시각·언어 토큰과 함께 이들을 문맥화하고, Action Expert는 world-token이 포함된 전체 prefix에 조건화된다. 별도의 world→action projector는 없다. 학습 전용 World Representation Head가 State 토큰 은닉상태를 Grid로 펼쳐 dense Gaussian feature field로 디코딩한다. 기하 분기가 메트릭 depth와 scale/rotation/opacity를, 외관 분기가 stop-gradient된 기하 특징 위에서 SH 계수를 예측한다(광도 단축경로가 기하를 오염시키지 않게 함). Gaussian 중심은 보정된 카메라로 depth를 unprojection해 얻는다.

## 5. 학습 목표

L = L_act + λ_cur·L_cur + λ_fut·L_fut. L_cur는 미분가능 Gaussian splatting 렌더의 RGB(광도+SSIM), depth, alpha, 정규화 항이며 유효 depth 영역만 감독한다. L_fut는 horizon h마다 Prediction 토큰 H^P_{t,h}, 현재 Gaussian 특징, horizon 임베딩으로 Δμ를 예측하고 motion 계수 m∈[0,1]로 게이팅한 뒤(μ_{t+h} = μ_t + m·Δμ), scale·회전·opacity·색은 현재 것을 상속한다. 각 미래 world는 해당 시점 카메라로 렌더링해 시점 변화가 물리 운동으로 흡수되지 않게 하며, RGB·depth·메트릭 3D flow·static-consistency 항을 쓴다. 미래 관측은 목표 구성에만 쓰이고 정책 forward에는 들어가지 않는다. 즉 π0.5 전체를 이 목적함수로 미세조정한 자체 학습 정책이다(스텝 수·배치 등 세부 하이퍼파라미터는 본문에 없음).

## 6. 배포와 효율 (Table 3)

배포 시 World Representation Head, 렌더러, 목표 생성기, 보조 손실, VGGT/TGE 경로가 모두 제거되고 20개 토큰만 남는다. 동일 로컬 파이프라인에서 action chunk당 지연은 π0.5 재현 286 ms → GaussianDream++ 330 ms(약 1.15배, +44 ms)다. GaussianDream의 531 ms는 원 논문 보충자료 수치를 참고로만 인용했으며, 저자들 스스로 동일 환경 재측정 없이 정확한 속도 향상을 주장할 수 없다고 명시한다.

## 7. 주요 결과 — LIBERO / LIBERO-Plus (Table 1)

| 방법 (matched protocol) | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| π0.5 재현 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| GaussianDream | 99.0 | 99.6 | 99.0 | 96.0 | 98.4 |
| **GaussianDream++** | **99.2** | **99.6** | **99.0** | **96.6** | **98.6** |

| LIBERO-Plus (zero-shot) | Camera | Robot | Lang | Light | BG | Noise | Layout | Overall |
|---|---|---|---|---|---|---|---|---|
| π0.5 재현 | 73.2 | 77.9 | 84.3 | 96.3 | 95.5 | 89.9 | 87.7 | 85.5 |
| GaussianDream | 77.3 | 72.1 | 86.9 | 98.9 | 98.5 | 93.6 | 88.4 | 87.0 |
| **GaussianDream++** | **80.1** | 73.0 | **87.4** | **99.1** | **98.8** | **94.2** | **90.0** | **87.8** |

LIBERO-Plus Overall은 공식 task-level 집계이며 7개 축의 단순 평균이 아니다. Table 1의 published 결과 중 LIBERO-Plus 최고치(ACoT-VLA 86.6)보다 높고, 향상은 Camera(+6.9 vs π0.5, +2.8 vs GaussianDream)와 Layout(+2.3, +1.6)에서 가장 뚜렷하다. Robot 축은 π0.5 재현(77.9)보다 낮다(73.0).

## 8. 실로봇 결과 (Table 2)

듀얼암 플랫폼에서 Bowl-Proximity(정밀 상대 배치)와 Eggplant-to-Pink-Plate(방해물 속 장기 조작) 2과제 × Standard/Layout/Camera 3조건 × 20회. 과제별 Overall: Bowl 25.0 → 46.7, Eggplant 33.3 → 58.3. 조건별 pooled: Standard 40.0 → 62.5, Layout 25.0 → 52.5, Camera 22.5 → 42.5. 전체 35/120(29.2%) → 63/120(52.5%). 비교 대상은 재현 π0.5뿐이며 GaussianDream 원작은 실로봇에서 비교되지 않았다.

## 9. 어블레이션 (Table 4)

구조: 감독 없는 world token만 추가해도 Overall 85.5 → 86.3(용량 효과), Current World만 86.9, 비결합 Current+Future 87.2, 결합(static consistency 없음) 87.5, 전체 87.8로 단조 증가한다. 감독 항: metric depth 제거가 가장 큰 손실(Overall 86.9, Camera 76.5, Layout 87.7), metric 3D flow 제거 87.2, alpha 제거 87.3, RGB 렌더링 제거 87.4. LIBERO 평균은 97.5~98.6 범위로 포화되어 차이가 작다.

## 10. 강점

- 선행작의 배포 시 VGGT/TGE 의존을 없애면서도 성능을 소폭 개선해, "학습 시 풍부한 3D 감독, 배포 시 가벼운 정책"이라는 비대칭 설계를 더 밀어붙였다.
- 동일 프로토콜 블록(π0.5 재현 / GaussianDream / ++)을 명확히 분리해 published 수치와의 비교를 과장하지 않는다.
- 토큰 용량 대비 감독 효과를 분리한 어블레이션이 설득력 있다.
- 실로봇에서 Camera·Layout 이동을 물리적으로 재현해 LIBERO-Plus 결과와 연결했다.

## 11. 한계

- 학습 하이퍼파라미터(스텝, 배치, λ 가중치, 데이터 규모)와 파라미터 수가 본문에 없다.
- LIBERO는 포화 상태여서 +0.2pt 차이의 통계적 의미가 불분명하며, 시드 분산이 보고되지 않는다.
- GaussianDream 대비 효율 우위는 서로 다른 환경의 지연을 비교한 참고치에 기대고 있다.
- LIBERO-Plus Robot 축에서는 π0.5 재현보다 약하다(73.0 vs 77.9); 선행작이 보고한 RoboCasa 평가는 이번에 없다.
- 실로봇은 2과제, 단일 베이스라인이다.

## 12. VLA-Tracker 관점 평가

π0.5를 3D Gaussian world 감독과 함께 미세조정한 자체 정책으로, GaussianDream의 후속작이므로 별도 모델 "GaussianDream++"로 등록한다. LIBERO avg 98.6(4 suite), LIBERO-Plus는 `libero_plus_*`와 Overall을 `libero_plus_total`(87.8)로 기록했다. 기하 강화 + 예측형 월드모델 계열에서 "배포 비용을 20토큰으로 줄인" 사례로서 GeoPredict, Spatial Forcing, Fast-WAM류와 비교할 가치가 있다. 다만 효율 주장의 정량 근거가 약하므로 지연 수치는 참고용으로만 보는 것이 좋다.

<!-- VERIFIED: pdf -->
