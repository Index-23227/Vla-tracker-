# V-Link — Recovering Lost Visual Representations in Action DiT for Vision-Language-Action Models

- arXiv: 2608.25308 (2026-08-26)
- 소속: AGIBOT, 저장대학교, 홍콩과기대(광저우), 사이먼프레이저 대학교

## 1. 한 줄 요약

V-Link는 GR00T N1.6처럼 VLM의 마지막 층 특징만 Action DiT로 넘기는 이중 시스템 VLA에서, 행동 전문가가 3D 기하와 2D 의미 정보에 제대로 접근하지 못한다는 병목을 진단하고 이를 복구한다. VLM 입력에 학습 가능한 Spatial Query와 Semantic Query를 붙여 학습 전용 깊이·분할 헤드로 특화시킨 뒤, Action DiT에 비대칭 경로로 주입한다. GR00T N1.6 대비 LIBERO +1.9%p(99.3%), LIBERO-Plus +31.2%p(75.0%), RoboTwin 2.0 6개 과제 +18.8%p(56.8%)를 기록했고, 추가 지연은 1.58 ms다.

## 2. 문제 설정

이중 시스템 VLA에서 VL→A 특징 전달은 행동 전문가가 무엇을 볼 수 있는지를 결정한다. 저자들은 Spatial Forcing에서 착안한 진단 프로토콜을 쓴다. GR00T N1.6의 VLM 특징과 Action DiT 특징을 고정한 채 약 2.3M 파라미터의 가벼운 깊이 추정·의미 분할 헤드만 학습한다. VLM 특징에서는 깊이 MAE 0.015, 분할 mIoU 0.665인데 Action DiT 특징에서는 각각 0.071과 0.290으로 나빠진다. 즉 VLM에 있던 지각 정보가 행동 전문가 쪽에서는 잘 꺼내지지 않는다. 저자들은 행동 손실만으로 학습하면 Action DiT가 빨리 손실을 줄이는 의미 단서에 기대고 3D 단서를 덜 쓰는 지름길을 택한다고 가정한다.

## 3. 핵심 아이디어

- **표현 분리**: VLM 안에 기하용 Spatial Query와 의미용 Semantic Query 두 세트를 따로 두고, 맞춤형 인과 마스크로 원래 이미지·텍스트 계산은 건드리지 않으면서 각 쿼리가 독립적으로 단서를 모으게 한다.
- **보조 감독**: 학습 때만 깊이 헤드와 분할 헤드로 각 쿼리를 특화시키고, 추론 때는 헤드를 제거한다.
- **비대칭 주입**: Semantic Query는 기존 VLM 이미지 토큰을 보완하는 병렬 교차 어텐션으로, Spatial Query는 반드시 거치는 별도의 기하 조건화 교차 어텐션으로 넣는다.

## 4. 방법 상세

- **입력**: E = [E_t; E_o; E_d; E_s](텍스트, 다시점 이미지, Spatial Query, Semantic Query). Qwen의 최종 RMSNorm을 거친 출력 E'_d, E'_s를 사용한다.
- **깊이 헤드**: LN(2048) → Linear(2048,256) → GELU → Dropout, 시점별 트랜스포머 인코더 2층, 합성곱 조밀 디코더로 구성되며 약 2.3M 파라미터다. softplus로 양의 metric depth를 출력한다. 손실은 smooth-L1 + 0.1·abs-rel + 0.5·top-k hard-patch다.
- **분할 헤드**: 같은 구조에 softmax 출력을 쓰고, 손실은 CE + 0.5·Dice다.
- **Action DiT 주입**: GR00T N1.6의 32층 AlternateVLDiT는 4층 주기(비이미지 교차 어텐션, 자기 어텐션, 이미지 교차 어텐션, 자기 어텐션)를 8번 반복한다. 주입은 이미지 교차 어텐션 층(l = 4k+2)에서만 한다. 상태·노이즈 행동 토큰 H_l이 VLM 이미지 토큰과 Semantic Query에 각각 교차 어텐션한 결과를 학습 가능한 게이트 g_l과 함께 합쳐 잔차로 더하고(Z_l), 이어서 Spatial Query에 교차 어텐션한 뒤 FFN을 거친다.
- **전체 손실**: L = L_act(flow matching) + 0.1·L_depth + 0.002·L_seg. 무작위 초기화된 쿼리를 안정화하려고 VLM LoRA, 쿼리, 헤드를 먼저 워밍업한 뒤 모든 학습 가능 요소를 함께 최적화한다.

## 5. 학습 설정

- GR00T N1.6 3B에서 초기화했다. 언어·시각 백본은 고정하고 VLM은 rank-128 LoRA로 적응시킨다. H100 4장, BF16, DeepSpeed ZeRO-2, AdamW(lr 1e-4, wd 1e-5, cosine, 5% 워밍업).
- LIBERO: 80K 스텝, 전역 배치 160, 외부·손목 시점. 쿼리는 세트당 2×5×5 토큰이고 16스텝 청크를 예측한다.
- RoboTwin 2.0: 6개 과제 × 시연 50개(총 300개)를 함께 학습한다. 120K 스텝, 배치 128, 쿼리 3×5×5, 15스텝 양팔 청크.
- 실로봇: AGIBOT A3 Ultra, 과제당 원격조작 시연 100개. 깊이는 Lingbot-Depth로 결손을 보완하고, 분할 의사 라벨은 GroundingDINO와 SAM3로 만든다.

## 6. 실험 설정

- LIBERO와 LIBERO-Plus는 공식 프로토콜을 따른다. 표준 LIBERO로만 학습하고 LIBERO-Plus는 추가 파인튜닝 없이 평가한다. 과제당 50회.
- RoboTwin 2.0은 저자들이 고른 6개 과제(Move Pillbottle Pad, Move Stapler Pad, Place Bread Skillet, Place Phone Stand, Press Stapler, Turn Switch)를 과제당 100회 평가한다.
- 실로봇은 자율 전원 켜기·끄기 2개 과제를 연속 50회씩 평가한다.

## 7. LIBERO와 LIBERO-Plus 결과 (Table I, II)

| 방법 | Spatial | Object | Goal | Long | 평균 |
|---|---|---|---|---|---|
| Spatial Forcing | 99.4 | 99.6 | 98.8 | 96.0 | 98.5 |
| GR00T N1.6 (Base) | 99.3 | 99.2 | 98.4 | 92.9 | 97.4 |
| **V-Link** | **99.5** | **100.0** | **99.8** | **98.0** | **99.3** |

가장 큰 향상은 LIBERO-Long(+5.1%p)이다.

LIBERO-Plus(7개 교란) 합계는 V-Link 75.0%로, OpenVLA-OFT와 Evo-Depth의 69.6%보다 5.4%p, GR00T N1.6의 43.8%보다 31.2%p 높다. 차원별로는 Camera 42.8, Robot 59.1, Language 82.8, Light 95.7, Background 94.9, Noise 87.8, Layout 74.4이다. 7개 중 6개 차원에서 1위이고, Camera는 π0-Fast(65.1)가 더 높다. 기저 모델 대비 향상은 Noise(+60.4), Language(+47.3), Light(+30.6)에서 특히 크다.

## 8. RoboTwin 2.0과 실로봇 결과 (Table III, Fig. 8)

| 방법 | Pillbottle | Stapler | Bread Skillet | Phone Stand | Press Stapler | Turn Switch | 평균 |
|---|---|---|---|---|---|---|---|
| π0 | 21 | 0 | 23 | 35 | 62 | 27 | 28.0 |
| EventVLA | 60 | 14 | 64 | 64 | 80 | 24 | 51.0 |
| GR00T N1.6 | 58 | 7 | 23 | 54 | 80 | 6 | 38.0 |
| **V-Link** | 74 | 22 | 55 | 60 | 97 | 33 | **56.8** |

같은 표의 π0.5 평균은 56.3이다(추출된 과제별 셀은 일부가 붙어 있어 여기서는 평균만 인용한다). V-Link는 6개 과제 모두에서 기저 모델보다 높다. 실로봇 AGIBOT A3 Ultra에서는 전원 켜기 98%, 끄기 94%로 GR00T N1.6(78%, 70%)보다 각각 20%p, 24%p 높다.

## 9. 어블레이션 (Table IV–VI, Fig. 6–7)

- **쿼리 종류(RoboTwin 6개 과제 평균)**: 기저 38.0 → Spatial만 51.2 → Semantic만 47.8 → 둘 다 56.8. Spatial은 Move·Place에서, Semantic은 Contact에서 이득이 크다(Contact 43.0 → 61.0).
- **감독과 주입**: 과제 헤드 없이 주입만 하면 40.3, 감독은 하되 주입을 하지 않으면 39.8, 둘 다 하면 56.8이다. 쿼리를 특화시키는 감독과 그것을 Action DiT가 쓸 수 있게 하는 주입이 모두 있어야 효과가 난다.
- **표현 진단(Fig. 6)**: Spatial Query가 깊이 MAE가 가장 낮고 Semantic Query가 mIoU가 가장 높아 각각 특화되었음을 보여준다. V-Link의 Action DiT 특징은 GR00T의 Action DiT 특징보다 두 과제 모두에서 낫다. 수치는 그림에만 있다.
- **쿼리 크기(Fig. 7)**: 3×1×1에서 37.8%, 3×2×2에서 51.8%, 3×5×5에서 56.8%이고, 3×6×6에서는 56.5%로 약간 떨어지면서 지연만 늘어난다.

## 10. 강점

- "VLM에는 정보가 있는데 행동 전문가는 못 쓴다"는 병목을 고정 특징 프로빙이라는 통제된 진단으로 정량화하고, 개선 후 같은 진단으로 복구를 확인했다. 문제 제기와 검증이 일관된다.
- 보조 헤드는 학습 때만 쓰고 추론 때는 제거하므로 깊이·분할 주석이 배포 시에는 필요 없다. 지연 증가도 1.58 ms로 작다.
- 표준 LIBERO로만 학습하고 LIBERO-Plus를 제로샷으로 평가해 강건성 이득(+31.2%p)을 보였다.
- 휴머노이드 실로봇에서도 향상을 확인했다.

## 11. 한계

- RoboTwin 2.0은 저자들이 "도전적인" 6개 과제를 골라 평가했으므로 전체 스위트 결과와 비교할 수 없다. 모든 어블레이션도 이 6개 과제에서만 했다.
- 학습 때 깊이·분할 정답이 필요하다. 시뮬레이션에서는 쉽게 얻지만, 실로봇에서는 Lingbot-Depth, GroundingDINO, SAM3 같은 외부 모델 의사 라벨에 의존한다.
- GR00T N1.6의 교대 어텐션 구조에 맞춰 설계했으므로 π0/π0.5 같은 층별 결합 구조로 옮길 수 있는지는 검증하지 않았다.
- LIBERO는 이미 포화에 가까워(기저 97.4%) 이득의 폭이 작다. LIBERO-Plus에서 Camera 교란은 여전히 42.8%로 약하다.
- 실로봇 과제는 2개뿐이고 과제 구조도 단순하다. 코드 공개 여부가 명시되지 않았다.

## 12. VLA-Tracker 관점 평가

V-Link는 GR00T N1.6을 LoRA와 새 모듈로 파인튜닝해 자체 정책을 학습하고 LIBERO, LIBERO-Plus, RoboTwin 2.0, 실로봇 결과를 보고하므로 수록 기준을 충족한다. LIBERO 평균 99.3%는 리더보드 최상위권이며, LIBERO-Plus 75.0%는 libero 블록 안에 libero_plus_* 키로 넣었다. RoboTwin 2.0 평균 56.8%는 6개 과제 부분 집합이라 robotwin_v2 순위 비교 때 주의해야 한다. GR00T N1.6 기준선과 어블레이션은 별도 블록으로 분리했다. Spatial Forcing, GeoVLA, Evo-Depth, GEAR-VLA 같은 3D 인식 강화 VLA와 비교할 때, "VLM이 아니라 VL→A 전달 경로를 고친다"는 설계 축의 대표 사례다.

<!-- VERIFIED: pdf -->
