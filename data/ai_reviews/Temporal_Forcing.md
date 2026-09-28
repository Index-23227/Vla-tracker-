# Temporal Forcing — 4D Representation Alignment for Vision-Language-Action Models

- arXiv: 2608.30643 (v1 2026-08-31)
- 소속: Nanjing University, 중국과학원 자동화연구소(CASIA), 중국과학원대학(UCAS) (Xingyu Ding, Yuzhong Zhao, Chunhai Zhao, Yinghuan Shi, Chaoyang Zhao, Yifan Zhang)

## 1. 한 줄 요약

Temporal Forcing은 Spatial Forcing류의 "프레임 단위 3D 표현 정렬"을 **4D(시간에 따라 변하는 3D)로 확장**한 학습 기법이다. 가벼운 history pathway로 과거 관측을 요약하고, 그 잠재 표현을 사전학습 4D 파운데이션 모델(StreamVGGT)의 인과적 기하 특징에 정렬한다. Qwen3-VL-4B 기반 StarVLA-OFT에 적용해 LIBERO 평균 96.6→98.8, RoboTwin 2.0 12개 과제 53.5→62.8, 실로봇 숨김 배치 과제 전체 성공률 20.0%→43.3%를 기록했다.

## 2. 문제 설정

현재 프레임의 3D 기하만 정렬하면 "어떻게 여기까지 왔는가"를 표현할 수 없다. 예컨대 똑같은 블록 두 개 중 하나를 불투명 상자에 넣고 나면 현재 프레임만으로는 "나머지 블록"이 무엇인지, 몇 단계를 끝냈는지 알 수 없다(observation aliasing). 이런 장기·다단계 과제에서는 물체 상태 전이와 과제 진행을 시간에 걸쳐 추적해야 한다.

## 3. 핵심 아이디어

(1) 현재 프레임을 제외한 과거 K=7 프레임을 소수 토큰으로 요약하는 history pathway를 붙이고, (2) 그 표현의 **시간적 변화**를 4D 모델의 특징 궤적에 맞추도록 직접 감독한다. 4D 모델과 정렬 head는 학습에만 쓰이고 추론에는 base VLA + history pathway만 남는다.

## 4. 아키텍처

Base: StarVLA의 QwenOFT 구현(Qwen3-VL-4B 백본 + L1 회귀 MLP 액션 헤드). History pathway: 과거 프레임을 224×224로 frozen DINOv2 ViT-L/14에 넣고 Q-Former로 프레임·카메라당 2개 gist 토큰(512-d) 생성 → 시간 오프셋·카메라 임베딩 추가 → 2층 causal temporal transformer → 16개 학습 query로 history 토큰 요약 → 첫 decoder layer 입력에서 zero-init tanh 게이트 cross-attention으로 primary 카메라 이미지 토큰에 주입(초기에는 identity). 백본 시퀀스 길이는 늘지 않는다. 4D 타깃: StreamVGGT가 LIBERO에서는 최대 12초, RoboTwin/실로봇에서는 에피소드 prefix 전체를 인과적으로 처리하고, layer 21 특징을 stride-4 앵커마다 오프라인 저장한다.

## 5. 학습과 추론

손실 L = L_act + 0.5·L_cur + 0.5·L_temp. L_temp = (L_state + 4·L_change + L_read)/6 로, state항은 각 시점 gist 평균을 타깃에 cosine 정렬, change항은 인접 시점 차분끼리 정렬(장면 정체성 등 창 내 상수 성분은 상쇄되어 창 안의 변화만 학습), readout항은 history 토큰 평균이 가장 최근 시점을 재현하도록 한다. L_cur는 VLA layer 24의 현재 프레임 이미지 토큰을 8×8로 풀링한 StreamVGGT dense 특징에 정렬한다. 게이트가 닫혀 있어도 pre-gate 특징에 손실을 걸어 pathway 전체가 gradient를 받는다. LIBERO 본 실험은 네 suite를 하나의 모델로 50k 스텝(배치 128, A100×8) 학습. 실로봇 추론은 RTX 5090에서 chunk당 58 ms(base 45 ms)로 10 Hz 예산 내.

## 6. 주요 결과 — LIBERO (Table 1)

| 방법 | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| Spatial Forcing | 99.4 | 99.6 | 98.8 | 96.0 | 98.5 |
| HAMLET | 99.0 | 100.0 | 99.2 | 92.2 | 97.6 |
| MemoryVLA | 98.4 | 98.4 | 96.4 | 93.4 | 96.7 |
| StarVLA-OFT (base) | 97.8 | 98.6 | 96.2 | 93.8 | 96.6 |
| **Temporal Forcing** | **99.6** | 99.8 | 98.4 | **97.2** | **98.8** |

가장 큰 향상은 이력 의존성이 큰 Long(93.8→97.2). 여러 베이스라인은 suite별 개별 모델이지만 Temporal Forcing은 단일 모델이다. 부록 Table 6(a) zero-shot LIBERO-Plus pooled total은 75.0→77.8이며, 센서 노이즈에서 73.1→81.3으로 가장 크게 오른다.

## 7. 주요 결과 — RoboTwin 2.0 12개 과제 (Supp. Table 5)

easy(clean), 과제당 100회. 평균 π0 36.8, DP3 39.0, StarVLA-OFT 53.5 → **Temporal Forcing 62.8**, 12개 중 9개 과제에서 개선. Handover Block 0→44, Handover Mic 39→79로 인계 중 가림이 있는 과제의 향상이 가장 크다. 반면 Stack Blocks Three 41→40, Place Dual Shoes 28→22, Place Burger Fries 96→92처럼 완료 단계가 계속 보이는 과제에서는 소폭 하락했다. 표준 50과제 프로토콜이 아니라 선택된 부분집합이라는 점에 유의해야 한다.

## 8. 통제 실험 (Table 2, 10k 스텝·배치 64)

| # | 구성 | Avg | Long |
|---|---|---|---|
| 1 | Base | 90.7 | 76.2 |
| 2 | 4D 타깃만(현재 프레임 정렬) | 90.5 | 68.2 |
| 3 | History만 | 88.2 | 72.8 |
| 4 | History + L_cur (L_temp 없음) | 83.7 | 63.0 |
| 5 | History + L_temp (L_cur 없음) | 90.8 | 77.2 |
| 6 | 3D 타깃만 | 87.7 | 64.6 |
| 7 | 3D 정렬 전체 | 85.0 | 63.2 |
| 8 | **Temporal Forcing** | **93.6** | **83.8** |

History만 넣으면 게이트가 0.6×10⁻³에 머물러 사실상 사용되지 않고 오히려 −2.5pt. L_temp를 넣어야 게이트가 13.4×10⁻³까지 열리고, 추론 시 게이트를 닫으면 −5.4pt(Goal·Long 각 −7.8)로 history를 실제로 쓰고 있음이 확인된다. 같은 StreamVGGT를 프레임 단독으로 돌린 3D 타깃은 history와 결합 시 오히려 해롭다(85.0).

## 9. 실로봇 결과 (Table 3)

UR3, 동일 블록 2개를 하나는 상자에, 다른 하나는 서랍에 넣고 서랍 닫기. 100개 시연으로 네 모델 모두 파인튜닝. 단계 격리 평가 합계: OpenVLA 58/90, π0 70/90, base 75/90, Ours 78/90으로 개별 기술은 비슷하다. 연속 과제(30회)에서 S1/S1:2/S1:3는 base 25/12/6, **Ours 27/20/13**. 첫 블록이 가려진 뒤부터 격차가 벌어져, 개선이 조작 능력이 아니라 과제 진행 기억에서 온다는 해석을 뒷받침한다.

## 10. 강점

- "History를 넣는 것만으로는 부족하고 4D 정렬이 있어야 history가 쓰인다"는 주장을 게이트 크기, 게이트 차단 실험, history-blind 예측기(change항 0.54 vs 0.21) 등 여러 방식으로 검증했다.
- 3D vs 4D 타깃을 동일 모델·동일 특징 공간에서 비교해 교란 요인을 잘 통제했다.
- 추론 시 외부 3D/4D 모델이 필요 없고 백본 시퀀스 길이도 유지되어 배포 부담이 작다.
- 격리/연속 평가를 분리한 실로봇 프로토콜이 기억 효과를 깔끔하게 분리한다.

## 11. 한계와 의문점

- RoboTwin은 12개 선택 과제(이력 의존 과제 위주 선정)만 평가해 표준 50과제 평균과 직접 비교할 수 없다. 베이스라인은 발표 수치이고 본 방법은 자체 평가다.
- 실로봇은 단일 과제·30회로 규모가 작다.
- LIBERO 본 결과는 단일 학습 run이며 시드 분산이 보고되지 않는다. Spatial Forcing(98.5)과의 차이는 0.3pt에 불과하다.
- History 창이 2.8~3.7초로 짧아, 더 긴 시간 의존성에는 여전히 4D 타깃이 제공하는 문맥에 간접 의존한다.

## 12. VLA-Tracker 관점 평가

StarVLA-OFT 위에 history pathway와 정렬 손실을 더해 처음부터 학습한 자체 정책이므로 수록 대상이다. `libero` 블록에는 논문이 보고한 4 suite 점수와 평균 98.8, 그리고 zero-shot LIBERO-Plus(libero_plus_*, pooled total 77.8)를 넣었다. RoboTwin 2.0은 12과제 부분집합이므로 표준 `robotwin_v2` 블록이 아닌 `robotwin_v2_12task_easy`로 분리해 50과제 평균과 섞여 순위화되지 않도록 했다. Base, 통제 실험, 실로봇은 별도 블록이다. "메모리/이력 VLA"와 "기하 표현 정렬" 두 계열을 잇는 사례로 가치가 있다.

<!-- VERIFIED: pdf -->
