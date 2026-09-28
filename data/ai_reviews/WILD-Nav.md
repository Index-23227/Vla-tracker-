# Exposing the Long-tail in Embodied Urban Navigation via Scalable Learning from In-the-Wild Videos (WILD-Nav)

> **한 줄 요약**: 30개 도시, 약 500시간의 웹 거리 보행 영상을 LoGeR+Pi3X 메트릭 궤적 복원, 규칙 기반 meta-action, Qwen3.6VL-27B 구조화 CoT 주석으로 500K+ point-goal 내비게이션 샘플로 변환하고, 이를 이용해 Qwen3.5VL-4B 기반 reasoning-aware 내비게이션 VLA **WILD-Nav**를 학습. 영상 테스트셋에서 ADE 0.086 m / FDE 0.068 m(WILD-Nav-think)로 UrbanNav(0.157/0.136)·SocialNav를 크게 앞서고, 희소성(rarity)과 모델 의존 난이도를 결합한 long-tail 발굴 + privileged reflection 기반 실패 귀인, hard-example 증분 fine-tuning(tail ADE 0.286→0.217)을 제시.

---

## 1. 배경 및 동기

- 도시 보행 공간은 밀집 보행자, 불규칙한 통행 가능 영역, 암묵적 사회 규칙이 섞여 있어 모듈형 파이프라인보다 end-to-end 내비게이션 foundation model이 유리하다.
- 그러나 teleop 데이터는 규모가 작고, 시뮬레이션은 sim-to-real 격차가 있으며, 둘 다 사전 정의된 시나리오 안에서만 수집되어 **예상하지 못한 long-tail 상황**을 드러내기 어렵다.
- 저자 주장: in-the-wild 1인칭 영상은 확장 가능한 학습 데이터일 뿐 아니라, 넓은 경험 분포 덕분에 **어떤 경험이 부족하고 어디서 모델이 실패하는지** 체계적으로 찾아내는 경험적 기반이 된다.

## 2. 방법론 심층 분석

### 2.1 문제 정의
- 입력: 현재 RGB $I_t$, 과거 4프레임, 과거 위치 궤적 $\tau^{hist}_t$(자기중심 지면 좌표), 로컬 point goal $g_t$.
- 출력: 5개 waypoint 궤적 $\hat\tau^{plan}_t$, 20개 클래스 meta-action $a_t$, 구조화된 내비게이션 근거 $c_t$.
- 지도나 환경 사전 정보 없이 실시간 단기 계획.

### 2.2 WILD-Nav 모델
- 백본: Qwen3.5VL-4B, 시각 인코더 **고정**. 5개 연속 프레임 토큰 + 과제 프롬프트 + 텍스트화된 궤적 이력/목표.
- 특수 토큰 `<PLAN_QUERY>`의 최종 hidden state $h_q \in \mathbb{R}^{2560}$를 action expert가 디코딩:
  - meta-action head: 선형 분류기 + softmax.
  - trajectory head: 512차원 은닉층 3개, GELU의 4층 MLP → 5개 waypoint 회귀.
- **think 변형**: 근거(perception–analysis–planning)를 자기회귀로 먼저 생성한 뒤 `<PLAN_QUERY>`를 붙여 한 번 더 forward → 계획이 생성된 근거 전체에 조건화.
- **inst 변형**: `<PLAN_QUERY>`를 근거 앞에 두고 학습, causal attention 때문에 뒤따르는 텍스트를 보지 못하므로 추론 시 한 번에 계획 가능(근거는 보조 학습 목표).

### 2.3 학습 목적
$\mathcal{L} = \lambda_{cot}\mathcal{L}_{cot} + \lambda_{act}\mathcal{L}_{act} + \lambda_{traj}\mathcal{L}_{traj}$ — CoT 토큰 언어모델링 loss, meta-action cross-entropy, waypoint MSE. Action expert는 LM adapter보다 높은 학습률.

### 2.4 Long-tail 발굴
- 평균 위험 $R(f)=\sum_z p(z) r_f(z)$ 분해: 빈도가 낮은 패턴은 조건부 오차가 커도 평균에 거의 기여하지 않는다.
- **희소성**: 정규화된 객체 범주 + 이미지/깊이 위치 + 운동 구성의 결합 특징에 smoothed IDF를 적용한 에피소드 점수.
- **난이도**: ADE/FDE 각각 상위 3% 합집합을 hard set으로 정의.
- **Privileged reflection**: hard set에 대해 교사 VLM이 미래 관측·참조 궤적을 보고 장면 분석/근거/궤적 일관성을 검증, 실패를 추론 단계별로 귀인. 채굴 지표(ADE, rarity)는 프롬프트에서 제외.

## 3. 데이터 전략

- 30개 도시·15개국 거리 보행 영상 약 500시간(Table 1: SCAND 8.7 h, CityWalker 200 h 대비). 5분 클립 분할, VLM으로 컷·인물 중심 클립·계단/엘리베이터 등 범위 밖 구간 제거.
- 궤적 복원: LoGeR(장문맥 기하 재구성) + Pi3X(메트릭 깊이·카메라 자세) → 지면 투영, 1 Hz 샘플링, 과거 5 + 미래 5 프레임 에피소드. 겹치는 윈도가 분할을 넘지 않도록 temporal block 단위 분할.
- 주석: GroundingDINO 검출 + 복원 깊이 → 정규화 perception 라벨; Qwen3.6VL-27B가 관측 가능한 증거만으로 근거를 작성하되 미래 관측·meta-action은 검증용 privileged 정보로만 사용.
- 분할: 학습용 부분집합 80/20, long-tail 채굴 세트 287,820 샘플, 평가 세트 104,655 샘플. 실세계 적응용 SCAND/SiT/GND 통합 45,096 샘플.

## 4. 실험 설계

- Baseline: UrbanNav, SocialNav(공개 가중치 = SocialNav-origin, 및 동일 데이터로 fine-tune한 버전).
- 지표: ADE, FDE(m), MAOE(도, 에피소드별 최대 방향 오차 평균), meta-action 정확도(WILD-Nav만).
- 설정: (1) 영상 테스트셋, (2) 실세계 테스트셋 직접 전이, (3) 실세계 데이터 fine-tune, (4) Long-Tail 테스트셋 증분 fine-tune.
- 모든 정량 평가는 **오프라인 open-loop**. 실제 로봇 배치는 정성적(Fig. 7).
- V100 32GB GPU 사용.

## 5. 주요 결과

### Table 2 (Meta %↑, ADE/FDE m↓, MAOE °↓)
| 설정 | Method | Meta | ADE | FDE | MAOE |
|---|---|---|---|---|---|
| Video | UrbanNav | – | 0.157 | 0.136 | 6.30 |
| Video | SocialNav | – | 0.470 | 0.728 | 12.65 |
| Video | WILD-Nav-inst | 96.5 | 0.094 | 0.071 | 5.90 |
| Video | **WILD-Nav-think** | 93.4 | **0.086** | **0.068** | **5.32** |
| Real (전이) | UrbanNav | – | 0.867 | 1.493 | 13.92 |
| Real (전이) | WILD-Nav-inst | 93.4 | 0.953 | 1.493 | 7.36 |
| Real (전이) | **WILD-Nav-think** | 93.3 | **0.619** | **0.834** | **7.14** |
| Real (FT) | UrbanNav | – | 0.257 | 0.286 | 8.29 |
| Real (FT) | **WILD-Nav-inst** | 89.13 | **0.206** | **0.127** | **4.49** |

- 영상 테스트셋: think가 inst 대비 ADE 8.5%, FDE 4.2% 감소. inst는 UrbanNav 대비 FDE 47.8% 감소.
- 실세계 FT: 차선인 UrbanNav 대비 FDE 55.6%, MAOE 45.8% 감소.
- 직접 전이 시 오차가 크게 증가 → 카메라 시점·운동 제약 차이로 인한 도메인 격차.

## 6. Ablation 분석

### Table 3: Long-Tail 테스트셋 증분 fine-tune (WILD-Nav-inst 기반)
| Method | All ADE | All FDE | Hard ADE | Hard FDE | Meta % |
|---|---|---|---|---|---|
| Base | 0.094 | 0.074 | 0.286 | 0.247 | 94.88 |
| +Random | 0.091 | 0.072 | 0.277 | 0.231 | 95.10 |
| +Hard | 0.121 | 0.102 | 0.230 | 0.165 | 96.83 |
| **+Hard+Random** | 0.093 | 0.079 | **0.217** | **0.160** | 96.73 |

- Random은 전체 성능 유지, tail 개선은 미미. Hard만 쓰면 tail은 좋아지나 전체가 악화(ADE 0.094→0.121). Hard+Random이 전체를 유지하면서 tail 최저 오차 → hard example은 표적 감독, replay는 분포 유지.
- think vs inst 비교는 CoT 조건화의 효과에 대한 ablation 역할(영상 셋 ADE 0.094→0.086).
- Long-tail 통계: hard set은 채굴 세트의 4.92%, hard 중 24.18%가 rare, rare 중 14.48%가 hard — 희소성과 난이도는 겹치지만 일치하지 않음. HDBSCAN 군집별로 밀집 보행자 정지(C18)는 계획/궤적 형태 오류, 교통 표지 장면(C60)은 규칙 위반, 스쿠터·자전거 상호작용(C105/C141)은 예측·충돌 위험 실패가 두드러짐.

## 7. 관련 연구와의 위치

- CityWalker(뉴욕 영상, 유효 약 200시간), UrbanNav(랜드마크 지시), SocialNav(사회적 통행성) 계열의 웹 영상 기반 도시 내비게이션을 **규모·지리적 다양성·구조화 근거 주석**으로 확장.
- FLAME, CityNav, RoomTour3D 등은 상위 결정만 제공 — WILD-Nav는 실행 가능한 메트릭 궤적을 제공.
- 자율주행의 AutoDrive-P3(perception–prediction–planning 추론), AutoDrive-R2(self-reflection)에서 영감을 받아 도시 보행 내비게이션에 이식.
- Long-tail 연구에서 사전 정의 corner-case taxonomy 대신 데이터 분포와 모델 오차로부터 tail을 발견한다는 점이 차별점.

## 8. 강점

- 500시간·30개 도시·500K+ 샘플이라는 대규모, 자동화된 주석 파이프라인과 누수 방지(temporal block 분할, privileged 정보 격리) 설계.
- 희소성(데이터 공백)과 난이도(모델 약점)를 분리 측정하고 그 교집합을 정량화한 long-tail 분석 프레임.
- think/inst 두 변형으로 추론 비용과 정확도의 트레이드오프를 제시.
- Hard+Random 증분 학습 실험이 "long-tail 발굴 → 데이터 개선"의 폐루프 활용 가능성을 실증.

## 9. 약점 및 한계

- 모든 정량 평가가 **오프라인 open-loop 궤적 오차**이며, 실제 로봇 폐루프 성공률·충돌률은 보고되지 않음(실기 결과는 정성적 그림뿐).
- 보행자 영상에서 복원한 궤적은 로봇 운동 제약과 다르며, 실세계 직접 전이 오차가 크게 증가(ADE 0.086→0.619)해 도메인 격차가 명확.
- 실세계 FT 결과는 inst만 제시되어 think의 FT 성능은 알 수 없음.
- Baseline이 UrbanNav·SocialNav 두 계열로 제한적이며, 영상 테스트셋은 자체 파이프라인으로 생성한 라벨이라 자기 분포에 유리할 수 있다.
- 파라미터 규모·학습 스텝·loss 가중치 등 세부 하이퍼파라미터가 본문에 부족, 코드/데이터 공개 여부 불명.
- 실패 군집 해석은 저자도 인정하듯 통계적 국소 연관이지 인과적 범주가 아님.

## 10. 실용적 시사점

- 배달 로봇 등 도시 보행 내비게이션 개발 시, 웹 보행 영상 + 메트릭 재구성 + VLM 근거 주석 파이프라인은 저비용 대규모 사전학습 데이터 확보 방안이 된다.
- 소량 실세계 데이터로의 fine-tune이 여전히 필수이며, 평가 셋 구축 시 rarity/난이도 기준의 tail 셋(WILD-LongTail)을 별도로 두면 평균 지표에 가려진 약점을 추적할 수 있다.
- 증분 학습에서 hard example만 쓰면 전체 성능이 떨어지므로 replay 혼합이 필요하다는 실무 교훈.

## 11. 예상 질문과 답변

**Q1. think 변형이 meta-action 정확도는 오히려 낮은 이유는?**
A. 영상 셋에서 think 93.4% vs inst 96.5%로 궤적 오차는 줄지만 분류 정확도는 떨어진다. 논문은 원인을 분석하지 않으며, 생성된 근거의 오류가 이산 결정에 전파되었을 가능성이 있다.

**Q2. 영상에서 복원한 궤적의 스케일은 신뢰할 수 있나?**
A. Pi3X가 메트릭 깊이와 참조 없는 카메라 자세를 추정하고 LoGeR가 장문맥 정렬로 일관성을 유지한다. 다만 복원 오차에 대한 정량 검증은 제시되지 않았다.

**Q3. Privileged 정보가 학습 주석에 새어 들어가지 않나?**
A. 교사 VLM에게 관측 가능한 단서로만 근거를 쓰도록 지시하고, 미래 정보는 검증·수정 용도로만 사용하며, 주석은 스키마·meta-action·GT 궤적과 대조 검증한다.

**Q4. Long-tail 정의가 WILD-Nav-inst에 의존하면 순환적이지 않나?**
A. 난이도는 의도적으로 모델 의존적으로 정의되며, 평가 시 tail 샘플 ID를 고정해 fine-tune 후 모델들을 동일 샘플에서 비교한다. 희소성은 모델과 독립적인 분포 지표로 보완한다.

## 12. 결론

WILD-Nav 논문은 in-the-wild 거리 보행 영상을 도시 내비게이션 VLA의 확장 가능한 감독원이자 long-tail 발견의 경험적 기반으로 동시에 활용한 연구다. Qwen3.5VL-4B 기반 reasoning-aware 정책은 영상 테스트셋에서 ADE 0.086 m, 실세계 FT 후 FDE 0.127 m로 UrbanNav·SocialNav를 앞섰고, 희소성–난이도 결합 분석과 privileged reflection은 평균 지표에 가려진 장면 의존적 실패를 드러냈다. 다만 평가가 오프라인 open-loop에 머물고 실세계 도메인 격차가 크므로, 폐루프 로봇 평가와 공개 데이터/코드로의 검증이 후속 과제다.

<!-- VERIFIED: pdf -->
