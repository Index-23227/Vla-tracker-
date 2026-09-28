# Act with Intent: Distilling Behavior Intent for Vision-Language-Action Models (INDI)

> **한 줄 요약**: 학습 시에만 쓰는 동결 teacher VLM(Cosmos-Reason2-8B)이 시연 구간(관측·지시·거친 행동 요약·실행 영상)을 해석해 만든 "행동 의도(intent)" 표현을 VLA 액션 디코더 중간층으로 증류하는 INDI(Intention Distillation). GR00T-N1.7 기준 SimplerEnv-Bridge 64.3% → 84.7%, RoboCasa Kitchen(100 demo/task) 64.1% → 70.3%, 실물 ID clean 71.0% → 76.0%.

- **arXiv**: 2608.23478 (2026-08-24)
- **소속**: POSTECH (GSAI, IME) — Sangoh Lee, Sangwoo Mo, Wook-Shin Han
- **프로젝트 페이지**: https://leesangoh.github.io/indi-project-page/
- **백본**: GR00T-N1.7 (주), π0.5 (보조)

---

## 1. 배경 및 동기

VLA의 액션 디코더는 대부분 behavior cloning으로 학습되어 "어떤 모터 명령이 시연되었는가"만 감독받고, 그 행동이 지시 아래에서 달성하려는 국소 목표(local objective)는 암묵적으로 남는다. 미래 프레임·잠재 관측·궤적·모션 표현을 예측하는 future-based 감독은 풍부한 신호를 주지만, 이는 "일어날 수 있는 특정 실현(realization)"이지 앞으로의 행동이 공유하는 의미적 목표가 아니다. 저자들은 행동 수준의 의도를 명시적으로 디코더 안에 형성시키면 행동 생성이 더 조직화된다고 주장한다.

## 2. 문제 정의

- 입력: RGB 관측 o_t, 언어 지시 ℓ, proprioceptive state q_t. 동결 VLM이 context 토큰 B_t를 만든다.
- 출력: flow matching 액션 청크 A_t, 그리고 부가적으로 시각 결과(V_t)·텍스트 목적(R_t) grounding 잠재.
- 목표: 디코더가 표준 입력만으로 중간층에서 intent 상태 I_t를 복원하고, 이를 이용해 (A_t, V_t, R_t) ~ p_θ(·|B_t, q_t, I_t)를 모델링.

## 3. 방법

- **Intent 타깃 생성**: H-step 시연 구간에 대해 teacher VLM이 E_t = (o_t, ℓ, 행동 요약 c_A, 실행 영상 vid_{t:t+H})를 보고 functional-purpose 문장을 생성. 입력 span의 **중간층** hidden state를 풀링한 것이 intent 타깃 I*, 생성 문장의 최종층 표현이 textual-purpose 타깃 R*, 끝점 관측을 동결 시각 인코더로 인코딩한 것이 visual 타깃 V*.
- **Intent-aware 디코딩**: 디코더 입력에 K_I=8개의 학습 가능한 intent query, K_R=8 textual grounding row, 카메라당 K_V=16 visual grounding row를 추가. 디코더 깊이 50% 지점에서 intent query 상태를 teacher 공간으로 사영해 cosine alignment(L_I).
- **Grounding 예측**: no-grad 패스로 grounding 상태를 먼저 예측한 뒤 detach·re-noise해 self-conditioning 입력으로 사용하고, 최종 상태를 REPA식 cosine alignment로 타깃에 정렬.
- **Intent 의존적 정보 흐름**: 비대칭 attention(intent query는 q_t, B_t, 서로만 참조; grounding은 intent 참조; action은 intent+grounding 참조). 이진 게이트 α~Bernoulli(0.5)로 비-intent row의 B_t 직접 접근을 막아 정보가 intent를 거치도록 강제. 또한 배치 셔플된 intent를 넣었을 때 하류 손실이 margin m=0.05 이상 커지도록 하는 hinge loss L_mis.
- **최종 손실**: L = L_A + λ_I L_I + λ_V L_V + λ_R L_R + λ_mis L_mis. 배포 시 α=1, teacher·타깃·사영 헤드·mismatch 분기 제거.

## 4. 학습 절차

- 각 백본의 optimizer·learning rate·VL 동결 설정을 그대로 유지하고 액션 디코더(+추가 row)를 학습.
- teacher 추론과 타깃 생성은 오프라인 1회 수행 후 캐시.
- Bridge는 8-step(5Hz) 구간, RoboCasa는 32-step 구간을 intent 타깃 단위로 사용.
- 추가 파라미터: GR00T-N1.7 +46.4M(1.3%), π0.5 +23.1M(0.64%).

## 5. 데이터

- **SimplerEnv-Bridge**: BridgeData V2 LeRobot 변환본(53,192 에피소드, 1,893,026 프레임, 5Hz), 256×256 단일 카메라, 8차원 state, 7차원 action.
- **RoboCasa Kitchen**: machine-generated 데이터 24개 과제 × 과제당 100 demo(2,400 에피소드, 689,595 프레임, 20Hz), 좌·우·손목 3카메라.
- **실물**: 양팔 SO-101(5-DoF + 1-DoF 그리퍼) 텔레오퍼레이션 시연, 4개 과제(Threading, Basket Nesting, Cross-Bin Stacking, Drawer Storage).

## 6. 실험 설정

- 비교 기준선: 동일 데이터·예산·평가 프로토콜의 GR00T-N1.7, π0.5, 그리고 intent 대신 끝점 시각 표현을 감독하는 "future supervision" 변형.
- 시뮬레이션은 3회 독립 평가 × 과제당 50 에피소드의 평균±표준편차.
- 실물: 과제당 50 trial, ID clean / held-out 물체 / distractor 조건.
- 분석: supervision·capacity 대조(Table 6a), intent 개입(Table 6b), zero-intent, teacher 교체, 표현 분석.

## 7. 주요 결과

**Table 1 – SimplerEnv-Bridge (%)**

| Method | Spoon | Carrot | Stack | EP-Basket | Avg |
|---|---|---|---|---|---|
| π0-FAST | 59.0 | 79.0 | 65.0 | 33.0 | 59.0 |
| GR00T-N1.7 | 84.7 | 79.3 | 57.3 | 36.0 | 64.3 |
| + future supervision | 81.3 | 73.3 | 56.7 | 60.7 | 68.0 |
| **GR00T-N1.7 + INDI** | **88.7** | **84.7** | **69.3** | 96.0 | **84.7** |
| π0.5 | 78.0 | 72.7 | 32.0 | 26.7 | 52.3 |
| π0.5 + INDI | 81.3 | 76.0 | 39.3 | 38.7 | 58.8 |

EP-Basket에서 36.0 → 96.0(+60pp)이 가장 크고, 나머지 3과제 평균 향상은 7.1pp.

**Table 2 – RoboCasa Kitchen 24과제 (%)**

| Method | Pick&Place | Open/Close | Others | Avg |
|---|---|---|---|---|
| GR00T-N1.7 (G3000, 참고) | 53.0 | 80.8 | 79.0 | 70.8 |
| GR00T-N1.7 (G100) | 39.4 | 75.9 | 76.7 | 64.1 |
| + future supervision (G100) | 47.0 | 80.3 | 72.1 | 65.8 |
| **+ INDI (G100)** | 49.8 | **82.8** | **79.1** | **70.3** |
| π0.5 → π0.5 + INDI | 14.0 → 15.3 | 55.1 → 56.1 | 39.5 → 53.5 | 34.9 → 41.4 |

과제당 100 demo로 3,000 demo 체크포인트(70.8)의 0.5pp 이내. Table 3의 보고치(RS-CL 69.7, FLARE 66.4 등; 300 demo)보다 높다.

**Table 4 – 실물 평균 (%)**: ID clean 71.0 → 76.0, Held-out 62.0 → 68.0, Distractors 53.0 → 62.0. 장기 과제 향상이 크다(세 조건 평균 Cross-Bin Stacking +7.3, Drawer Storage +10.7pp).

**Table 6a – 대조 실험(단일 평가 run)**: GR00T-N1.7 61.5, Groundings only 60.0, Future supervision 68.0, Free latent 57.0, Intent only 76.0, INDI 85.5 → 이득의 주원천은 teacher 유래 intent.

**개입 분석**: 동일 목표 intent 주입 시 84.5%, 다른 목표 45.2%, Gaussian 노이즈 1.0%; zero-intent 시 85.5% → 45.5%. 비용: GR00T-N1.7 추론 56.1 → 61.5ms, π0.5 24.4 → 29.7ms.

## 8. Related Work 상의 위치

- Future/world-model 감독(GR-1/Seer류 미래 프레임, FLARE의 미래 잠재 정렬, DreamGen·Video Policy)과 대비: 특정 미래의 실현이 아니라 "행동의 의미적 목적"을 감독.
- 언어 추론 계열(ECoT, 계층적 subtask 예측)과 달리 추론 시 텍스트 생성이 없고, 의도가 디코더 내부 잠재로 존재.
- 중간층 representation alignment(REPA, FLARE)를 VLA 디코더에 적용하되 타깃을 teacher VLM의 멀티모달 intent로 바꾼 것.

## 9. 강점

1. **대조군 설계가 촘촘함**: future supervision, free latent(용량만 추가), groundings only 등으로 "추가 용량/보조 손실 효과"와 intent 효과를 분리.
2. **두 백본·두 시뮬레이터·실물에서 일관된 향상**, 3회 평가 평균과 표준편차 보고.
3. **인과적 검증**: intent 교체·위상 강제·zero-intent 개입으로 복원된 잠재가 실제로 실행을 좌우함을 보임.
4. **배포 부담이 작음**: teacher 제거, +1.3% 파라미터, +5ms 수준 지연.
5. 데이터 효율: RoboCasa 100 demo/task로 3,000 demo 모델에 근접.

## 10. 약점 및 한계

1. **EP-Basket 편중**: Bridge 평균 +20.4pp 중 상당 부분이 한 과제(+60pp)에서 나오며, 해당 과제는 SpatialVLA가 100%를 내는 등 특이한 분포를 보인다.
2. **teacher 의존성**: 부록 Table 17에서 teacher 선택에 따라 성능이 크게 바뀐다고 명시 — Cosmos-Reason2-8B 같은 강한 멀티모달 추론 모델이 필요.
3. **공개 기준선과의 비교는 프로토콜 불일치**: Table 1·3의 외부 수치는 원 논문 프로토콜을 그대로 옮긴 것.
4. **실물 규모 제한**: 저가 SO-101 4개 과제, 단일 플랫폼. π0.5 실물 결과 없음.
5. 코드 공개 여부가 본문에 명시되지 않음(프로젝트 페이지만 제시).
6. LIBERO/CALVIN 등 널리 쓰이는 벤치마크 결과 부재.

## 11. 재현 및 확장 아이디어

- teacher 크기·종류(일반 VLM vs 추론 특화 VLM) 스케일링과 intent 추출 층 위치 스윕.
- LIBERO-Long, CALVIN ABC→D 같은 장기 과제 벤치마크에서의 검증.
- intent 잠재를 계층적 플래너 출력 또는 인간 교정 인터페이스로 사용하는 확장.
- Discrete/autoregressive 액션 헤드(OpenVLA류)에도 동일한 중간층 intent 증류가 효과적인지 확인.

## 12. 총평

INDI는 "무엇을 했는가"를 넘어 "왜 그 행동을 하는가"를 디코더 안에 명시적인 중간 표현으로 심는 깔끔한 학습 기법이다. 추론 시 부담이 거의 없고, 대조 실험과 개입 분석으로 이득이 단순 용량 증가가 아닌 intent 감독에서 온다는 점을 설득력 있게 보인다. 다만 Bridge 향상이 특정 과제에 치우치고 강력한 teacher VLM에 의존한다는 점은 일반화 판단 시 유의해야 한다.

**한 문장 요약**: 동결 teacher VLM의 행동 의도 표현을 flow-matching 액션 디코더 중간층에 증류해 GR00T-N1.7을 SimplerEnv-Bridge 84.7%, RoboCasa(100 demo) 70.3%로 끌어올린 경량 학습 기법.

<!-- VERIFIED: pdf -->
