# V2V-VLA: Vehicle-to-Vehicle Vision-Language-Action Model for Cooperative Autonomous Driving

> **한 줄 요약**: 단일 에이전트에 머물러 있던 주행 VLA를 **차량 간 협력(V2V)** 설정으로 확장해, ego와 선택된 통신 상대 CAV의 카메라 기반 BEV occupancy map을 좌표변환 후 element-wise-or로 합친 뒤 InternViT + Qwen2에 넣고, **주행 행동·미래 waypoint·언어 추론·통신 정책을 단 한 번의 forward pass로 동시 생성**하는 모델. 저자들이 함께 공개한 CMU-Drive(Bench2Drive 기반, 220 route, 최대 16 CAV, GPU 1장) 클로즈드루프 벤치마크에서 비협력 baseline SimLingo의 DS 56.32 / SR 30.91%를 **DS 63.67 / SR 34.55%**로 끌어올렸다.

- **arXiv**: 2608.07621v1 (2026-08-07, cs.AI)
- **소속**: Carnegie Mellon University, Robotics Institute (Hsu-kuang Chiu, Stephen F. Smith)
- **형식**: 11 pages
- **코드**: 미공개 (코드·벤치마크·모델 체크포인트 공개 예정이라고 명시)

---

## 1. 배경 및 동기

VLA 모델은 시각 인지·언어 추론·차량 제어를 하나로 묶어 end-to-end 자율주행에서 강한 성능을 보여왔다. 그러나 기존 주행 VLA는 거의 전부 **단일 에이전트** 세팅에서 평가된다. 반면 실제 협력 자율주행(cooperative autonomous driving)은 여러 대의 자율주행 차량이 **부분 관측(partial observability)** 하에서 서로 지각·추론·조율해야 하는 문제다.

여기서 새로운 질문이 생긴다. VLA 모델이 이웃 차량으로부터 **상보적인(complementary) 관측을 능동적으로 획득**하면서도 효율적인 클로즈드루프 주행을 유지할 수 있는가?

기존 연구의 공백은 세 갈래다.

1. **Bench2Drive / SimLingo**: 다양한 도시 시나리오에 대해 포괄적인 단일 에이전트 클로즈드루프 평가를 확립했지만 협력이 없다. SimLingo는 주행·비전언어 이해·language-action alignment를 한 모델로 통합해 Bench2Drive SOTA를 달성했다.
2. **CoLMDriver / InterDrive**: 멀티에이전트로 확장했고 다중 라운드 LLM 협상으로 주행 의도 충돌을 해소한다. 그러나 배경 교통 참여자와 취약 도로 사용자가 얽힌 **안전 필수(safety-critical) 시나리오**를 충분히 다루지 않는다.
3. **MDrive**: agentic하게 생성한 상호작용 시나리오에 InterDrive와 V2X 시나리오를 결합했지만, **새 모델이나 알고리즘 없이 기존 방법들을 벤치마킹**하는 데 그친다.

게다가 두 기존 협력 벤치마크 모두 CAV 수가 도심 교차로의 실제 차량 수보다 훨씬 적다(Table 1 기준 평균 3.80대, 3.36대). 이는 평가 시나리오의 복잡성과 현실성을 제한한다.

---

## 2. CMU-Drive 벤치마크 구축

CMU-Drive는 Bench2Drive를 **단일 에이전트 시뮬레이션·평가 프로토콜에서 협력 멀티에이전트 설정으로 변환**하면서 다양한 도시 교통 시나리오는 그대로 보존한다.

- Bench2Drive의 **44개 시나리오 타입**을 상속: 보행자 횡단, 응급차량의 신호 무시 통과 등 안전 필수 도심 주행 상황.
- 각 시나리오를 **5가지 날씨·조명 조건**으로 인스턴스화 → 총 **220개 평가 route**.
- 하나의 CARLA 클로즈드루프 환경 안에서 **2~16대의 협력 자율주행 차량**을 동시에 시뮬레이션·평가. 각 CAV는 개별 시작 지점과 목적지를 갖고, 배경 차량·보행자·자전거가 있는 동일한 동적 환경을 공유한다.
- **첫 번째 CAV**는 원본 Bench2Drive route 설정을 그대로 따라가며 안전 필수 배경 이벤트의 트리거 역할을 유지 → 벤치마크 일관성 보존.
- 추가 CAV들의 시작·목적지는 각 시나리오의 도로 위상(road topology)에서 **자동 생성**. 예컨대 교차로 시나리오에서는 서로 다른 진입 차선에서 출발해 대응하는 출구로 향한다.
- 중요한 설계: 추가 CAV의 시작 위치를 **안전 필수 이벤트 트리거를 방해할 수 있는 차선에는 두지 않는다**. 원본 벤치마크가 의도한 상호작용 행동과 multi-ability 벤치마킹을 보존하기 위함이다.

📌 [Figure 1 삽입] — 보행자 횡단(CAV 3대), 응급차량 신호위반(8대), 자전거 횡단(12대), 혼잡 교차로 직진/회전(16대) 예시 route

### Table 1 비교

| Benchmark | # Routes | Min # CAVs | Max # CAVs | Avg # CAVs | # GPUs |
|---|---|---|---|---|---|
| InterDrive | 92 | 2 | 8 | 3.80 | 3 |
| MDrive | 225 | 1 | 8 | 3.36 | 3 |
| **CMU-Drive (ours)** | **220** | **2** | **16** | **6.56** | **1** |

주목할 점은 GPU 수다. 저자들은 선행 연구와 **다른 코딩 아키텍처 설계**를 채택해, 최대 16개 협력 에이전트가 도는 클로즈드루프 평가를 **GPU 1장**으로 수행한다. 평가 진입 장벽을 3배 낮춘 셈이다.

---

## 3. 멀티에이전트 평가 지표

CARLA Leaderboard 2.0 / Bench2Drive / SimLingo가 쓰는 단일 에이전트 프로토콜을 멀티에이전트로 확장한다.

**에이전트 수준**: route j의 에이전트 i에 대해
- RC_i^j = 에이전트 i가 완주한 할당 route의 비율
- IS_i^j = 기저 점수 1.0에서 출발해 위반마다 곱셈 페널티. 위반 항목은 보행자·자전거·차량·정적 물체와의 충돌, 적색신호·정지신호 위반, 도로 이탈, 응급차량 양보 실패.

**Route 수준**:
- RC^j = (1/N^j) Σ_i RC_i^j  — 에이전트 평균
- IS^j = Π_i IS_i^j  — 에이전트 **곱**
- DS^j = RC^j · IS^j
- SS^j = 1 if DS^j = 100 else 0

**전체**: DS = (1/R) Σ_j DS^j, SR = (1/R) Σ_j SS^j, R = 220.

여기서 설계상 가장 중요한 선택은 **infraction score를 곱으로 집계**한다는 점이다. 에이전트가 늘어날수록 route-level IS는 기하급수적으로 가혹해진다. 16대 중 한 대만 사고를 내도 route 전체 점수가 무너진다. 즉 CMU-Drive의 DS는 "평균적으로 잘하는가"가 아니라 **"전원이 무사한가"** 를 묻는 지표이며, 이것이 SR이 30%대에 머무는 이유이기도 하다.

---

## 4. V2V-VLA 아키텍처

각 협력 주행 에이전트는 **자기만의 V2V-VLA 모델**을 갖는다(중앙 집중식이 아니다). 클로즈드루프 시뮬레이션의 매 프레임마다 각 에이전트는 주행 행동, 언어 추론, 미래 waypoint, 그리고 **다음 timestep에 어떤 CAV에 질의할지에 대한 선택적 통신 정책**을 동시에 생성한다. 각 에이전트는 RGB 카메라, IMU, GPS를 장착한다.

📌 [Figure 2 삽입] — V2V-VLA 전체 아키텍처

### 입력 3종

**Language input (L_input)**: ego가 취해야 할 주행 행동을 서술하도록 요청하는 프롬프트 템플릿. 여기에 **다른 협력 차량들의 상대 위치를 ego 좌표계로** 함께 넣는다. 실제 예시(Figure 4): `What should the ego do next? Cooperative Autonomous Vehicles: CAV_4 at front (17.7, -2.2). CAV_5 at back (-30.1, -0.1). Target waypoint: (74.78, -0.34). <IMAGE> <OCCUPANCY_MAP>`

**Action input (A_input)**: ego의 target waypoint를 MLP로 인코딩.

**Vision input (V_input)**: 두 부분.
1. ego의 전방 RGB 카메라 이미지 I_ego
2. **병합된 카메라 기반 BEV occupancy map O_merged**

O_merged 생성 절차가 이 논문의 협력 지각 핵심이다.
- ego와 선택된 통신 차량 CAV_c가 각각 카메라 기반 BEV occupancy map O_ego, O_c를 생성 (**UniAD** 사용)
- 두 차량의 pose P_ego, P_c로 좌표 변환: `O_{c->ego} = CoordinateTransform(O_c, P_ego, P_c)`
- element-wise-or로 병합: `O_merged = ElementwiseOr(O_ego, O_{c->ego})`
- O_merged를 **RGB 이미지로 렌더링**해 I_ego와 함께 vision encoder(**InternViT**)에 투입: `V_input = VisionEncoder([I_ego, O_merged])`

occupancy를 이미지로 렌더링해 VLM에 넣는 선택은 아키텍처를 단순화한다. 별도 fusion 모듈 없이 기존 VLM의 vision 경로를 그대로 재활용하고, LLM은 두 이미지를 나란히 읽는다.

### LLM과 출력

`L_output, A_output = LLM(L_input, A_input, V_input)`, LLM은 **Qwen2**.

**Language output**: 주행 결정을 설명하고 필요 시 안전 필수 물체를 식별한다. 특히 **ego에게 보이지 않는 critical object**에 대한 추론 코멘트와 **권장 통신 대상 CAV_c**를 생성한다.

**Action output**: SimLingo와 동일한 포맷 — 미래 **isochronous** waypoint와 **equidistant** waypoint. 이 waypoint들이 PID 컨트롤러로 들어가 steer/throttle/brake를 만든다.

### 통신 알고리즘 (3.1절)

- **기본값**: ego는 자신의 **가장 가까운 전방 CAV**를 통신 상대 CAV_c로 선택.
- **모델 기반 갱신**: V2V-VLA가 "다른 협력 차량 CAV_k가 ego에게 보이지 않는 critical object를 감지할 수 있다"고 언어 출력에서 지시하면, **다음 timestep**에서 ego는 기본 선택 대신 CAV_k를 CAV_c로 사용한다.

통신 정책이 별도 헤드가 아니라 **언어 출력 문장 안에 녹아 있다**는 점이 설계상 가장 특이하다. 추론(왜 감속하는가)과 통신 결정(누구에게 물을 것인가)이 같은 토큰 시퀀스에서 나오므로, 다중 라운드 협상 없이 단일 forward pass로 끝난다.

---

## 5. 학습 데이터 수집

1. SimLingo의 단일 에이전트 **학습** route 일부를, Bench2Drive 평가 route를 CMU-Drive 평가 route로 확장한 것과 **동일한 방식**으로 멀티에이전트 학습 route로 확장.
2. 각 자율주행 에이전트의 expert 주행 모델로 **PDM-lite** 사용. PDM-lite는 CARLA 환경 내 **모든 물체의 ground-truth 위치·속도**를, 그 물체가 보이든 안 보이든 상관없이 사용해 주행을 결정하는 규칙 기반 플래너다.
3. 매 timestep마다 PDM-lite 에이전트의 미래 waypoint를 A_output의 ground-truth 주석으로 저장.
4. L_output의 권장 행동과 추론 부분은 DriveLM·SimLingo와 유사한 **규칙 기반** 방식으로 생성.
5. **선행 연구와 다른 지점 — 통신 supervision 주석**: 매 timestep마다 expert 에이전트의 행동에 영향을 준 critical object를 식별하고, 그 물체가 **ego에게는 보이지 않지만 다른 협력 차량에게는 관측 가능**하면 그 차량을 통신 파트너로 권장하도록 주석한다.

총 **180K SFT 샘플**.

여기서 PDM-lite의 특권 정보(privileged information) 사용이 통신 supervision을 자동 생성 가능하게 만드는 열쇠다. expert는 가시성과 무관하게 모든 물체를 알기 때문에, "무엇이 결정에 영향을 줬는가"와 "그것이 누구에게 보였는가"를 시뮬레이터 상태에서 정확히 라벨링할 수 있다.

---

## 6. 학습 세부

- **손실**: A_output의 미래 waypoint에 대한 **smooth-L1** + L_output에 대한 **cross-entropy**
- **초기화**: **SimLingo**와 **UniAD**의 가중치로 시작
- **동결/미세조정**: LLM의 **LoRA** 부분만 fine-tune, **UniAD는 freeze**, V2V-VLA의 나머지 학습 가능 레이어는 모두 fine-tune
- **12 epochs**, **batch size 8**
- 나머지 하이퍼파라미터는 SimLingo와 동일
- **8× NVIDIA H100-80GB, 48시간**

즉 V2V-VLA는 frozen 정책 위에 덧씌운 래퍼가 아니라, 180K 샘플로 자체 supervised fine-tuning을 수행한 **자체 학습 주행 policy**다. UniAD만 occupancy 생성기로 동결되어 있다.

---

## 7. 실험 결과

### 정량 (Table 2)

| Method | Cooperation | DS ↑ | SR (%) ↑ |
|---|---|---|---|
| SimLingo | ✗ | 56.32 | 30.91 |
| **V2V-VLA (ours)** | ✓ | **63.67** | **34.55** |

220개 CMU-Drive 평가 route 기준. 협력을 하지 않는 baseline 대비 **DS +7.35 (상대 +13.0%)**, **SR +3.64%p (상대 +11.8%)**.

앞서 짚은 IS의 곱셈 집계를 감안하면 DS 7점 상승은 겉보기보다 큰 폭이다. route 안의 어느 한 에이전트가 위반을 줄여도 route-level IS 전체가 곱으로 개선되기 때문에, 이 향상은 특정 차량이 아니라 **에이전트 전반의 위반 감소**를 시사한다.

### 시나리오별 (Figure 3)

44개 시나리오 타입 각각에 대한 V2V-VLA의 평균 driving score가 막대 그래프로 제시된다. 여전히 어려운 안전 필수 시나리오가 여럿 남아 있으며, 논문은 **NonSignalizedJunctionLeftTurnEnterFlow**(비신호 교차로에서 좌회전으로 교통 흐름에 진입)를 대표 사례로 지목하며 추가 연구가 필요하다고 밝힌다.

📌 [Figure 3 삽입] — 시나리오 타입별 평균 driving score

### 정성 (Figure 4)

ego 차량 CAV_0은 **장거리와 비로 인한 이미지 blur** 때문에 선행 자전거를 명확히 관측하지 못한다. 그러나 위험 대상에 더 가까운 선행 협력 차량 CAV_4가 자전거를 성공적으로 식별하고, V2V 통신을 통해 공유된 지각 정보를 이용해 CAV_0이 안전하게 감속해 충돌을 예방한다.

모델 출력 예시: *"Follow the route. Maintain the reduced speed to stay behind the bicycle that is to the front. The critical object is occluded from the ego car's view but detected by CAV_4 at front (17.7, -2.2) via V2V communication. Future isochronous waypoints: [...]. Future equidistant waypoints: [...]."*

이 한 문장 안에 행동(감속 유지) · 이유(전방 자전거) · 통신 근거(가려진 물체를 CAV_4가 감지) · 통신 대상(CAV_4)이 모두 들어 있다. 통신 정책을 언어에 실은 설계의 의도가 그대로 드러나는 예시다.

---

## 8. Related Work 상의 위치

- **단일 에이전트 주행 VLA** (EMMA, DriveVLM, OpenEMMA, AutoVLA, SimLingo, Alpamayo-R1, SpaceDrive): 협력 지각·추론·계획 미지원. V2V-VLA는 이 계보를 멀티에이전트로 확장.
- **협력 자율주행** (V2V-LLM, V2V-GoT, CoLMDriver, V2X-Real, V2V4Real, QuantV2X, V2XPnP, CoopRe, TurboTrain, TUMTraf-V2X): 다수가 지각 데이터셋이거나, LLM 기반이어도 다중 라운드 협상에 의존. V2V-VLA는 협상을 없애고 **단일 forward pass**로 통합.
- **클로즈드루프 벤치마크** (Bench2Drive → InterDrive → MDrive): CMU-Drive는 CAV 수(최대 16, 평균 6.56), 안전 필수 배경 시나리오 보존, GPU 요구량(1장) 세 축에서 차별화.

같은 1저자의 V2V-LLM·V2V-GoT가 협력 LLM의 **지각·QA** 측면을 다뤘다면, 본 논문은 그 흐름을 **클로즈드루프 제어**까지 밀어붙인 후속작으로 읽힌다.

---

## 9. 강점

1. **통신 결정을 언어로 학습한 발상**: 통신 대상 선택을 별도 모듈이나 협상 프로토콜이 아니라 언어 출력의 일부로 만들어, 추론과 통신을 하나의 autoregressive 생성으로 통합했다. 다중 라운드 LLM 추론 대비 지연 이점이 구조적으로 명확하다.
2. **PDM-lite 특권 정보를 통신 supervision으로 전환**: "critical object가 ego에겐 안 보이고 다른 CAV에겐 보인다"는 라벨을 시뮬레이터에서 자동 생성한 것은 데이터 파이프라인 측면의 실질적 기여다. 사람 주석 없이 180K 샘플을 만든 근거.
3. **평가 비용을 3배 낮춤**: 최대 16 CAV 클로즈드루프를 GPU 1장으로 돌리는 코드 아키텍처는 커뮤니티 재현성에 직접적으로 기여한다.
4. **Bench2Drive의 안전 필수 트리거를 훼손하지 않는 확장 설계**: 추가 CAV를 트리거 방해 차선에서 배제한 세심함 덕에 원 벤치마크의 multi-ability 성격이 보존된다.
5. **표현 수준 융합의 단순함**: occupancy를 좌표변환 후 or 연산으로 합치고 RGB로 렌더링해 VLM에 넣는 방식은 구현이 가볍고 통신 대역폭 논의로 확장하기 쉽다.

---

## 10. 약점 및 한계

1. **Baseline이 단 하나**: Table 2는 SimLingo 한 줄뿐이다. CoLMDriver나 MDrive에서 벤치마킹된 협력 방법들과의 비교가 없어, 향상이 "협력 일반"의 효과인지 "V2V-VLA 특유 설계"의 효과인지 분리되지 않는다.
2. **Ablation 전무**: 통신 정책(기본 전방 선택 vs 모델 제안), occupancy 병합, 언어 추론 supervision, CAV 수 변화 각각의 기여도가 측정되지 않았다. 특히 "가장 가까운 전방 CAV" 기본 규칙만으로 얼마나 나오는지가 빠져 있어, 모델이 학습한 통신 선택의 실제 가치가 불명확하다.
3. **Figure 3의 수치 미제공**: 44개 시나리오별 점수가 막대 그래프로만 제시되어 후속 연구가 정확한 수치를 재사용할 수 없다.
4. **element-wise-or 융합의 조악함**: occupancy를 이진 or로 합치면 각 차량의 confidence, 관측 불확실성, 좌표변환 오차가 모두 소실된다. 실제 V2V에서는 localization 오차가 핵심 난제인데 이 논문은 정확한 pose를 가정한다.
5. **통신 비용·지연·실패 모델 부재**: 대역폭, 패킷 손실, 통신 지연, 악의적/고장 CAV 같은 현실적 V2V 제약이 전혀 모델링되지 않았다.
6. **한 프레임에 통신 상대 1대**: O_merged는 ego + CAV_c 단 하나의 병합이다. 16 CAV 시나리오에서 한 번에 한 대만 참조하는 제약이 성능 상한이 될 수 있다.
7. **파라미터 수·추론 속도 미공개**: Qwen2 크기, InternViT 크기, 클로즈드루프 실행 주파수가 명시되지 않아 실시간성 판단이 불가능하다.
8. **Sim-only**: CARLA 전용이며 실차·실데이터 검증은 없다. PDM-lite 특권 expert에 의존하므로 실세계 이전 시 supervision 생성 경로가 그대로 통하지 않는다.

---

## 11. 재현 및 확장 아이디어

- **통신 정책 ablation**: (a) 통신 없음 (b) 항상 최근접 전방 CAV (c) 모델 제안 CAV (d) 오라클(critical object를 실제로 보는 CAV) 4단 비교. 논문이 남긴 가장 큰 공백이자 가장 저렴한 실험.
- **CAV 수 스케일링 곡선**: 2/4/8/16 CAV 구간별 DS·SR을 분해하면, IS의 곱셈 집계가 만드는 난이도 상승과 협력 이득의 상충을 정량화할 수 있다.
- **다중 상대 병합**: O_merged를 상위 k개 CAV로 확장하고 or 대신 confidence-weighted soft fusion 적용.
- **Pose noise robustness**: P_ego, P_c에 GPS 수준 노이즈(0.1~1 m, 1~5°)를 주입해 좌표변환 융합의 강건성 측정.
- **통신 예산 제약**: timestep당 통신 횟수에 예산을 걸고, 모델이 "언제 물을 가치가 있는가"를 학습하도록 RL로 fine-tune.
- **비신호 좌회전 특화**: Figure 3에서 지목된 NonSignalizedJunctionLeftTurnEnterFlow에 데이터를 집중 증강해, 협력이 가장 큰 이득을 주는 시나리오 클래스를 규명.
- **Bench2Drive 단일 에이전트 역검증**: CMU-Drive로 학습한 V2V-VLA가 원본 Bench2Drive 단일 에이전트에서 SimLingo 대비 퇴행하지 않는지 확인.

---

## 12. 총평

이 논문의 실질적 기여는 두 개이고, 둘 다 "인프라"에 가깝다. 하나는 **GPU 1장으로 최대 16 CAV 클로즈드루프를 돌리는 CMU-Drive**이고, 다른 하나는 **협력 주행 VLA의 첫 baseline인 V2V-VLA**다. 저자들 스스로도 "first benchmark and baseline"이라는 표현을 쓴다.

모델 쪽에서 가장 인상적인 아이디어는 **통신 결정을 언어 출력에 녹인 것**이다. CoLMDriver류의 다중 라운드 협상을 단일 forward pass로 접는 이 설계는, "누구에게 물을지"라는 이산 결정을 별도 정책망으로 분리하지 않고 이미 학습 중인 추론 토큰 시퀀스에 얹었다는 점에서 우아하다. PDM-lite의 특권 정보로 그 supervision을 자동 생성한 것도 실무적으로 영리하다.

다만 논문의 실험 파트는 기여 대비 확연히 얇다. Table 2가 두 줄이고 ablation이 하나도 없다. DS 56.32 → 63.67이라는 숫자만으로는 "협력 정보를 넣으면 좋아진다"는 자명한 명제를 확인할 뿐, V2V-VLA의 어떤 설계 요소가 얼마나 기여했는지는 알 수 없다. 특히 기본 통신 규칙("가장 가까운 전방 CAV")이 이미 강력한 휴리스틱이므로, 모델이 학습한 통신 제안이 그 위에 얼마나 더 얹었는지가 검증되지 않은 것은 아쉽다. 융합 방식(binary or)과 pose 정확도 가정도 실세계 V2V의 핵심 난제를 우회한다.

그럼에도 협력 주행 VLA라는 문제 설정 자체가 신규하고, 벤치마크·코드·체크포인트 공개를 약속했으며, 평가 비용을 3배 낮춘 점은 분야 진입 장벽을 실제로 낮춘다. **"완성된 방법론"보다는 "잘 설계된 출발점"으로 읽는 것이 정확한 논문**이며, 후속 연구가 붙기 좋은 형태로 열려 있다는 점이 최대 미덕이다.

<!-- VERIFIED: pdf -->
