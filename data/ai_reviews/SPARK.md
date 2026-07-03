# SPARK: Sequential Planning via Anchored Robotic Keypoints

> **한 줄 요약**: 학습이 전혀 없는(training-free) 뉴로심볼릭 조작 시스템. Gemini 1회 호출로 5개 기본 프리미티브 위의 **typed behavior tree(BT)** 를 커밋하고, 남는 test-time compute를 전부 **지각(perception)** 에 투자 — 객체당 3개의 대안 SAM3 텍스트 프롬프트를 생성해 가장 확신도 높은 검출을 선택(adaptive perception self-consistency)하고, 실패 시 새 LLM 호출 없이 재검출-재시도한다. LIBERO-PRO 6개 position-and-task 셀에서 43.7%로 CaP-Agent0(18.2%)·MolmoAct2(18.6%)·π0.5(12.8%)를 2배 이상 앞서고 RATS(43.8%)와 대등하며, 시행당 비용은 ~$0.048로 CaP-Agent0의 ~1/20. 동일 프리미티브가 3개 로봇 패밀리(UR10e, FR3, 양팔 Franka)에서 재학습 없이 평균 68%를 기록. Case Western Reserve University.

---

## 1. 배경 및 동기

- OpenVLA·π0·π0.5 등 end-to-end VLA는 표준 LIBERO에서 95%+를 기록하지만, 객체 위치와 태스크 서술을 섭동하는 **LIBERO-PRO**에서는 0에 가깝게 붕괴 (OpenVLA·π0 0.0%, π0.5 12.8%) — 기계론적 해석 연구들은 VLA가 latent feature를 **절대 end-effector 위치로 인코딩**해 단일 레이아웃에 묶인 궤적을 암기한다는 점을 원인으로 지목
- Code-as-Policy 계열의 CaP-Agent0은 3개 프론티어 모델 앙상블 + 멀티턴 코드 재합성으로 18.2%까지 회복하지만, 턴당 ~9회 프론티어 모델 호출 + VDM ~10회 호출로 시행당 ~$1 수준의 비용이 들고, 실패하면 모델을 다시 질의하는 것 외의 탈출구가 없음
- 저자들의 핵심 관찰: 위치·언어 섭동은 **플랜의 구조를 거의 바꾸지 않는다**. "그릇을 접시에 올려라"는 그릇이 왼쪽에 있든 오른쪽에 있든 같은 상위 프로그램이다. 바뀌는 것은 대상 객체가 차지하는 **픽셀의 위치** — 즉 가장 많이 실패하는 층은 플랜이 아니라 지각
- 따라서 SPARK는 test-time compute를 플랜 재생성(VLA는 VLM 백본에, CaP는 매 턴 코드에)이 아닌 **지각 층**에 배분한다

## 2. 방법론 심층 분석

### 2.1 파이프라인 개요 (4 컴포넌트)
1. **SAM3** — 멀티카메라(버드아이+사이드+손목, 640×480 RGB-D; LIBERO-PRO에서는 표준 agentview + eye-in-hand)에서 open-vocabulary 텍스트 프롬프트로 마스크 생성 → 각 마스크의 centroid 픽셀 + median masked depth를 intrinsics로 역투영해 월드 좌표 3D 키포인트 산출
2. **Gemini 플래너** — 주석 이미지(마스크+라벨) + 3D 키포인트 + 태스크 언어를 받아 **단일 호출**(temperature 0.3, JSON mime)로 YAML BT를 방출. 시뮬레이션은 Gemini 3.1 Pro, 실기는 Gemini 3.5 Flash. 실행 전 파싱·타입체크되며 malformed 출력은 복구 층으로
3. **IK 컨트롤러** — 자세 제약 이동은 Pyroki(JAX 기반 6-DOF 제약 솔버, w_pos=1.0/w_ori=0.5), 비제약 이동은 MuJoCo Jacobian pseudoinverse; Cartesian 이동은 Robosuite OSC(병진 강성 300 N/m). 연속 waypoint는 Ruckig time-optimal 궤적으로 연결
4. **계층적 복구 층** — §2.4

### 2.2 Typed BT 문법: "로봇이 초견 연주하는 악보"
- **5개 기본 프리미티브**(move_to_keypoint, move_relative, grasp, release, wait)가 시스템이 표현 가능한 모션 공간을 스팬하고, 여기서 30+개의 typed skill(grasp_se3, insert, pour, sweep, constrained_scrub, open_drawer, screw, 양팔용 pick_with_arm/sync_barrier/handoff 등)이 합성됨 — LIBERO-PRO 보고 수치는 기본 프리미티브만으로 구성
- 각 프리미티브는 **키포인트 라벨 + 스칼라 파라미터**(force, offset, angle, duration)만 받음. 쿼터니언 연산·depth 투영 같은 저수준 제어는 문법 안에 캡슐화 — CaP-Agent0이 LLM에 solve_ik()와 쿼터니언 산술을 직접 노출해 매 시행 런타임 에러를 유발하는 것과 대조
- 라벨은 **실행 순간** 현재 SAM3 검출 맵에 대해 3D 좌표로 해석(anchored keypoint) → 계획 후 객체가 이동해도 실행 전에 재검출됨. "wash" 스킬이 없어도 Gemini가 move_relative 진동 4개로 접시 문지르기 래스터 패턴을 스스로 합성하는 사례 보고
- 성공한 BT는 수동적으로 캐시되어 이후 태스크의 in-context 예시로 제공 가능(Voyager 정신), 단 파인튜닝은 전무

### 2.3 Adaptive Perception Self-Consistency (§3.4)
- SAM3 검출이 전체 성능의 병목 → self-consistency를 지각 접지에 적용. 시뮬레이션에서 시행당 1회의 추가 Gemini 호출이 장면 이미지+지시문을 보고 객체당 **3개의 대안 프롬프트**(색·형태·재질 변형, K=3)를 제안, SAM3가 각각을 평가해 **단일 확신 검출**을 내는 프롬프트→라벨 쌍을 채택. 약한 다중 매치를 내는 모호한 프롬프트는 폐기. 실기는 단일 프롬프트
- 효과: spatial 평균 +27.7pt (64.2 vs 36.5), object 평균 +10.0pt (39.9 vs 29.9). 같은 추가 컴퓨트를 플랜 재생성에 쓰면 **측정 가능한 이득 없음** — 논문의 핵심 주장을 뒷받침하는 통제 실험

### 2.4 3단 복구 (Tiered Recovery)
1. **In-place perturbation**: wiggle / grasp_perturb로 아슬아슬하게 빗나간 접촉 재안착
2. **Perception re-grounding**: z축 10cm 후퇴 → SAM3 재실행 → **같은 플랜**을 교정된 위치로 재시도 (LLM 호출 없음)
3. **Replan**: 새 Gemini 호출 — 보고된 실험에서는 미사용
- 재접지 비활성화 시 LIBERO-PRO ~5pt 하락; 회복된 실패 대부분은 팔이 카메라 시야에서 물러나면 해소되는 첫 프레임 SAM3 미검출

## 3. 실험 설정

- **LIBERO-PRO**: 16셀 중 CaP-Agent0(Fu et al.)의 프로토콜을 따라 6개 position-and-task 셀(object/goal/spatial 스위트 × position/task 섭동, 셀당 10태스크 × 50시행) 평가. **S2 프로토콜**: vision-only, ground-truth 조회 없음. 시뮬레이션은 시행당 Gemini 2회 호출(프롬프트 합성 + BT 생성)
- **CaP-Bench**: CaP-Agent0 자체 robosuite 벤치마크 7태스크, 태스크당 100시행
- **물리 실험**: UR10e(Robotiq 2F-85), Franka FR3(Franka Hand), 양팔 Franka(Panda 좌 + FR3 우, MSG compliant gripper). 9개 고유 태스크, 11개 태스크-임바디먼트 셀, 셀당 20시행. Azure Kinect 구조광 depth + RealSense D435i 손목(양팔은 ZED Mini + OV9732). 객체 위치·회전 무작위, 일부 태스크는 객체 카테고리 교체
- **Ablation 축**: 지각 소싱 3구성 — Fair(태스크 언어만, CaP-Agent0 매칭) / +BDDL names(LIBERO 정식 객체명 사전 추가) / Adaptive(3-프롬프트 self-consistency, 완전체) — 플래너·컨트롤러 고정

## 4. 핵심 결과 — LIBERO-PRO (Table 1)

| Method | Obj-Pos | Obj-Task | Goal-Pos | Goal-Task | Spa-Pos | Spa-Task | Mean |
|---|---|---|---|---|---|---|---|
| OpenVLA / π0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| π0.5 | 17 | 1 | 38 | 0 | 20 | 1 | 12.8 |
| MolmoAct2 | 47.2 | 0.0 | 29.0 | 12.0 | 23.0 | 0.4 | 18.6 |
| CaP-Agent0 (원 보고) | 22 | 18 | 26 | 17 | 12 | 14 | 18.2 |
| RATS | 61.0 | 63.0 | 43.0 | 36.0 | 29.0 | 31.0 | **43.8** |
| SPARK Fair | 36.4 | 23.4 | 36.4 | 22.4 | 30.0 | 43.0 | 31.9 |
| **SPARK Adaptive** | 43.4 | 36.4 | 40.0 | 14.0 | **56.0** | **72.4** | **43.7** |

- 최강 보고 기법 RATS(43.8%)와 사실상 동률 — 단 RATS는 오프라인 "play" 단계로 재사용 코드-스킬 라이브러리를 구축하는 반면 SPARK는 플래닝 1회 호출뿐
- **spatial 스위트 64.2%로 RATS(30.0%)의 2배 이상** — 객체 정체성은 유지된 채 위치만 바뀌는 섭동에서 지각 투자 전략의 수익이 최대
- +BDDL names(31.2)가 Fair(31.9)와 사실상 동일 → 정식 객체명 사전은 도움이 안 되고, 프롬프트 **선택 메커니즘**이 핵심
- 유일한 역효과 셀: **goal-task 14.0%** (Fair 22.4%보다 낮음) — 목표 재작성이 객체가 아닌 지역·기구(다른 서랍, 랙 대신 스토브)를 재지정하므로 외관 변형 프롬프트가 노이즈만 추가. 일부 goal-task 태스크는 지각과 무관하게 kinematic cap 존재

## 5. CaP-Bench 및 비용 결과

| Task | CaP-Agent0 (M4) | RATS | SPARK |
|---|---|---|---|
| Lift | ~100 | 84 | **100** |
| Stack | ~95 | 60 | **97** |
| CubeRestack | ~95 | 46 | **100** |
| Wipe | **~85** | 100 | 60 |
| NutAssemblySquare | 0 | 0 | 0 |
| TwoArmLift | **~70** | 34 | 63 |
| TwoArmHandover | **~30** | 20 | 24 |

- pick-and-place 계열은 **단일 호출**로 멀티턴 앙상블(M4: GPT-5.2 + Claude Opus 4.5 + Gemini-3-Pro 각 3회/턴 + VDM ~10회)과 동률 이상. NutAssemblySquare 0%는 OSC 컨트롤러 z 하한이라는 양측 공유 kinematic ceiling
- 열세 3태스크(Wipe/TwoArmLift/TwoArmHandover)는 **실행 중 관측 피드백**이 필요한 태스크 — 얼룩이 지워졌는지는 행동 후 이미지에서만 읽히는데 CaP의 VDM이 이를 턴 간 텍스트로 노출하는 반면 SPARK는 플랜을 한 번에 커밋. 구조적 한계이며 turn-level observation gate로 회복 가능하다고 논함
- **비용**: 시뮬 시행당 ~$0.048 (BT 생성 ~$0.038 + 변형 프롬프트 ~$0.010, Gemini 3.1 Pro $2/1M in·$12/1M out 기준), 실기 ~$0.028 (Flash + implicit cache). CaP-Agent0은 문서화된 호출 수에 2026 공개 단가 적용 시 시행당 ~$1 규모 → **약 20배 차이**

## 6. 물리 실험 (Table 2, 셀당 20시행)

| 플랫폼 | 태스크 | 성공률 |
|---|---|---|
| UR10e | Utensils in bowl / in tray / Plushie in bowl / Stack blocks | 55 / 90 / 100 / 65 |
| FR3 | Utensils in tray / Sponge-wash / Mug pour / Sweep / T-shirt fold | 80 / 100 / 65 / 55 / 50 |
| Bimanual | T-shirt fold / Silverware sort | 30 / 60 |
| **평균 (11셀)** | | **68** |

- Utensils-in-tray가 UR10e(90%) vs FR3(80%)로 **IK 솔버·그리퍼·워크스페이스만 다른 통제된 크로스-임바디먼트 비교** 제공 — 넓은 Robotiq 조가 식기 형상에 유리하다는 해석
- 지배적 실패는 거의 전부 SAM3 접지: 포크 손잡이를 나이프로 0.94 확신도 오분류, 어두운 천의 hem 경계 미해석(검정/회색 셔츠 0% 폴드), 좁은 머그 손잡이의 depth 노이즈. 얇은 물체는 마스크 OBB 주축이 프레임 간 뒤집혀 grasp_se3(SE(3)-equivariant grasp 생성기)로 대체
- 이 실패 국지화 자체가 논문의 주장 — BT 트레이스와 프리미티브별 post-condition이 실패를 플래너/지각/kinematic limit 중 하나로 **판독 가능하게 귀속**시키며, 플래너와 컨트롤러는 거의 잘못이 없음을 보임

## 7. Ablation 종합

- **Adaptive self-consistency**: spatial +27.7pt, object +10.0pt, goal-task −8.4pt — 이득은 "객체는 그대로, 위치만 이동"한 섭동에 집중
- **복구 루프**: 비활성화 시 전체 ~5pt 하락, 대부분 retract-and-re-detect 1사이클로 해소되는 첫 프레임 SAM3 미검출
- **BDDL 정식 명칭 사전**: 무이득 (31.2 vs Fair 31.9) — 무엇을 아느냐보다 어떤 표현이 검출기에 먹히는지를 고르는 것이 중요
- **동일 컴퓨트를 플랜에 투자**: 측정 가능한 이득 없음 — compute 배분 주장의 직접 근거

## 8. 부산물: 텔레오퍼레이션 없는 학습 데이터

- 모든 시행이 typed grammar를 통과하므로 매 트라이얼이 **의미 라벨링된 에피소드 레코드**(BT, 궤적, 타임스탬프된 프리미티브 트레이스, 객체 접지, SAM3 검출 맵)를 남김
- VLA들이 이미 이기고 있는 training-free 플래너가, 그 VLA들을 학습시킬 검증된 시연 데이터를 사람 텔레오퍼레이션 없이 크로스-임바디먼트로 공급할 수 있다는 제안 — 시스템 논문을 넘어 데이터 엔진으로서의 포지셔닝

## 9. 약점 / 한계

1. **표준 벤치마크 부재**: 표준 LIBERO/CALVIN 수치가 없어 기존 VLA 리더보드와 수평 비교 불가 — LIBERO-PRO 43.7%는 섭동 프로토콜 수치로 읽어야 함
2. **goal-task 취약**: adaptive 프롬프트가 오히려 −8.4pt; 태스크 자체가 바뀌는 섭동에는 지각 투자 전략이 실패함을 저자 스스로 인정
3. **단일-샷 플랜의 관측 맹점**: 실행 중 성공 신호가 필요한 Wipe·양팔 핸드오버류에서 멀티턴 CaP에 열세
4. **크로스-임바디먼트 전이가 부분적**: 플래너는 무학습 전이되나 워크스페이스 한계·그리퍼 힘 프로파일·IK 파라미터는 플랫폼별 튜닝 잔존
5. **SAM3 단일 실패점**: 어두운 천·투명/반사면 depth 바이어스·유사 객체 30px centroid 병합 등 지배적 실패가 전부 지각에 집중 — 복구 루프가 프롬프트 집합을 바꾸지 않아 결정론적 오검출은 재시도에도 지속
6. **프론티어 API 의존**: Gemini 3.1 Pro/3.5 Flash 의존이라 재현성과 장기 비교 가능성이 모델 버전에 묶임; monocular depth(DA3)는 cross-view 불일치로 실기에서 하드웨어 depth 필수
7. 물리 실험이 셀당 20시행으로 표본이 작고, 적대적 강건성은 명시적으로 범위 외 (Appendix G는 논의만)

## 10. 다른 연구와의 위치

- **vs CaP-Agent0 (Code-as-Policy 계열)**: 같은 test-time compute 회복 전략이지만 층이 다름 — CaP는 플랜 층(멀티턴 코드 재합성, 3모델 앙상블), SPARK는 지각 층(프롬프트 self-consistency + 재접지). 단일 호출로 pick-and-place 동률, 비용 ~1/20
- **vs RATS**: 오프라인 self-directed play로 스킬 라이브러리를 증류하는 RATS와 전체 평균 동률(43.7 vs 43.8)이나, SPARK는 온라인 예산만 사용하고 spatial에서 2배 이상
- **vs ReKep/MOKA**: 같은 keypoint-접지 training-free 계열이나, ReKep은 ~10Hz로 재최적화되는 암묵적 궤적(단일 로봇 역학에 종속), MOKA는 시행마다 VLM이 키포인트 선택. SPARK는 명시적 typed BT 하나를 커밋하고 고정 라벨을 open-vocabulary 재검출로 재접지 — 3개 임바디먼트에서 동일 플랜
- **vs NS-VLA**: 유사한 프리미티브 집합으로 LIBERO-PLUS에서 좋은 성능이나 BC 사전학습 + 온라인 RL 필요; SPARK는 같은 구조적 prior(키포인트 위 객체 수준 추상화)를 무학습으로 회복
- **vs LLM-to-BT 계열 (LLM-OBTEA, BETR-XP-LLM 등)**: 텍스트 목표 해석 수준이던 단일 호출 BT 생성을 지각 접지 조작 + 현대 VLA/코드 에이전트 베이스라인 비교로 확장
- **vs RoboMonkey**: test-time compute를 액션 층(K개 섭동 액션 VLM 검증)에 쓰는 것과의 3자 대비 — SPARK는 지각 층이 위치·언어 섭동 하에서 호출당 수익이 최대라는 실증

## 11. 향후 연구 방향

- **Turn-level observation gate**: 실행 중 성공 조건 재확인으로 Wipe/양팔 핸드오버 격차 회복 (저자 명시)
- BT 전이에 대한 학습된 world model(V-JEPA 2, DINO-WM 계열) 결합, 다단계 조립을 위한 part-level 접지(PartNeXt 등), 촉각 피드백(FlexiTac) — 모듈이 typed plan 뒤에 있어 재학습 없이 drop-in 가능하다는 모듈성 주장
- goal-task형 섭동에서 프롬프트 다양화가 아닌 다른 지각 전략(예: 지역/기구 접지) 탐색
- 로그된 검증 궤적으로 실제 VLA를 학습시켜 "데이터 엔진" 주장을 폐루프로 입증
- 17–28종 섭동 벤치마크(RobustVLA, Eva-VLA, STRONG-VLA)에 대한 적대적 강건성 평가

## 12. 종합 평가

"섭동이 바꾸는 것은 플랜이 아니라 픽셀"이라는 한 문장의 관찰을, 앵커드 키포인트 위의 typed BT + 지각 self-consistency + 무-LLM 복구라는 일관된 시스템으로 밀어붙인 논문. 지각 소싱만 바꾸는 3구성 통제 실험, 같은 컴퓨트를 플랜에 썼을 때 무이득이라는 반사실 검증, post-condition 기반 실패 귀속과 실기 3플랫폼 검증까지 시스템 논문으로서의 완성도가 높다. goal-task 역효과(−8.4pt)를 숨기지 않고 메커니즘까지 분석한 점도 신뢰를 더한다. 다만 표준 LIBERO 수치가 없어 본 트래커의 주 리더보드에는 오르지 않으며(LIBERO-PRO 43.7%는 별도 섭동 프로토콜), 프론티어 API 의존과 SAM3 단일 실패점, 관측 피드백이 필요한 태스크의 구조적 열세가 명확한 한계. 그럼에도 training-free 시스템이 학습된 VLA를 섭동 하에서 2배 이상 앞서면서 그 VLA를 학습시킬 데이터까지 생산한다는 구도는, test-time compute 배분과 뉴로심볼릭 설계 논의에서 중요한 레퍼런스가 될 것이다.

<!-- VERIFIED: pdf -->
