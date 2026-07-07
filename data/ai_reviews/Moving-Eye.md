# The Moving Eye: Enhancing VLA Spatial Generalization via Hybrid Dynamic Data Collection

> **한 줄 요약**: VLA의 취약한 공간 일반화의 원인을 **shortcut learning**(Camera-Base / Camera-Object / Object-Position coupling)으로 규명하고, 이중팔 셋업(조작용 So-101 + 환경 카메라를 움직이는 Airbot "moving eye")으로 **Moving View와 Multi-Fixed View 데이터를 1:k로 혼합**하는 Hybrid Dynamic Data Collection 전략을 실기기에서 체계 검증. 순수 Moving 데이터는 수렴 실패(54.8%), 순수 Multi-Fixed는 80.5%, **Gr00t 기준 1:3 혼합("Golden Ratio")이 89.0%**로 최고. 효과는 ACT(+8.1), Diffusion(+26.8), Pi0(+13.8), Gr00t(+8.5) 등 **아키텍처 전반에 보편적**이며, 보조 pen 데이터로 학습한 공간 불변성이 미학습 태스크로 전이되어 Moving Test 43%→83%. (Lion Rock AI Lab, CMRIAT; HKUST; Nankai Univ. — IROS 2026)

---

## 1. 배경 및 동기

- VLA 모델은 의미 이해·조작 능력은 인상적이나, **카메라 포즈나 물체 배치의 미세한 섭동**만으로 성능이 붕괴하는 공간 일반화 취약성을 보임 (LIBERO-Plus, LIBERO-Pro, RADAR 등 로버스트니스 벤치마크가 지적).
- 저자들은 핵심 원인을 **shortcut learning**으로 규명: 모델이 태스크 관련 공간 관계 대신 카메라–로봇–물체 간의 **허위 상관(spurious correlations)**을 학습.
- 단순히 시점 수를 늘리는 것(Multi-Fixed)만으로는 불충분 — 물체 간 상대 위치가 고정되어 있으면 여전히 shortcut에 빠짐(Exp. 2에서 실증).
- 실전 배포 시나리오(고정 카메라의 기계적 진동, 서로 다른 리그에서 수집된 데이터셋 혼합, VR/AR·ego-centric·모바일 조작의 연속 시점 변화)는 모두 시점 강건성을 요구.

## 2. 문제 설정: 세 가지 암묵적 커플링

| 커플링 | 내용 | 검증 |
|---|---|---|
| **Camera-Base** | 배경 대비 로봇의 정적 외관을 암기 | Exp. 1 |
| **Camera-Object** | 고정 카메라 각도에 의존한 물체 인식 → 새 시점에서 실패 | 시점 다양화로 완화 |
| **Object-Position** | 물체 간 고정 상대 위치(예: 펜–홀더)에 과적합 | Exp. 2 |

카메라 구성은 세 가지로 분류 (Table I): **Fixed View**(전 에피소드 단일 정적 포즈, Point), **Multi-Fixed View**(에피소드 내 정적·에피소드 간 이산 변화, Bounded Region), **Moving View**(에피소드 내 연속 궤적, Bounded Region — "decoupling zone").

## 3. 방법론: Hybrid Dynamic Data Collection

문제 설정: 정책 π(a|o, l)은 손목 카메라 + 이동식 환경 카메라 관측과 언어 지시를 입력받음. 환경 카메라 포즈 P_c는 작업공간 W 내에서 제어됨. 전략은 세 요소로 구성:

1. **Hierarchical Viewpoint Sampling**: Fixed / Multi-Fixed / Moving 구성 전반에서 수집. Multi-Fixed 데이터는 수렴 안정성을, Moving 데이터는 공간 불변성을 강제하는 **regularizer** 역할.
2. **Multi-dimensional Diversity Injection**: 카메라 운동 외에 **물체–수용체 상대 위치를 명시적으로 랜덤화**하여 Object-Position coupling 차단. Camera-Base 디커플링은 시점 변화로 달성(로봇 베이스 자체는 이동하지 않음).
3. **Optimal Composition**: 혼합비 Moving:Multi-Fixed = 1:k, 즉 D_train = k/(k+1)·D_MultiFixed + 1/(k+1)·D_Moving (식 1). Gr00t n1 기준 **1:3이 최적("Golden Ratio")**이며, 최적 k는 아키텍처마다 다름(Exp. 4).

수집 알고리즘(Algorithm 1): 각 에피소드에서 r~U(0,1) 샘플, r < 1/(k+1)이면 Moving 모드(경계 내 연속 궤적 τ_c(t) 추종), 아니면 Multi-Fixed 모드(Uniform(W_bounded)에서 정적 포즈 샘플 후 유지); 매 에피소드 물체 배치 P_obj 랜덤화.

## 4. 실험 셋업

- **플랫폼**: So-101 로봇팔(손목 카메라, 조작) + Airbot 팔(환경 카메라 "eye" 제어). WowRobo 2MP USB 카메라, 640×480, 30 FPS.
- **태스크**: (i) Pen pick-and-place — 펜을 잡아 홀더에 삽입하는 contact-rich 정렬; (ii) 5종 multi-object(시계, 롤리팝, 손톱깎이, 테이프, 큐브) pick-and-place.
- **Moving 수집**: 카메라를 end-effector 공간에서 등속 0.05 m/s(평균 각속도 0.198 rad/s, 최대 0.419 rad/s)로 연속 이동; LeRobot SO-101 파이프라인으로 30 FPS MP4 연속 기록(이산 스냅샷 아님).
- **평가 프로토콜**: 고정된 40개 카메라 이동 궤적(Moving Test) + 40개 정적 포즈(Multi-Fixed Test) + 5개 target-container 상대 위치 + 8개 펜 방향(0°~315°). Pen 태스크 400 평가 에피소드, Multi-Task 100 에피소드(5종 × 20회).
- **학습**: Gr00t 모델 전부 8×H800, batch 4, lr 1e-4 cosine decay; pen 2400 에피소드 약 34–37시간, multi-task 약 14시간(펜 에피소드는 1–5회 파지로 프레임 ~2.5배).

## 5. Exp. 1: Camera-Base Coupling 검증

Fixed View 데이터로 학습 후 두 조건 평가 (Table II, 2400 samples):

| Method | ID-Test (Fixed) | OOD-Test (Moving) |
|---|---|---|
| Baseline (Fixed) | 85.0 | **43.0** |
| Ours (Mixed Data) | 86.0 | **83.0** |

Fixed 베이스라인은 ID 85%에서 움직이는 카메라에 **43%로 붕괴** — 자기중심 포즈를 오판해 pick/place 위치를 놓침. 혼합 데이터는 고정 시점 성능(86%)을 유지하면서 동적 시점에서도 83%로 강건.

## 6. Exp. 2: Object-Position Coupling 검증

카메라 포즈는 그룹 내 랜덤·홀더 위치는 고정인 Multi-Fixed 데이터로 학습한 진단 실험 (Table III):

| Test Condition | Baseline (Multi-Fixed) | Ours (Mixed 1:3) |
|---|---|---|
| ID (Fixed Holder) | 95.0 ± 3.5 | 91.9 ± 2.4 |
| OOD (Shifted Holder, 1 직경 이동) | **71.9 ± 5.2** | **90.6 ± 6.3** |

**핵심 발견**: 카메라 시점이 다양해도(Multi-Fixed) 물체 간 상대 위치가 고정이면 모델은 홀더의 절대 위치 또는 카메라/베이스와의 관계라는 shortcut을 학습 → 홀더를 한 직경만 옮겨도 95.0→71.9로 급락. 다양성 주입을 포함한 혼합 전략은 90.6%로 커플링을 성공적으로 차단.

## 7. Exp. 3: 혼합비와 Golden Ratio

Pen 태스크, Gr00t, Moving Test 기준 (Table IV):

| Moving:Multi-Fixed | 1:0 | 1:1 | 1:3 | 0:1 |
|---|---|---|---|---|
| 성공률 (%) | 54.8 ± 10.7 | 83.3 ± 7.1 | **89.0 ± 5.7** | 80.5 ± 6.1 |

1. **순수 Moving은 불충분(54.8%)**: 시각 입력의 높은 분산이 수렴을 방해 — 가설과 달리 동적 데이터만으로는 실패.
2. **순수 Multi-Fixed는 강한 베이스라인(80.5%)**: Moving Test에서조차 순수 Moving을 상회하는 의외의 결과.
3. **혼합이 최적(89.0%)**: Multi-Fixed가 수렴 안정성을, Moving이 공간 불변성 regularizer를 제공하는 상보 구조를 확인.

**Moving-View가 작동하는 이유(저자 해석)**: 조밀하고 준균일한 다중 시점 샘플링. 프레임률 저하는 데이터 감소처럼 완만한 성능 저하로 이어질 것이며, 큰 프레임 간 시점 점프(과도한 속도)가 발생하는 임의 고속 시점 운동은 범위 외로 규정.

## 8. Exp. 4-1: Cross-Task 전이와 샘플 효율

Multi-task 학습 데이터는 **순수 Fixed View**. 여기에 보조 Pen 데이터(1:3 Golden Ratio로 수집)를 50% 혼합 시 Moving Test 성공률 (Table V):

| Episodes | 600 | 1200 | 1800 | 2400 |
|---|---|---|---|---|
| Baseline (Fixed Data) | 18.0 ± 5.2 | 31.0 ± 11.5 | 28.0 ± 5.7 | 43.0 ± 16.1 |
| Ours (Mixed Data) | **51.0 ± 7.6** | **62.0 ± 12.0** | **71.0 ± 10.5** | **83.0 ± 6.0** |

**Skill decoupling 해석**: 모델은 "무엇을 잡을지"(semantics/affordance)는 Fixed-View multi-task 데이터에서, "어떻게 지각할지"(공간 표현/hand-eye coordination)는 Moving-View pen 데이터에서 학습하고, 혼합 학습 중 두 스킬을 합성 → 미학습 물체(시계 등)를 동적 시점에서 파지. 600 에피소드만으로 베이스라인 2400 에피소드(43%)를 상회(51%)하는 **샘플 효율**도 확보.

## 9. Exp. 4-2: 아키텍처 보편성

Pen 태스크에서 ACT, Diffusion Policy, Pi0(π0.5), Gr00t n1 비교 (Table VI, Moving Test):

| Model | Multi-Fixed | Best Mix | Ratio | Gain |
|---|---|---|---|---|
| ACT | 39.4 ± 6.3 | 47.5 ± 4.6 | 1:3 | +8.1 |
| Diffusion | 33.8 ± 21.8 | 60.6 ± 7.2 | 1:1 | **+26.8** |
| Pi0 | 45.0 ± 8.9 | 58.8 ± 12.5 | 1:11 | +13.8 |
| Gr00t (VLA) | 80.5 ± 6.1 | **89.0 ± 5.7** | 1:3 | +8.5 |

- 순수 Moving의 수렴 난항과 혼합의 이득은 **모든 아키텍처에서 일관** — shortcut learning 취약성과 공간 일반화 난제가 아키텍처 보편적 특성임을 시사.
- 최적 혼합비는 모델별로 상이(Diffusion 1:1, Pi0 1:11, ACT/Gr00t 1:3). ACT/Diffusion/Pi0는 기본 튜닝만 수행(모델 용량 비교가 아닌 데이터 전략의 보편성 입증이 목적).

## 10. 관련 연구와의 위치

- **3D/기하 명시 주입**: SPA, GeoAware-VLA, OG-VLA, RVT, ManiVID-3D — 아키텍처 개입으로 시점 불변 표현 학습.
- **카메라 정보 주입/증류**: FTM, 다중뷰 교사 증류(Acar et al.), Plücker embedding 카메라 조건화(Jiang et al.), Vantage.
- **데이터 중심 증강**: VISTA(diffusion novel-view synthesis), RoboSplat(3DGS 편집), Invariance Co-training, ADC(수집 중 실시간 섭동).
- **MOVE**: 시뮬레이션에서 motion 기반 수집을 제안했으나 실기기 배포는 미해결 — 본 논문은 이를 **실제 로봇에서 구현**하고 Object-Position 디커플링 차원을 추가.
- **Active perception과의 관계**: Vision-in-Action, ActiveUMI 등 "어디를 볼지"의 온라인 제어와 **직교·상보적** — 본 논문은 훈련 데이터 분포를 재구성해 어떤 viewing policy 하에서도 지속되는 shortcut 자체를 제거.

## 11. 한계 및 향후 과제

- 평가가 탁상 pick-and-place(5종 multi-object)와 contact-rich 정렬(펜 삽입)에 국한 — 전략 자체는 task-agnostic이나 장기 호라이즌·복잡 접촉 태스크로의 체계적 검증은 미완.
- 로봇 베이스 자체의 이동은 다루지 않음(Camera-Base 디커플링은 시점 변화로만 달성).
- 급격/고속 시점 운동(큰 프레임 간 시점 점프)은 명시적으로 범위 외.
- 폐색(occlusion) 하 관측성 개선은 부수 효과로 기대만 하고 별도 평가 없음.
- 향후: moving viewpoint를 VLA 내부의 온라인 모듈로 격상 — 데이터가 제공한 시점 불변성을 유지하면서 폐색을 해소하는 **policy-conditioned active viewpoint control**(학습된 next-best-view).

## 12. 결론 및 기여 요약

1. **Real-World Realization**: 시뮬레이션 패러다임(MOVE)과 물리 배포 간극을 메우는 동적 시점 데이터 수집 시스템의 체계적 실기기 검증.
2. **Shortcut Breaking**: 나이브 증강으로 해결 불가한 허위 상관(특히 Object-Position coupling)을 Hybrid Dynamic Data Collection이 효과적으로 완화함을 통제 실험으로 입증 (OOD 71.9→90.6).
3. **Transfer & Efficiency**: 보조 태스크의 디커플링된 동적 데이터가 미학습 태스크의 정적 정책을 강건화하며(43→83%), 소량 데이터로 고성능 일반화 달성.
4. **보편성**: ACT, Diffusion, Pi0, Gr00t 전 아키텍처가 혼합 전략으로 이득 — 공간 일반화는 "카메라를 움직이는 것"만이 아니라 **암묵적 커플링을 끊는 것**을 요구한다는 메시지.

**핵심 메시지**: Multi-Fixed 데이터는 안정적 수렴을, Moving View 데이터는 다양성(regularization)을 담당하며, 둘의 전략적 혼합(Gr00t 기준 1:3 Golden Ratio)이 shortcut learning을 깨는 가장 실용적인 데이터 중심 해법이다.

<!-- VERIFIED: pdf -->
