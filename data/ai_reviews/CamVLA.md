# CamVLA — From Fixed to Free Cameras: Calibration-Free View-Robust Vision-Language-Action Model

> **한 줄 요약**: 기존 view-robust VLA들이 배포 시 정확한 camera extrinsics를 알아야 한다는 "deployment-fragile" 가정에 의존하는 문제를 지적하고, 정책이 **스스로 카메라 위치를 추론**하도록 설계한 calibration-free / depth-free / single-view VLA 프레임워크를 제안. (1) 카메라 좌표계에서 camera-centric delta action을 예측하는 Action Head와 (2) monocular RGB에서 6-DoF hand-eye matrix를 회귀하는 auxiliary Geometric Head를 병렬로 두고, 결정론적 기하 변환으로 base-frame action을 합성. RLBench unseen viewpoint에서 π0 33.2→**51.4%** (+18.2%p), GR00T N1.7 28.4→**38.4%** (+10.0%p), 실제 로봇 15° 카메라 오프셋에서 π0 16.0→29.3%, GR00T 14.7→33.0% 달성. 오버헤드는 +6.30M 파라미터(0.19%), +1 ms(62 vs 61 ms)에 불과.

---

## 1. 배경 및 동기

### The Viewpoint Trap
- π0를 단일 canonical view로 학습하면 RLBench 학습 뷰에서 ~65.3% 성공률이지만, 카메라를 단 **15° 회전시키면 6.3%로 붕괴** (5°에서 23.0%, 10°에서 9.3%; Fig. 1). 장면이 완전히 관측 가능하고 semantic goal이 동일해도 실패가 지속됨.
- 실제 배포에서는 센서가 부딪히고, 다른 플랫폼에 장착되고, 사람이 손에 들고, 모바일 베이스에서 pose가 계속 drift함 → 학습 시의 통제된 카메라 셋업이 유지되지 않음.

### 구조적 원인: 숨겨진 hand-eye 변환
- 표준 VLA는 camera-perspective 시각 관측으로부터 **robot base frame action**을 예측 → 입력 프레임과 출력 프레임의 불일치. 이 간극을 잇는 hand-eye 변환 T_t가 입력에 없으므로 네트워크 가중치에 **암묵적으로 entangle**되고, 배포 시 T_t가 학습 분포에서 벗어나면 매핑이 붕괴.
- 기존 해법들(OC-VLA의 calibrated camera 좌표 재표현, Jiang et al.의 ray embedding, 4D-VLA의 intrinsics/extrinsics 기반 back-projection, AnyCamVLA의 canonical view 합성)은 모두 **배포 시 정확한 extrinsics를 알아야 함** — hand-held/drifting/remounted 카메라라는, view robustness가 가장 필요한 바로 그 상황에서 깨지는 가정.

### 핵심 주장
> "정책에게 카메라가 어디 있는지 알려주지 말고, 스스로 알아내게 하라." 인간의 vision-guided manipulation이 egocentric frame에서 작동하면서 allocentric 시스템이 head pose를 암묵적으로 유지하는 생물학적 factorization에서 영감.

📌 [Figure 1 삽입] — 카메라 각도 오프셋에 따른 π0 성공률 붕괴 곡선

---

## 2. 방법론 개요: 두 하위 문제로의 분해

CamVLA는 정책을 monocular RGB 한 장에서 추론 가능한 두 하위 문제로 분해한다 (Fig. 2):

1. **Camera-Centric Action Generation** ("어떻게 움직여야 하는가?"): end-effector action을 **로컬 카메라 좌표계**에서 natively 예측 → 시각 관측과 정렬되어 외부 카메라 pose와 무관.
2. **Camera-Perspective Geometric Grounding** ("나는 어디서 보고 있는가?"): 카메라-로봇 base 간 **6-DoF hand-eye matrix**를 회귀 → viewpoint 의존 변동을 단일 relative pose에 격리.

두 출력은 결정론적 기하 변환으로 base-frame action에 합성된다. 배포 시 calibrated extrinsics, depth 센서, view synthesis가 전혀 불필요 — 표준 VLA와 동일한 monocular RGB + language command만 사용.

📌 [Figure 2 삽입] — CamVLA 아키텍처 개요 (VLM backbone → Action Head + Geometric Head → Geometric Transformation)

---

## 3. Camera-Centric Action Generation

- 표준 VLA 공식화: 관측 I_t, proprioceptive state s_t = [p_b,t, r_b,t], 언어 목표 L로부터 base-frame delta action ΔA_b,t = [Δp_b,t, Δr_b,t, g_t] 예측 (axis-angle 회전, gripper state g_t ∈ [0,1]).
- 문제: f_θ가 ΔA_b,t를 출력하도록 학습되면 어려운 **cross-frame mapping**을 배워야 하며, multi-view 데이터로 학습해도 상충하는 visual flow들을 동일한 base-frame action에 매핑해야 해서 geometric entanglement 발생.
- 해법: 시각 관측과 action이 모두 카메라 프레임에서 표현되는 **same-frame 형태** 학습. Camera-centric delta action ΔA_c,t = [Δp_c,t, Δr_c,t, g_t] (Eq. 1). 이미지에서 왼쪽으로의 visual translation이 항상 카메라 로컬 X축 음의 변위에 대응하는 일관된 공간 관계 → visual-action confusion 방지.

---

## 4. Camera-Perspective Geometric Grounding과 결정론적 변환

### Geometric Head
- 로봇 팔은 base frame에서 정의된 kinematics로 작동하므로 ΔA_c,t를 그대로 컨트롤러에 보낼 수 없음. Auxiliary Geometric Head가 시각 특징으로부터 T_t ∈ SE(3)를 회귀 — translation τ_t ∈ R³ + axis-angle ω_t ∈ R³로 파라미터화, (ΔA_c,t, T_t) = f_θ(I_t, s_t, L) (Eq. 2).
- 구현: **3-layer MLP (GELU, hidden dim 1024)**, 백본 image encoder의 visual token을 mean pooling한 특징 위에서 작동.

### 결정론적 기하 변환 (핵심 수학적 성질)
- Δp_c,t와 Δr_c,t는 free vector이므로 예측된 회전 R_t ∈ SO(3)에 대해 선형 변환: Δp_b,t = R_t Δp_c,t, Δr_b,t = R_t Δr_c,t (Eq. 3-4).
- **translation τ_t는 delta action 실행에서 정확히 소거됨** (Supp. B에서 유도: 상대 translation에서 τ_t 상쇄, 상대 회전은 rotation conjugation 하 matrix logarithm의 equivariance로 Δr_b,t = R_t Δr_c,t). → 테스트 시 τ_t drift는 물리적 영향이 **0**이고, viewpoint 오차는 R_t에만 국한. τ_t 회귀는 시각 백본의 geometric grounding 강화 및 absolute-action 변형 지원 목적으로 유지.

### 학습
- Joint objective **L = L_act + λ L_ext (λ = 0.1)**. L_act는 베이스 VLA의 action loss 상속(π0의 flow matching 등), L_ext는 translation/axis-angle에 대한 MSE. GT hand-eye matrix는 **학습 supervision으로만** 사용, 배포 시 extrinsic 정보 없음.

---

## 5. 실험 설정

### 시뮬레이션 (RLBench)
- Front camera를 로봇 base 중심으로 −90°~90°, 5° 간격 회전. **학습은 15° 간격 (0°, ±15°, ..., ±90°)**, 평가는 나머지 unseen viewpoint에서 zero-shot.
- 6개 태스크: slide block, push buttons, take umbrella, close laptop lid, lamp off, put knife. 태스크·뷰포인트당 100 expert demo (CoppeliaSim + OMPL), 평가 50 에피소드. 제어 20 Hz, 추론당 예측 궤적의 첫 5 스텝 실행.

### 실제 로봇
- **Franka Research 3** + parallel gripper, calibrated Intel RealSense D435i. **5개 학습 카메라 뷰**에서 demo 수집(뷰당 100개), 3개 테스트 카메라(Cam 2/3/4)를 0°/5°/10°/15° 회전시켜 평가 (태스크·오프셋·카메라당 20 에피소드).
- 5개 가사 태스크: basket upright, pick & place banana, push cabbage, wipe table, pick & place cup. 수집 30 Hz → 학습/배포 10 Hz, 추론당 20 스텝 실행.

### 공통 구현
- 8× NVIDIA H100 80GB, multi-task 학습. 입력은 **단일 third-person monocular RGB 224×224** — wrist 카메라는 의도적으로 제외(wrist view로 viewpoint 변화를 우회하는 것 방지). Action은 delta 6-DoF EE pose (axis-angle).

---

## 6. 시뮬레이션 결과 (Table 1)

Zero-shot unseen viewpoint 평균 성공률 (%):

| Model | Slide Block | Push Buttons | Take Umbrella | Close Laptop | Lamp Off | Put Knife | **Mean** |
|---|---|---|---|---|---|---|---|
| π0 | 18.3 | 51.5 | 32.3 | 57.0 | 29.8 | 10.0 | 33.2 |
| **π0 + CamVLA** | 44.5 | 72.3 | 39.2 | 69.0 | 58.0 | 25.3 | **51.4** |
| GR00T N1.7 | 27.5 | 13.5 | 41.8 | 47.7 | 28.2 | 11.5 | 28.4 |
| **GR00T N1.7 + CamVLA** | 44.7 | 30.5 | 50.3 | 56.0 | 35.0 | 14.0 | **38.4** |

- π0 기준 **+18.2%p 절대 향상** (33.2→51.4). 베이스라인이 심하게 붕괴하는 Slide Block(18.3→44.5), Lamp Off(29.8→58.0)에서 특히 큰 개선.
- GR00T N1.7에서도 **+10.0%p** (28.4→38.4) — 서로 다른 두 아키텍처에서 일관된 이득으로 프레임워크의 범용성 입증.

---

## 7. 실제 로봇 결과 (Table 2-4)

### Repositioned camera 일반화 (Table 2, 3개 카메라 평균)
| Offset | π0 | π0+CamVLA | GR00T N1.7 | GR00T+CamVLA |
|---|---|---|---|---|
| 0° (canonical) | 63.3 | **79.0** | 64.7 | **80.7** |
| 5° | 53.3 | **68.0** | 52.0 | **72.3** |
| 10° | 39.3 | **55.3** | 35.7 | **53.0** |
| 15° | 16.0 | **29.3** | 14.7 | **33.0** |

- Canonical view(0°)에서도 CamVLA가 크게 우세 — camera-centric 표현 자체가 학습을 돕는다는 신호. 15° 극단 오프셋에서 베이스라인이 붕괴(16.0/14.7%)해도 CamVLA는 29.3/33.0% 유지.

### Hand-eye 추정 오차 (Table 3, GT calibration 대비)
- Translation: 1.35 / 2.12 / 7.91 / 27.16 cm, Rotation: 2.49 / 4.73 / 5.98 / 9.39° (0°/5°/10°/15°). 15°에서 translation 오차가 커도 **relative-action 공식화가 translation 오차를 물리적으로 격리**하고 회전 오차(<10°)는 closed-loop 정책의 허용 범위 내.

### 계산 효율 (Table 4, RTX 4090)
- π0: 3238.1M params, 660.9 GFLOPs, 61 ms ↔ CamVLA: 3244.4M (+6.30M, **0.19%**), 661.9 G (+0.15%), **62 ms (+1 ms)**. 10 Hz에서 20-step 궤적 실행이므로 62 ms 추론은 로봇 동작과 병렬 실행 가능 — 실시간 배포에 병목 없음.

---

## 8. Ablation: 학습 뷰포인트 밀도와 구간별 분석 (Table 5-6)

### 학습 뷰포인트 밀도 (Table 5, π0 베이스)
| 학습 간격 | π0 | CamVLA | CamVLA† (GT extrinsics) | Trans. 오차 | Rot. 오차 |
|---|---|---|---|---|---|
| 15° | 33.2 | 51.4 | 52.3 | 4.69 cm | 1.41° |
| 30° | 25.5 | 34.0 | 40.0 | 19.71 cm | 4.77° |
| 45° | 16.8 | 21.2 | 26.3 | 34.83 cm | 8.28° |

- 15° 간격에서 self-predicted와 GT의 격차가 **0.9%p** (51.4 vs 52.3) — 자체 예측 extrinsics의 높은 정확도(회전 1.41°) 확인. 극단적으로 sparse한 45°에서도 GT 대비 근접(21.2 vs 26.3)하며 베이스라인 대비 +4.4%p.
- GT extrinsics를 쓴 CamVLA†도 유사하게 저하됨 → sparse 학습에서의 성능 하락은 hand-eye 회귀 오차가 아니라 **unseen 시점의 시각 표현 shift가 주 원인**.

### 뷰포인트 구간별 relative drop (Table 6)
- 인접 학습 뷰 대비 unseen 뷰의 상대 하락: CamVLA 평균 **4.1%** vs π0 8.9%. [30°, 45°] 구간에서 π0가 20.8% 하락할 때 CamVLA는 **0.9%**로 거의 완벽한 view invariance.

---

## 9. Ablation: 표현 선택 (Table 7-9, Fig. 5)

### State/Action 공간 (Table 7)
- Action을 base→camera frame으로만 바꿔도 33.2→51.4%. GT vs self-predicted extrinsics 격차는 0.9%p(52.3 vs 51.4), camera-frame state 사용 시 0.2~0.3%p 차이. **Base-frame state + camera-frame action + self-predicted hand-eye** 조합(51.4%)만이 유일하게 완전한 calibration-free이며, 0.3%p의 비용으로 state 관측 단계의 calibration 필요성까지 제거.

### Hand-eye 표현 (Table 8)
- Rotation-only 회귀: 51.3% / 1.6° — full 6-DoF(51.4% / 1.4°)와 사실상 동일. Translation 예측이 성능을 해치지 않으면서 absolute-action 변형 지원 등 확장성 제공.

### 시각 특징 소스 (Table 9)
- **Image Encoder 특징 (default)**: 51.4%, 4.7 cm / 1.4° — 가장 정밀한 기하 추정. VLM Backbone 특징은 성공률이 소폭 높지만(53.5%) 기하 추정이 크게 나빠지고(14.7 cm / 3.7°) unseen 각도에서 geometric discontinuity/spike 발생 → 실기 안전성 때문에 Image Encoder 선택. Gradient detach는 일관되게 해로움(Image Encoder detach 42.6%, VLM detach 20.9% + 45.0 cm/36.0° 오차) → Geometric Head는 end-to-end feature adaptation이 필수.

### Extrinsic 회전 노이즈 강건성 (Fig. 5)
- GT hand-eye 회전에 매 planning step마다 랜덤 회전 노이즈 주입: 무노이즈 64.0% → 1°에서 63.3%, 5°에서 58.7%. **12° 노이즈까지도 무노이즈 π0(36.0%)를 상회**. Geometric Head의 실제 평균 회전 오차가 1.41°이므로 사실상 성능 저하가 없는 고정밀 영역에서 작동 — calibration-free 프레임워크의 타당성 검증.

---

## 10. 관련 연구와의 차별점

- **3D 구조 도입 계열** (Perceiver-Actor, 3D-VLA, Lift3D, 3D Diffuser Actor 등): depth/point cloud/multi-view/calibrated geometry 의존 + 계산 오버헤드.
- **View synthesis/augmentation, view-invariant 표현 계열**: 이미지 레벨 우회로, 기하를 명시적으로 다루지 않음.
- **Camera-aware conditioning 계열** (OC-VLA, camera-conditioned ray embedding, 4D-VLA, AnyCamVLA): 모두 배포 시 known & accurate intrinsics/extrinsics 필요. CamVLA는 hand-eye matrix를 **monocular RGB에서 self-predict**하여 외부 calibration/depth/view synthesis/3D 재구성 없이 view-robust manipulation 달성 — "given geometry를 learned geometry로 대체".
- 프레임워크는 온라인 hand-eye calibration 시스템과도 호환(예측 extrinsics를 대체해 성능 추가 향상 가능).

---

## 11. 한계

- **Wrist 카메라 미고려**: 단일 third-person 카메라만 사용하며, multi-camera 시스템에서 wrist-mounted 카메라의 viewpoint perturbation은 다루지 않음.
- **극단적 시점 변화·고정밀 태스크에서 취약**: out-of-distribution 시각 특징과 hand-eye 회귀 오차로 인해 15°를 넘는 실세계 오프셋에서는 베이스라인과 함께 성능이 급락 (실험도 15°까지만 수행).
- 실패 모드(Fig. 15): FOV 경계의 물체, workspace 초과 action, self-occlusion.
- 코드 미공개 (프로젝트 페이지만 존재), venue 미정 arXiv preprint.

---

## 12. 종합 평가

CamVLA는 view-robust VLA 연구의 공통 가정 — "배포 시 카메라 extrinsics를 안다" — 이 정작 view robustness가 필요한 배포 상황에서 깨진다는 날카로운 문제 인식에서 출발해, **camera-centric action + self-predicted hand-eye + 결정론적 합성**이라는 깔끔한 factorization으로 해결한 연구다. Delta action이 hand-eye translation과 수학적으로 독립이라는 성질(Supp. B)을 활용해 viewpoint 오차를 회전 성분에만 격리한 설계가 특히 우아하며, 12° 회전 노이즈까지 견디는 closed-loop 허용 범위 분석이 calibration-free 주장을 정량적으로 뒷받침한다. π0와 GR00T N1.7 두 아키텍처, 시뮬레이션과 실기 모두에서 일관된 대폭 개선을 0.19% 파라미터/1 ms 오버헤드로 달성했고, ablation이 밀도·표현·특징 소스·노이즈 축을 빠짐없이 커버한다. 다만 RLBench 수치는 표준 프로토콜이 아닌 자체 viewpoint-robustness 프로토콜(6개 태스크, unseen view 평균)임에 유의해야 하며, wrist 카메라 perturbation과 15° 초과 시점 변화는 미해결이다. Action head는 베이스 VLA의 action expert(π0 인스턴스 기준 flow matching)가 camera-centric action을 생성하므로 **flow_matching**으로 분류된다.

**Score: 8.0 / 10**

<!-- VERIFIED: pdf -->
