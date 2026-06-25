# OpenHLM: An Empirical Recipe for Whole-Body Humanoid Loco-Manipulation

> **한 줄 요약**: 휴머노이드 whole-body loco-manipulation을 위한 **VLA 레시피를 한 변수씩 통제 실험**으로 구축한 연구. **(I) joint-based whole-body teleop + 0.2 s preview** → **(II) π0.5 init + weight-surgery action projection + multi-step flow matching** → **(III) stationary teleop / HuMI co-training**의 3단계 로드맵을 따라 OpenHLM이 도출되며, Unitree G1에서 **HLM-12 8 학습 태스크 평균 task progress 89%**, **long-horizon 태스크에서 GR00T N1.6 (57.5%)·Ψ0 (48.8%)을 87.5%로 능가**하면서 데모 시간은 절반 이하(1.14 h vs 2.70 h)에 불과.

---

## 1. 배경 및 동기

### 문제 정의: Whole-Body Humanoid Loco-Manipulation
- 기존 대부분의 휴머노이드 시스템은 **상체와 하체를 분리된 컨트롤러로 처리** (IK로 팔, RL controller로 다리, navigation command + root-height로 stitching) → 휴머노이드를 사실상 **wheeled dual-arm platform**으로 격하시킴.
- 사람처럼 **페달을 발로 누르거나 낮은 선반에 닿기 위해 스쿼트**하는 동작 = 분리형 컨트롤러의 표현 가능 공간 밖.
- 해결책으로 등장한 **two-level hierarchy**: 고수준 VLA(언어+픽셀 → whole-body command) + 저수준 motion-tracking controller. 그러나 설계 공간은 거의 미탐색.

### 본 논문이 던지는 세 가지 질문
1. **컨트롤러/teleop 인터페이스**를 어떻게 설계해야 좋은 whole-body 데모를 모을 수 있는가?
2. **static / wheeled dual-arm용 VLA**를 휴머노이드의 full DoF에 적응시키려면 어떤 design choice가 실제로 중요한가?
3. Whole-body teleop은 비싸다 — **싼 데이터 소스**가 그 자리를 채울 수 있는가?

---

## 2. 방법론 심층 분석 — 3단계 통제 실험 로드맵

### 2.1 Phase I: Low-Level Controller & Teleoperation (§3.1)

**Two-level 구조**:
- 고수준 정책 (사람 operator 또는 학습된 VLA) → 10 Hz로 whole-body reference command 출력
- 저수준 controller (SONIC 기반 motion-tracking) → 50 Hz로 target joint position 출력, PD tracking

**Teleop 인터페이스 비교 (Table 1)** — Cola Placement / Shelf Cup Transfer / Bottle Disposal:

| 방법 | DoF 노출 | Cola Prog. | 보행 footsteps | Pedal 가능 |
|---|---|---|---|---|
| Decoupled control (GR00T variant) | 21-D | 66.7% | 42.3 | ✗ |
| VR 3-point (SONIC variant) | 24-D | 40.0% | 12.3 | ✗ |
| **Joint-based whole-body** (PICO + GMR) | **32-D** | **86.7%** | **12.0** | **✓** |

→ **Joint-based whole-body teleop이 3개 태스크 모두 수행 가능한 유일한 인터페이스**. Decoupled는 stuttering walking (3.5× 더 많은 걸음), VR 3-point는 indecisive stalling.

**Joint-space retargeting vs. SMPL recording (Fig. 3)**:
- Joint-based 88% vs SMPL-based 75% (4-task 평균).
- 원인: SMPL의 81-D는 32-D 대비 redundant — kinematic chain 제약이 있지만 VLA는 모든 차원을 학습해야 하므로 difficulty 증가.

**Future-frame preview latency 스윕 (Fig. 4)**:
- Δt ∈ {0, 0.2, 0.4, 0.6} s → **0.2 s가 best (67% progress, 35.2 s/demo)**.
- 0 s: stationary manip은 좋지만 보행에서 stuttering/ground-stomping.
- 0.6 s: operator delay 누적 → demo 41 s, progress 13%로 붕괴.

### 2.2 Phase II: Whole-Body VLA Policy Design (§3.2)

**Default 구성**: π0.5 백본 + weight-surgery action projection + pretrained bimanual ordering + absolute joint targets + proprioception input + 10-step flow matching.

**(A) Interface ablations (amber group, Fig. 5)** — 4-task subset에서 한 변수씩 default에서 변경:
- Random-init projection / humanoid-native ordering / relative action targets / no proprioception → 모두 **소폭의 progress 감소만** 발생 (단일 bottleneck 없음).
- 단, **proprioception 제거 + relative actions를 동시에** 적용 시 catastrophic failure (OOD drift) — design choice 간 상호작용 존재.

**(B) Pretraining ablation (rose group, Fig. 5)** — 가장 중요한 발견:
| Init | 4-task Avg Progress |
|---|---|
| π0.5 (non-humanoid robot 데이터로 사전학습) | **91%** |
| PaliGemma (동일 아키텍처, robot 데이터 없음) | 60% |
| Random init | 42% |

- **놀라운 점**: π0.5와 PaliGemma의 **validation action MSE는 거의 동일**한데 on-robot 성능은 크게 다름.
- π0.5의 "see error → correct → retry" closed-loop manipulation prior가 embodiment gap (dual-arm → humanoid)을 넘어 전이됨.
- **Action MSE는 robot pretraining의 가치를 평가하는 데 나쁜 proxy** — 같은 MSE의 두 모델이 실제로 매우 다르게 행동.
- Random init: 거친 stepping gait는 학습하지만 manipulation 능력이 거의 붕괴.

**(C) Faster action generation (sage group, Fig. 5)**:
- One-step flow matching (no retraining) / drifting model 모두 inference latency 90 → 60 ms로 줄지만 progress가 ~20점 하락.
- 두 one-step 대안 모두 **validation MSE는 더 낮음 (~0.007 vs ~0.009)** — 또 한 번 MSE-progress 디커플링.
- 가설: ℓ2는 가깝지만 시간적으로 jittery한 action → robot에서 destabilizing. → **multi-step (10) flow matching 유지**.

**(D) Data scaling (Fig. 6)**: 5 → 10 → 20 → 40 demos per task: 52% → 60% → 85% → 91%. **10→20 구간이 가장 큰 점프**, 40에서 평탄화. 40 demos = skilled operator 약 1.5 시간 / medium-task.

→ 8-task 전체 학습 시 **평균 89% progress** (Fig. 7, 8-task baseline 막대).

### 2.3 Phase III: Heterogeneous Co-Training (§3.3)

**Whole-body teleop never covers**되는 4개 held-out 태스크 (Pig Placement, Gum Can Placement, Shelf Cube Transfer, Pouring) 기준 측정.

**(a) Stationary same-embodiment teleop** (feet planted, manipulation only):
- 평균 demo 13 min vs full teleop 21 min (overheating shutdown / scene reset 제외하면 격차 더 큼).
- 결과: 8-task 평균은 regress 없음, **4 held-out 평균 progress 33% → 87%** (oracle 94%에 근접).
- Pouring (새로운 vessel-tilt motion 필요)도 oracle 수준 달성 → **새로운 motion + 새로운 semantic 모두 공급**.

**(b) HuMI co-training** (handheld UMI grippers + body trackers + IK retargeting, robot-free):
- 7 min/40 demos (stationary 13 min의 절반).
- Tasks 9–11 (motion-reuse, 새로운 object/prompt)에서 stationary와 동등.
- Pouring (새로운 motion 필요)에서 실패 → **새로운 semantic은 공급하지만 새로운 motion은 공급 못함**.
- 원인: 시각 도메인 갭 (RealSense vs rectified GoPro fisheye, parallel gripper vs rigid handheld) + action 도메인 갭 (IK 후에도 human motion 잔존).

---

## 3. 데이터 / 시스템

| 항목 | 값 |
|---|---|
| Robot | Unitree G1 humanoid |
| 카메라 | head- + wrist-mounted RGB |
| Low-level controller | SONIC general motion tracker (50 Hz, PD) |
| High-level VLA | 10 Hz inference, RTX 5080, ~90 ms/call |
| Teleop hardware | PICO 4 Ultra VR + body trackers + GMR retargeting |
| HuMI rig | 2 handheld UMI grippers + 3 body-pose trackers (pelvis + 양 발) |
| HLM-12 데모 | 4 capability family × 3 tasks = 12 tasks, 40 demos/task |
| Long-horizon demo | 6 demos × 20 ordered fruit pairs (5 fruits 중 2 선택) |

**HLM-12의 4 capability families**:
1. Pick-and-place with locomotion (e.g., Cola Placement) — 분리형 컨트롤러도 원칙적으로 가능.
2. Whole-body workspace extension (e.g., Shelf Cube Transfer) — hip flexion + knee bend + torso pitch 협응.
3. Using body parts as manipulators (e.g., Bottle Disposal — 발로 페달 누름) — 분리형 컨트롤러의 표현 공간 밖.
4. Loco-manipulation under constraint (e.g., Cart Pushing) — 환경/접촉 제약.

---

## 4. 실험 결과 — HLM-12

### 4-task subset 핵심 ablation 요약
- Teleop 방법: joint-based whole-body 채택.
- SMPL recording: 거부 (88% vs 75%).
- Preview latency: 0.2 s 채택.
- VLA init: π0.5 (91%) ≫ PaliGemma (60%) ≫ scratch (42%).
- Action 생성: multi-step flow matching 유지 (one-step 대안 -20점).
- Data budget: 40 demos/task.

### 8-task 전체 학습 (Fig. 7)
- Whole-body teleop baseline: **평균 89% progress**.
- 디자인 선택이 더 넓은 distribution에도 generalize됨을 확인.

### 4 held-out 태스크
| 조건 | Tasks 9–11 평균 | Tasks 9–12 평균 |
|---|---|---|
| 8-task baseline (no co-training) | — | 33% |
| Stationary co-training | ~89% | **87%** |
| HuMI co-training | ~89% | 80% |
| 12-task oracle | — | 94% |

---

## 5. System-Level 비교 — Long-Horizon Task (§4)

**Task**: home pose → low coffee table에서 {fruit 1} (오른손) → mid table에서 {fruit 2} (왼손) → tall shelf의 상이한 container에 배치. 5개 fruit 중 2개 ordered pair = 20조합.

| Method | Progress (%) | Demo Duration |
|---|---|---|
| Ψ0 | 48.8 ± 4.4 | 2.70 h |
| GR00T N1.6 | 57.5 ± 4.6 | 2.70 h |
| **OpenHLM (HuMI co-training)** | **87.5 ± 3.7** | **1.14 h** |
| OpenHLM (teleop oracle) | 97.5 ± 1.7 | 2.73 h |

- OpenHLM은 14개 lemon/tomato pair를 HuMI로만 커버 (whole-body teleop은 banana/peach/mangosteen 6 pair = 36 demos만).
- 두 baseline 모두 **stereotyped arm trajectory** — 언어 지정 과일 추적 실패, weak grasping.
- Ψ0는 tall shelf 앞에서 stops short — 보행 능력도 부족.
- **GR00T N1.6 / Ψ0 모두 사전학습에 Unitree G1 데이터 포함, π0.5(OpenHLM)는 미포함** — "휴머노이드 데이터를 pretraining에 섞는다"가 만능 해법이 아니라 design detail이 결정적임을 입증.

---

## 6. 주요 기여

1. **Systematic empirical study** — 세 phase에 걸친 controlled ablations로 whole-body 휴머노이드 VLA의 design space를 체계적으로 매핑.
2. **OpenHLM 레시피** — joint-based whole-body teleop + π0.5-init humanoid-adapted VLA + heterogeneous co-training. 코드/데이터/체크포인트 전부 공개 예정.
3. **Heterogeneous co-training for data efficiency** — stationary teleop은 motion + semantic 모두, HuMI는 semantic만을 절반 비용으로 공급한다는 정량화.

---

## 7. 핵심 인사이트

### (A) Teleop 인터페이스 = data-collection의 표현 공간을 결정
- 분리형 / VR 3-point는 **태스크 자체를 표현 불가**하게 만듦 (footpedal, squat).
- Joint-based whole-body teleop은 32-D로 full DoF 노출, 가장 자연스러운 보행 (footsteps 12 vs 42).

### (B) Non-humanoid robot pretraining → humanoid VLA로의 전이
- π0.5 (dual-arm 데이터)가 PaliGemma (robot 데이터 없음)보다 평균 31점 우위.
- **MSE가 같아도 on-robot 행동이 크게 다름** — manipulation prior (특히 retry 행동)가 embodiment gap을 넘음.

### (C) MSE-Performance 디커플링이 두 번 등장
1. π0.5 vs PaliGemma (같은 MSE, 다른 progress).
2. Multi-step vs one-step flow matching (one-step이 MSE는 더 낮지만 progress는 -20점).
   → Validation MSE는 robot 정책 평가에 부적합한 metric.

### (D) Heterogeneous data의 정체성
- Stationary same-embodiment = 새 motion + 새 semantic 모두.
- HuMI (robot-free) = 새 semantic만 (현 데이터 규모에서). 비용은 stationary의 절반.
- Scale up시 HuMI도 motion 전이 가능성 있음 → future work.

---

## 8. 한계 및 비판

1. **단일 플랫폼**: Unitree G1만 사용 — 다른 휴머노이드로의 generalization 미검증.
2. **HuMI motion 갭**: 현재 data budget에서 새로운 motion 전이 실패. 시각/액션 도메인 갭의 정확한 메커니즘 미규명.
3. **메커니즘 미해명**: action MSE ↔ on-robot 디커플링과 one-step generation 저하의 정확한 원인 불명 — future work로 명시.
4. **VLA 아키텍처 자체는 미탐색**: π0.5 내부 구조는 untouched, loco-manipulation 전용 아키텍처 설계 미수행.
5. **평가 표본**: per (policy, task) 5 rollout — standard error는 보고하지만 표본이 작음 (long-horizon은 pair당 1 rollout만).

---

## 9. 후속 연구 방향

- HuMI 데이터 **scaling**으로 motion 전이까지 가능한지 검증.
- AR/VR 헤드셋 / 스마트 글래스 등 다른 cheap data source 통합.
- Loco-manipulation 전용 VLA 아키텍처 설계 (action chunk horizon, hierarchical action head 등).
- Validation MSE를 대체하는 on-robot proxy metric 개발.
- 다양한 휴머노이드 플랫폼·다양한 scene으로 일반화 검증.

---

## 10. 다른 연구와의 관계

| 연구 | 차이점 |
|---|---|
| GR00T N1.6 [7] | Cosmos-2B + DiT action head, decoupled WBC, Unitree G1 pretraining 포함. OpenHLM에 long-horizon에서 30점 열세 — humanoid pretraining만으로는 부족. |
| Ψ0 [8] | Egocentric human video pretrain + humanoid mid-train. OpenHLM에 long-horizon에서 39점 열세, 보행에서도 약함. |
| TWIST2 [9] | Whole-body teleop 데이터 수집은 잘하지만 single-task visuomotor IL과 결합 (multi-task VLA 아님). |
| SONIC [13] | Motion-tracking controller scale-up. OpenHLM이 SONIC을 low-level controller로 채택 + VLA 통합 평가. |
| π0.5 [22] | Static/wheeled dual-arm용. OpenHLM이 humanoid에 적응한 첫 사례 중 하나로 검증. |
| HuMI [1] | Robot-free 데이터 수집 시스템. OpenHLM이 이를 co-training 소스로 통합. |
| EgoHumanoid [17] | Human + teleop pairing으로 environment generalization. OpenHLM은 cheap data로 새 object/instruction 확장 — orthogonal. |
| HOIST | Foundation VLA + 고정 WBC + 작은 latent-space RL module로 contact-rich task 풀이. OpenHLM은 RL 없이 imitation + co-training만으로 daily-life loco-manipulation 커버. |

---

## 11. Action Head Category 판정 근거

- **action_head**: π0.5의 **multi-step (10-step) flow matching** action head를 그대로 사용. Inference 시 학습된 vector field를 통합.
- **action_head_category**: `flow_matching`.
- 일관성: π0.5 [22] / HOIST 등 동일 카테고리. Discrete tokenization / autoregressive head / pure diffusion DDPM 아님.

---

## 12. 결론

OpenHLM은 *"휴머노이드 VLA를 어떻게 만들 것인가"* 라는 질문에 **scale도 fancy architecture도 아닌, 통제된 실험으로 design detail을 찾자**고 답한다. 세 phase에 걸친 one-variable-at-a-time ablation은 다음과 같은 구체적 레시피로 수렴한다.

- **데이터 측**: joint-based whole-body teleop + 0.2 s preview, SMPL recording은 피하고 joint-space로 online retargeting.
- **정책 측**: π0.5 init + weight-surgery action projection + bimanual ordering + absolute targets + proprioception + multi-step flow matching.
- **확장 측**: stationary teleop은 새 motion + 새 semantic, HuMI는 새 semantic을 절반 비용으로.

결과적으로 동일한 (혹은 절반의) 데모 예산으로 GR00T N1.6 / Ψ0를 long-horizon 태스크에서 큰 격차로 능가하며, "휴머노이드 데이터를 pretraining에 섞는다"는 흔한 접근이 design detail을 대체할 수 없음을 보였다. 또한 두 번에 걸쳐 등장한 **action MSE ↔ on-robot 성능 디커플링**은 향후 휴머노이드 VLA 평가 방법론 자체를 다시 보게 만드는 흥미로운 관찰이다. 코드/데이터/체크포인트 공개와 함께, 본 로드맵은 후속 휴머노이드 loco-manipulation 연구의 출발점 역할을 할 것이다.

---

<!-- VERIFIED: pdf -->
