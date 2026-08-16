# Ego2Robot: Scalable Robot Data Synthesis from Egocentric Human Data

> **한 줄 요약**: egocentric 인간 손 영상을 **retarget + robot-arm 렌더링 + 3단 품질 큐레이션**으로 로봇 학습 데이터로 변환하는 파이프라인. ~1,940h ego 영상 → **15개 morphology × 18,561h** 합성 데이터(~9.6배 증폭, 역대 최대 ego-to-robot 데이터셋). 이 데이터로 Qwen3.5-4B + DiT(flow matching) VLA를 직접 사전학습하고, RoboTwin2.0을 **11개 독립 perturbation 축**으로 분해한 벤치마크에서 robot-only 대비 Clean 62.2→68.1, Randomized 50.9→53.5, EBench 39.6→51.7을 보고.

- arXiv: 2608.02580v1 (2026-08-03)
- 소속: AIM3 Lab (Renmin University of China), Qwen Team (Alibaba Inc.), ShanghaiTech, BIGAI, Beihang
- Project: https://www-ye.github.io/ego2robot_blog/

---

## 1. 배경 및 동기

- VLA의 일반화 능력은 결국 **로봇 데이터의 규모와 다양성**에 종속. Open X-Embodiment, DROID, AgibotWorld 같은 대규모 시도에도 teleoperation 비용·하드웨어 가용성·상호작용 다양성 제약은 그대로.
- Egocentric 인간 영상은 대안. 객체·환경·태스크 다양성이 압도적이고 수집 비용이 낮음.
- 그러나 human ↔ robot embodiment gap이 큼. 기존 retarget-and-render 계열(Phantom, EgoMimic)은 **소규모 / per-task 정책**에서만 효과를 보였음.
- 본 논문의 질문: **"ego2robot 합성 데이터가 대규모 VLA *사전학습* 데이터로서 OOD 일반화를 개선하는가?"** — 이 질문이 미탐구 영역이라는 점이 논문의 출발점.

---

## 2. 선행 연구와의 차별점

| 계열 | 대표 | 한계 / Ego2Robot의 차별 |
|---|---|---|
| 인간 영상 시각 사전학습 | R3M, MVP, VIP | representation만 제공, 여전히 robot 데이터로 embodiment gap을 메워야 함 |
| Human video co-training | EgoMimic, Being-H0, ViTRA, EgoScale | 영상을 그대로 섞음. embodiment masking 수준의 정렬 |
| Retarget-and-render | Phantom [14], EasyMimic [30] | **소규모 / 개별 태스크 한정**. Ego2Robot은 18,561h × 15 morphology로 스케일 |
| Robot-to-robot 증강 | RoviAug, Mirage, OXE-AugE | robot→robot의 **좁은** 시각 gap. Ego2Robot은 훨씬 큰 ego→robot gap을 다룸 |
| 분해형 일반화 벤치마크 | Colosseum, LIBERO-Plus, LIBERO-PRO | **single-arm 한정**. RoboTwin2.0/EBench는 dual-arm이지만 bundled 결과만 보고 |

즉 (a) retarget-and-render의 **스케일링**, (b) dual-arm + cross-embodiment를 동시에 지원하는 **분해형 평가 프로토콜**, 두 축이 기여.

---

## 3. 파이프라인 (1) — Action Alignment

21개 hand keypoint → parallel-gripper EEF 궤적.

**Hand-to-Gripper Retargeting**
- 가상 fingertip: `p_vf = 0.7·p_index + 0.3·p_middle`
- TCP: `p_tcp = (p_thumb + p_vf)/2`, 개폐 폭: `w = ||p_thumb − p_vf||`
- 우수 직교 프레임 `R = [x y z]`:
  - `z = s(p_thumb − p_vf)/w` (grasp axis, jaw line)
  - `d = p_vf − p_wrist`, `y = (z×d)/||z×d||` (gripper normal)
  - `x = y×z` (approach axis)
  - `s = +1` (오른손) / `−1` (왼손) → **양손이 동일 gripper frame으로 수렴**

**Temporal Smoothing**: 위치·폭에 Savitzky-Golay, 회전에 Gaussian-weighted SLERP.

**Action Speed Alignment**: 인간 손 동작이 teleoperation보다 훨씬 빠름 → 소스별 프레임 서브샘플링. ANT/EgoDex 60%(~1.7× 느리게), EgoVerse 45%(~2.2×), ViTRA 25%(~4×).

---

## 4. 파이프라인 (2) — Visual Alignment

인간 팔을 지우고 로봇 팔을 같은 장면에 합성.

1. **Arm Segmentation**: SAM 3로 프레임별 인간 팔 마스크 (시간적 일관성 유지)
2. **Hand Removal**: ProPainter 비디오 인페인팅으로 팔 제거 + 배경 복원
3. **Robot Base Pose Search** — 이 논문의 가장 실질적인 기술 난제. ego 궤적은 **embodiment-free**라 참조할 로봇 base가 아예 없음.
   - `T*_base = argmax (1/|K|) Σ_k 1[IK(T_base^{-1} T_ee_k) feasible]`, K는 궤적의 공간적 극단을 덮는 keyframe 집합
   - 후보 생성: 좌우 `r×{0.3…1.2}` 7단계, 전후 `r×{−0.1…0.9}` 7단계, 상하 `r×{0.4…−0.4}` 5단계, orientation pitch{30,45,60}×yaw{−45…45}×roll{−15,0,15}
   - 점수: `S = FR(T_base) − 5.0·|ρ̄ − 0.65|` — **평균 도달거리를 최대 reach의 65%로 유도**해 조작 여유(kinematic margin) 확보
   - mink IK solver(quadprog, 100 iter, 1e-5), 팔당 top-5 후보 → 좌우 25조합 joint 검증
4. **IK + Rendering**: MuJoCo에서 프레임별 IK, 원본 카메라 시점에서 렌더
5. **Depth-Aware Compositing**: arm body 마스크는 항상 덮어씀(카메라-작업공간 사이에 위치하므로 가려질 일 없음), gripper 마스크만 per-pixel depth 비교(Depth Anything V3 장면 depth vs MuJoCo 렌더 depth). hand 마스크를 5×5 커널로 dilate해 인페인팅 경계 노출 방지.

---

## 5. 파이프라인 (3) — Quality Curation & 데이터 구성

- **L1 (pipeline-internal)**: hand 미검출 / IK tracking error ≥ 0.05m / 렌더된 로봇 0픽셀 / self-collision / bimanual cross-arm contact > 1 / robot mask > 70% 면적 → invalid. Stability erosion: `<⌊0.3×fps⌋` 길이의 짧은 valid 구간이 invalid 사이에 끼면 invalid로 침식.
- **L2 (statistical)**: 차원별 Q1/Q99 기반 `[Q1−3(Q99−Q1), Q99+3(Q99−Q1)]` 이탈 프레임, residual/acceleration/jerk 급변 프레임 플래그. **invalid 비율 60% 초과 에피소드는 통째로 폐기**.
- **L3 (VLM consistency)**: Qwen3.5가 4fps 샘플 영상과 subtask description을 대조해 JSON(`is_consistent`, `confidence`, `reasoning`)으로 판정. 가짜/장난감 객체는 tolerate, 행동 유형·객체 카테고리·목표 위치·실행 실패만 major mismatch로.

**입력 경로**: Path A = hand pose 주석이 있는 ego 데이터셋 / Path B = raw 영상 → WiLoR 프레임별 MANO 복원 + DynHaMR 시간적 최적화, 긴 영상은 Qwen3.5로 subtask 분절.

**소스**: ANT 7h (자체 pick-and-place) + EgoDex 732h + ViTRA 249h + EgoVerse 954h ≈ **1,940h** → 15 morphology 렌더 + 큐레이션 → **18,561h**.

**15 morphology**: Panda, Kinova Gen3, IIWA, Sawyer, FR3, xArm7 (7-DOF) / UR5e, UR10e, Jaco, ViperX, WidowX, ARX-L5, Piper, YAM, Aloha-Agilex (6-DOF). Reach 0.787m(WidowX) ~ 1.627m(UR10e).

---

## 6. 모델 아키텍처 및 학습

- **Backbone**: Qwen3.5-4B VLM, timestep당 2–3 view RGB + 구조화 프롬프트. 카메라 intrinsic K와 extrinsic T_wc를 **mRoPE로 visual feature에 주입** → 카메라-장면 3D 관계 추론.
- **Action head**: DiT. context generator가 8개 learnable query token으로 H를 조건 feature C로 압축, DiT가 cross-attention으로 C에 attend해 **32-step action chunk** 예측.
- **목적함수**: flow matching. `a_t = (1−t)a_0 + t·ε`, `L = E||v_θ(a_t,t,c) − (ε − a_0)||²`. 학습 8 step / 추론 4 Euler step.
  - ⚠️ 본문 5.1은 "8 diffusion steps", 부록 C.3은 flow matching이라고 명시. 실체는 flow matching이며 `action_head_category: flow_matching`으로 분류.
- **Action 표현**: **camera-frame relative EEF** 7-dim (Δp 3 + Δω 3 + gripper 1). `Δp_cc = R_ce Δp_ee`, `ΔR_cc = R_ce ΔR_ee R_ce^T`. 카메라 배치가 미지·다양한 ego 영상을 **per-video calibration 없이** 통합하는 핵심 설계.
- **프롬프트**: `embodiment: {type}_{model}` (robot_aloha / h2r_arx / human_ego, 15% 확률로 "None" 드롭) + `instruction:` (드롭 없음).
- **Pretraining**: 8×A100, batch 12/GPU, backbone lr 1e-5→1e-6 cosine(5K warmup), action head lr 10배, AdamW(0.9/0.95), bf16, ZeRO-1, **200K step ≈ 19.2M frame**. 이미지 증강 없음.
  - **모든 설정이 동일 step·동일 batch → 처리 프레임 수가 동일**. 데이터셋 크기 차이가 아니라 *혼합 비율* 효과만 남기는 공정 비교 설계. 이 논문에서 가장 칭찬할 만한 실험 위생.
- **Finetuning**: 사전학습 가중치 로드, ColorJitter, ZeRO-2, 50K step.

---

## 7. 평가 프레임워크 (분해형 벤치마크)

RoboTwin 2.0을 **11개 독립 perturbation**으로 확장 + EBench Table Top 7태스크 보완.

| 축 | 설정 | 성격 |
|---|---|---|
| Visual Appearance | Background texture / Lighting | bundled에서 **분리** |
|  | Robot Color (전 링크 hue shift [0°,360°)) | **신규** |
| Scene Layout | Table Height (±4cm) / Clutter (3–5 distractor) | 분리 |
|  | Camera Offset (축당 최대 5cm) | 신규 |
| Embodiment | ARX-X5 / UR5-WSG / Franka Panda | **신규**, zero-shot. 기본 Aloha-Agilex를 두 개의 독립 single-arm URDF로 대체, base 간격 0.6/0.59/0.65m, 초기 EEF pose를 IK로 정렬, 3개 카메라 extrinsic 정렬 |
| Task Semantics | Unseen Objects (50 태스크) / Paraphrased Lang (505 구어체 지시) | 신규 |
| External | EBench Table Top (Isaac Sim, 더 높은 head camera) | 외부 |

- 프로토콜: 태스크당 50 episode, binary success. Visual = mean(BG, Light, Color), Scene = mean(Height, Clutter, Camera), Embodiment = mean(ARX, UR5, Franka), Task = mean(Obj, Lang).
- Finetuning: RoboTwin Clean 50태스크 × 50 demo(2,500 ep), chunk 20 / replan 16. EBench는 7태스크 × 400 demo(2,800 ep), chunk 32.

---

## 8. 주요 결과 (Table 1)

| Pretraining | Clean | Rand | Visual | Scene | Embody | Task | EBench |
|---|---|---|---|---|---|---|---|
| Robot-only | 62.2 | 50.9 | 61.4 | 52.9 | 23.8 | 46.2 | 39.6 |
| Ego2R+Robot (1:3) | 61.4 | 51.0 | 61.2 | 52.5 | 21.9 | 49.5 | **47.4** |
| Ego2R+Robot (3:1) | 64.1 | 49.2 | 62.7 | 54.3 | **28.2** | 51.6 | **51.7** |
| Ego2R+Robot (1:1) | **68.1** | **53.5** | **67.3** | **56.9** | 27.2 | **54.1** | 49.8 |

- **1:1이 7개 열 중 5개에서 선두.** Clean +5.9, Randomized +2.6. 합성 데이터를 섞어도 clean 성능이 떨어지지 않고 오히려 오르는 점이 중요.
- **1:3은 사실상 무효** (Clean −0.8, Embody −1.9). ego 데이터가 소수면 신호가 묻힘. 3:1과 1:1에서 비로소 실질 이득.
- **EBench는 3:1이 최고 (+12.1)**. EBench의 head camera가 더 높아 egocentric 시점에 가깝다 → **시점 bias가 부분적으로 일치할 때 co-training 이득이 증폭**된다는 해석. 이 논문에서 가장 통찰적인 관찰.

---

## 9. Perturbation별 분해 (Table 2, vs Robot-only)

| | BG | Light | Color | Height | Clutter | Camera | ARX | UR5 | Franka | Obj | Lang |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Robot-only | 66.6 | 58.2 | 59.4 | 60.1 | 48.3 | 50.4 | 44.1 | 20.2 | 7.0 | 29.3 | 63.1 |
| 1:3 | 65.0 | 58.3 | 60.3 | 58.6 | 49.3 | 49.6 | 43.7 | 17.6 | 4.5 | 36.8 | 62.2 |
| 3:1 | 65.5 | 60.9 | 61.8 | 62.0 | 49.2 | 51.6 | 47.6 | **31.4** | 5.6 | **40.0** | 63.1 |
| 1:1 | **70.3** | **65.8** | **65.8** | **62.4** | **52.0** | **56.3** | **51.2** | 25.0 | 5.3 | 39.6 | **68.5** |

- **Visual이 가장 크게 이득**: lighting +7.6, robot color +6.4, background +3.7. 조명은 ego 영상의 장면 다양성에서, robot color는 15개 morphology 렌더링에서 온다는 설명.
- **Camera offset +5.9**: ego 영상이 자연히 다양한 head pose/시점을 포함하기 때문.
- **Embodiment**: ARX 44.1→51.2 (+7.1), UR5는 3:1에서 31.4로 정점. 그러나 **Franka는 전 설정에서 7% 이하이며 오히려 하락**(−1.7). 저자는 학습 embodiment와의 kinematic gap 때문이라 설명하는데, 이는 camera-frame action 표현의 embodiment 불변성이 무제한은 아님을 보여줌.
- **Unseen object 29.3→40.0 (+10.7 at 3:1)** — 가장 큰 단일 이득. ego 영상의 객체 다양성이 직접적으로 전이된다는 가장 깨끗한 증거.

---

## 10. Ablation: 파이프라인 자체의 가치 (Figure 3)

robot 데이터를 **완전히 배제**하고 ego 계열만으로 사전학습 → RoboTwin Randomized 성능.

| 설정 | 데이터량 | Rand SR |
|---|---|---|
| Raw ego (렌더 없음) | ~1,940h | 28.1 |
| Ego2R, 1 morphology (ARX-L5) | ~1,237h | 31.7 |
| Ego2R, 5 morphologies | ~6,187h | (도표만) |
| Ego2R, 10 morphologies | ~12,374h | (도표만) |
| Ego2R, 15 morphologies | ~18,561h | 33.5 |
| Ego2R (15) + Raw ego | ~20,501h | **37.3** |

- **파이프라인 정렬이 필수**: raw ego 28.1 → Ego2R 1-morph 31.7 (+3.6). 데이터량은 오히려 줄었는데(1,940→1,237h) 성능이 오름 → **시각·행동 정렬의 순수 기여**.
- **morphology 증가가 단조 개선**: 1→15에서 31.7→33.5.
- **가장 흥미로운 결과**: 15-morph Ego2R에 raw ego를 *다시* 더하면 37.3으로 점프. 저자 해석 — raw ego가 시각 외형과 행동 분포가 약간 다른 **"16번째 morphology"** 역할을 하며 다양성을 추가로 공급. 정제된 데이터가 원본을 대체하는 게 아니라 **보완**한다는 뜻.

---

## 11. 실기 실험 (ARX ACone, Figure 4)

- 플랫폼: ARX ACone dual-arm (팔당 6-DOF, parallel gripper, head + wrist ×2 RGB, 15Hz 제어, 32 step마다 replan)
- 5개 long-horizon 태스크: Put Fruits(3단계), Put Blocks(4단계, 서랍 열기→블록 2개→닫기), Fold Towel(2단계), Sweep Trash(4단계), Insert Screw(4단계, bimanual handover + 삽입)
- 데이터: 태스크당 teleop 20 demo(총 100) + **~35분 ego play 영상** → Path B → **675개 ACone 합성 에피소드**. Finetuning 시 1:1 혼합.
- 채점: 하위 단계 부분 점수(태스크당 100점), 20 trial.
- 결과: **Mix + Ego2R Play가 5개 태스크 전부에서 최고**. 텍스트로 명시된 수치는 Put Blocks **+14**, Insert Screw **+13** (vs Robot-only). Mix 사전학습만으로도 이미 Robot-only 상회.
- 시사점: 특별한 장비 없이 **캐주얼하게 찍은 7분/scene 짜리 ego play 영상**이 유효한 학습 신호가 된다는 것 — 파이프라인의 현장 실용성을 보여주는 부분.

---

## 12. 한계, VLA-Tracker 맥락, 세미나 Q&A 예상

**저자 명시 한계**
1. Parallel-jaw gripper로만 retarget → **finger articulation 폐기**. Dexterous multi-finger 확장 필요.
2. Inpainting + depth compositing 의존 → 심한 occlusion/복잡 조명에서 artifact.
3. 평가가 RoboTwin2.0 태스크 범위에 국한.

**리뷰어 관점 추가 지적**
- **Franka zero-shot 5–7%**는 사실상 실패. camera-frame action이 morphology gap을 자동으로 넘지 못한다는 반례이며, 논문은 이를 "kinematic gap"으로 한 문장 처리하고 넘어감.
- Rand에서 3:1이 robot-only보다 **낮음**(49.2 vs 50.9). 비율 민감도가 크며 최적 비율을 사전에 알 방법이 제시되지 않음.
- 실기 결과 절대 수치가 막대그래프에만 있어 텍스트로 검증 가능한 값이 델타 2개뿐. 재현·인용에 불리.
- 비교 대상이 사실상 자체 robot-only ablation. Pi0.5 비교(Fig. 9)는 있으나 도표 전용이라 정량 인용 불가.

**VLA-Tracker 맥락**
- 데이터 합성 논문이지만 **자체 VLA(Qwen3.5-4B + DiT flow matching)를 학습하고 정책 성공률을 보고**하므로 SiMDex와 동일 논거로 등재 대상.
- 벤치마크는 `benchmarks.robotwin_v2`(+ baseline/ablation 형제 블록)와 `benchmarks.ebench`. LIBERO/CALVIN/SimplerEnv 수치는 **없음** → 리더보드 랭킹에는 반영되지 않음.
- 관련 논문: Phantom(2503), EgoMimic(ICRA25), Being-H0/H0.5, EgoScale, ViTRA, EgoVerse, RoboTwin 2.0.

**세미나 Q&A 예상**

| # | 질문 | 답변 |
|---|---|---|
| 1 | 데이터셋 논문인데 왜 VLA로 취급? | 파이프라인 산출물로 끝나지 않고 Qwen3.5-4B + DiT flow-matching 정책을 200K step 사전학습해 RoboTwin2.0/EBench/실기 성공률을 보고. 정책 산출물이 존재. |
| 2 | 18,561h가 진짜 "새 데이터"인가? | 아니다. 원본은 1,940h이고 15 morphology 렌더로 ~9.6× 증폭한 것. 행동 궤적의 정보량은 늘지 않고 **시각적 embodiment 다양성**만 늘어남. Fig. 3에서 morphology 1→15 이득이 +1.8에 그치는 것이 이 한계를 정직하게 드러냄. |
| 3 | 왜 1:1이 최적이고 3:1은 Rand에서 나쁜가? | ego 데이터는 시각 다양성은 크지만 물리적으로 부정확(인페인팅 artifact, IK 근사). 비중이 커지면 robot 데이터의 정밀한 dynamics 신호가 희석됨. 반대로 EBench처럼 시점이 ego에 가까우면 3:1이 이김. 즉 **최적 비율은 타깃 도메인의 시점 bias에 의존**. |
| 4 | Base pose search가 왜 어려운 문제인가? | robot→robot 전이는 소스 base가 주어지지만 ego 궤적은 embodiment-free라 참조점이 없음. 15개 morphology 각각 arm length/joint 구조가 달라 동일 궤적에도 서로 다른 base가 필요 → morphology별 독립 grid search + IK 검증. `ρ̄=0.65` 페널티로 reach 여유를 남기는 설계가 실무적 포인트. |
| 5 | camera-frame relative EEF를 쓰는 이유는? | ego 영상은 카메라 배치가 미지이고 매번 다름. world-frame action은 영상마다 calibration이 필요하고 소스 간 action space가 호환되지 않음. 관찰자 좌표계 변위로 표현하면 서로 다른 카메라 설정과 morphology가 자연히 통합됨. |
| 6 | L3 VLM 필터가 실제로 필요한가? | 논문에 L1/L2/L3 각각의 정량 기여도(ablation)가 **없다**. 파이프라인 전체(raw ego → Ego2R) 효과 +3.6만 있음. 큐레이션 단계별 기여 분해는 미해결. |
| 7 | Franka가 왜 실패했나? | 학습 embodiment(Aloha-Agilex)와 kinematic gap이 큼. 다만 Franka는 사전학습 morphology 15종에 포함되어 있고 DROID도 Franka 기반인데 5–7%에 그친 점은 저자 설명(kinematic gap)만으로는 불충분. 평가 시 dual single-arm URDF 배치(0.65m)와 학습 분포의 불일치도 의심 가능. |
| 8 | 재현 가능성은? | EgoDex/ViTRA/EgoVerse/DROID/AgibotWorld/InternData/RoboTwin2.0은 공개, WiLoR·DynHaMR·SAM3·ProPainter·mink·MuJoCo도 공개. 반면 ANT 데이터셋, 합성 데이터, 학습 코드/가중치 공개 여부는 미명시(`open_source: false`). 8×A100 × 200K step 사전학습 비용도 진입 장벽. |

<!-- VERIFIED: pdf -->
