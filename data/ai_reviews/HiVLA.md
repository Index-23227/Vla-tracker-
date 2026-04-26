# HiVLA: Visual-Grounded-Centric Hierarchical Embodied Manipulation

## 1. 배경 및 동기

- 현재의 end-to-end VLA(π0, OpenVLA, GR00T-N1.5 등)는 VLM을 좁은 manipulation 데이터로 fine-tune 하면서 web-scale reasoning 능력을 catastrophic forgetting 하는 근본적 trade-off를 안고 있다(§1).
- 이를 회피하려는 hierarchical approach(RT-H, Hi-Robot, HAMSTER)는 textual subtask나 keypoint 같은 intermediate representation을 사용하나, visual fidelity와 spatial context를 동시에 살리지 못한다(§2.2).
- 예컨대 DexGraspVLA는 down-sampled global image에 mask를 적용해 fine detail을 잃고, InterleaveVLA는 local crop에서 absolute spatial coordinate를 상실한다(§1).
- HiVLA는 "high-resolution local appearance + precise global spatial awareness + explicit skill-level subtask"를 동시에 활용할 수 있는 visual-grounded 중심의 계층 구조를 제안한다.
- 핵심 통찰: VLM이 natively 내는 bounding box를 intermediate로 쓰고, DiT action expert 내부에 cascaded cross-attention으로 세 가지 conditioning을 순차적으로 주입하면 각 역할이 깔끔하게 분리된다.

## 2. 방법론

- **High-Level VLM Planner** (§4.1): Qwen3-VL 8B fine-tuned. 입력은 instruction L, gripper state, 이전 subtask, before/after visual history. 출력은 structured JSON: subtask description L_sub,t, action type, target object name, normalized bbox B_t=[y_min,x_min,y_max,x_max]. Bbox가 Image Crop tool을 invoke 하여 1080×1920 원본에서 object-centric patch I_t^local을 추출.
- **DiT Action Expert** (§4.2): H-RDT 기반 LLaMA-style backbone (RMSNorm+SwiGLU), EgoDex pretrained weight로 초기화. Global image cross-attention의 weight를 복사해 novel local image cross-attention layer를 초기화.
- **Conditional Flow Matching** (Eq. 1-3): linear interpolation path x_τ=τA_t+(1−τ)z, z∼N(0,I). vθ가 velocity field u=A_t−z를 예측, L_CFM=E[‖vθ(x_τ,τ,C_t)−(A_t−z)‖²]. Inference는 forward Euler ODE solver.
- **Cascaded Cross-Attention DiT Block** (Fig. 2b, §4.2): 각 block에서 순차적으로 (1) Global Img. Cross-Attn (DINOv2+SigLIP feature C^global), (2) Local Img. Cross-Attn (1080p 원본에서 crop된 C^local + DETR-style absolute sinusoidal PE, Eq. 4: C^local-pos=C^local+PE(p)), (3) Language Cross-Attn (L_sub,t embedding C^lang). AdaLN으로 timestep modulation.
- **Latency** (§5.1): VLM planner 1.9s/step (asynchronous), DiT 16-step action chunk 0.162s → 8Hz control.

## 3. 실험 결과

Table 1 (RoboTwin 2.0, 9-task, 100 trials per task, domain randomization):

| Task | π0 | π0.5 | StarVLA | H-RDT | Ours w/o Skill | **Ours** |
|---|---|---|---|---|---|---|
| Click Bell | 45 | 65 | 71 | 88 | 95 | 94 |
| Click Clock | 53 | 66 | 83 | 93 | 97 | 97 |
| Press Stapler | 60 | 69 | 63 | 89 | 98 | 97 |
| Lift Pot | 59 | 21 | 18 | 92 | 96 | 96 |
| Easy Avg | 54.3 | 55.3 | 58.8 | 90.5 | 96.5 | **96.0** |
| Place Shoe | 75 | 68 | 61 | 88 | 94 | 95 |
| Move Stapler | 15 | 17 | 15 | 34 | 42 | 60 |
| Stamp Seal | 61 | 42 | 25 | 43 | 68 | 76 |
| Stack 3 Blocks | 1 | 1 | 16 | 20 | 26 | 37 |
| Click 3 Bells | 41 | 54 | 66 | 88 | 92 | 98 |
| Hard Avg | 38.6 | 36.4 | 36.6 | 54.6 | 64.4 | **73.2** |
| **Total Avg** | 45.6 | 44.8 | 46.4 | 70.6 | 78.7 | **83.3** |

- H-RDT 대비 +12.7pp, π0 대비 +37.7pp.
- **Robustness (Table 2)**: bbox에 100% noise 주입해도 57.0% 유지(global feature로 self-correct), language noise는 선형적으로 성능 저하(12.0% at 100%) — 언어 명령에는 strict adherence.
- **Guidance Injection Ablation (Table 4A)**: Local→Text 70.4%, Global→Text 70.6%, Global→Local→Text(ours) **83.3%**. "Coarse-to-Fine"(Global→Local→Language) ordering이 최적.
- **Visual Grounding Components (Table 4B)**: w/o HD Crop 75.2%, w/o Abs.PE 76.8%, Full 83.3%. Click 3 Bells에서 Abs.PE 없이 동일 객체 disambiguation 실패 확인.
- **VLM Planner Validation** (§5.1): Qwen3-VL 8B를 210K-instance dialogue dataset으로 FT → bbox mIoU 90.37%, exact-match subtask 98.57%.
- **Real-world (Table 3, 30 trials/task)**: 360 teleoperated episode(GroundingDINO+SAM2 자동 annotation), 80K step FT. 3 Cups task에서 H-RDT 0/30 vs HiVLA 6/30, 3 Blocks에서 H-RDT 0/30 vs HiVLA 7/30. Cluttered 환경에서의 decisive advantage.

## 4. 한계 및 미해결 문제

- **RoboTwin 2.0의 커스텀 9-task subset**: 9개 task 평균 83.3%는 표준 RoboTwin 전체(50 task)가 아닌 HiVLA-HD라는 저자 custom subset 결과이다. 다른 visual-grounded 오픈소스가 드물어 baseline 선택도 제한적(§5.1)이며, 일반적인 RoboTwin SOTA와의 직접 비교는 부적절하다.
- **VLM latency & dual-model 인프라**: unoptimized planner가 1.9s/step이며 asynchronous tracking으로 bridge 한다지만 real-time 요구가 엄격한 high-DoF task에서는 병목이 될 수 있다. 모델 두 개를 동시 운용해야 하는 deployment 복잡성도 존재.
- **Planner 오류의 언어 의존성 (Table 2)**: Bbox noise에는 매우 강인하지만(57% @ 100%), language noise에는 proportional degradation (12% @ 100%)을 보인다. 이는 policy가 언어 명령을 엄격하게 따르도록 설계된 trade-off로, open-world에서 planner의 hallucination이 직접적으로 실패로 이어질 수 있다.

## 5. 총평

- **Novelty ★★★★☆** — Hierarchical VLA 자체는 새롭지 않지만, (1) native VLM bounding box를 intermediate로 사용, (2) 1080p 원본에서 HD crop + absolute sinusoidal PE로 spatial awareness를 유지, (3) cascaded cross-attention의 coarse-to-fine ordering까지 — 세 요소의 조합은 독창적이고 ablation도 촘촘하다. H-RDT weight 복사로 local attention 초기화한 엔지니어링 디테일도 인상적.
- **Practical impact ★★★★☆** — RoboTwin Hard task 평균 73.2%(H-RDT 54.6% 대비 +18.6pp)는 cluttered bimanual manipulation에서 실질적인 도약이다. 실세계 360 episode로 3 Cups/3 Blocks 등 기존 VLA가 거의 0%인 task를 작동시킨다는 점에서 impact는 크다. 다만 RoboTwin 커스텀 subset이라는 제약, 그리고 planner+policy dual 인프라 부담은 상용 배포 시 고려 요소.

## 6. 예상 질문

- **Q1. "RoboTwin 2.0에서 83.3%"는 standard benchmark 결과인가?** A. 아니다. 9-task custom subset(HiVLA-HD, Easy 4 + Hard 5)이며 Hard-mode domain randomization 설정. 공식 RoboTwin 2.0 전체 suite 결과가 아니므로 다른 논문과의 직접 비교에는 주의가 필요하다. 해당 9개 task는 fine-grained grasping과 long-horizon skill composition을 stress-test 하도록 선택되었다.
- **Q2. Cascaded cross-attention 순서(Global→Local→Text)가 정말 중요한가?** A. Table 4A의 6가지 순서 ablation은 명확하다. Local→Text 70.4%, Global→Text 70.6%(단일 modality): 두 visual을 결합하지 않으면 ~70%에서 고원. 순서가 Local→Text→Global 80.1%, Global→Text→Local 78.3%, Local→Global→Text 74.1%, Global→Local→Text 83.3%. "coarse-to-fine" 직관이 실증된다.
- **Q3. 왜 H-RDT weight 복사 초기화가 필요한가?** A. Local image cross-attention은 architecturally novel layer이다. Random init 시 EgoDex-pretrained backbone과 mismatch. Global cross-attention weight가 이미 scene→state-action mapping을 학습했으므로 이를 local branch의 warm start로 활용하면 수렴 속도와 안정성 모두 향상된다(명시적 비교는 없지만 설계 의도).
- **Q4. Ours w/o Skill(78.7%) vs Ours(83.3%) 차이(+4.6pp)는 subtask decomposition 때문인가?** A. 그렇다. "Ours w/o Skill"은 global instruction을 skill 대신 사용. Easy task에서는 차이 미미(96.5 vs 96.0)지만 Hard task에서 64.4%→73.2%(+8.8pp). 즉 one-to-one subtask skill이 diffusion policy의 cognitive load를 줄인다. 또한 VLM planner가 실패한 sub-skill을 re-issue 할 수 있는 emergent error-correction property도 관찰됨(§5.2).

<!-- VERIFIED: pdf -->
