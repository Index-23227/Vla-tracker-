# LaMP: Learning VLA Policy with 3D Scene Flow as Latent Motion Prior

## 1. 배경 및 동기

- 기존 VLA는 VLM의 2D semantic feature로부터 action을 직접 regression 하며, 3D physical interaction을 implicit 하게 학습한다. 이 implicit learning은 unfamiliar spatial dynamics(carmera view 변화, layout perturbation)에서 brittle 하다(§1).
- 중간 표현으로 2D optical flow(FlowVLA, TraceVLA), pixel-space future(F1, FLARE), 3D point-cloud diffusion 등이 시도되었으나 각각 depth 부재, pixel 재구성 비용, static feature 한계, task-specific pipeline 제약을 가진다.
- **LaMP의 가설**: dense 3D scene flow는 action의 byproduct가 아니라 policy learning을 guide 하는 **latent motion prior**로 작용해야 한다.
- **효율 아이디어**: 전체 multi-step denoising으로 3D flow를 완전히 생성하지 말고, **1-step partially denoised hidden state**만 추출해 gated cross-attention으로 VLM 마지막 layer에 주입하면 latency 비용 없이 geometric foresight를 얻을 수 있다.
- 자연스럽게 VLM pretraining(Qwen3-VL-4B-Instruct)의 semantic 이해를 유지하면서 cross-embodiment에 robust한 policy를 구축한다.

## 2. 방법론

- **Factorization** (Eq. 1): P(A, M | z) = P(A | z, M) · P(M | z) — action과 motion의 joint를 chain rule로 분해. Motion Expert가 P(M | z), Action Expert가 P(A | z, M).
- **3D Scene Flow Representation** (§III-C): K_h×K_w=20×20 uniform grid의 keypoint를 T=32 미래 timestep에 걸쳐 트래킹. 각 keypoint는 Δp=(Δu, Δv, Δd) 예측. 최종 M_t∈R^(K×T×3), K=400. Reference camera frame으로 정규화하여 embodiment-agnostic.
- **Motion Expert** (§III-C): CogVideoX-style 3D transformer, spatial patchification 2×2 → 10×10 spatial tokens/timestep. 12-layer, hidden dim 1024. VLM의 last-layer feature z_t를 AdaLN으로 conditioning. Conditional flow matching (Eq. 2-3): M_τ=(1−τ)ε+τM_t, v_θ가 M_t−ε 예측.
- **One-step Denoising** (§III-D): M_0∼N(0,I)에서 N=10 solver 중 1 step만 실행(τ=0.1), intermediate hidden state z_m∈R^(B×N×d_m) 추출. Full denoising 없이 task-relevant dynamics 유지.
- **Gated Motion Guidance** (§III-E, Eq. 4): z_guided = z + σ(g) · CA(Q=LN(z), K=V=LN(W_proj z_m)). Learnable scalar gate g, sigmoid로 (0,1) 제약, 초기값 0 → weak motion injection에서 시작하여 필요할 때만 gain 증가. Representational collapse 방지.
- **Action Expert** (Eq. 5): a_τ=(1−τ)ε_a+τa, L_action=E‖v_φ(a_τ, τ, z_guided)−(a−ε_a)‖². Flow matching 기반.
- **Two-stage Training** (§III-F):
  - Stage 1: Motion Expert pretraining on 1.6M observation-language-motion triplets (TraceForge 파이프라인) from LIBERO, BridgeV2, DROID, InternData-A1. Batch 32/GPU × 16 GPU = 512, hidden 1024, 12 layer, AdamW, lr 2e-4, 30 epoch, bfloat16 (Table III).
  - Stage 2: VLM(Qwen3-VL-4B-Instruct)과 Motion Expert freeze, Motion Guidance module + Action Expert만 학습. LIBERO/SimplerEnv/Real은 각각 H=9/29/15, 15K/15K/20K steps.
- **Inference** (Algorithm 2): (1) VLM feature z 추출 → (2) Motion Expert 1-step, z_m 추출 → (3) Guide(z, z_m) → (4) Action Expert N=10 step ODE integration.

## 3. 실험 결과

**Table I (LIBERO, SimplerEnv-WidowX)**:

| Method | Spatial | Object | Goal | Long | **Avg** | Stack | Carrot | Spoon | Eggplant | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 | 0.0 | 0.0 | 4.2 | 12.5 | 4.2 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 | – | – | – | – | – |
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 | 16.7 | 0.0 | 29.1 | 62.5 | 40.1 |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 | 44.7 | 64.7 | 49.3 | 69.7 | 57.1 |
| GR00T N1 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 | 16.7 | 45.8 | 62.5 | 20.8 | 49.5 |
| UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 | 29.2 | 62.5 | 83.3 | 100.0 | 68.7 |
| F1 | 98.2 | 97.8 | 95.4 | 91.3 | 95.7 | 50.0 | 70.8 | 50.0 | 66.7 | 72.9 |
| FlowVLA | 93.2 | 95.0 | 91.6 | 72.6 | 88.1 | 62.5 | 62.5 | 70.8 | 100.0 | 74.0 |
| **LaMP** | **99.4** | **99.8** | **97.4** | **96.7** | **98.3** | **75.0** | 66.7 | 79.1 | 95.8 | **79.2** |
| w/o motion | 95.8 | 98.9 | 96.6 | 78.2 | 92.4 | 25.0 | 45.8 | 66.7 | 87.5 | 56.3 |

- LIBERO 98.3% (최고), LIBERO-Long 96.7% — π0.5(92.4%) 대비 +4.3pp.
- SimplerEnv-WidowX 79.2% zero-shot, FlowVLA(74.0%) 대비 +5.2pp. Stack Block 75.0% vs FlowVLA 62.5%.
- **w/o motion 제거 시 LIBERO 98.3→92.4% (−5.9pp), Long 96.7→78.2% (−18.5pp)**, SimplerEnv 79.2→56.3% (−22.9pp), Stack Block 75.0→25.0% (−50pp). Motion prior가 결정적.

**Table II (LIBERO-Plus zero-shot OOD, 7 perturbation dims)**:

| Method | Camera | Robot | Language | Light | Background | Noise | Layout | **Avg** |
|---|---|---|---|---|---|---|---|---|
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| π0-Fast | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| **LaMP** | 64.5 | **69.6** | **88.2** | **95.3** | **97.4** | **76.9** | 73.8 | **79.3** |
| w/o motion | 46.7 | 56.0 | 82.5 | 95.3 | 95.4 | 69.3 | 71.0 | 71.6 |

- Avg 79.3% (strongest baseline +9.7pp). Robot perturbation 69.6% vs π0-Fast 21.6% — camera-frame motion prior(Δu,Δv,Δd)가 joint-space 대비 kinematic variation에 강인.

**Fig. 4 (2D-Flow ablation)**: 3D scene flow → 2D optical flow(depth dim mask out)로 교체 시 Stack Block 75.0→58.3% (−16.7pp), Put Spoon 79.1→70.8% (−8.3pp). Depth-aware 3D prior가 contact-rich task에서 필수.

**Fig. 5 (Fusion strategy)**: Gated CA(ours) 75.0% vs Add 33.3% vs Concat+MLP 45.8% on Stack Block. Unweighted fusion은 conflict 시 오히려 harm.

**Fig. 7 (Real-world, Flexiv Rizon 4)**: Pick-and-Place 80%, Fold Towel(Deformable) 50%, Making Bread(Long-Horizon) 80%, OOD avg 62.5%. π0(70/40/70/52.5) 및 3D FDP(53/40/60/26.8) 상회.

## 4. 한계 및 미해결 문제

- **고정된 spatial/temporal resolution**(§VI): 20×20 grid와 T=32로 고정. 매우 fine-grained local motion(예: threading, 바느질)이나 매우 long-horizon(>32 step planning)에는 expressivity 부족. Adaptive-resolution motion representation은 향후 과제.
- **TraceForge pipeline 의존**: 1.6M triplet이 모두 TraceForge [20]로 offline 생성된 supervised 3D scene flow. Unlabeled in-the-wild video에서 motion prior를 배우는 것은 아직 구현되지 않았다. Data pipeline 공백이 존재.
- **Motion Expert freezing의 cost**: Stage 2에서 Motion Expert와 VLM을 freeze — task-specific motion pattern이 pretraining 분포 밖에 있다면 adaptation이 어려울 수 있다. 1-step denoising 또한 information bottleneck이 될 가능성.

## 5. 총평

- **Novelty ★★★★★** — "3D scene flow를 world model로 완전히 생성하지 말고 1-step partially denoised hidden state를 latent prior로 주입"이라는 포인트는 참신하고, 이를 gated cross-attention(learnable scalar gate, sigmoid, init=0)으로 안전하게 VLM에 bind 하는 설계도 세련됐다. Representational collapse 회피와 latency 회피를 동시에 달성한 fusion design이 본 논문의 백미.
- **Practical impact ★★★★☆** — LIBERO 98.3%(#4 on leaderboard) + LIBERO-Plus OOD 79.3%(차기 baseline +9.7pp) + zero-shot SimplerEnv-WidowX 79.2%. 특히 LIBERO-Plus Robot perturbation에서 21.6%→69.6%로 kinematic variation에 극적으로 강인한 generalization을 보여주어 실세계 배포 가치가 높다. Real-world Flexiv Rizon 4에서 160 episode만으로 쓸 만한 성능을 내는 data efficiency도 고무적.

## 6. 예상 질문

- **Q1. 왜 full multi-step motion reconstruction 대신 1-step denoising인가?** A. 두 가지 이유. (1) Latency: N=10 step ODE는 action expert에도 필요하므로 motion까지 full solve 하면 inference cost가 두 배. 1-step(τ=0.1)은 single forward pass와 동등. (2) Information quality: VPP, mimic-video 선행 연구에서 fully denoised latent보다 partial hidden state가 오히려 policy-relevant structure를 더 잘 보존한다는 empirical evidence. 저자는 "retain task-relevant dynamics while circumventing computational cost and bias"(§III-D)라 표현.
- **Q2. Gated cross-attention이 왜 필요한가? Add/Concat으로는 왜 안 되는가?** A. Fig. 5에서 명확. Add(unweighted) → Stack Block 33.3%, Concat+MLP 45.8%, Gated 75.0%. 핵심은 σ(g) gate가 0으로 초기화되어 학습 초기에 motion injection이 거의 없다가 유익할 때만 점진적으로 증가한다는 점. Motion prediction이 noisy할 때 강제 주입하면 VLM representation을 오히려 손상(representational collapse)한다는 저자 관찰과 일치.
- **Q3. LIBERO-Plus에서 Robot perturbation +48pp(21.6→69.6)는 왜 이렇게 크게 나오는가?** A. Camera-frame (Δu, Δv, Δd) space에서 reasoning 하므로 arm의 kinematic configuration 변화에 독립적이다. 반대로 action을 joint space에서 regression 하는 모델(π0-Fast)은 kinematic change에 그대로 노출된다. 이는 embodiment-agnostic motion representation의 직접적 효과.
- **Q4. Qwen3-VL-4B-Instruct 선택 이유와 backbone freeze의 영향은?** A. Qwen3-VL은 최신의 강한 VL backbone이고 4B는 H100 16장에서 운용 가능한 compromise. Freeze 하는 이유는 VLM의 semantic understanding과 motion prior를 cleanly 분리하기 위함 — 만약 full fine-tune 하면 HiVLA가 지적한 catastrophic forgetting이 발생. 대신 gated cross-attention이 last layer에만 motion cue를 주입하는 최소 침습 접근을 택한다.

<!-- VERIFIED: pdf -->
