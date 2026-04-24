# MMaDA-VLA: Large Diffusion VLA with Unified Multi-Modal Instruction and Generation

## 1. 배경 및 동기

- 기존 VLA는 크게 두 패러다임으로 나뉜다: (1) VLM + policy head의 hierarchical continuous VLA(π0, OpenVLA-OFT, VLA-Adapter), (2) action 토큰을 autoregressive 하게 생성하는 discrete VLA(OpenVLA, π0-FAST, RT-1)(§2.2).
- Hierarchical 방식은 module 경계의 information fidelity 저하와 computational overhead, autoregressive 방식은 action dimension 간의 weak temporal consistency와 long-horizon error accumulation이 문제(§1).
- 두 방식 모두 **environment dynamics (future observation)**를 explicit 하게 모델링하는 내장 메커니즘이 없다. Visual generation을 도입한 World Model VLA(WorldVLA, CoT-VLA, UniVLA, Seer, VPP)도 대부분 **autoregressive backbone에서 fine-tune** 되어 train/inference mismatch가 남는다.
- **MMaDA-VLA의 가설**: fully native pre-trained large diffusion backbone(MMaDA-8B-Base)을 사용하면 language·image·continuous robot control을 **하나의 discrete token space**에서 masked-token denoising으로 통합할 수 있고, goal observation과 action chunk를 parallel로 함께 refine 하여 world-model-like dynamics를 auxiliary module 없이 얻을 수 있다.

## 2. 방법론

- **Unified Discrete Tokenization** (§3.2): LLaDA textual tokenizer + MAGVIT-v2 image quantizer(Show-o에서 차용) + action tokenizer(각 dimension 256 bins, bin width는 training range로 결정). 모든 modality가 하나의 vocabulary V로.
- **Multi-Modal Sequence** (Eq. 2): x = [SOO]õ_t[EOO] [SOL]l̃[EOL] [SOO]õ_t'[EOO] [SOA]ã_{t:t'−1}[EOA]. Instruction part는 current obs + language, Generation part는 goal obs + action chunk.
- **Hybrid Attention** (§3.2): **intra-modal bidirectional full attention** + **inter-modal causal attention**. Action dimension은 inherently unordered이므로 autoregressive ordering의 inductive bias를 피함. Goal-image generation과 action 생성을 decouple 하여 feature foundation 안정화.
- **Masked Token Denoising** (Eq. 3): Generation part의 토큰을 mask ratio(cosine schedule) γ로 랜덤 replace → L(θ) = −E[(1/N)Σ 1[x_i^m=[M]] log π_θ(x_i | x^m)]. 단일 objective로 pre-training과 fine-tuning 모두.
- **Problem Formulation** (Eq. 1): π_θ(o_{t'}, a_{t:t'-1} | o_t, l), t' = t + k, k는 action chunk size. Action과 goal observation이 parallel joint 분포.
- **Iterative Denoising Inference** (Eq. 4-5): Generation part를 모두 [M]으로 초기화, D denoising step. 각 step에서 π_θ가 clean sequence 분포 estimate → greedy decoding → confidence 하위 β 토큰만 다시 mask(confidence-based remasking).
- **KV Cache** (§3.4): Instruction part는 denoising 동안 stable하므로 K_l, V_l, AttnOut_l, FFNOut_l cache. Generation part는 λ step마다 refresh, selective refresh ratio ρ. Real-time 요구에 대응.
- **Training setup** (Table 3): MMaDA-8B-Base backbone, MAGVIT-v2 tokenizer, global batch 640(pretrain) / 80·LIBERO, 320·CALVIN(FT), lr 1e-4 cosine decay, AdamW, BFloat16. Action chunk 5(LIBERO), 10(CALVIN). Pretraining 1 epoch × 8 nodes × 8 H800 80GB ≈ 30 hours.
- **Pretraining data** (Table 2, ~61M steps): DROID 49.94%, BC-Z 9.95%, Language Table 9.88%, Furniture Bench 7.35%, Fractal 6.44%, Bridge V2 3.15%, Kuka 3.03%, FMB 2.06%, CALVIN 1.87%, LIBERO 1.55% + 13 smaller datasets. CALVIN ABC→D 공정성 위해 environment D data는 제외.

## 3. 실험 결과

**Table 1 LIBERO (4 suite, fine-tuned)**:

| Method | Spatial | Object | Goal | Long | **Avg** |
|---|---|---|---|---|---|
| OpenVLA | 84.9 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.8 |
| Discrete Diffusion VLA | 97.2 | 98.6 | 97.4 | 92.0 | 96.3 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| VLA-Adapter | 97.8 | 99.2 | 97.2 | 95.0 | 97.3 |
| UniVLA (World Model) | 95.4 | 98.8 | 93.5 | 94.0 | 95.5 |
| MM-ACT | 97.8 | 99.4 | 94.8 | 93.0 | 96.3 |
| **MMaDA-VLA (Ours)** | **98.8** | **99.8** | **98.0** | **95.2** | **98.0** |

**Table 1 CALVIN (ABC→D, avg completed chain length)**:

| Method | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | **Avg Len** |
|---|---|---|---|---|---|---|
| OpenVLA-OFT | 96.3 | 89.1 | 82.4 | 75.8 | 66.5 | 4.10 |
| VLA-Adapter | 99.1 | 94.6 | 88.8 | 82.8 | 76.5 | 4.42 |
| DreamVLA (WM) | 98.2 | 94.6 | 89.5 | 83.4 | 78.1 | 4.44 |
| Seer (WM) | 96.3 | 91.6 | 86.1 | 80.3 | 74.0 | 4.28 |
| VPP (WM) | 95.7 | 91.2 | 86.3 | 81.0 | 75.0 | 4.33 |
| **MMaDA-VLA (Ours)** | **99.8** | **98.6** | **96.3** | **93.5** | **89.7** | **4.78** |

- LIBERO Avg 98.0% — VLA-Adapter(97.3%) 대비 +0.7pp, 전체 SOTA.
- CALVIN Avg Len 4.78 — DreamVLA(4.44) 대비 +0.34, 5/5 success rate 89.7%로 80% 장벽 돌파.

**Table 4 Ablation (CALVIN, without pretraining)**:

| Variant | Avg Len |
|---|---|
| MMaDA-VLA (w/o Pre-Training) | 4.56 |
| w/o World-Model | 4.08 (−0.48) |
| w/o Parallel Denoising | 4.38 (−0.18) |
| w/ Causal Attention | 4.49 (−0.07) |
| w/ Bidirectional Attention | 4.52 (−0.04) |

- World-model(goal image generation) 제거가 가장 큰 drop(−0.48), parallel denoising이 sequential 대비 +0.18, hybrid attention이 causal/bidirectional 단일 대비 각각 +0.07/+0.04.

**Table 5 Pre-training 효과**:
- LIBERO 94.5% → 98.0% (+3.5pp), CALVIN 4.56 → 4.78 (+0.22).

**Real-world (Fig. 3, AgileX Piper 6-DoF, 30 trials/task)**:
- Pick-and-Place, Stacking, Storage, Organizing 4개 task에서 MMaDA-VLA 83.3~93.3%. GR00T N1.6 baseline 56.7~70%. 각 task 300 demonstration으로 FT.

## 4. 한계 및 미해결 문제

- **Visual generation의 저해상도 한계**(§5.3, Fig. 6): 계산 효율성을 위해 compact image representation을 쓰다 보니 generated goal image가 fine-grained detail(gripper geometry, small object)에서 blurry. Task progression은 capture 하지만 pixel-level fidelity는 낮아 세밀한 affordance prediction에는 부족.
- **Iterative denoising latency**: 24 denoising step(Table 3)이 필요하며 KV cache로 완화하지만 여전히 autoregressive single-pass 대비 연산량이 크다. Real-time 고주파 control에는 inference engineering이 필수.
- **MMaDA-8B-Base 의존**: 본 method는 natively pre-trained large diffusion VLM을 전제로 한다. 향후 더 큰 스케일의 unified diffusion backbone이 나와도 MMaDA에 tight coupling 되어 있어 swap 비용이 크다. 또한 LIBERO 1.55%, CALVIN 1.87% 같은 low-ratio dataset이 pretraining mixture에 있어 downstream task adaptation은 주로 FT에 의존.

## 5. 총평

- **Novelty ★★★★★** — "Language·image·continuous action을 discrete token space에 unify하고 masked-denoising 단일 objective로 goal obs + action을 parallel 생성"이라는 formulation은 매우 참신하다. 기존 discrete diffusion VLA(LLaDA-VLA, UD-VLA)들은 autoregressive backbone에서 fine-tune 되었지만 MMaDA-VLA는 native pretrained diffusion backbone을 쓴다는 점이 핵심 차별화. Hybrid attention(intra-modal full + inter-modal causal)으로 action의 unordered nature와 modality 경계의 clean separation을 동시에 해결한 설계도 영리하다.
- **Practical impact ★★★★★** — LIBERO 98.0%와 CALVIN 4.78은 각각 새로운 SOTA로, CALVIN 5/5 89.7%는 처음으로 90%에 근접한 수치다. World model variant 전반(DreamVLA 4.44, Seer 4.28, VPP 4.33)을 큰 차이로 넘어서며, auxiliary module 없는 단일 backbone이 world-model 기능을 자연스럽게 흡수할 수 있음을 입증. Open-source code/model release도 긍정적. Real-world GR00T N1.6 대비 +20pp 수준의 gain은 재현 가능성 검증이 중요하지만 설계의 일반성에 대한 좋은 신호.

## 6. 예상 질문

- **Q1. Discrete diffusion이 discrete VLA(OpenVLA)와 다른 점은?** A. OpenVLA는 autoregressive left-to-right로 action token을 생성하여 action dimension 간에 인위적 ordering을 강제한다. MMaDA-VLA는 **masked token denoising**으로 action chunk 전체를 parallel refinement 한다. Hybrid attention의 intra-modal full attention이 action 7D 사이의 unordered interaction을 허용하고, iterative denoising이 global consistency를 개선한다. 결과적으로 Discrete Diffusion VLA(96.3%) 대비 LIBERO +1.7pp.
- **Q2. World model(goal obs generation)이 CALVIN에서 −0.48로 LIBERO보다 훨씬 큰 영향을 주는 이유는?** A. CALVIN ABC→D는 long-horizon(5-step) sequential task로 environment dynamics 모델링이 필수. LIBERO는 task당 단일 long instruction이라 goal obs prediction의 이득이 상대적으로 작다. 또한 CALVIN은 training data가 풍부해(1000K+ episode) visual generation이 pretrain 이득을 볼 수 있는 반면, LIBERO는 data 규모가 작아 visual prediction이 overfitting 위험을 증가시킬 수 있다(§4.3 저자 언급).
- **Q3. Parallel denoising이 sequential(goal 먼저 생성 후 action) 대비 왜 우월한가?** A. Sequential은 (a) goal image를 deterministic 하게 먼저 생성하므로 그 error가 action prediction에 누적되고, (b) action이 goal generation의 intermediate hidden state를 활용할 수 없다. Parallel은 매 denoising iteration에서 goal과 action이 서로의 partially-determined token을 condition으로 삼아 co-refine 한다. Ablation에서 4.38 vs 4.56으로 +0.18 gap을 보여 이를 실증.
- **Q4. KV cache refresh 전략(interval λ, selective ratio ρ)은 어떻게 결정하나?** A. Table 3의 refresh interval=6(즉 매 6 denoising step마다 refresh). Selective refresh는 cosine similarity 하위 ⌊ρn'⌋ token만 업데이트. 저자는 "only a small subset of response tokens changing substantially at each step"이라는 empirical observation에 기반(참조 [51]). 구체적 ρ 값은 본문에 명시 안 됨(hyperparameter section). Instruction part는 denoising 동안 완전 stable이므로 cache 재활용 전략이 가장 크게 효과.

<!-- VERIFIED: pdf -->
