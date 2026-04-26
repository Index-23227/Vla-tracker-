# dVLA: Diffusion Vision-Language-Action Model with Multimodal Chain-of-Thought

> **한 줄 요약**: MMaDA(discrete diffusion language model) + FAST 액션 토크나이저를 결합하여 시각·언어·액션을 단일 discrete diffusion 목적함수로 학습하는 8B 모델. Subgoal 이미지 + 텍스트 reasoning + 액션을 동시에 생성하는 multimodal CoT로 LIBERO 96.4% 달성, 실세계 bin-picking 등 4개 task에서 65% 평균. ICLR 2026 submission (Midea Group).

## 1. 배경 및 동기

- 기존 VLA 발전은 두 단계: (1) VLM을 단순 feature extractor로 사용, (2) image-text와 action trajectory를 co-train하면서 sub-step CoT 추가 (§1).
- 그러나 두 가지 한계: (i) co-training의 두 목적함수(next-token prediction vs continuous action denoising) 간 gradient conflict, (ii) auto-regressive VLM에 image generation을 통합하기 어려움 → cross-modal physical law 학습 실패.
- Discrete Diffusion Language Models (DLMs)이 LLM 대비 flexible speed-quality trade-off와 controllability를 제공한다는 최근 진전(LLaDA, MMaDA)에 주목.
- 저자들은 vision/language/action 세 modality를 모두 discrete token으로 통일하고 단일 discrete diffusion objective로 학습하면 cross-modal CoT 일관성이 자연스럽게 확보된다고 가정.

## 2. 방법론

### 2.1 통일된 확률적 정식화 (§3.1, Eq. 1)
- 입력 시퀀스 x = {o, l, s, o_goal, r, a_chunk}: 현재 observation, language instruction, robot state, subgoal 이미지(visual CoT), 텍스트 reasoning(textual CoT), action chunk.
- 모든 modality를 random masking 후 unmasked token 기반으로 masked token을 복원:
  L_unify(θ) = −E_{t,x_0,x_t}[(1/t) Σ_i I[x_t^i = MASK] · log p_θ(x_0^i | x_t)] (Eq. 1).
- 단일 목적함수로 visual reasoning + textual reasoning + action prediction을 모두 학습.

### 2.2 dVLA 아키텍처 (§3.2, Fig. 1)
- **Backbone**: MMaDA (Yang et al., 2025) — discrete diffusion으로 image generation + multimodal understanding 통합.
- **Vision tokenizer**: MAGViT-v2 (Yu et al., 2023). 압축률 16, codebook 8192. 256×256 → 256 tokens.
- **Text tokenizer**: LLaDA-8B-Instruct, vocab 126,464.
- **Action tokenizer**: FAST (Pertsch et al., 2025) — DCT + BPE로 continuous action을 discrete token화. Vocab 2048.
- **확장 vocab**: 126,464 → 136,704 (모든 modality 통합).

### 2.3 Multimodal CoT (§3.3, Fig. 2)
- 입력 sequence 형식:
  `[BOS][BOI]{image}[EOI] {text}{state} ×M [BOI]{Subgoal}[EOI]{Reasoning} [BOA]{action}...{action}[EOA] [EOS]`
- **Visual subgoal**: 미래 timestep t ∈ [0.9C, 1.1C]에서의 top-view image. CFG scale 3.5 사용.
- **Textual subgoal**: SEED-1.5VL로 3초 간격 video segmentation 주석 자동 생성. Bin picking 같은 long-horizon에서만 사용, simpler task는 inference 가속을 위해 omit.
- 추론 시 dVLA는 visual CoT(미래 모션을 시각화) + textual CoT(step-by-step instruction)를 평행 생성한 뒤 액션을 grounding.

### 2.4 가속 전략 (§3.4)
- **Prefix attention mask**: blockwise causal mask. Block 1 = [o, l, s] (입력), Block 2 = [o_goal, r, a_chunk]. 각 block 내부는 bidirectional, block 간은 causal. KV caching 부분 적용 가능.
- **KV caching from dLLM-Cache (Liu et al., 2025b)**: denoising step 간 K/V 변화가 작다는 관찰을 이용해 cache + 저빈도 refresh. Training-free plug-and-play.

### 2.5 학습 설정 (§3.5)
- Action chunk length C=5 (LIBERO), C=50 (real-world). 입력 이미지 256×256.
- 실세계: 1100 trajectories(Bin Picking 600, Open Box 100, Hang Cups 200, Pick&place 200) Franka 7-DoF + 2 ZED + 1 Realsense wrist.

## 3. 실험 결과

- **LIBERO (Table 1, 500 trials)**: dVLA Avg **96.4%**. Spatial 97.4 / Object 97.9 / Goal 98.2 / Long 92.2.
  - Continuous baseline 대비: π₀ 94.2 (+2.2), GR00T-N1 93.9 (+2.5), OpenVLA-OFT(Cont) 95.4 (+1.0).
  - Discrete baseline 대비: WorldVLA 81.8 (+14.6), CoTVLA 81.1 (+15.3, 본문은 27.4로 다른 평균 기준), Discrete Diffusion VLA 96.3 (+0.1).
  - **Vanilla dVLA (MMCoT 미사용)**: 89.8% — Multimodal CoT가 +6.6pp 기여.
- **실세계 (Table 2, 40 trials)**: dVLA **26/40 = 65%**. Bin Picking 7/10, Open Box 5/10, Hang Cups 7/10, Pick&place 7/10.
  - GR00T 19/40 (47.5%), Diffusion Policy 14/40 (35%), OpenVLA 14/40 (35%), Vanilla dVLA 21/40 (52.5%) 모두 능가.
  - MMCoT가 +12.5pp 기여 (52.5 → 65). Bin picking에서 cluttered 상황 grasping 정확도가 핵심 차이.
- **가속 효과 (Table 3)**: Full Attention 1.3 Hz → Prefix + KV caching **2.9 Hz** (~2.2× 가속). LIBERO Spatial 97.4→96.9, Object 97.9→97.3 (마진 손실). 실세계 bin picking 7/10 유지, Hang Cups 8/10 → 7/10. 1.5 Hz → 3 Hz.
- **Failure prediction 능력 (§4.2, Fig. 4)**: Visual CoT가 실제 실패 실행 결과(예: 그리퍼와 박스 모서리 사이 끼인 물체, 잘못된 방향 이동)와 정확히 일치. 통일된 학습으로 physical law를 capture.

## 4. 한계 및 미해결 문제

1. **추론 비용**: 가속 후에도 3 Hz로 reactive control(보통 10-25 Hz)에는 여전히 느림. MMaDA 백본의 iterative denoising 본질적 한계.
2. **CoT 데이터 의존**: Textual reasoning은 SEED-1.5VL로 자동 생성하지만 long-horizon task에만 적용. 자동 라벨링 품질이 검증되지 않으며 도메인마다 prompt 튜닝 필요. Visual subgoal은 top-view camera로 제한.
3. **실세계 평가의 통계적 한계**: Task당 10 trials, 총 40 trials. 신뢰 구간이 좁고 task 다양성도 4개로 한정됨. 또한 closed-source / not open-source(`open_source: false` in YAML)이라 재현·확장이 제한적.
4. **CoTVLA 대비 +27.4 보고와 Table 1 수치 불일치**: 본문은 dVLA 96.4 vs CoTVLA 69.0(평균)으로 +27.4pp 차이를 주장하지만 Table 1의 CoTVLA 행은 Spatial 81.13 / Object 87.5 / Goal 91.6 / Long 87.6 / Avg 69.0으로 Avg 계산이 다른 정의(아마 별도 protocol)에서 나온 듯하여 직접 비교 시 주의 필요.

## 5. 총평

- **Novelty: ★★★★☆** — Discrete Diffusion VLA(Liang et al., 2025)와 LLaDA-VLA가 선례를 만들었지만, **이미지 생성을 통한 visual CoT까지 통합한 unified diffusion VLA**는 dVLA가 처음. Failure prediction이 단일 학습 목적의 부산물로 emergent하게 나타나는 것은 흥미로운 발견.
- **Practical impact: ★★★☆☆** — LIBERO 96.4%는 SOTA권이지만 Discrete Diffusion VLA(96.3) 대비 +0.1pp로 압도적이지는 않음. 실세계 bin-picking에서 65%로 vanilla 대비 +12.5pp는 실용적이지만 코드 미공개로 후속 연구가 어려움.

Discrete diffusion이 VLA 백본으로 작동할 수 있음을 강하게 시연한 작업. 특히 visual CoT가 unsafe execution까지 예측한다는 점은 안전성 평가 측면의 미래 응용을 시사한다. 다만 inference 속도와 closed-source 정책이 광범위 채택에 걸림돌.

## 6. 예상 질문

| Q | A |
|---|---|
| Vanilla dVLA(89.8%)와 dVLA(96.4%)의 +6.6pp 차이는 정말 multimodal CoT 덕분인가? | Table 1에서 vanilla = MMCoT 미사용, dVLA = visual + textual CoT 모두 사용. 다른 모든 조건(MMaDA backbone, FAST tokenizer)은 동일하므로 차이는 CoT 구성요소에서 옴. 실세계에서는 +12.5pp(Table 2)로 더 큼. |
| Discrete Diffusion VLA(96.3)와 거의 같은데 dVLA의 차별점은? | Discrete Diffusion VLA는 액션 디코딩에만 discrete diffusion을 적용. dVLA는 vision/language/action 셋 모두를 단일 discrete diffusion으로 통합 + visual CoT subgoal 이미지 생성을 추가. 이로 인해 failure prediction 같은 emergent 능력 확보. |
| 왜 visual subgoal을 top-view로 제한했나? | §3.5에서 inference 가속 목적. 다른 view 추가는 token 수를 늘려 denoising 비용을 가중. 후속 연구에서 multi-view subgoal 확장이 필요. |
| Prefix Attention + KV caching의 정확도 손실은 수용 가능한가? | LIBERO Spatial -0.5pp, Object -0.6pp로 매우 작고, 추론 속도가 ~2.2× 빨라짐. 실세계 bin picking은 동일(7/10), Hang Cups는 -1 trial로 통계적 유의성 낮음. |

<!-- VERIFIED: pdf -->
