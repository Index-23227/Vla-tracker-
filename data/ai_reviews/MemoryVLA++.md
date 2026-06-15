# MemoryVLA++: Temporal Modeling via Memory and Imagination in Vision-Language-Action Models

> **한 줄 요약**: MemoryVLA의 Perceptual-Cognitive Memory Bank(과거)에 Stable Video Diffusion 1.5B 기반 latent-space imagination(미래)을 결합한 full temporal VLA. LIBERO 5-suite 평균 98.4%, SimplerEnv 73.9%, CALVIN ABC→D 4.29, Mikasa-Robo 44.4%, Libero-Plus zero-shot 73.1% / SFT 82.7%를 달성.

---

## 1. 배경 및 동기

### 기존 VLA의 한계
- OpenVLA, π₀, CogACT 등 대부분의 VLA는 **현재 관측에만 의존**하는 reactive policy → long-horizon, temporally-dependent task에서 실패.
- 단순히 N개 history frame을 입력에 concat하는 방법: self-attention quadratic cost + temporal redundancy.
- Future video prediction(SuSIE, UniPi, ForeAct 등): pixel-level 예측이 비싸고, control-irrelevant 디테일까지 학습하며 예측 오차가 액션으로 전파됨.

### 인지과학적 영감
- Working memory(작업 기억) ↔ Hippocampal episodic memory(일화 기억) ↔ Internal model(예측 모델, 소뇌).
- 인간은 working memory가 episodic memory에서 decision-relevant context를 인출하고, internal model로 미래 상태를 예측한 뒤 cerebellum이 motor execution을 조율.

### 핵심 질문
- **과거 메모리만으로는 부족한 imagination-dependent task(예: 동적 컨베이어 grasping)에서 latent-space future modeling이 효과적인가?**
- **Pixel decoding 없이 partial denoising latent만 사용해도 control-useful한 future cue를 뽑을 수 있나?**

📌 [Figure 1 삽입] — Button Pressing(memory 필요) + Dynamic-Conveyor Grasping(imagination 필요) 예시.

---

## 2. 방법론 심층 분석

### 2.1 전체 아키텍처 (Fig. 3)

```
Current RGB + Language
        ↓
   7B Prismatic VLM
   (DINOv2 + SigLIP + LLaMA-7B)
        ↓
[Perceptual tokens p ∈ R^{Np×dp}] + [Cognitive token c ∈ R^{1×dc}]   ← Working Memory M_wk
        ↓
   ┌────────────────────────────┐    ┌──────────────────────────┐
   │ PCMB Retrieval (×2 layers) │    │ SVD 1.5B World Model     │
   │  cross-attn + timestep PE  │    │  partial denoising (1step)│
   │  → H^p, H^c                │    │  → multi-scale {U_s}     │
   └────────────┬───────────────┘    └──────────┬───────────────┘
                │                                │
        Gate Fusion (σ-MLP)                FPN + Imagination Former
                │                                │
        [p̃, c̃] memory-aware             z_img ∈ R^{K×Nq×dp}
                │                                │
                └────── Memory-Guided ───────────┘
                       Cross-Attn + Gate Fusion
                                │
                     [p̄, c̃]  = F_temp (full temporal-aware)
                                ↓
                Full Temporal-Aware Action Expert (DiT, ~300M)
                  cognition-attn + perception-attn
                  DDIM 10 steps + CFG 1.5
                                ↓
                     7-DoF action sequence (chunk=16)
```

### 2.2 Perceptual-Cognitive Memory Bank (PCMB)

| 컴포넌트 | 정의 | 역할 |
|---|---|---|
| Perceptual entry m^p_i | R^{Np×dp} | 저수준 visual detail 보존 |
| Cognitive entry m^c_i | R^{1×dc} | 고수준 semantic 요약 |
| Capacity L | 메모리 길이 (default 16 in Libero-Long-90; 256 in real-temporal) | 시간 순서로 저장 |
| Retrieval | cross-attention (timestep PE 사용, 2-layer Transformer) | working memory가 query |
| Gate Fusion | g^x = σ(MLP(concat[x, H^x])); x̃ = g⊙H + (1-g)⊙x | 적응적 결합 |
| Consolidation | 가장 유사한 인접 쌍(코사인) 평균 merge | redundancy-aware 압축 |

> ❓ **예상 질문**: PCMB가 단순 FIFO buffer와 어떻게 다른가?
> **답변**: (1) two-stream(perceptual + cognitive) 분리 저장, (2) 길이 초과 시 인접 유사 쌍 merge로 정보 손실 최소화, (3) cross-attention retrieval로 모든 step이 query에 따라 다른 가중치를 받음. 단순 FIFO와 달리 의미적 압축이 일어난다.

### 2.3 World-Model-Based Imagination (Fig. 5)

- **Backbone**: Stable Video Diffusion 1.5B (대규모 인터넷 영상 사전학습).
- **Adaptation**: manipulation video로 SVD를 v-prediction MSE로 추가 학습 (CLIP-encoded instruction을 spatio-temporal UNet에 cross-attn injection).
- **Latent imagination 핵심 트릭**: partial denoising → multi-scale intermediate UNet feature {U_s}^S_{s=1} 추출 → FPN으로 aggregate → K-step latent token z ∈ R^{K×Nz×dp}.
- **Imagination Former**: learnable query q ∈ R^{K×Nq×dp}가 z̄에 query-based spatial attention → temporal attention → FFN. 결과 z_img ∈ R^{K×Nq×dp}.
- **Memory-Guided Integration**: memory-augmented perceptual p̃가 z_img를 cross-attend → FFN → sigmoid gate g로 p̃와 h를 fusion: `p̄ = g⊙p̃ + (1-g)⊙h`.
- **Pixel 디코딩 X**: SVD가 latent에서 멈춤 → pixel-level FVD/LPIPS는 reference용으로만 사용.

> ❓ **예상 질문**: Pixel 예측 없이 latent만으로 decision-relevant인지 어떻게 보장?
> **답변**: VLA training 시 world model은 frozen(Tab. VIII(c): freeze 44.4 > unfreeze 42.8). 즉 SVD pretrained prior를 보존하면서 downstream gradient는 imagination former와 integration module만 통과. 따라서 latent가 action MSE를 줄이는 방향으로 정제된다. Tab. IX에서 imagination quality(PSNR 20.36 avg, FVD 105)가 어느 정도 보장됨을 확인.

### 2.4 Full Temporal-Aware Action Expert (Fig. 6)

- **Diffusion Transformer (DiT) + DDIM**: noisy action A_τ에서 시작해 점진적 denoising.
- **Cognition Attention**: `h_c = CogAttn([c̃ + TE(τ); A_τ])` — cognitive token이 high-level semantic guidance.
- **Perception Attention**: `Â_0 = FFN(PerAttn(h_c, p̄, p̄))` — perceptual tokens가 fine-grained visual detail 제공.
- **출력**: 7-DoF continuous action (Δx, Δy, Δz, Δθ_x, Δθ_y, Δθ_z, gripper). Dual-arm은 concat.
- **Action expert 크기**: ~300M (논문 명시).

---

## 3. 데이터 전략

| 벤치마크 | 학습 데이터 | 학습 step | World model step |
|---|---|---|---|
| LIBERO | 50 demo/task × 4 suites + Long-90 별도 | 60k(4 suites) + 30k(Long-90) | 40k |
| SimplerEnv | BridgeData-v2 (~60k WidowX teleop) | 50k | 40k |
| Mikasa-Robo | 250 demo/task × 5 tasks 합학습 | 20k | 20k |
| CALVIN | ABC→D (A,B,C 학습, D 평가) | 60k | VPP 기반 |
| Libero-Plus | 4 standard suites(zero-shot) / mixed(SFT) | 60k | 40k |
| Real-robot | Franka/WidowX/Dual-ARX5 teleop demo (50-300) | 5k-30k task별 | 20k |

> ❓ **예상 질문**: World model adaptation을 task별로 따로 하나?
> **답변**: 그렇다. Libero/SimplerEnv/Mikasa/Calvin/real-robot 각각 별도 학습 step 사용. CALVIN은 VPP[66] 결과를 그대로 사용. 일반화된 single world model이 아님.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Backbone VLM | Prismatic 7B (DINOv2 + SigLIP) |
| LLM | LLaMA-7B (default); 일부 ablation에서 Qwen2.5 |
| Action expert | DiT ~300M |
| World model | Stable Video Diffusion 1.5B (frozen at VLA training) |
| Hardware | 8× NVIDIA A100 또는 H20 |
| Global batch | 208-256 (26-32/GPU, FSDP) |
| Learning rate | 2×10⁻⁵ |
| Inference DDIM | 10 steps, CFG 1.5 |
| Action chunk | 16 |
| Inference latency (RTX 4090) | 0.241 s (66.4 Hz) — baseline 0.187 s, MemoryVLA 0.194 s |
| GPU memory | 21.7 GB |

---

## 5. 실험 설계 및 평가 프로토콜

- **시뮬레이션 5종**: Libero (general manipulation, 5 suites 130 tasks), SimplerEnv (real-to-sim general), Mikasa-Robo (memory-dependent), Calvin (long-horizon language), Libero-Plus (robustness, 7 OOD axes).
- **실세계 3 카테고리**: general manipulation (6 tasks), long-horizon memory (6 tasks), long-horizon imagination (5 tasks: Conveyor Pick Low/Mid/High, Conveyor Scan-Pick, Bag Pack & Zip).
- **로봇 플랫폼 3종**: Franka, WidowX, Dual-ARX5.
- **총 task 수**: ~200 task with diverse variations.
- **Baseline**: CogACT가 주 baseline, 부재 시 reported best.
- **MemoryVLA++는 real-robot에서 imagination-dependent에만 평가**(나머지는 MemoryVLA 점수).

---

## 6. 실험 결과 심층 분석 (PDF Table 직접 인용)

### 6.1 LIBERO (Table I)

| Method | Spatial | Object | Goal | Long-10 | Long-90 | Avg |
|---|---|---|---|---|---|---|
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | — | 94.2 |
| GR00T-N1.5 | 94.4 | 97.6 | 93.0 | 90.6 | — | 93.9 |
| UniVLA | 96.4 | 98.0 | 90.8 | 89.6 | — | 93.7 |
| CogACT (baseline) | 97.2 | 98.0 | 90.2 | 88.8 | 92.1 | 93.2 |
| MemoryVLA | 98.4 | 98.4 | 96.4 | 93.4 | 95.6 | 96.5 (+3.3) |
| **MemoryVLA++** | **99.8** | **100.0** | **98.2** | **96.0** | **97.8** | **98.4 (+5.2)** |

- 모든 suite에서 best. Long-10 +7.2pp over CogACT.
- Object 100.0% — 완벽 수렴.

### 6.2 SimplerEnv (Table II) — BridgeData-v2 → Real-to-Sim

| Method | Spoon | Carrot | Stack Cube | Eggplant | Avg |
|---|---|---|---|---|---|
| π₀ | 84.6 | 55.8 | 47.9 | 85.4 | 68.4 |
| CogACT | 58.3 | 45.8 | 29.2 | 95.8 | 57.3 |
| MemoryVLA | 75.0 | 75.0 | 37.5 | 100.0 | 71.9 (+14.6) |
| **MemoryVLA++** | **83.3** | **66.7** | 45.8 | **100.0** | **73.9 (+16.6)** |

- CogACT 대비 +16.6pp. Tab. XI에서 Qwen2.5+Dexbotic 사전학습으로 교체 시 **84.4%**까지 향상(별도 ablation).

### 6.3 Mikasa-Robo (Table III) — Memory-Dependent

| Method (frames) | SGT | IM | RC3 | RC5 | RC9 | Avg |
|---|---|---|---|---|---|---|
| π₀ (1) | 33 | 42 | 35 | 22 | 15 | 29.4 |
| OpenVLA-OFT (1) | 47 | 14 | 59 | 16 | 6 | 28.4 |
| MemoryVLA (1) | 88 | 24 | 44 | 30 | 20 | 41.2 |
| **MemoryVLA++ (1)** | **97** | **40** | 50 | 19 | 16 | **44.4 (+15.0)** |

- ShellGameTouch에서 +50pp(prev best 47 → 97). 메모리 정보 retrieval이 결정적인 task.
- RC5/RC9에서는 MemoryVLA보다 떨어짐 — color sequence 길어질수록 imagination이 오히려 distractor 가능성.

> ❓ **예상 질문**: RC5(19), RC9(16)는 왜 떨어졌나?
> **답변**: RememberColor 시리즈는 순수 sequential memory recall이라 imagination이 도움이 안 될 뿐 아니라 latent fusion이 noise로 작용했을 수 있다. PCMB만 강화한 MemoryVLA(30, 20)가 오히려 더 나음. Imagination이 모든 temporal task에 universal하지 않다는 증거.

### 6.4 CALVIN ABC→D (Table IV)

| Method | 1 | 2 | 3 | 4 | 5 | Avg Len |
|---|---|---|---|---|---|---|
| OpenVLA-OFT | 89.1 | 79.4 | 67.4 | 59.8 | 51.5 | 3.47 |
| π₀ | 93.8 | 85.0 | 76.7 | 68.1 | 59.9 | 3.92 |
| MemoryVLA | 94.8 | 87.4 | 81.4 | 75.9 | 69.4 | 4.09 (+0.84) |
| **MemoryVLA++** | **95.6** | **90.2** | **85.7** | **81.7** | **76.1** | **4.29 (+1.04)** |

- 5-task chaining에서 76.1% — long-horizon consistency 강력. CogACT 기준 +1.04 길이.

### 6.5 Libero-Plus (Table V) — 7-axis OOD

| Setting | Camera | Robot | Language | Light | Background | Noise | Layout | Avg |
|---|---|---|---|---|---|---|---|---|
| **Zero-shot** | | | | | | | | |
| OpenVLA-OFT | 55.6 | 21.7 | 81.0 | 92.7 | 91.0 | 78.6 | 68.7 | 67.9 |
| MemoryVLA | 42.7 | 44.9 | 84.4 | 92.8 | 95.0 | 62.1 | 84.7 | 70.2 |
| **MemoryVLA++** | 36.4 | **68.9** | **88.7** | **93.8** | 90.6 | 63.5 | 83.8 | **73.1** |
| **SFT** | | | | | | | | |
| OpenVLA-OFT | 92.8 | 30.3 | 85.8 | 94.9 | 93.9 | 89.3 | 77.6 | 79.6 |
| MemoryVLA | 91.4 | 48.6 | 79.4 | 95.2 | 95.3 | 94.0 | 75.7 | 81.9 |
| **MemoryVLA++** | **96.8** | **49.7** | 71.0 | **96.6** | **97.0** | **96.0** | **78.6** | **82.7** |

- Robot init 변경에 매우 강함(zero-shot 68.9, prev best 46.2). Camera는 zero-shot에서 36.4로 약함.

### 6.6 실세계 (Table VI)

- **General(6 tasks)**: MemoryVLA 85% (CogACT 76, +9). MemoryVLA++ 평가 X.
- **Long-horizon memory(6 tasks)**: MemoryVLA 83% (CogACT 57, +26). MemoryVLA++ 평가 X.
- **Long-horizon imagination(5 tasks)**: MemoryVLA++ 77% (+28 over baseline per abstract; baseline value table에 명시 안 됨).

> ❓ **예상 질문**: MemoryVLA++가 일반/메모리 카테고리에서 평가 안 된 이유?
> **답변**: 논문 명시 — "MemoryVLA++ is evaluated only on imagination-dependent tasks, where future imagination is explicitly required". 즉 imagination 모듈의 marginal contribution이 보장되는 영역에서만 비교. 다른 카테고리에서 MemoryVLA 대비 추가 이득이 있는지는 검증되지 않음 (논문의 약점).

---

## 7. Ablation 분석

### 7.1 Memory Length (Table VII(a))

| Length | SimplerEnv | Long-90 | Real-Temporal |
|---|---|---|---|
| Small (4/8/64) | 67.7 | 94.2 | 78 |
| **Default (16/16/256)** | **71.9** | **95.6** | **84** |
| Large (64/32/512) | 67.7 | 95.6 | 81 |

- Default(중간)가 최적. 너무 길면 outdated context가 noise.

### 7.2 Imagination Ablation (Table VIII) — Mikasa-Robo

| Denoise step | Avg | | Horizon K | Avg |
|---|---|---|---|---|
| 1 | 44.4 | | 4 | 43.4 |
| 3 | 44.6 | | 8 | 43.8 |
| 5 | 43.6 | | **16** | **44.4** |

| WM update | Avg | | Integration | Avg |
|---|---|---|---|---|
| w/o Freeze | 42.8 | | Add | 41.2 |
| **w/ Freeze** | **44.4** | | **Mem-Guided** | **44.4** |

- **단 1-step denoising만으로도 충분**(44.4 ≈ 3-step 44.6). Pixel-perfect generation 불필요.
- **World model freeze 필수**. 같이 학습하면 SVD prior가 망가짐.
- **Memory-guided 통합이 +3.2pp** over naive add — gate fusion 가치.

### 7.3 Backbone 교체 (Table XI)

| Backbone | Pretraining | SimplerEnv | Libero Avg |
|---|---|---|---|
| LLaMA2 | CogACT | 71.9 | 96.7 |
| Qwen2.5 | Dexbotic | **84.4** | **97.0** |

- SimplerEnv에서 +12.5pp — 더 강력한 VLM pretraining이 latent imagination에 큰 leverage.

### 7.4 Imagination Quality (Table IX)

| Dataset | PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ | EPE↓ |
|---|---|---|---|---|---|
| Libero | 20.26 | 0.820 | 0.182 | 101.93 | 0.5104 |
| Mikasa-Robo | 26.39 | 0.838 | 0.174 | 189.38 | 0.1540 |
| Calvin | 22.22 | 0.833 | 0.185 | 29.69 | 0.2049 |
| Avg (all) | 20.36 | 0.794 | 0.216 | 105.00 | 0.8829 |

- Bridge/Real 데이터는 PSNR ~17 수준으로 낮음 — 어디까지나 latent prior 역할.

---

## 8. 관련 연구 비교

| 모델 | Past Memory | Future Imagination | Method | LIBERO Avg |
|---|---|---|---|---|
| OpenVLA / π₀ / CogACT | ✗ | ✗ | Current observation only | 75.9 / 94.2 / 93.2 |
| TraceVLA / 4D-VLA | △ (frame stack) | ✗ | History concat | 74.8 / 92.2 |
| VPP / Mimic-Video / Seer | ✗ | △ (latent or video) | Future visual feat | — |
| DreamVLA | ✗ | △ | Visual subgoal prediction | 92.6 |
| MemoryVLA (이전 버전) | ✓ (PCMB) | ✗ | Working-Episodic Memory | 96.5 |
| **MemoryVLA++** | ✓ (PCMB) | ✓ (SVD latent) | **Full Temporal Modeling** | **98.4** |

### 핵심 차별점
- **유일하게 memory + imagination을 모두 명시적으로 modeling**.
- **Latent-space imagination**: pixel decoding 비용 회피, 1-step denoising으로 충분.
- **Memory-guided integration**: imagination의 noise를 memory가 filter.

---

## 9. 한계 및 미해결 문제

### 방법론적 한계
1. **Real-robot에서 MemoryVLA++가 imagination-dependent task만 평가**. General/memory 카테고리에서는 이전 버전(MemoryVLA)만 비교됨 → imagination 모듈이 generic case에서 손해를 끼치는지 불명.
2. **RC5/RC9 회귀**: Mikasa-Robo의 일부 순수 sequential-recall task에서 MemoryVLA보다 떨어짐. Imagination이 distractor가 되는 case 존재.
3. **Camera OOD 약함**(zero-shot 36.4): viewpoint shift에 취약. SVD prior가 카메라 일관성에 의존하는 가능성.
4. **World model task별 별도 adaptation 필요** (Libero, SimplerEnv, Mikasa, Calvin 각각). Generalist world model 부재.
5. **Inference cost 증가**: RTX 4090에서 0.241s (66Hz). 일부 고주파 제어에는 부담.
6. **MemoryVLA[26] 대비 ablation 부족**: 논문은 MemoryVLA 대비 imagination 모듈 추가만 강조 — PCMB와 imagination이 독립적인지(orthogonality) 직접적 분리 ablation 없음.

### Reporting 이슈
- SimplerEnv abstract에서 "74.0%"이라 했지만 Table II는 73.9%. Tab. XI의 84.4%는 다른 backbone 결과.

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Latent-space partial denoising으로 control-relevant future를 효율적으로 추출. Memory-guided integration의 design이 깔끔. |
| **Technical depth** | ★★★★☆ — Imagination former, gate fusion, redundancy-aware consolidation 등 컴포넌트별 ablation 체계적. |
| **Experimental rigor** | ★★★★☆ — 5 sim + 3 real category, ~200 task, 3 robot platform. CogACT를 일관된 baseline으로 사용. 단, MemoryVLA++의 real-robot 평가가 1 카테고리에 한정. |
| **Practical impact** | ★★★★☆ — 66Hz @ RTX 4090로 실 시스템 가능. World model freeze로 deployment 단순화. |
| **Writing quality** | ★★★★☆ — 인지과학 비유부터 수식 전개까지 일관성 있음. |

**강점**: VLA에 memory + imagination을 통합한 가장 완전한 framework. Latent partial denoising으로 future modeling의 cost 문제 해결. 5 simulation benchmark + real-robot 3 카테고리에서 광범위 검증.

**약점**: MemoryVLA(자기 prior)와의 imagination 모듈 ablation이 일부 task에서 negative(RC5/RC9). Real-robot에서 MemoryVLA++의 범용성 검증 부족. World model이 task-specific adaptation 필요.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|---|---|
| 1 | 1-step partial denoising으로 future를 잡는 게 신뢰 가능한가? | Tab. VIII(a): 1-step 44.4 ≈ 3-step 44.6. SVD의 multi-scale UNet feature가 이미 풍부한 spatio-temporal prior를 담고 있어 full denoising 불필요. Tab. IX의 PSNR/SSIM은 reference용. |
| 2 | World model을 freeze해야 한다는 의미는? | Tab. VIII(c): freeze 44.4 vs co-train 42.8. SVD pretrained prior가 robust한 visual dynamics를 제공하므로 policy gradient가 이를 망가뜨리면 안 됨. 다만 task별 adaptation은 별도로 수행. |
| 3 | RC5/RC9에서 회귀한 이유? | Pure sequential color recall은 visual imagination이 도움 X. PCMB만으로 충분한 task에서 imagination latent가 fusion noise로 작용. Imagination 모듈의 task-specific gain. |
| 4 | Memory-guided integration이 왜 add보다 큰 차이? | Tab. VIII(d): 41.2 vs 44.4 (+3.2). Imagination latent에는 control-irrelevant content가 섞여 있고, memory-augmented perceptual token이 query로 작용해 decision-relevant fragment만 골라 cross-attend → sigmoid gate로 fusion 정도 조절. |
| 5 | PCMB consolidation의 정보 손실은? | 가장 유사한 인접 쌍만 평균 merge → cosine sim이 높은 redundant pair만 압축, 의미적으로 다른 entry는 보존. Capacity 16-512 사이에서 가장 강력 (Tab. VII(a)). |
| 6 | Latency 0.241s가 실 시스템에 충분한가? | 66.4Hz @ RTX 4090, 53.2Hz @ H20. 일반 manipulation(10-30Hz 제어)에는 충분. 고주파 control(>100Hz)에는 부담. GPU memory 21.7GB도 deployment에서 고려 필요. |
| 7 | Qwen2.5+Dexbotic 결과(SimplerEnv 84.4)는 어떻게 봐야? | LLaMA2(71.9) 대비 +12.5pp. VLM pretraining이 latent imagination에 크게 leverage됨. 논문 main number는 LLaMA2 73.9이지만, 실제 best는 84.4. |
| 8 | Real-robot에서 imagination-dependent에만 평가하는 게 fair한가? | 논문 명시 — imagination이 explicit하게 필요한 task에서만 비교. 다른 카테고리에서 MemoryVLA++ 대 MemoryVLA 직접 비교가 없어 imagination 모듈의 범용성은 검증 안 됨. 향후 연구 과제. |
| 9 | Camera OOD(zero-shot 36.4)가 약한 이유? | SVD가 internet video 사전학습이라 카메라 viewpoint에 어느 정도 robust하지만, Libero-Plus의 camera variation은 학습 데이터에 없는 distribution shift. SFT(96.8)에서는 거의 완벽 → adaptation으로 해결 가능. |
| 10 | MemoryVLA 대비 어떤 task에서 imagination이 가장 결정적? | (1) Mikasa SGT: +9pp (88→97); (2) Calvin 5-task: +6.7pp (69.4→76.1); (3) Real conveyor task: +28pp over baseline. 즉 future state evolution이 의미 있는 task에서 가장 두드러진 향상. |

---

<!-- VERIFIED: pdf -->
