# MemoryVLA: Perceptual-Cognitive Memory in VLA Models

> **한 줄 요약**: 인지과학의 working memory + episodic memory 이중 시스템에 영감을 받아 Perceptual-Cognitive Memory Bank(PCMB)를 도입한 7B Cognition-Memory-Action 프레임워크. 메모리 조건부 diffusion transformer 액션 expert로 SimplerEnv-Bridge 71.9, Fractal 72.7, LIBERO 96.5, 실세계 long-horizon 83% (+26 vs CogACT) 달성. ICLR 2026.

## 1. 배경 및 동기

- 로봇 조작은 본질적으로 non-Markovian이지만 OpenVLA, π₀ 등 주류 VLA는 현재 관측만 사용하여 long-horizon temporal task에서 실패. Push Buttons처럼 push 전후 시각 차이가 거의 없는 경우 "이미 눌렀나?"를 판별하지 못함 (Fig. 1a).
- 단순 frame concatenation은 (1) self-attention의 quadratic 복잡도로 context length 제한, (2) 단일 frame로 사전학습된 VLM 분포와 misalign 두 문제를 야기 (§1).
- Cognitive science (Baddeley & Hitch, 1974; Tulving et al., 1972)는 인간이 working memory(전두 신경 활성)와 hippocampus 기반 episodic memory(verbatim + gist) 이중 체계를 사용한다고 제시. MemoryVLA는 이를 그대로 차용.
- 기존 temporal modeling(Octo, RoboFlamingo, TraceVLA, UniVLA)은 frame interleaving이나 latent compression에 의존해 fine-grained perceptual history를 잃거나 효과적으로 활용 못 함.

## 2. 방법론

### 2.1 전체 프레임워크 (§3.1, Fig. 2)
- Input: 단일 third-person RGB 224×224 + language instruction → Output: 16-step 7-DoF action sequence (Δx,y,z, Δθ_x,y,z, gripper).
- 3단계: (1) Vision-Language Cognition → working memory, (2) PCMB retrieval/fusion/consolidation → memory-augmented tokens, (3) memory-conditioned diffusion action expert.

### 2.2 Vision-Language Cognition Module (§3.2)
- 7B Prismatic VLM (Karamcheti et al., 2024) + Open-X Embodiment 추가 사전학습.
- DINOv2 + SigLIP 병렬 인코더 → SE-bottleneck perceptual compression → perceptual token p ∈ ℝ^{256×d_p}.
- Raw visual token + tokenized instruction → LLaMA-7B → EOS 위치의 cognitive token c ∈ ℝ^{1×d_c}.
- Working memory M_wk = {p, c}.

### 2.3 Perceptual-Cognitive Memory Module (§3.3, Fig. 3)
**Memory Bank**: M_pcmb = {m^per_i, m^cog_i}_{i=1..L}. 각 stream은 최대 L 엔트리.

**Retrieval (Eq. 5-6)**: Sinusoidal timestep PE를 더한 K^x = [m_1^x + TE(t_1); …]와 working memory query 간 scaled dot-product attention. 2-layer Transformer로 H^p, H^c 산출.

**Gate Fusion (Eq. 7-8)**: g^x = σ(MLP(concat[x, H^x])), x̃ = g^x ⊙ H^x + (1-g^x) ⊙ x. 학습된 gate로 현재 vs 과거 정보 비율 조절.

**Consolidation (Eq. 9, Fig. 3c)**: 용량 초과 시 인접 엔트리 cosine similarity 최대 쌍을 평균하여 merge. FIFO 대비 정보 보존.

### 2.4 Memory-Conditioned Diffusion Action Expert (§3.4)
- DiT(Peebles & Xie, 2023) + DDIM 10-step denoising. ~300M 파라미터.
- 매 denoising step에서 noisy action token + sinusoidal denoising timestep encoding을 cognitive c̃와 concat → cognition-attention layer (high-level guidance) → perception-attention layer (fine-grained details from p̃) → FFN.
- MSE loss로 학습, MLP head로 7-DoF 액션 출력. CFG scale 1.5.
- Training: 8×A100, batch=256, lr=2e-5.

## 3. 실험 결과

3 robots, 6 benchmarks, 150+ tasks, 500+ variations (Fig. 4):

- **SimplerEnv-Bridge (Table 1)**: Avg **71.9%**, +14.6 vs CogACT-Large(57.3). Spoon-on-Towel 75.0, Carrot-on-Plate 75.0, Stack-Cube 37.5, Eggplant-in-Basket 100.0. π₀-Beta(68.4)도 능가.
- **SimplerEnv-Fractal (Table 2)**: Overall **72.7%** (VM 77.7 / VA 67.7). VM Open/Close Drawer 84.7 (+12.9 vs CogACT). VA에서 Open/Close Drawer +24.9, Put in Drawer +11.7로 OOD에서 더 큰 이득.
- **LIBERO (Table 3)**: Avg **96.5%**. Spatial 98.4 / Object 98.4 / Goal 96.4 / Long-10 93.4 / Long-90 95.6. CogACT 93.2 +3.3, π₀ 94.2 능가. Wrist view나 proprioceptive state 없이 third-person RGB만으로 달성. (본 데이터셋 키지의 libero_avg는 96.50.)
- **Mikasa-Robo (Table 4)**: Avg **41.2%** (π₀ 29.4 +11.8). ShellGameTouch는 88로 최고 +41.0pp. Memory 기반 reasoning이 핵심인 벤치마크.
- **실세계 (Table 5)**: General 6 tasks **85%** (+9 vs CogACT 76), Long-horizon Temporal 6 tasks **83%** (+26 vs CogACT 57). Long-horizon에서 Seq Push Buttons +43, Change Food +38, Guess Where +32, Clean Table & Count +17. π₀와 OpenVLA(9%)를 큰 폭으로 능가.

**Ablation**:
- Table 6: Cognitive only 63.5, Perceptual only 64.6, **Both 71.9**. Memory length 4=67.7, **16=71.9**, 64=67.7 (적절한 capacity 필요).
- Table 7: Timestep PE 없이 69.8 → with 71.9. Add fusion 67.7 → **Gate 71.9**. FIFO 66.7 → **Token Merge 71.9**.

## 4. 한계 및 미해결 문제

1. **Memory length 16의 단기 horizon**: Table 6에서 length=64는 오히려 성능 저하. 매우 긴 horizon(분/시간 단위) task에서는 token merge가 핵심 정보 손실을 야기할 수 있음. 저자도 결론에서 "lifelong memory" 개발을 future work로 명시.
2. **Single third-person view 제약**: 모든 실험이 단일 RGB. Wrist view, depth, proprioception을 통합하면 더 나아질 가능성이 있으나 본 논문은 미적용. 정밀 manipulation(Stack Cube 37.5%)에서는 한계.
3. **PCMB는 task별 임시 메모리**: Episode 종료 시 초기화. Task 간 transfer 가능한 long-term memory가 부재 — 인지과학 motivation을 따르면 결국 hippocampal consolidation이 영구화되어야 하지만 그 단계는 미구현.

## 5. 총평

- **Novelty: ★★★★☆** — Perceptual + Cognitive dual stream memory bank와 retrieval-fusion-consolidation 3-step의 깔끔한 구성. 인지과학 motivation을 단순 비유 이상으로 architecture에 매핑한 점이 좋음.
- **Practical impact: ★★★★★** — 실세계 long-horizon에서 +26pp 향상은 매우 실용적 의미. SimplerEnv, LIBERO, Mikasa-Robo, 실세계 6+ 벤치마크 광범위 SOTA. 코드 + 모델 + log + 비디오 모두 공개(shihao1895/MemoryVLA).

VLA의 "현재만 본다"는 근본 한계를 인지과학 inspired 설계로 명료하게 푼 작업. 특히 Mikasa-Robo와 실세계 Push Buttons 같은 memory-critical task에서의 압도적 성능은 단순한 SOTA 갱신을 넘어서 비-Markovian 로봇 학습의 새로운 표준이 될 가능성.

## 6. 예상 질문

| Q | A |
|---|---|
| Perceptual + Cognitive 둘 다 필요한가? | Table 6 ablation: Cognitive only 63.5, Perceptual only 64.6, Both 71.9. 양쪽 모두 약 7-8pp 기여. 인지과학의 verbatim + gist 이중 표현 가설을 실증. |
| Memory consolidation의 token merge가 정보 손실을 야기하지 않나? | Table 7에서 FIFO 66.7 vs Token Merge 71.9. 인접 + 의미적으로 유사한 쌍만 merge하므로 redundant 정보 위주로 압축됨. |
| 왜 length=64에서 성능이 떨어지나? | 너무 긴 history는 attention noise를 늘리고, current decision-relevant 정보가 희석됨. 16이 sweet spot으로 보고됨. Long-horizon 작업에서도 16으로 충분한 이유는 token merge가 정보 압축을 잘 수행하기 때문. |
| LIBERO에서 wrist 없이 SOTA인 비결? | Memory bank가 historical perception을 누적하여 wrist view의 보완 기능을 일부 대신함. 또한 7B Prismatic VLM의 강력한 사전학습 + diffusion expert의 16-step 예측이 단일 view 한계를 보완. |

<!-- VERIFIED: pdf -->
