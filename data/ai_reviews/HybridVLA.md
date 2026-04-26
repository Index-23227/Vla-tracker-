# HybridVLA: Collaborative Diffusion and Autoregression in a Unified Vision-Language-Action Model

> **한 줄 요약**: Diffusion 기반 연속 action 생성과 next-token prediction 기반 autoregressive action 생성을 **단일 LLM(Llama-2 7B)** 안에서 collaborative training으로 통합하고, autoregressive token confidence로 두 action을 적응적 ensemble하여 RLBench 10 task SR 74%(OpenVLA 대비 +33%p, CogACT 대비 +14%p)를 달성한 PKU/BAAI/CUHK 논문(arXiv 2503.10631, 2025).

---

## 1. 배경 및 동기

- **Autoregressive VLA**(RT-2, OpenVLA, ManipLLM): VLM의 reasoning을 살리지만 action을 discrete bin으로 quantize → 연속성 손실.
- **Diffusion-based VLA**(π₀, CogACT, DiVLA, TinyVLA): 별도 diffusion head를 VLM 뒤에 부착 → VLM의 next-token prediction 능력을 활용하지 못하고 head는 internet-scale pretrain의 혜택 없음.
- 핵심 질문(Sec. 1): **"두 패러다임을 단일 LLM 안에서 mutual reinforce 시킬 수 있는가?"**

---

## 2. 방법론

### 2.1 Architecture (Sec. 3.1, Fig. 2)

- **VLM base**: Prismatic VLM(DINOv2 + SigLIP). HybridVLA(7B)는 LLAMA-2 7B, HybridVLA(2.7B)는 Phi-2 2.7B.
- **Action**: 7-DOF (single-arm) / 14-DOF (dual-arm) end-effector pose: $a = [\Delta x, \Delta y, \Delta z, \text{Roll}, \text{Pitch}, \text{Yaw}, 0/1]$.
- **Robot state**: discretize 대신 learnable MLP로 LLM embedding 공간(B×1×4096)에 투영 — Table 1의 Type 1 vs Type 3 비교에서 Diffusion SR 0.66 vs 0.61로 우월.

### 2.2 Collaborative Training Recipe (Sec. 3.2)

- **Token sequence**: `[multimodal] [robot_state] <BOD> [diffusion_tokens] <EOD> [autoregressive_tokens]`. <BOD>/<EOD> marker가 두 paradigm 경계를 명시.
- **순서가 중요**: diffusion → autoregressive로 배치(Type 1). 역순(Type 4)에서는 AR ground-truth가 diffusion conditioning으로 leak되어 SR 0.57로 하락(Table 1).
- **Hybrid objective**:
  $$L_{\text{hybrid}} = L_{\text{dif}} + L_{\text{ce}}, \quad L_{\text{dif}} = \mathbb{E}_{a,i,c}\Vert \epsilon - \epsilon_\pi(a^i_t, i, c)\Vert^2$$
  Classifier-free guidance는 stability를 위해 미사용(RDT-1B 따라).
- **Structured training**: 35 dataset, 760K trajectory, 33M frame으로 5 epoch pretrain → self-collected RLBench/real로 fine-tune. 8× A800 GPU, 300 epoch fine-tune.

### 2.3 Collaborative Action Ensemble (Sec. 3.3)

- 추론 시 두 action 모두 생성:
  - Diffusion: <BOD> 이후 DDIM **4 step** denoising(KV cache로 가속).
  - Autoregressive: <EOD> 이후 standard next-token prediction, mean confidence $c^{ar}_{t+1}$ 계산.
- Ensemble 규칙:
  $$a_{t+1} = \begin{cases} (a^d_{t+1} + a^{ar}_{t+1})/2 & \text{if } c^{ar}_{t+1} > \theta \\ a^d_{t+1} & \text{otherwise} \end{cases}, \quad \theta = 0.96$$
- 관찰(Sec. 3.3): 성공 sample의 80%에서 AR confidence > 0.96. Diffusion은 정밀 조작(Phone on base, Close laptop lid)에서, AR은 의미 추론(Water plants, Frame off hanger)에서 우월.

> ❓ **예상 질문**: 두 paradigm이 negative interfere하지 않는가?
> **답변**: Table 3에서 Ex1(diffusion-only with CTR) 0.66 vs Ex2(no CTR) 0.60, Ex3(AR-only with CTR) 0.62 vs Ex4 0.57 — CTR이 양쪽 모두를 개선, mutual reinforcement 정량 검증.

---

## 3. 실험 결과

### 3.1 RLBench 10 task Multi-task SR & Inference (Table 2)
- **HybridVLA (7B) Mean SR 0.74 @ 6.1 Hz** vs CogACT(7B) 0.60 @ 9.8 Hz vs π₀(2.6B) 0.55 @ 13.8 Hz vs OpenVLA(7B) 0.41 @ 6.3 Hz vs ManipLLM 0.38.
- **HybridVLA-dif (7B) 0.66 @ 9.4 Hz** — CogACT 대비 +6%p, π₀ 대비 +11%p로 collaborative training이 **diffusion-only inference도 향상**.
- 어려운 task에서 차이 큼: Wine at rack 0.50 vs CogACT 0.25, Frame off hanger 0.70 vs 0.35, Water plants 0.50 vs 0.25.

### 3.2 Real-world (Table 5)
- **Franka 단일팔 5 task**: HybridVLA 0.83, HybridVLA-dif 0.80 vs CogACT 0.61, π₀ 0.45. Pour water +35%p over π₀.
- **AgileX 양팔 5 task**: HybridVLA 0.71 vs π₀ 0.55. (CogACT는 multi-view 미지원으로 dual-arm 비교 불가.)

### 3.3 Ablation (Table 3)
- Ex0(full) 0.74, Ex1(dif + CTR) 0.66 vs Ex2(dif, no CTR) 0.60, Ex3(AR + CTR) 0.62 vs Ex4(AR, no CTR) 0.57.
- **Ex5(no LSP) 0.22로 급락** — 760K trajectory 사전학습이 결정적. RSE 제거 시 0.68.

### 3.4 Generalization (Table 4)
- 단일팔 Pick&Place에서 HybridVLA의 절대 drop이 모든 axis에서 CogACT보다 작음(예: Object -33% vs CogACT -43%, Background -11% vs -37%).
- 양팔 Lift ball: HybridVLA의 Object/Lighting drop은 -6%로 π₀(-8%/-15%) 대비 우월.

---

## 4. 한계 및 미해결 문제

1. **추론 속도**: HybridVLA(7B) 6.1 Hz — π₀(13.8)·HybridVLA-dif(9.4) 대비 느림. AR generation이 bottleneck(저자 명시, Sec. 5).
2. **DDIM step의 robustness**: 4 step으로 충분하다고 주장하나 더 dexterous task(plug insertion 등)에서 미검증.
3. **Confidence threshold θ=0.96** 고정 — task-adaptive tuning 미탐구(Appendix B에 일부 분석).
4. **AR-leakage 위험**: Type 4 token order에서는 GT leak — 향후 streaming/online 학습 시 유사 issue 가능.
5. **CogACT가 multi-view 미지원**으로 dual-arm 비교 baseline 부재 — 직접적 SOTA 비교는 π₀에 한정.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — 단일 LLM 내에서 diffusion + AR을 token-level로 통합하는 첫 robotic-specific 시도 |
| **Technical depth** | ★★★★★ — Token formulation 4-way ablation(Table 1), Hybrid loss, KV cache 가속 |
| **Experimental rigor** | ★★★★★ — RLBench 10 task + Franka 5 task + AgileX 5 task + 4 generalization axis |
| **Practical impact** | ★★★★☆ — Mean SR 0.74로 SOTA, 단 inference 6.1 Hz |

**강점**: 두 paradigm의 mutual reinforcement를 단일 LLM에서 정량 입증, dual-arm 확장. **약점**: 7B + AR로 인한 latency, multi-view CogACT 비교 부재.

---

## 6. 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | "Single LLM" 통합의 실질적 이득은? | Table 3에서 Ex1(0.66) vs Ex2(0.60), Ex3(0.62) vs Ex4(0.57) — CTR로 양쪽 paradigm 모두 +5%p 이상. CogACT(separate diffusion head, 0.60)와 HybridVLA-dif(0.66) 비교가 핵심 증거. |
| 2 | Token order(diffusion → AR)가 강제되는 구조의 한계? | 역순(Type 4)은 GT leak으로 SR 0.57(Table 1). Streaming/online에서는 별도 masking 전략 필요. |
| 3 | DDIM 4 step만으로 정밀도 충분한가? | Sec. 3.3에서 step 수에 무관한 성능 보고. Appendix B.2에 추가 분석. 그러나 plug insertion 같은 sub-mm 정밀 task는 미검증. |
| 4 | AR confidence θ=0.96의 generic threshold인가? | 80%+ 성공 sample이 0.96 초과(Sec. 3.3); task별 분포 차이는 Appendix에 분석. |
| 5 | LSP 없이 0.22 — 사전학습 의존성? | 760K trajectory, 33M frame 필수로 소규모 lab reproduce가 약점. |

<!-- VERIFIED: pdf -->
