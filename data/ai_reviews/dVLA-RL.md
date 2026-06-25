# dVLA-RL: Reinforcement Learning over Denoising Trajectories for Discrete Diffusion Vision-Language-Action Models

> **한 줄 요약**: Discrete Diffusion VLA(dVLA)의 K-step masked denoising 과정을 Markov chain으로 unroll하여 intractable한 marginal action probability 대신 **denoising trajectory의 joint probability**를 PPO objective로 직접 최적화함으로써, native dVLA에 최초로 RL을 적용하고 LIBERO 99.7% SoTA + RoboTwin 2.0 VLA-기반 SoTA(SFT 대비 +30.6pt)를 달성한 프레임워크.

---

## 1. 배경 및 동기

### VLA의 action 생성 패러다임 4종
1. **Autoregressive / parallel decoding** (OpenVLA, OpenVLA-OFT): token-wise / 1-step 출력 → exact likelihood가 분석적으로 얻어지므로 표준 policy gradient 직접 적용 가능.
2. **Continuous diffusion / flow-matching** (π0, π0.5): likelihood approximation, variational objective, 또는 ODE→SDE 변환으로 RL 적용.
3. **Discrete Diffusion VLA(dVLA)** (MM-ACT, LLaDA-VLA류): vision·language·action을 단일 discrete token space에 통합하고 **masked generative modeling**으로 생성. SFT에서 continuous 대비 경쟁력 입증되었으나 RL은 미개척.

### 핵심 난점: marginal probability의 intractability
- dVLA의 최종 action a_t = x_0는 K-step stochastic denoising (x_K → x_{K-1} → … → x_0)을 거쳐 생성.
- exact marginal π_θ(a_t | s_t) = Σ_paths p(x_K) Π pθ(x_{k-1} | x_k, s_t)은 가능한 intermediate path가 조합적으로 폭발 → 계산 불가.
- 단순 last-step approximation은 intermediate transition을 무시하므로 rollout 시 실제 실행된 generation path와 mismatch → 본 논문이 Figure 3에서 학습 불안정/plateau로 실증.

### 본 논문의 질문
**"intermediate denoising state들을 hidden variable로 marginalize 하지 않고, 실제로 sample된 단 하나의 trajectory에 대한 joint probability를 PPO ratio로 쓰면 어떻게 되는가?"**

---

## 2. 방법론 심층 분석

### 2.1 Trajectory-level joint probability (Sec. 4.1, Eq. 4-5)

각 denoising step k에서 transition x_k → x_{k-1}을 두 메커니즘으로 분해:

```
P_θ(x_{k-1} | x_k, s_t) = P_unmask(U_k | p_θ, k) · Π_{i ∈ U_k} p_θ(x^i_{k-1} | x_k, s_t)
```

- **U_k**: scheduler가 step k에서 unmask할 token index 집합. Gumbel-TopK로 선택, network confidence p_θ에 의존.
- **p_θ(x^i_{k-1} | x_k, s_t)**: 선택된 위치 i의 token 생성 확률 (categorical softmax).
- 선택 안 된 위치 (j ∉ U_k)는 deterministic하게 유지 (P=1) → product에서 자동 소거.

전체 trajectory τ = (x_K, …, x_0)의 log probability:

```
log π_θ(τ | s_t) = Σ_{k=K}^{1} [ log P_unmask(U_k | p_θ, k) + Σ_{i ∈ U_k} log p_θ(x^i_{k-1} | x_k, s_t) ]
```

### 2.2 Implicit unmasking optimization (Sec. 4.2, Eq. 6)

Unmasking term을 명시적으로 backprop하면 두 가지 문제:
1. **PPO trust region 하에서 gradient 무의미**: θ ≈ θ_old이므로 선택 확률이 거의 invariant.
2. **Gumbel-TopK의 non-differentiable sorting**: backprop이 high-variance impulse gradient를 만들어 PPO clip을 violate, 학습 destabilize.

→ Unmasking process를 **intrinsic non-differentiable system dynamic**으로 취급하고 token-generation term만 gradient에 사용. 추가로 이미 decoded된 token도 제외하고, **[MASK] → specific token으로 phase transition한 부분집합 M_k ⊆ U_k**에만 gradient 흐름:

```
L^{dVLA-RL}(θ) = E_τ [ Σ_{k=K}^{1} Σ_{i ∈ M_k} log p_θ(x^i_{k-1} | x_k, s_t) · A_t ]
```

A_t는 **chunk-level advantage** (value head로 추정, 모든 denoising step에 uniform weight).

### 2.3 Implicit scheduler 학습
명시적 loss는 M_k에만 있으나 network confidence p_θ는 unmasking과 token generation **양쪽에서 공유**됨. token prediction이 정확해지면 contextual representation이 sharpen → Gumbel-TopK가 더 uncertain region을 정확히 target. **즉 explicit gradient estimator 없이도 scheduler가 implicitly co-adapt**.

### 2.4 Hybrid denoising-step strategy (Sec. 5.4)

Trajectory-level formulation의 부산물: K값이 다른 denoising도 같은 probability framework에서 동일하게 처리됨.
- 1-step decoding: parallel decoding과 수학적으로 동일 (degenerate path).
- 2/4-step: 더 긴 trajectory, 더 풍부한 iterative refinement.

**Hybrid dVLA-RL**: 각 task마다 SFT checkpoint의 초기 SR을 보고 최소 denoising step을 자동 할당. 쉬운 task → 1-step (싸고 빠름), 어려운 task → 2/4-step (refinement quality 확보, 초기 positive feedback 확보로 sparse reward에서 학습 가속).

---

## 3. 데이터 전략

| 항목 | 내용 |
|---|---|
| **Benchmarks** | LIBERO (Spatial/Object/Goal/Long), RoboTwin 2.0 (8개 선택 task, bimanual) |
| **SFT scale** | LIBERO: 500 demo/suite · RoboTwin 2.0: 1,000 demo/task (SimpleVLA-RL과 동등) |
| **SFT 차이점** | SimpleVLA-RL은 per-task / per-suite, dVLA-RL은 **multi-task joint training** 단일 checkpoint |
| **RL rollout** | LIBERO: 512 episode/task, RoboTwin 2.0: 64 episode/task, 1,000 training scenario/task |
| **Reward** | binary sparse outcome (성공 1, 실패 0) |
| **RL infra** | RLinf (Yu et al., 2025) PPO 구현 위에 trajectory-level probability 계산 모듈 추가 |

---

## 4. 시스템/학습 세부사항

- **Backbone**: MM-ACT (Liang et al., 2026) — unified discrete diffusion VLA, action chunk를 discrete token으로 표현.
- **MM-ACT* (논문 내 SFT baseline)**: action-only SFT, action chunk size 16, 4개 LIBERO suite에 대해 jointly 학습 (원본 MM-ACT는 suite별 separate).
- **PPO**: clipped surrogate (Eq. 1) + value head 기반 advantage.
- **Computational profile (Table 3, RoboTwin 2.0)**:
  | Setting | Train (s) | Inference delay (ms) | Inference FLOPs (G) |
  |---|---|---|---|
  | 1-step | 259.33 | 182.47 | 17,547.82 |
  | 2-step | 452.06 | 364.21 | 35,095.64 |
  | 4-step | 845.95 | 728.92 | 70,191.28 |
  | **Hybrid** | 607.89 | 387.24 | 37,289.12 |
- **Compute infra**: Baidu AIHC + D-Robotics RoboGo (정량적 GPU 수는 미공개).

---

## 5. 실험 결과 (PDF Table 1, 2 직접 확인)

### 5.1 LIBERO (Table 1, 4 suite × 50 trial × 10 task = 2,000 trial)

| Method | Spatial | Object | Goal | Long | **Avg** |
|---|---|---|---|---|---|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| SimpleVLA-RL | 99.4 | 99.1 | 99.2 | 98.5 | 99.1 |
| MM-ACT | 97.8 | 99.4 | 94.8 | 93.0 | 96.3 |
| MM-ACT* (SFT) | 91.2 | 82.6 | 90.0 | 88.7 | 88.1 |
| **dVLA-RL** | **99.8** | **100.0** | **99.6** | **99.2** | **99.7** |
| (WAM-기반) Cosmos Policy | 98.1 | 100.0 | 98.2 | 97.6 | 98.5 |
| (WAM-기반) LingBot-VA | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |

→ VLA·WAM 통합 전체 1위. SFT 대비 +11.6pt (특히 Object +17.4pt).

### 5.2 RoboTwin 2.0 (Table 2, 8 bimanual task)

| Method | Beat Block | Place Phone | Pick Dual | Lift Pot | Move Can | Place A2B | Place Empty | Handover Mic | **Avg** |
|---|---|---|---|---|---|---|---|---|---|
| π0 | 59.0 | 22.0 | 50.0 | 51.0 | 41.0 | 38.0 | 60.0 | 96.0 | 52.1 |
| RDT | 22.0 | 13.0 | 18.0 | 45.0 | 33.0 | 21.0 | 42.0 | 95.0 | 36.1 |
| OpenVLA-OFT | 28.1 | 17.1 | 29.7 | 10.1 | 28.1 | 37.5 | 77.3 | 45.3 | 34.2 |
| SimpleVLA-RL | 87.5 | 39.6 | 68.3 | 64.1 | 61.2 | 45.3 | 94.2 | 89.2 | 68.7 |
| π0.5 | 88.0 | 73.0 | 98.0 | 88.0 | 72.0 | 74.0 | 74.0 | 32.0 | 74.9 |
| MM-ACT* (SFT) | 85.0 | 62.0 | 71.0 | 46.0 | 40.0 | 57.0 | 81.0 | 49.0 | 61.4 |
| **dVLA-RL** | **95.3** | **96.9** | **95.3** | **89.1** | **87.5** | **79.7** | **95.3** | **96.9** | **92.0** |
| (WAM 참고) Motus | 88.0 | 86.0 | 90.0 | 99.0 | 74.0 | 79.0 | 98.0 | 63.0 | 84.6 |
| (WAM 참고) LingBot-VA | 98.0 | 97.0 | 99.0 | 99.0 | 97.0 | 93.0 | 100.0 | 96.0 | 97.4 |

→ SFT 대비 **+30.6pt** (61.4 → 92.0). VLA 기반 method 중 1위. WAM-based(다른 학습 setting)와는 경쟁적이지만 SoTA는 아님. 가장 큰 gain: Handover Mic +47.9, Move Can Pot +47.5, Lift Pot +43.1.

---

## 6. Ablation 분석

### 6.1 Trajectory-level vs last-step proxy (Figure 3, LIBERO-Spatial / Object × 2-step / 4-step)
- **2-step Spatial**: trajectory-level이 ~0.97 수렴, last-step은 후반 불안정·하락.
- **4-step Object**: trajectory-level이 saturation 근접 유지, last-step은 0.7 근처로 degrade.
- 핵심: intermediate transition을 무시하면 actual rollout과 mismatch → optimization signal이 noisy.

### 6.2 Denoising-step 효과 (Figure 4, RoboTwin 2.0 multi-task PPO)
| Setting | Peak SR |
|---|---|
| 1-step | 0.885 |
| 2-step | 0.908 |
| **Hybrid** | **0.920** |

- 1-step은 가장 저렴하지만 sparse reward에서 초기 positive feedback이 부족해 학습 늦음.
- 다단계는 더 강한 SFT initialization → RL이 빨리 informative gradient를 받음.
- Hybrid는 task별로 최적 K를 선택해 sample efficiency도 가장 우수.

### 6.3 Efficiency (Table 3, §5.5)
- Hybrid는 4-step 대비 train 28% 단축 (846 → 608s), inference delay 47% 단축 (729 → 387ms), FLOPs 47% 감소.
- 동시에 peak SR은 4-step보다도 높음 → 단순 trade-off가 아닌 **win-win**.

---

## 7. Related Work 비교

| 접근 | 대표 | 한계 vs dVLA-RL |
|---|---|---|
| **Autoregressive token VLA + RL** | SimpleVLA-RL (GRPO), RLinf | exact token likelihood가 있으나 dVLA의 parallel masked decoding엔 적용 불가 |
| **Continuous diffusion / flow + RL** | Flow Q-Learning, ODE→SDE 변환, RL Token | likelihood approximation 또는 lightweight actor-critic interface 필요. 본질적으로 surrogate. |
| **Discrete diffusion LM RL (NLP)** | Zhao et al. 2025, Yang et al. 2026, Gong et al. 2026, Zhang et al. 2026 | endpoint-conditioned denoising surrogate 사용 → 실제 inference path와 distribution mismatch, cumulative gradient imbalance |
| **Step-wise value estimation for diffusion** | Wang et al. 2026 | intermediate state마다 value head 추가 → credit assignment 복잡, dVLA처럼 K가 짧을 땐 overkill |
| **dVLA-RL** | 본 논문 | **실제 sampled trajectory의 joint probability** 직접 사용. Scheduler를 system dynamic으로 취급해 surrogate 없이 standard PPO. K가 다양해도 같은 framework. |

핵심 위치 잡기: NLP DDM의 endpoint surrogate들과 달리 dVLA는 K가 짧고(2-4 step) intermediate state가 standalone semantic이 약한 transient refinement → **path-faithful objective**가 자연스럽고 효율적.

---

## 8. Limitations (자체 언급 + 분석)

1. **MM-ACT 단일 backbone 검증**: dVLA-RL의 일반성은 다른 discrete diffusion VLA (Wen et al. 2025, Liu et al. 2026b 등)에서 별도 실증 필요.
2. **Sparse outcome reward 의존**: SimpleVLA-RL과 마찬가지로 초기 SFT SR이 너무 낮으면 RL이 학습할 positive signal을 받지 못함 (논문은 cold-start 한계를 명시적으로 다루지 않음).
3. **RoboTwin 2.0에서 WAM-based methods (LingBot-VA 97.4, Fast-WAM 96.6)에 못 미침** — 논문은 "different training settings"로 caveat 처리하지만, 동일 setting 비교는 부재.
4. **8 task로 RoboTwin 2.0 평가 제한**: 전체 RoboTwin 2.0 task suite의 부분집합.
5. **코드/체크포인트 미공개 (현재)**: 재현성·확장성 측면 미흡.
6. **추론 cost**: 가장 효율적인 Hybrid도 1-step의 ~2.1×(inference delay 기준). 고주파 reactive 제어엔 여전히 부담.
7. **Implicit scheduler 학습의 정량 검증 부족**: "p_θ가 sharpen되면 Gumbel-TopK가 자동으로 좋아진다"는 주장은 직관적이나 별도 ablation 없음.

---

## 9. 종합 평가

| 항목 | 평가 |
|---|---|
| **Novelty** | ★★★★★ — Discrete diffusion VLA에 처음으로 RL을 native하게 적용. Marginal → trajectory joint probability shift는 NLP DDM RL의 endpoint surrogate 계열과 본질적으로 다른 새 formulation. |
| **Technical depth** | ★★★★★ — Eq. 3 (intractability) → Eq. 4-5 (path factorization) → Eq. 6 (gradient restriction)의 수학적 reduction이 깔끔하고, Gumbel-TopK를 system dynamic으로 둔 implicit optimization 논리도 설득력 있음. |
| **Empirical rigor** | ★★★★☆ — LIBERO 2,000 trial + RoboTwin 2.0 64 trial/task. Figure 3·4 ablation은 명확. 다만 backbone 다양성 부족, RoboTwin 2.0 task 부분집합. |
| **Practical impact** | ★★★★☆ — LIBERO 99.7%로 VLA·WAM 전체 1위, RoboTwin 2.0 VLA 1위(+30.6 vs SFT). Hybrid는 compute efficiency까지 확보. 다만 code 미공개. |
| **Reproducibility** | ★★☆☆☆ — code/weights 미공개, MM-ACT backbone 의존 + Baidu AIHC infra 의존. RLinf만 open. |

### 핵심 기여
1. **dVLA용 RL의 핵심 수학적 장애물 (combinatorial marginalization)을 path-level joint probability로 우회**.
2. **Gumbel-TopK scheduler를 non-differentiable system dynamic으로 분리**해 PPO clip을 violate하지 않는 stable gradient를 확보.
3. **K-flexible objective로 Hybrid denoising-step strategy를 가능케 함** → task별 최적 trade-off (효과+효율).
4. **MM-ACT* SFT 88.1% → dVLA-RL 99.7% (LIBERO), 61.4% → 92.0% (RoboTwin 2.0)** 의 큰 실증 gain으로 discrete diffusion VLA가 RL-ready paradigm임을 입증.

---

## 10. 예상 세미나 질문

| # | 질문 | 답변 요점 |
|---|---|---|
| 1 | Marginal probability가 intractable이라는데, last-step probability를 proxy로 쓰면 안 되나? | Figure 3에서 직접 실증: last-step proxy는 학습 후반 unstable·plateau. 이유: rollout 시 실제 실행되는 path는 multi-step Markov chain이고, last-step은 그 chain의 단 하나의 conditional. distribution mismatch → noisy gradient. |
| 2 | Gumbel-TopK term을 그냥 버려도 되는가? Bias가 생기지 않나? | Theoretical bias는 있으나 (i) trust region 하에서 P_unmask 변화가 미미해 gradient 기여가 numerically negligible, (ii) Gumbel sort backprop은 high-variance impulse를 생성해 PPO clip violation → 오히려 학습 destabilize. 또한 p_θ가 두 메커니즘에서 공유되므로 token-pred gradient가 implicit하게 scheduler를 개선 (§4.2). |
| 3 | Hybrid가 fixed multi-step보다 peak SR이 높은 이유는 무엇인가? Cheap한 1-step도 섞는데 왜 더 잘 되나? | Easy task에서 1-step decoding이 이미 saturate면, 불필요한 denoising step이 오히려 exploration noise를 추가하지 않음. Hard task에는 4-step refinement로 sparse reward 환경에서 초기 positive feedback 확보. Task별 optimal allocation의 합이 fixed보다 우수. |
| 4 | SimpleVLA-RL (autoregressive VLA + GRPO)과 직접 비교 가능한가? 어느 쪽이 본질적으로 우수한가? | LIBERO Avg: SimpleVLA-RL 99.1 vs dVLA-RL 99.7 (둘 다 saturate 영역). RoboTwin 2.0: SimpleVLA-RL 68.7 vs dVLA-RL 92.0으로 큰 차이. 단 SFT backbone (OpenVLA-OFT vs MM-ACT*)과 학습 setting (per-task vs multi-task joint)이 달라 "RL 알고리즘 자체의 우열"로 환원하기는 어렵다. 본 논문도 directly head-to-head로 주장하지는 않음. |
| 5 | RoboTwin 2.0에서 WAM-based LingBot-VA(97.4)에 비해 dVLA-RL(92.0)이 낮다. 본질적 한계인가? | 논문은 "different RoboTwin training settings"로 caveat 처리. WAM은 video-pretrained world prior 사용 (5-step DiT denoising 등 추가 compute). dVLA-RL은 순수 VLA + RL로만 작동 → architectural ceiling이 다를 가능성. 동일 setting 비교는 future work. |
| 6 | 왜 newly unmasked token (M_k)에만 gradient를 흘리고, 이미 decoded된 token은 제외하나? | 이미 decoded된 token은 step 간 동일하게 유지(deterministic) → 그 위치의 log p에 gradient를 흘리면 (a) 같은 sample이 여러 번 reweighted되어 step 수에 비례한 redundant gradient noise 발생, (b) intermediate state x_k의 통계를 왜곡. M_k 제한은 **각 token의 contribution을 정확히 한 번** carry. |
| 7 | RL이 SFT 88.1% → 99.7%로 +11.6pt 향상시켰는데, 이게 RL의 본질적 효과인가 아니면 단순히 더 많은 trial(2,000+)로 generalize한 결과인가? | RL은 outcome reward를 통해 policy gradient를 받지만 SFT는 단순 trajectory mimicry. Multi-task joint SFT가 단일 suite 전용보다 약한 (MM-ACT* 88.1 < MM-ACT 96.3) 상태에서 시작해 RL로 99.7까지 끌어올렸다는 것이 핵심. 동일 backbone + 추가 SFT data로 같은 성능 도달 가능한지는 직접 비교 부재. |
| 8 | Discrete diffusion + RL이 continuous diffusion (π0, π0.5) + RL 대비 갖는 본질적 장점이 있나? | (i) Token-wise categorical softmax는 exact conditional probability를 제공 → likelihood approximation·SDE 변환 같은 surrogate 불필요. (ii) Denoising step 수 K가 짧음 (2-4) → trajectory enumeration 비용 적음. (iii) Unified discrete token space는 vision/language/action 통합 modeling을 깔끔하게. 단, 연속 action의 fine-grained precision은 quantization으로 제약. |
| 9 | Implicit scheduler 개선은 실증되었나? | Sec. 4.2 끝부분에 직관적 논증만 있음 ("p_θ가 sharpen되면 Gumbel-TopK target이 정확해진다"). 별도 ablation 없음 → 정량 검증은 후속 연구 과제. |
| 10 | 다른 dVLA backbone (e.g., LLaDA-VLA, dVLA-Midea)에도 그대로 적용되는가? | 원리적으로는 가능 (masked denoising + categorical softmax만 있으면 됨). 다만 실증은 MM-ACT 단일 backbone에 제한 → backbone-specific tuning이나 hyperparameter sensitivity는 미지수. |

---

## 11. 코드 & 재현

- **arXiv**: https://arxiv.org/abs/2606.23623 (v1, 2026-06-22)
- **Code/weights**: 현재 공개 정보 없음 (논문 PDF 내 GitHub 링크 부재).
- **External 의존성**:
  - Backbone: **MM-ACT** (Liang et al., 2026) — discrete diffusion VLA, 자체로 별도 코드/체크포인트 필요.
  - RL infra: **RLinf** (Yu et al., 2025) — PPO 구현.
  - Compute: **Baidu AIHC** + **D-Robotics RoboGo** (training acceleration, low-latency inference).
- **재현 핵심 포인트**:
  1. Trajectory-level log-prob 계산: 각 denoising step에서 newly unmasked token (M_k) 추적 + 해당 위치의 softmax log-prob만 sum.
  2. PPO ratio: trajectory-level r_t(θ) = exp(log π_θ(τ|s_t) − log π_{θ_old}(τ|s_t)). value head는 chunk-level advantage 추정.
  3. Gumbel-TopK는 forward만 사용, gradient는 unmask term에 흐르지 않음.
  4. Hybrid 할당: 각 task SFT checkpoint를 K=1/2/4로 평가 후 동등 SR 달성하는 최소 K 선택.
- **하드웨어**: 구체 GPU 수/메모리 미공개. 8개 task × 64 episode × multi-K rollout이 동시 진행 → 분산 환경 필수.

---

## 12. 결론

dVLA-RL은 "discrete diffusion VLA에 RL을 어떻게 적용하는가"라는 명확한 미개척 문제에 대해, **marginal action probability 대신 sampled denoising trajectory의 joint probability를 PPO objective로 쓴다**는 깨끗한 수학적 답을 제시한 작업이다. 핵심은:

1. **수학적 reduction**: combinatorial marginalization (Eq. 3) → trajectory factorization (Eq. 4-5) → gradient-tractable surrogate (Eq. 6).
2. **System dynamic 분리**: non-differentiable Gumbel-TopK scheduler를 학습 그래프 밖으로 빼냄으로써 standard PPO clip의 stability를 보존.
3. **K-flexibility의 부산물**: 1/2/4-step이 같은 framework에서 같은 probability 정의로 통합되어 task-adaptive Hybrid 전략이 자연스럽게 도출.

LIBERO 99.7%로 VLA·WAM 통합 SoTA, RoboTwin 2.0 92.0%로 VLA 기반 SoTA (+30.6pt vs SFT)라는 실증 결과는 **native discrete diffusion VLA가 RL-ready paradigm**임을 명확히 보여준다. 향후 발전 방향은 (i) 다양한 dVLA backbone에서의 일반화 검증, (ii) WAM-based methods와의 동일-setting 직접 비교, (iii) implicit scheduler 개선의 정량 분석, (iv) 코드/모델 공개를 통한 community 확장. 이 작업이 자리매김할 위치는 **SimpleVLA-RL (autoregressive VLA + RL)** 과 **continuous diffusion/flow RL** 사이에서 **"discrete diffusion VLA + RL"** 라는 제3의 축을 명시적으로 개척한 reference 논문이다.

<!-- VERIFIED: pdf -->
