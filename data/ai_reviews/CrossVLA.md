# CrossVLA: Cross-Paradigm Post-Training and Inference Optimization for VLA Models 세미나 리뷰

> **한 줄 요약**: 이산 토큰 자기회귀(OpenVLA)와 연속 행동 flow-matching(π₀.₅) 두 VLA 패러다임을 동일한 인터페이스로 묶어, (i) DPO를 flow-matching 백본에 ODE 적분 없이 적용하는 surrogate log-probability 추정기, (ii) LoRA vs DoRA의 정면 비교, (iii) sample_actions latency의 78.6%가 denoise 루프이고 prefix K/V cache의 가속 한도가 21%임을 정량화한 inference anatomy를 동시에 제시한 실증 연구. LIBERO 4-suite 평균을 OpenVLA SFT 62.75% → DoRA+DPO 73.2%로 +10.4pp 향상.

---

## 1. 배경 및 동기

최근 VLA는 (a) **이산 토큰 AR**(OpenVLA 류) — Llama-2 7B + DINO/SigLIP 비전 타워에서 7×256 bin으로 행동을 디코딩, (b) **연속 행동 flow-matching**(π₀.₅) — PaliGemma 인코더 + 10-step ODE action expert로 7-DoF chunk를 생성, 두 패러다임으로 수렴해 왔다. 그러나 LLM에서 사실상 표준이 된 **DPO** 사후학습은 거의 전적으로 AR VLA에 머물러 있었는데, 이는 flow-matching policy의 log-likelihood가 closed-form이 아니라 probability-flow ODE를 적분해야 얻어진다는 비용 문제 때문이다. CrossVLA는 이 cross-paradigm 격차를 해소하면서 동시에 "VLA 추론 latency를 무엇이 실제로 결정하는가"를 분해 측정해, 최근 활발한 prefix K/V cache 류 가속 연구의 천장을 짚어주는 것을 목표로 한다.

---

## 2. 방법론 심층 분석

### 2.1 Cross-Paradigm VLA 인터페이스

저자는 두 패러다임을 다음 5개 메소드로 추상화한 Protocol을 정의한다:

| 메소드 | 역할 |
|---|---|
| `policy_logp()` / `policy_logp_with_ref()` | (현재/참조) policy log-prob 평가 |
| `policy_sample()` | 행동 chunk 샘플링 |
| `encode_obs()` | (이미지 + 언어 + state) 인코딩 |
| `sample_actions()` | 추론 시 행동 산출 (denoise 루프 포함) |

OpenVLA는 토큰 cross-entropy를 그대로, π₀.₅은 surrogate(아래 §2.2)를 통해 동일한 형식의 `policy_logp`를 노출한다 — 이로써 DPO 손실 함수 코드를 두 패러다임에서 그대로 공유.

### 2.2 Surrogate Flow-Matching Log-Probability (§3.2)

flow-matching log-likelihood를 ODE 적분 없이 근사하는 핵심 식은:

$$
\log \tilde p_\theta(x_1 | \text{obs}) = -\frac{1}{T_\text{eval}} \sum_{t \in \mathcal T_\text{eval}} \| v_\theta(x_t, t, \text{obs}) - v_\text{target} \|^2
$$

여기서 $x_t = (1-t)x_0 + t x_1$, $v_\text{target} = x_1 - x_0$, **Teval=4**의 stratified t-sampling: $\{0.125, 0.375, 0.625, 0.875\}$. 즉 "training 시 사용하는 flow-matching loss와 동일한 형태의 음의 squared velocity error"를 log-prob의 대용으로 사용한다. ODE를 풀지 않아도 backprop이 깨끗하게 흐르고, DPO 비율 $\beta \log(\pi_\theta/\pi_\text{ref})$가 정상적으로 정의된다.

### 2.3 PEFT 계층: LoRA vs DoRA (§3.3)

DoRA는 LoRA를 magnitude(스칼라) × direction(LoRA-style ΔW)으로 분해한다:

$$
W_\text{eff} = m \odot \frac{W_0 + (\alpha/r) BA}{\| W_0 + (\alpha/r) BA \|_\text{col}}
$$

구현 디테일이 중요한데, 저자는 **forward마다 $W_\text{eff}$를 한 번만 materialize**해 memory peak를 관리한다. 그래도 r=32 기준 peak memory가 LoRA 17.93 GB → DoRA 26.17 GB로 늘어나는 비용은 분명히 명시되어 있다 (Table 1).

| 메트릭 | LoRA-r32 | DoRA-r32 |
|---|---|---|
| Trainable params | 33.55M (0.44%) | 34.08M (0.45%) |
| Peak GPU mem (eval) | 17.93 GB | 26.17 GB |
| Initial cur≡ref diff | 0.0 | 0.0 |

### 2.4 DPO 학습 설정 (§3.4)

- LR **5e-5**, batch size **1**, max steps **500**, warmup **100**, β=**0.1**
- DPO 페어: suite당 ~200쌍, rejected는 action noise σ를 0.1 → 0.4까지 키워서 생성
- 4 suite × 3 seed = 12회 학습, 총 600 trial 평가

### 2.5 Multi-View + Temporal Contrastive 사전학습 (§3.5)

저자는 downstream 초기화용 프로젝션 head를 별도 사전학습한다:

- Dual-stream InfoNCE: **(a) multi-view** (동일 timestep agent-view vs wrist-view), **(b) temporal** (agent-view, Δ=5 step 간격)
- $\mathcal L = 0.5\,\mathcal L_\text{mva} + 0.5\,\mathcal L_\text{tc}$, τ=0.07, B=128
- 데이터: 50 ep × 30 anchor × 4 suite = **6000 LIBERO 프레임**
- Projection head 656K params, 2.6 MB 체크포인트, H20-3e 1대에서 ~30분

---

## 3. 데이터셋 및 평가 프로토콜

- **베이스라인**: OpenVLA-7B per-suite 체크포인트 (~15 GB), π₀.₅ PyTorch 변환본 (6.8 GB)
- **평가**: LIBERO 4 suite × 10 task × 5 trial = 50 trial/seed, **600 trial × 3 seed**
- **렌더링**: MuJoCo + EGL
- 저자는 OpenVLA SFT 재현치(Spatial 72 / Object 56 / Goal 70 / Long 53)가 원 논문(84.7 / 88.4 / 79.2 / 53.7)보다 낮음을 명시 — Δ는 "동일 코드/데이터의 ours-vs-ours 비교"로 해석해야 한다.

---

## 4. 실험 결과

### 4.1 LIBERO 4-suite 메인 결과 (Table 3)

| Suite | OpenVLA SFT (재현) | +LoRA s=42 | +LoRA multi-seed | **+DoRA pool (메인)** | Δ vs SFT |
|---|---|---|---|---|---|
| Spatial | 72% | 78% | — | **74.7%** (112/150) | +2.7 |
| Object | 56% | 62% | 75% | **76.0%** (114/150) | **+20.0** |
| Goal | 70% | 76% | 77% | **78.0%** (117/150) | +8.0 |
| Long-horizon | 53% | 54% | 64% | **64.0%** (96/150) | +11.0 |
| **Mean** | 62.75 | 67.50 | — | **73.2%** | **+10.4** |

- Object suite에서 seed 42 / 1337 / 2026 모두 정확히 38/50 — **seed variance 0**.
- LoRA에서 DoRA로 바꾼 것만으로 평균 +5.7pp의 추가 향상 (multi-seed LoRA 대비).

### 4.2 π₀.₅ 추론 latency 분해 (Table 4)

| 단계 | 시간 | 비중 |
|---|---|---|
| Image preprocess + tokenize | ~5 ms | 1.8% |
| embed_prefix + PaliGemma prefix forward | ~60 ms | 21.4% |
| **Denoise loop ×10 (action expert)** | **~220 ms** | **78.6%** |
| **합계** | ~280 ms | 100% |

> **함의**: VLA-Cache 류 prefix K/V 가속의 이론적 상한은 **약 21%**. 진짜 가속은 denoise 루프 자체를 줄여야 한다(consistency distillation, fewer-step flow 등).

### 4.3 KV-Cache 전략의 부정 결과

**Strategy 1 — Chunk-level cache**:

| 메트릭 | 결과 |
|---|---|
| Success | 40/50 = **80%** (baseline 100%) |
| Wall time (50 trials) | 1796s (baseline 1258s, **+30% 느려짐**) |
| Cache reuse rate | 82.1% |

**Strategy 2 — Token-level prefix cache**:

| sim threshold | max reuses | Success | Hit rate |
|---|---|---|---|
| 0.999 (sanity) | 1 | 1/1 | 0% |
| 0.92 | 50 | 0/1 | 86% |
| 0.98 | 5 | 0/2 | 64% |

캐시 적중률은 높아도 **suffix attention이 stale K/V로 망가지면서 success rate가 0%까지 떨어진다**. 단순 prefix cache는 VLA 추론을 도울 수 없다는 분명한 부정 결과.

### 4.4 Multi-view 사전학습 (Tables 5–6)

10 epoch (~470 step), 30분 만에 수렴:

| 메트릭 | Step 10 | Step 460 |
|---|---|---|
| $\mathcal L_\text{total}$ | 2.418 | 0.366 |
| $\mathcal L_\text{mva}$ | 3.527 | 0.508 |
| $\mathcal L_\text{tc}$ (Δ=5) | 1.309 | 0.223 |

k-NN retrieval (random baseline @1 = 2.75%):

| 메트릭 | R@1 | R@5 | R@10 |
|---|---|---|---|
| Same-task | **99.5%** | 99.9% | 99.95% |
| Same-episode | 91.4% | 97.7% | 99.0% |
| Same-task & |Δt|≤10 | 92.4% | 98.2% | 99.0% |

Per-suite recall@1: Spatial 99.3 / Object 100.0 / Goal 99.3 / Long-horizon 99.5.

---

## 5. Ablation 및 부정 결과 정리

| 가설 | 결과 |
|---|---|
| DPO를 flow-matching에 ODE 없이 적용 가능 | ✅ (surrogate logp + Teval=4) |
| DoRA가 LoRA보다 우월 | ✅ Object +20pp 포함 평균 +5.7pp |
| Prefix K/V cache로 큰 가속 가능 | ❌ 이론 상한 ~21%, 실측은 success rate 붕괴 |
| Token-level cache가 throughput을 살림 | ❌ 0–64% success — 사실상 사용 불가 |

---

## 6. 한계 및 의의

**한계**:
- 단일 저자 연구(Zhi Liu, Tianjin University)로 ablation budget이 제한적 — π₀.₅ 위에 직접 DoRA+DPO를 돌린 절대 점수는 없고 OpenVLA SFT 위의 결과만 제시됨 (cross-paradigm 검증의 한쪽 절반).
- OpenVLA SFT 재현치가 원 논문보다 낮아(예: Object 56% vs 88.4%) Δ 절대값을 다른 VLA의 Δ와 직접 비교하기 어려움.
- π₀.₅ latency 분해는 specific 하드웨어/구현 의존이므로 다른 환경에서 비중이 바뀔 수 있음.

**의의**:
- "flow-matching VLA에도 DPO를 붙일 수 있다"는 가능성을 코드/체크포인트(https://github.com/lz-googlefycy/vla-lab) 공개와 함께 입증.
- **denoise 루프가 latency의 80%**라는 정량적 결과는 SnapFlow, consistency-distillation 등 후속 가속 연구의 방향을 정해줌.
- LIBERO Object suite에서 3 seed 모두 38/50으로 **variance 0**은 DPO가 결정적(deterministic-like) 행동 분포로 수렴함을 시사 — flow-matching policy의 mode collapse 측면에서도 분석 가치 있음.

---

## 7. 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 |
|---|---|---|
| 1 | Teval=4가 너무 적지 않은가? | Stratified t-sampling으로 분산 감소. 학습 시 사용하는 FM loss와 동형이라 추가 노이즈 없이 ratio가 정의됨 |
| 2 | DoRA의 추가 메모리(26GB vs 18GB)는 정당화되나? | LIBERO 평균 +5.7pp(LoRA 대비), Object +14pp — 메모리 1.5배는 받아들일 만 |
| 3 | OpenVLA SFT 재현치가 왜 그렇게 낮나? | 동일 코드/데이터의 self-comparison용. Δ는 ours-vs-ours로 해석 |
| 4 | Object suite seed 분산 0은 의심스럽지 않은가? | 50 trial × 3 seed = 150 trial 모두 38/50. DPO가 deterministic-like한 분포로 수렴했다는 신호 |
| 5 | 21% 가속 상한은 어떤 가정에 의존하나? | prefix forward 60ms가 0이 되는 극한 가정. 실제 chunk-level cache는 30% 더 느림 |
| 6 | k-NN recall 99.5%가 실제 정책 성능으로 이어지는가? | 본 논문에서는 projection head를 downstream에 직접 적용한 정책 결과는 미보고 — future work |
| 7 | π₀.₅에도 DoRA+DPO를 붙이면 성과가 같은가? | abstract/본문에서 절대 점수 미보고. surrogate logp가 검증된 정도는 OpenVLA SFT 위에서만 |
| 8 | DPO 페어 200쌍(σ 0.1→0.4 노이즈)은 어떻게 검증되는가? | rejected가 진짜 더 나쁜 행동이라는 근거는 σ 크기에 의존 — 더 정교한 rejection sampling은 future work |

<!-- VERIFIED: pdf -->
