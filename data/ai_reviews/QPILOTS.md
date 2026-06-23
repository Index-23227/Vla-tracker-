# QPILOTS: Efficient Test-Time Q-Steering for Flow Policies

> **한 줄 요약**: Flow-matching VLA를 한 줄도 수정하지 않고, Euler 적분기 매 스텝마다 **Tweedie 디노이즈된 깨끗한 액션 위에서 평가한 Q-gradient**를 drift에 더해 KL-정규화 최적 정책 π* ∝ π_β · exp(τ Q̄)에서 샘플링하는 inference-time Q-steering 기법. OGBench 50태스크 online에서 aggregate 90% SR로 BPTT/distillation/fine-tuning 계열을 모두 제치고, 3B π0.5-LIBERO를 frozen 상태로 둔 채 6개 LIBERO-90 태스크에서 zero-shot 26% → 76.5% (QPILOTS-U) 달성. Latency overhead는 단 1.04배.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- **BPTT 계열** (FBRAC, DPPO 등): K-step denoising chain을 통째로 미분 → 수치적으로 불안정, scaling 불가
- **One-step distillation** (FQL, Consistency Policy): multi-step flow를 한 스텝으로 압축 → expressivity/multi-modality 상실
- **Policy fine-tuning** (QAM, QAM-E, ReinFlow): critic이 갱신될 때마다 base flow 가중치를 재학습 → critic-policy 결합이 강해 3B급 VLA에서 prohibitive
- **Inference-time steering** (DSRL, EXPO, BoN, CGQL): 학습 없이 추론에서 steering하지만, 지금까지 fine-tuning 계열에 aggregate에서 밀려왔고, **noisy intermediate x_t에서 critic을 평가**해서 critic이 calibrate되지 않은 영역에서 잘못된 gradient를 따라감

### 핵심 질문
- **Base flow를 동결한 채** 어떻게 critic의 action-gradient 정보를 손실 없이 활용할 수 있나?
- noisy x_t에서 ∇Q를 직접 평가하지 않고, **clean action 공간에서 평가한 gradient**를 어떻게 denoising 과정에 주입할 수 있나?
- 3B-scale VLA (π0.5) 위에서도 online RL이 실용적으로 동작하는가?

📌 [Figure 1 삽입] — QPILOTS는 policy extraction을 flow-time tilted sampling으로 재정식화. π* ∝ π_β exp(τQ)의 log-tilt potential V_t의 gradient를 base velocity field v_θ에 drift correction으로 더함.

---

## 2. 방법론 심층 분석

### 2.1 핵심 정식화: KL-정규화된 Tilted Sampling

KL-제약 정책 개선 문제
$$\max_\pi \mathbb{E}_{a\sim\pi}[Q(s,a)] \text{ s.t. } D_{KL}(\pi \| \pi_\beta) \le \delta$$
의 closed-form 해는 exponentially tilted policy
$$\pi^*(a|s) \propto \pi_\beta(a|s) \exp(\tau \bar Q(s,a))$$
이고, 이는 flow-matching의 SDE 정식화에서 **drift correction** σ_t²/2 · ∇V_t(x_t)를 더한 steered probability-flow ODE
$$\dot x_t^* = v_\theta(x_t^*, t, s) + \tfrac{\sigma_t^2}{2}\nabla V_t(x_t^*; s)$$
의 endpoint marginal로 정확히 실현된다. 여기서 log-tilt potential은
$$V_t(x_t; s) = \log \mathbb{E}_{x_1 \sim p_{1|t}}[\exp(\tau \bar Q(s, x_1))]$$
**핵심 통찰**: ∇V_t는 *clean action* x_1에 대한 critic gradient의 posterior expectation이므로, critic을 noisy x_t에서 평가할 필요가 없다.

### 2.2 QPILOTS-U: Tweedie Point Estimate

가장 간단한 추정 — posterior 평균을 Tweedie identity로 한 점에서 평가:
$$\hat x_1 = x_t + (1-t) v_\theta(x_t, t, s)$$
$$\widehat{\nabla V_t}^{UG}(x_t) = \tau \nabla_{x_t} \bar Q(s,\, \text{clip}[\hat x_1, -1, 1])$$
- **추가 학습 0**: critic + base flow의 단일 forward-backward만 추가 (per Euler step)
- **편향원**: (i) log-expectation을 점 추정으로 대체, (ii) 학습된 v_θ가 true conditional mean velocity와 일치할 때만 정확
- π_β가 Gaussian일 때만 Bayes-optimal

### 2.3 QPILOTS-M: Meta Flow Map Posterior Sampling

Tweedie 평균을 differentiable posterior 샘플로 대체:
$$\widehat{\nabla V_t}^{MFM}(x_t) = \nabla_{x_t} \log \frac{1}{N}\sum_{n=1}^N \exp\bigl(\tau \bar Q(s, \hat X_{0,1}(\epsilon_n; t, x_t, s))\bigr)$$
- **Meta Flow Map (MFM)** [Potaptchik 2026]: amortized one-step posterior sampler X̂_{0,1}이 (t, x_t, s) 조건부로 p_{1|t}에서 differentiable 샘플 생성
- N=4 posterior 샘플 + log-sum-exp Monte Carlo로 편향 점진 감소
- **이론 보장** (Proposition 1, Appendix B): exact posterior + unrescaled σ_t²/2 schedule 하에서
  $$W_2(\hat p_1^s, \pi^*) \le C(1/\sqrt K + 1/\sqrt N),\quad KL \le C(1/K + 1/N)$$
- 학습 코스트: K-step마다 N forward + 1 backward through MFM·critic

### 2.4 Gradient Rescaling — 실용적 안정화

이론 schedule σ_t²/2는 t→0에서 발산, t→1에서 0이라 endpoint-singular. Potaptchik et al.를 따라 **drift-magnitude matching** 적용:
$$v^{\text{steered}}_\theta = v_\theta + \alpha \cdot \frac{\|v_\theta\|}{\|\widehat{\nabla V_t}\| + \epsilon}\widehat{\nabla V_t}$$
- t=0에서는 steering skip (V_0가 state-independent)
- τ=1 고정, α만 domain별 sweep
- 정식 SDE 분석 밖이지만 실험적으로 더 안정

### 2.5 Pessimistic Critic Ensemble

Overestimation 억제용 critic ensemble (J=10, Li & Levine 2026 따름):
$$\bar Q(s,a) = \tfrac{1}{J}\sum_j Q_{\phi_j}(s,a) - \rho \cdot \text{std}_j Q_{\phi_j}(s,a)$$
ρ=0.5 기본, humanoid-maze는 ρ=0.

> ❓ **예상 질문**: noisy x_t에서 ∇Q 평가가 왜 문제인가? Universal Guidance/DSRL은 이미 그렇게 한다.
> **답변**: critic은 *demonstration data* 또는 *replay buffer*의 clean action 분포에서 학습됐다. noisy x_t는 학습 분포 밖이므로 Q(s, x_t)는 calibrate되지 않은 외삽치를 내놓는다. QPILOTS는 ∇Q를 항상 ε-clip된 clean action에서만 평가해 이 문제를 회피. CGQL/DSRL이 plateau하는 LIBERO 결과(Table 3)가 이 차이의 실증적 증거다.

---

## 3. 데이터 전략

### OGBench (from-scratch)
- 10 domain × 5 task = **50 single-task settings**
- 1×10⁶ offline gradient step + 5×10⁵ online step (1 update / env step)
- 기본 OGBench 데이터셋, cube-quadruple-play와 puzzle-4x4-play-sparse는 100M 추가 데이터셋
- 5 navigation (single-step action) + 5 manipulation (action chunking H=5)

### LIBERO (frozen base policy)
- **공개 π0.5-LIBERO 체크포인트** (Black et al. 2025, 3B params, PaliGemma-class VLM + flow-matching action expert, LIBERO-40 finetuned) 동결
- 평가: LIBERO-90 중 **π0.5 zero-shot SR이 5~50% 사이인 6 태스크** (26, 31, 38, 59, 60, 64) 선별 — head-room이 있는 태스크에 한정
- QPILOTS-M의 MFM은 **공개 LIBERO-90 데모**로 distill (4×10⁴ step, batch 256, ~2일 / H100). π0.5의 원 pretraining mixture는 closed-source라 mismatch 존재
- Critic은 SARSA target (Bellman target은 3B π0.5를 매 update마다 샘플링해야 해서 prohibitive)

> ❓ **예상 질문**: LIBERO 6 태스크 cherry-picking 아닌가?
> **답변**: 부분적으로 그렇다. zero-shot 50% 이상인 태스크는 이미 ceiling 가까워서 향상을 측정하기 어렵고, 5% 미만은 critic이 신호를 얻지 못한다. 다만 6 태스크 모두에서 zero-shot 대비 +25%~+85%p로 큰 향상이 일관되게 나타나 selection 효과만으로 설명하긴 어렵다. 표준 LIBERO-4 (Spatial/Object/Goal/Long)는 평가 X.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Base flow (OGBench) | flow-matching policy, 처음부터 공동학습 |
| Base flow (LIBERO) | π0.5-LIBERO 3B, **완전 동결** |
| Critic ensemble | J=10, LayerNorm, pessimism ρ=0.5 (humanoid는 0) |
| MFM (QPILOTS-M, LIBERO) | π0.5 action expert와 동일 transformer 구조, π0.5 가중치로 init, mfm_num_heads=8 |
| MFM distillation | GLASS reparameterization, λ_diag=1.0, λ_cons=0.5, τ_mfm=10⁻⁴ |
| Steering coefficient α | OGBench: domain별 {0.1, 0.2, 0.3, 0.5} (puzzle-4x4만 2.0); LIBERO: 태스크별 {0.1, 0.2, 0.3} |
| τ | 1 고정 |
| N (M variant) | 4 posterior samples / Euler step |
| K | base policy와 동일 (수정 X) |
| Action chunking H | OGBench manipulation 5, navigation 1; LIBERO 10 (π0.5 매칭) |
| γ | 0.99 (giant/humanoid maze 0.995) |
| Compute (OGBench) | TPU v4-8, 8h(U) / 12h(M) per task per 4 seeds |
| Compute (LIBERO) | 6h(U) / 8h(M) per task per seed |
| MFM distillation | ~2일 / H100 (frozen 3B teacher가 매 batch) |
| Latency analysis | NVIDIA RTX PRO 6000 |

---

## 5. 실험 설계 및 평가 프로토콜

### OGBench (Table 1)
- 8 seeds, 50 evaluation episodes/seed, 95% bootstrap CI
- end-of-offline (회색) vs end-of-online (컬러) 분리 보고
- 모든 baseline 숫자는 Li & Levine 2026에서 직접 가져옴 (QPILOTS-U/M만 새로 실행)

### LIBERO (Table 3, Fig. 2)
- 5 seeds × 5×10⁵ online steps
- **EMA time-weighted smoothing (α=0.99)** 후 seed 평균 — 마지막 step의 운빨 제거
- Per-task α/b_W는 Table 4

### Latency (Table 5)
- Task 59 single-step inference on RTX PRO 6000
- Best-of-N과 같은 plate 위에서 비교

---

## 6. 실험 결과 심층 분석

### OGBench Aggregate (50 tasks online SR, Table 1)

| Method | Aggregate Online SR |
|--------|---------------------|
| ReBRAC (Gaussian) | 64% |
| FBRAC (BPTT) | 64% |
| FQL (one-step distill) | 82% |
| QAM | 70% |
| **QAM-E** (prev SOTA fine-tune) | **85%** |
| DSRL (steering) | 60% |
| IFQL (BoN N=32) | 49% |
| FEdit | 79% |
| CGQL / CGQL-M / CGQL-L | 67 / 64 / 58% |
| **QPILOTS-U** | **89%** |
| **QPILOTS-M** | **90%** |

- **모든 카테고리(BPTT, distillation, fine-tuning, steering)를 제치고 aggregate 1위**. QAM-E 대비 +5%p
- humanoid-large-navigate에서 closest baseline 대비 **+25%p**의 큰 격차 → high-dimensional 액션에서 효과 가장 큼
- puzzle-4x4-play-sparse (희소 보상 + 큰 action space): QPILOTS-U/M만이 100%, 다른 inference-time 방법은 모두 0%
- offline phase에서는 QPILOTS-U가 약간 부족하지만 (point estimate의 한계), QPILOTS-M이 이 gap을 메움

### LIBERO 6-task 평균 (Table 3)

| Method | T26 | T31 | T38 | T59 | T60 | T64 | Mean |
|--------|-----|-----|-----|-----|-----|-----|------|
| π0.5 zero-shot | 6 | 14 | 32 | 19 | 50 | 36 | 26 |
| DSRL | 36 | 22 | 70 | 93 | 95 | 34 | 58 |
| BoN (N=5) | 10 | 19 | 53 | 35 | 66 | 57 | 40 |
| EXPO | 7 | 29 | 55 | 28 | 62 | 52 | 39 |
| CGQL-LinEx | 11 | 12 | 25 | 27 | 64 | 49 | 31 |
| **QPILOTS-U** | **43** | **55** | **91** | 82 | **99** | **89** | **76.5** |
| QPILOTS-M | 41 | 42 | 78 | **87** | 98 | 86 | 72 |

- zero-shot 26% → **QPILOTS-U 76.5%** (+50.5%p), 6태스크 중 5태스크에서 SOTA
- T64 ("stack right bowl on left bowl…") 같은 long-horizon에서 DSRL 34% vs QPILOTS-U 89% — **+55%p**
- 평균에서 QPILOTS-M < QPILOTS-U 인 점은 흥미로움 → §7에서 분석

### Latency (Table 5, π0.5 + task 59)

| Method | SR (%) | Inference Time | Overhead |
|--------|--------|----------------|----------|
| π0.5 baseline | 19.2 | 112.7 ms | 1.0× |
| Best-of-5 | 36.4 (+89.5%) | 127.2 ms | 1.14× |
| Best-of-10 | 41.4 (+115.6%) | 137.0 ms | 1.22× |
| Best-of-20 | 45.5 (+136.9%) | 158.0 ms | 1.44× |
| **QPILOTS-U** | **83.3 (+333.8%)** | **114.1 ms** | **1.04×** |
| QPILOTS-M | 88.5 (+360.9%) | 240.1 ms | 2.14× |

- **QPILOTS-U: BoN-5보다도 빠르면서 SR은 2.3배** — 가장 실용적 결과
- QPILOTS-M은 latency 2.14배지만 SR이 가장 높음 — quality-latency tradeoff 명확

> ❓ **예상 질문**: LIBERO에서 왜 -M이 -U보다 떨어지나? 이론적으로 -M이 더 정확한데?
> **답변**: 저자가 직접 인정한 한계. MFM 이론은 *base flow와 동일한 데이터*로 MFM을 distill한다고 가정하지만, π0.5의 pretraining mixture는 closed-source이고 LIBERO-90 demo는 그 부분 집합일 뿐이다. distillation gap이 클수록 MFM 샘플이 true posterior에서 멀어져 결국 noisy 추정이 된다. OGBench에서는 같은 데이터로 MFM/base를 공동학습하기 때문에 -M이 일관되게 우위 (특히 offline phase).

---

## 7. Ablation 분석

### Value Learning Ablation (Fig. 3, LIBERO-90 task 64)

VLA에서 standard Bellman target은 매 update마다 π0.5를 batch_size번 샘플링해야 해 infeasible. 4개 critic objective 비교:

| Objective | T64 SR at 5×10⁵ steps |
|-----------|-----------------------|
| **SARSA** (chosen) | **~90%** |
| Monte Carlo | unstable, dominated by 초기 negative returns |
| IQL | high variance under non-stationary replay |
| FQL (distill steered π0.5 to one-step) | high variance |

- **결정적 발견**: offline RL용 IQL/FQL이 non-stationary online replay 하에서 무너진다. SARSA는 buffer 저장된 a'를 재사용해 안정.
- 이는 VLA 기반 online RL의 일반 처방으로 확장 가능한 인사이트

### Steering Coefficient Sensitivity (Fig. 4)

OGBench 10 domain × 2 tuning task × α ∈ {0.1, 0.2, 0.3, 0.5, 1.0, 2.0}, 4 seeds.

- 대부분 도메인이 α=0.1~0.5에서 peak, 양쪽으로 gracefully 감소 → **low sensitivity**
- 예외: **puzzle-4x4-play-sparse** — α<1에서 거의 0, α=2에서만 emerge. 희소 보상 + 큰 action space에서는 steering 강도가 결정적

### Critic Ensemble Pessimism
- ρ=0.5 OGBench 기본, humanoid-maze ρ=0 (저자가 인정하듯 domain-specific tuning)
- Adjoint matching (QAM/QAM-E)과 같은 calibration 의존

### 이론 보장의 가정 검증 (Appendix B)
- Assumption 1 (bounded τQ): batch-normalize Q then clip — 엄밀히는 IID 가정 위반이지만 per-batch guarantee로 처리
- Assumption 2 (smooth MFM/v_θ): GeLU + gradient clipping으로 만족
- Assumption 3 (bounded σ_t): t=0과 t=1-ε에서 truncate
- 상수 C는 action dimension에 **지수적으로** 의존 가능 → high-dim에서 bound는 vacuous할 수 있음 (실용 결과는 양호)

---

## 8. 관련 연구 비교

| 모델 | Policy 갱신 | Critic gradient 평가 위치 | Base flow 수정 | π0.5-scale RL 가능? | LIBERO 6-task |
|------|-------------|---------------------------|-----------------|---------------------|----------------|
| FBRAC | BPTT through K steps | noisy x_t | full update | ✗ (unstable) | — |
| FQL | one-step distill | clean (one-step actor) | replace with one-step | △ (expressivity loss) | — |
| QAM-E | adjoint matching (step-wise) | noisy x_t at sampled steps | edit-flow update | △ (critic 갱신마다 재학습) | — |
| DSRL | noise-space SAC | noise-space critic | frozen | ✓ | 58 |
| EXPO | residual edit policy | full action | frozen | ✓ | 39 |
| BoN | sampling + rerank | full action | frozen | ✓ but slow | 40 |
| CGQL-LinEx | classifier guidance | noisy x_t | frozen | ✓ | 31 |
| **QPILOTS-U** | **inference-time drift** | **Tweedie clean x̂_1** | **frozen** | **✓** | **76.5** |
| **QPILOTS-M** | inference-time drift | MFM posterior sample | frozen | ✓ | 72 |

### 핵심 차별점
- **유일하게 frozen base flow + clean-action gradient + multi-step structure 보존**을 동시 달성
- DSRL/EXPO는 noise-space 또는 후처리라 "어느 denoising step에 어느 정도 개입할지" 표현력 부족
- BoN은 expensive (N×inference) + action-dim에 poor scaling
- Universal Guidance를 RL critic으로 일반화한 첫 사례 (Bansal 2024는 classifier; Attarian 2024는 success classifier에 적용)

---

## 9. 한계 및 미해결 문제

### 방법론적 미비점
1. **α는 여전히 수동 튜닝**: per-domain (OGBench), per-task (LIBERO). adaptive choice (e.g. critic curvature로) 미해결.
2. **Gradient rescaling은 이론 밖**: 실용은 잘 되지만 formal SDE 분석은 σ_t²/2 schedule에만 적용. 이 둘이 같은 π*에 수렴한다는 증명 없음.
3. **τ=1 고정**: KL constraint level을 명시적으로 조절할 hyperparameter 없음.
4. **QPILOTS-M의 MFM distillation 데이터 의존성**: π0.5처럼 closed-source pretraining인 경우 MFM이 잘 동작하지 않을 수 있음 (Table 3에서 -M < -U 가 그 증거).
5. **고차원 액션에서 상수 C 지수 폭발**: 이론 bound는 action_dim ≪ 10인 경우만 의미 있을 가능성. π0.5의 H=10 chunking으로 70-DoF는 실증적으로 동작하지만 이론 보장 약함.

### 평가 한계
1. **LIBERO에서 표준 4-suite 평가 없음** — Spatial/Object/Goal/Long에 적용 안 함. LIBERO-90 6-task cherry-picking으로 head-room 확보.
2. **Real-world 실험 0**: OGBench, LIBERO 시뮬레이션만. π0.5의 강점인 real-world generalization에서의 검증 부재.
3. **다른 VLA backbone에서 검증 안 됨**: π0, OpenVLA, GR00T, RDT 같은 다른 flow/diffusion VLA에서의 transferability 미확인. 다만 critic-agnostic + frozen-base 디자인이라 plug-and-play 적용 가능성은 높음.
4. **No code release** (paper 시점): open_source=false, 재현 의존성 큼.

### Attribution
- LIBERO 향상이 (a) clean-space ∇Q (b) SARSA critic (c) action chunking H=10 중 어느 게 결정적인지 — SARSA ablation은 있지만 (a) vs (b)의 분리는 불완전. EXPO/BoN/CGQL이 동일 SARSA critic을 공유한다는 점이 부분 해소.

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Tweedie + classifier guidance를 flow RL에 응용한 점이 단순하지만 깨끗. MFM-G estimator의 RL 도입은 처음. |
| **Technical depth** | ★★★★★ — KL-정규화 정책개선의 SDE 정식화, MFM-G의 W2/KL bound, gradient rescaling의 schedule 회피 등 분석이 두텁다. Appendix B의 가정/증명도 정직. |
| **Experimental rigor** | ★★★★☆ — OGBench 50 task × 8 seeds, bootstrap CI, baseline 12종 비교는 매우 강력. 다만 LIBERO 6태스크 cherry-pick과 표준 4-suite 부재가 아쉬움. |
| **Practical impact** | ★★★★★ — QPILOTS-U는 **1.04× latency overhead로 π0.5 zero-shot SR을 +334% 끌어올림**. 3B-scale VLA를 frozen으로 두고 online RL 하는 것이 진짜 작동한다는 첫 강한 증거. |
| **Writing quality** | ★★★★☆ — Algorithm 1, Figure 1, Table 1이 readable. SDE 정식화 부분은 prior(Potaptchik 2026, Bansal 2024) 의존이 크지만 self-contained. |

**강점**:
- Inference-time steering이 fine-tuning을 aggregate에서 능가한 첫 결과 (OGBench 90% > QAM-E 85%)
- Frozen 3B VLA + critic-only training이라는 실용적 recipe — π0.5의 모든 generalization을 보존
- W2/KL bound로 이론적 정당화
- QPILOTS-U는 단 1.04× 오버헤드로 실제 로봇 제어 주기 (10Hz)에 직접 적용 가능

**약점**:
- α tuning manual + per-task
- LIBERO 평가가 cherry-picked 6 task에 한정, 표준 suite 없음
- MFM distillation의 데이터 의존성이 LIBERO에서 -M < -U 현상으로 드러남
- 코드 미공개

**Bottom line**: "frozen pretrained flow VLA + online RL"이라는 매우 현실적인 setting에서 동작하는 첫 inference-time 방법. QAM-E 같은 fine-tuning 계열보다 aggregate에서 우월하면서 base 가중치를 건드리지 않는다는 점에서, 향후 3B+ generalist VLA의 task-specific deployment에 표준 recipe가 될 가능성이 높다.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Frozen π0.5에서 critic만 학습하는 게 진짜 의미 있나? 결국 critic이 모든 task-specific 정보를 들고 있게 되는데. | 그렇다. critic이 task-specific signal을 담당하고 base flow의 generic motor prior는 보존. QAM-E처럼 base를 재학습하면 다른 태스크 generalization이 무너질 위험. π0.5의 6태스크 zero-shot이 5~50%인 데서 시작해 76.5%로 올라간 게 그 가치를 보여줌. |
| 2 | QPILOTS-U가 QPILOTS-M보다 LIBERO에서 더 좋은 게 unsettling하지 않나? 이론적으론 -M이 unbiased. | 저자가 §5.2에서 honest하게 인정. MFM이 base flow와 다른 데이터로 distill되면 (LIBERO-90 ⊊ π0.5 pretraining mix) MFM 샘플이 biased. OGBench에선 같은 데이터로 공동학습하기에 -M이 우위. 시사점: closed-source teacher는 -M의 아킬레스건. |
| 3 | clean action에서 ∇Q 평가가 핵심 차별점이라는데, 정말 그게 결정적인가? | Table 3에서 같은 SARSA critic 공유하는 BoN/EXPO/CGQL-LinEx가 평균 31-40%인데 QPILOTS-U가 76.5%. 차이는 (a) per-step drift correction (one-shot rerank vs continuous steering), (b) clean-space critic eval. CGQL-LinEx(31)은 noisy x_t에서 critic 평가 → noisy 영역 외삽치를 따라가 무너짐. |
| 4 | α=0.1~0.5가 대부분 잘 된다는데, puzzle-4x4가 α=2.0이어야 하는 건 어떻게 발견? | Domain별 2 tuning task × 4 seeds × {0.1, 0.2, 0.3, 0.5, 1.0, 2.0} sweep. puzzle-4x4는 sparse reward + 큰 action space라 약한 drift로는 보상 region에 못 도달. **이런 경우를 어떻게 자동 식별할지가 future work**. |
| 5 | Gradient rescaling이 이론에 없는데 왜 그게 같은 π*에 수렴하나? | 엄밀히는 증명 없음. 직관: σ_t²/2와 ‖v_θ‖/‖∇V_t‖ 둘 다 drift 방향(direction)을 유지하고 magnitude만 다름. 같은 vector field의 reparameterization이면 endpoint distribution은 보존된다고 *경험적으로* 관찰. Formal한 증명은 §6 Limitation에 open. |
| 6 | OGBench 90%인데 QAM-E 85%와 5%p 차이가 진짜 의미 있나? bootstrap CI 안에 들어가지 않나? | aggregate에서 5%p는 50태스크 평균이라 무시할 수치 아님. 특히 humanoid-large +25%p, puzzle-4x4-play-sparse +0→100 같은 큰 jump가 평균을 견인. CI는 per-task로 reported (Table 1 bracket). |
| 7 | SARSA를 쓰면 a' ∼ π*에서 샘플링 안 하니까 off-policy bias 있지 않나? | Yes, π_β(a'|s')에서 evaluate하니 true Q* 가 아닌 behavior Q^β를 학습. 저자는 이를 **"behavior Q-value"** 라고 명명하며 §5.2에서 명시적 인정. 그래도 LIBERO에서 잘 동작하는 이유는 π_β=π0.5가 이미 expert-quality라 behavior Q가 좋은 ranking을 주기 때문. low-quality π_β에선 깨질 수 있는 가정. |
| 8 | π0 이나 OpenVLA-OFT 같은 다른 VLA에 plug-and-play? | π0: flow-matching action expert라 same recipe 적용 가능. π0.5와 거의 동일. OpenVLA-OFT: parallel decoding head는 flow가 아니라 다중 token regression이라 직접 적용 불가 (denoising chain이 없음). DiffusionPolicy, RDT, GR00T는 다 가능해 보임. |
| 9 | 이 방법은 BPTT의 instability를 회피한다는데, 그 instability가 정말 그렇게 큰 문제였나? FBRAC도 64% (online) 받는데. | FBRAC 64%는 50태스크 평균이고, OGBench 일부 도메인 (e.g. antmaze-giant) 에서 거의 0%로 collapse. BPTT는 평균에서 동작해도 long-horizon/sparse에서 catastrophic failure 빈번. 그래서 reliability 측면에서 QPILOTS가 우위. |
| 10 | Latency 분석이 task 59 한 태스크라 generalize되나? | 단일-step inference latency라 모든 태스크에서 비슷할 것 (모델 크기·N에 의존). 다만 액션 차원이 다른 humanoid 같은 환경에선 약간 다를 수 있음. 1.04× 자체가 워낙 작아 "거의 free" 라는 메시지는 robust. |

<!-- VERIFIED: pdf -->
