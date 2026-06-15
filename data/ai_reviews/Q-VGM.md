# Q-VGM: Q-Guided Value-Gradient Matching for Flow-Matching VLA Policies

> **한 줄 요약**: Few-shot SFT pi_0.5 flow-matching VLA를 고정 rollout buffer만 가지고 off-policy로 fine-tuning하는 RL 기법. Action-sensitive Cal-QL critic의 action-space gradient ∇_A Q를 denoising-time velocity correction으로 변환(VGG-Flow optimal-control view)하여, denoising chain backprop도 action likelihood도 필요 없이 residual velocity matching으로 정책을 개선. LIBERO 75.0%→92.5%, RoboTwin 2.0 76.4%→87.2%, real-robot 40.0%→67.5%.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- **Diffusion-QL / 직접 Q-maximization** ([15]): critic 값을 policy loss에 더해 denoising chain 전체로 backprop. VLA scale(억 단위 파라미터)에서는 gradient path가 길고 불안정 — 논문 Table 1에서 75.0%→69.5%로 오히려 **퇴화**.
- **Test-time Q selection / Q-guidance** ([16,17,21,33,34]): critic을 추론 시 sample re-ranking이나 ∇_A Q refinement에만 사용. 정책 자체는 갱신 안 됨 → SFT 잠재력 한계.
- **Q-guided action distillation (PA-RL)** ([18]): critic-improved clean action을 supervised label로 distill. 그러나 velocity field에 대한 감독 신호가 없어 flow 구조를 무시.
- **Policy-gradient 계열** ([9-13]): tractable action likelihood가 필요. Flow policy는 K-step Euler denoising으로 액션을 생성하므로 likelihood가 노출되지 않음. Stochastic-flow 변형이나 surrogate로 근사만 가능.

### 핵심 질문
- **Denoising chain backprop 없이, action likelihood 없이, off-policy로 flow-matching VLA를 어떻게 value-gradient(1차 정보)로 개선할 것인가?**
- **Critic의 ∇_A Q (clean action endpoint에서만 정의)를 어떻게 noisy intermediate denoising state의 velocity-field 감독으로 변환할 것인가?**

📌 [Figure 1 삽입] — (a) RLT readout 위 frozen-VLM+action-injection Cal-QL critic 학습, (b) ∇_A Q → denoising-time velocity correction 변환, (c) Few-shot SFT pi_0.5 → 자가 rollout으로 정책 amortize.

---

## 2. 방법론 심층 분석

### 2.1 전체 구조
- **Base policy**: pi_0.5 — frozen VLM backbone + flow action expert vθ(x_t, t, c). Linear flow path x_t = (1-t)A + tε, t∈[0,1], **t=0이 clean action** (논문의 시간 컨벤션).
- **Critic state**: VLA prefix를 그대로 입력 못 함 → RL Token([28]) 디자인 따라 readout vector z_rl ∈ R^2048로 압축. 여기에 projected proprioception 연결: s̄ = LayerNorm([z_rl ‖ W_p p]) ∈ R^2304.
- **Critic head**: action chunk A를 **모든 hidden layer에 재주입**(per-layer action injection) — h_0 = g_0([s̄;A]), h_{ℓ+1} = g_{ℓ+1}([h_ℓ;A]). State dim이 action dim보다 훨씬 크기 때문에 입력 단에만 concat하면 critic이 A 변화에 둔감해진다는 관찰 기반.
- **Ensemble Cal-QL**: M개 head 평균 Q(s,A) = (1/M) Σ Q_m. 2-layer (1024 → 512), 1 head당 ~3.9M params. Cal-QL([30]) = TD loss + α_cql · CQL conservative penalty (MC return 이하만 underestimation 회피).

### 2.2 VGG-Flow optimal-control view ([1])
- KL-regularized policy improvement (목적식 (1)): p*(x_0) ∝ p_base(x_0) · exp(r(x_0)/λ).
- Flow에서 이를 실현 = base velocity에 residual h를 더하는 stochastic optimal control 문제 (식 (2)).
- **최적 residual velocity h*(x_t, t) = β ∇_{x_t} V(x_t, t)**, β = 1/λ (식 (3)).
- Clean endpoint t=0에서 V(x_0, 0) = -r(x_0) → ∇_{x_0} V(x_0,0) = -∇_{x_0} r(x_0) (식 (4)). 본 논문은 **r(x_0) = Q_ψ(o, x_0)**로 인스턴스화.

### 2.3 Q-VGM 알고리즘 (Algorithm 1)
한 training iteration마다:
1. **Look-forward clean-action estimate** (식 (5)): 각 denoising state x_k에서 **frozen base velocity**로 one-step Euler projection — Â^k_base = x_k - t_k · sg[v_base(x_k, t_k, s)]. Frozen base를 쓰는 게 critic 학습 분포 안에 머무는 데 핵심(ablation: 이 anchor 빼면 92.5→86.5).
2. **Iterative projected Q-gradient ascent** (식 (6)): Â^{k,j+1} = Π_A[Â^{k,j} + α · clip_G(∇_A Q|_{Â^{k,j}})], J번 반복. Π_A는 valid action range로 clip, clip_G는 gradient magnitude bound.
3. **Keep-best selection**: j* = argmax_j Q(s, Â^{k,j}). j=0이 unmodified base 예측이므로 ascent가 Q를 개선 못 하면 **자동으로 원래 액션으로 fallback**. 이게 per-sample adaptive β_eff 역할 — discrete line search on Q landscape.
4. **Velocity correction** (식 (7)): h^k_eff = (Â^k_base - Â^{k,j*}) / t_k. Base 도착점에서 critic-improved 도착점으로의 평행이동을 t_k로 나눠 velocity 단위로 변환.
5. **Residual velocity matching loss** (식 (8)): L_align = Σ_k s(t_k) · ‖ (vθ(x_k,t_k,s) - v_base(x_k,t_k,s)) - sg[h^k_eff] ‖². **Denoising-time gate s(t_k) = (1-t_k)^p**가 clean에 가까울수록(즉 t_k 작을수록) 강한 가중. Look-forward 근사가 reliable한 영역에 guidance를 집중.

### 2.4 Stop-gradient 구조
- Denoising states {x_k}, base policy v_base, critic Q, velocity targets h_eff 모두 **stop-gradient**.
- Gradient는 오직 local prediction vθ(x_k, t_k, s)를 통해서만 흐름 → **denoising chain backprop 완전 회피**, 학습 안정성 확보.
- 추론 시에는 vθ = v_base + hθ로 sampling만 — critic guidance가 action expert에 amortize되어 test-time search 불필요.

> ❓ **예상 질문**: 왜 clean action에서만 정의되는 ∇_A Q로 모든 denoising step의 velocity를 supervise할 수 있나?
> **답변**: 핵심 trick은 (a) frozen base velocity로 x_k → Â^k_base를 **clean action으로 projection**하고 (b) 거기서 ∇_A Q로 ascent하여 Â^{k,j*}를 얻은 뒤 (c) 둘의 차이를 1/t_k로 나눠 **velocity로 환산**한 것. 이는 VGG-Flow의 ∇_{x_t} V 전체 trajectory 전파를 풀지 않고 **clean endpoint에서의 ∇_A Q만으로 근사**한 것이며, gate s(t)=(1-t)^p가 t→0 부근(즉 look-forward가 정확한 영역)에 가중을 집중하여 근사 오류를 통제한다.

> ❓ **예상 질문**: PA-RL의 distillation과 본질적 차이는?
> **답변**: PA-RL([18])도 ∇_a Q ascent로 개선된 액션을 만들지만, 이를 **terminal supervised label**로 distill — flow의 velocity field는 무시. Q-VGM은 같은 ascent 결과를 **velocity 단위로 변환하여 residual velocity matching**으로 학습 → flow parameterization을 보존. Table 1에서 86.3 vs 92.5의 6.2pp 차이가 이 차이의 효과.

---

## 3. 데이터 전략

### 학습 데이터
- **Few-shot SFT 단계**: LIBERO 각 suite마다 소수 demos로 pi_0.5 fine-tuning(정확한 demo 수 LIBERO에 명시 X; real-robot은 task당 **30 teleop demos**).
- **Rollout buffer**: LIBERO suite당 **300 rollout episodes** (SFT 정책으로 수집), RoboTwin 2.0 task별 SFT-policy rollouts, real-robot task당 **100 rollouts**.
- **Critic 학습**: 이 fixed buffer 위에서 Cal-QL. 추가 환경 interaction 없음(fully offline).

### Cal-QL 학습 디테일 (Appendix A)
- Chunk-level transitions (s_t, A_t, R_t, s_{t+H_a}, A_{t+H_a}, d_t), R_t = Σ r_{t+j}.
- TD target에 **dataset의 next action chunk** 사용 — 완전한 off-policy.
- CQL penalty: L_CQL = E_s[log Σ_A exp Q_m(s,A) - E_{A∼D}[Q_m(s,A)]].
- Cal-QL calibration: MC return을 초과하는 Q-value만 penalty → in-support action의 불필요한 underestimation 방지.

### RLT autoencoder
- 2-layer transformer encoder-decoder, 2048-dim tokens, 8 heads. Frozen VLA prefix embedding에 대한 MSE 복원 학습 → held-out cosine similarity >0.95.

> ❓ **예상 질문**: Fixed rollout buffer만 쓴다는데 critic이 학습 분포 밖에서 부정확하면?
> **답변**: 논문 Limitations에서 명시적으로 인정 — ∇_A Q는 **rollout이 supporting하는 영역에서만 신뢰**. 완화책: (a) gradient clipping(clip_G), (b) keep-best fallback(개선 못 하면 base로 복귀), (c) frozen-base anchor로 look-forward를 behavior 분포에 묶음. 그러나 path-space deviation을 sampling dynamics에 internalize하는 trust region(Trust-Region Q Adjoint Matching [35] 같은) 추가 안전장치는 향후 과제로 남김.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Base policy | pi_0.5 (VLM frozen, action expert만 update) |
| Critic state | s̄ ∈ R^2304 = LayerNorm([z_rl(2048) ‖ W_p·proprio]) |
| Critic head | 2 layers (1024→512), per-layer action injection, ensemble M heads |
| Params per critic head | ~3.9M |
| Critic objective | Cal-QL (TD + α_cql · CQL) on chunk-level transitions |
| Action chunk H_a | 5 (real-robot 7-DoF 설정에서 35-dim flattened input) |
| Denoising steps K | Euler steps for sampling (논문에 specific K 명시 안 됨) |
| Q-gradient ascent J | iterative steps, step size α |
| Gate | s(t) = (1-t)^p (clean 근처 강조) |
| Velocity matching | residual matching with stop-gradient on x_k, v_base, Q, h_eff |
| LIBERO rollouts/suite | 300 episodes |
| Real-robot SFT demos | 30 / task |
| Real-robot rollouts | 100 / task |
| Hardware | 논문 본문에 명시 X |
| Open source | No (코드/체크포인트 공개 정보 없음) |

---

## 5. 실험 설계 및 평가 프로토콜

세 환경:
1. **LIBERO** — 4 suite (Spatial, Object, Goal, Long), 7-DoF EEF action, **각 task당 50 independent rollouts**.
2. **RoboTwin 2.0** ([32]) — 14-DoF dual-arm qpos, 10개 bimanual task, task당 50 rollouts.
3. **Real-robot tabletop** — 7-DoF arm + 2 RGB-D cameras (head + front), 2 task(Pick Peach, Stack Bowls), task당 **20 physical trials**.

**공정 비교 원칙**: 모든 critic-기반 baseline은 **동일 SFT checkpoint, 동일 rollout data, 동일 RLT features, 동일 Cal-QL critic** 공유 — critic을 어떻게 쓰는지만 다르게.

비교 baseline:
- Test-time Q Selection ([16,33]) — sampled chunk를 critic value로 re-rank.
- Test-time Q Guidance ([21,34]) — inference-time ∇_A Q refinement.
- Q-Improved Action Distillation (PA-RL [18]) — critic-improved action을 supervised label로 distill.
- Diffusion-QL ([15]) — denoising chain backprop with Q-maximization.

---

## 6. 실험 결과 심층 분석

### LIBERO (Table 1, 평균 성공률 %)

| Method | Spatial | Object | Goal | Long | Avg |
|--------|---------|--------|------|------|-----|
| pi_0.5 few-shot SFT (init) | 82.0 | 84.0 | 78.0 | 56.0 | 75.0 |
| Test-time Q Selection | 88.0 | 91.0 | 85.0 | 64.0 | 82.0 |
| Test-time Q Guidance | 91.0 | 93.0 | 88.0 | 68.0 | 85.0 |
| Q-Improved Action Distillation | 91.0 | 92.0 | 88.0 | 74.0 | 86.3 |
| Diffusion-QL | 73.0 | 78.0 | 74.0 | 53.0 | 69.5 |
| **Q-VGM (Ours)** | **96.0** | **95.0** | **95.0** | **84.0** | **92.5** |

- **Long suite에서 +28pp (56→84)** — 가장 큰 절대 향상. Long-horizon에서 velocity correction이 누적 오차 보정에 결정적.
- Diffusion-QL이 SFT(75)보다 **5.5pp 하락** → VLA scale에서 denoising chain backprop이 실제로 불안정함을 정량적으로 입증.
- 두 번째로 좋은 PA-RL(86.3) 대비 **+6.2pp** — velocity field supervision의 가치.

### RoboTwin 2.0 (Table 2, 10 task)

| Task | SFT | Q-Sel | Q-Guid | Q-Distill | **Ours** |
|------|-----|-------|--------|-----------|----------|
| adjust bottle | 80.0 | 84.0 | 86.0 | 88.0 | **94.0** |
| shake bottle | 88.0 | 92.0 | 90.0 | 92.0 | **94.0** |
| lift pot | 54.0 | 56.0 | 58.0 | 60.0 | **70.0** |
| place container plate | 86.0 | 88.0 | 90.0 | 90.0 | **94.0** |
| stack bowls two | 84.0 | 86.0 | 88.0 | 88.0 | **92.0** |
| handover mic | 70.0 | 72.0 | 76.0 | 76.0 | **86.0** |
| place empty cup | 86.0 | 90.0 | 88.0 | 90.0 | **94.0** |
| beat block hammer | 78.0 | 82.0 | 84.0 | 82.0 | **90.0** |
| place shoe | 50.0 | 52.0 | 54.0 | 56.0 | **66.0** |
| click bell | 88.0 | 90.0 | 90.0 | 92.0 | **92.0** |
| **Average** | **76.4** | 79.2 | 80.4 | 81.4 | **87.2** |

- **+10.8pp 평균 향상**. 가장 큰 gain은 SFT 성능이 낮은 task들 — place shoe(+16), handover mic(+16), lift pot(+16). Contact-rich/coordination failure modes에서 critic guidance가 효과적.
- Test-time 방법들은 80% 부근에서 plateau, training-time velocity-field supervision이 그 위 headroom을 잡음.

### Real-Robot (Table 3)

| Task | SFT (30 demos) | **Ours** | Δ |
|------|---------------|----------|---|
| Pick Peach | 9/20 (45.0%) | 15/20 (75.0%) | **+30.0** |
| Stack Bowls | 7/20 (35.0%) | 12/20 (60.0%) | **+25.0** |
| **Avg** | **40.0** | **67.5** | **+27.5** |

- Sim 결과(LIBERO/RoboTwin)와 일관 — 실세계에서도 fully-offline value-gradient matching이 작동.
- 한 task당 100 rollouts라는 비교적 적은 데이터로 +27.5pp.

> ❓ **예상 질문**: Diffusion-QL이 75→69.5로 오히려 떨어진 게 단지 hyperparameter 문제 아닌가?
> **답변**: 가능성은 있지만, 논문은 모든 baseline이 **same SFT checkpoint, same critic**을 쓰는 통제된 비교. 저자 주장은 "denoising chain 전체로 gradient propagation이 billion-scale에서 unstable" — 이는 Diffusion-QL 원 논문([15])이 작은 diffusion policy 기준이라는 점과 부합. 큰 폭 하락은 instability의 정성적 증거.

> ❓ **예상 질문**: RoboTwin "easy/hard" 구분 없이 v2 평균만 보고하는데?
> **답변**: 논문은 RoboTwin 2.0의 10 task **개별 점수**를 보고하며 easy/hard 분류는 명시하지 않음. 평균 87.2%는 10 task 단순 평균.

---

## 7. Ablation 분석 (Table 4, LIBERO Avg %)

| Variant | Avg SR (%) | Δ from Full |
|---------|-----------|-------------|
| **Full method** | **92.5** | — |
| _Critic-side_ | | |
| ResNet encoder (RLT 대신) | 82.5 | -10.0 |
| No per-layer action injection | 87.5 | -5.0 |
| Single critic head | 89.5 | -3.0 |
| _Policy-side_ | | |
| No keep-best (last iterate) | 88.5 | -4.0 |
| All-step alignment (s(t)=1) | 86.0 | -6.5 |
| No frozen-base anchor | 86.5 | -6.0 |

해석:
- **RLT vs ResNet (-10.0pp)**: VLA-derived rich state feature가 multi-task value learning에 결정적. ResNet은 visual feature는 잡지만 multimodal task context는 부족.
- **Per-layer action injection (-5.0pp)**: 입력 단 concat만으로는 high-dim state feature에 액션이 묻혀 ∇_A Q signal이 약해짐.
- **Ensemble (-3.0pp)**: smoother gradient + 후보 selection 안정성.
- **Keep-best (-4.0pp)**: gradient ascent가 overshoot할 때 base로 fallback 못 하면 정책이 잘못된 방향으로 끌려감.
- **Gate s(t)=(1-t)^p (-6.5pp)**: clean에 가까울수록 look-forward 근사가 정확하다는 가정의 정량적 뒷받침. 모든 denoising step에 일률 적용은 noisy 영역 supervision으로 오염.
- **Frozen-base anchor (-6.0pp)**: 현재 정책으로 look-forward 추정하면 정책이 drift하며 critic 학습 분포 밖으로 빠져나감 → unstable.

> ❓ **예상 질문**: 6개 ablation이 모두 양의 효과를 보이는데 누가 가장 중요한가?
> **답변**: **RLT encoder (-10pp)가 단일 최대 기여** — VLA representation의 풍부함이 모든 후속 critic/policy 학습의 기반. 다음이 gate(-6.5)와 frozen anchor(-6), 즉 **look-forward 근사를 신뢰할 수 있는 영역으로 제한**하는 두 메커니즘. 이는 방법의 핵심 가정(t→0에서만 clean-projection이 valid)이 실험적으로 sharp하게 검증됨을 의미.

---

## 8. 관련 연구 비교

| Method | Flow VLA용 | Likelihood 불필요 | Chain backprop 회피 | Velocity field 감독 | LIBERO Avg |
|--------|-----------|------------------|---------------------|--------------------|-----------:|
| Diffusion-QL [15] | △ (diffusion 원본) | ✓ | ✗ (전체 chain) | ✗ | 69.5 |
| Test-time Q Selection [16,33] | ✓ | ✓ | ✓ (정책 X) | ✗ | 82.0 |
| Test-time Q Guidance [21,34] | ✓ | ✓ | ✓ | ✗ | 85.0 |
| PA-RL [18] | ✓ | ✓ | ✓ | ✗ (terminal label) | 86.3 |
| Adjoint Matching [25] / Q-Adjoint [26] | ✓ | ✓ | ✓ (adjoint) | ✓ (different formulation) | n/a |
| **Q-VGM** | ✓ | ✓ | ✓ | **✓ (residual velocity matching)** | **92.5** |

### 핵심 차이
- **Q-Adjoint Matching [26]과의 차이**: 둘 다 chain backprop 회피, adjoint vs value-gradient view라는 다른 수학적 framing. Q-VGM은 **iterative Q-gradient ascent + keep-best**라는 discrete optimization을 velocity correction의 source로 삼아 per-sample β_eff를 자동 조절.
- **PA-RL과의 차이**: 같은 ∇_A Q ascent를 쓰지만 PA-RL은 terminal action으로 distill, Q-VGM은 **velocity 단위로 변환하여 flow 구조를 보존**.
- **VGG-Flow [1]과의 차이**: VGG-Flow는 unconditional generative model에서의 value-gradient matching을 제안. Q-VGM은 이를 **conditional VLA action flow + off-policy critic + look-forward clean-action projection**으로 인스턴스화.

---

## 9. 한계 및 미해결 문제

### 저자 명시 한계
1. **Critic 신뢰 영역 의존**: ∇_A Q는 rollout-supported 영역 밖에서 부정확. clip + keep-best로 완화하지만 근본적 해결은 trust region 도입 필요(미래 과제).
2. **Critic scaling**: task horizon과 환경 다양성이 늘면 Q-learning 자체가 scalable하지 않음 ([36]). World model과 결합한 multi-step TD가 후속 방향.

### 추가 미비점
3. **Hyperparameter 미보고**: K(Euler steps), J(ascent steps), α(ascent step size), p(gate exponent), λ(KL coeff)의 실제 값과 sensitivity가 본문에 sharp하게 정리 안 됨.
4. **Real-robot 규모**: 2 task × 20 trial = 40 episodes는 통계적 신뢰성 측면에서 작음. 더 다양한 task/object/lighting 확장 필요.
5. **Backbone 의존성**: pi_0.5에서만 검증. 다른 flow VLA(pi_0, GR00T-Flow 등)에 plug-and-play 가능한지 확인 안 됨.
6. **No code release**: 재현성 한계. action injection 방식, ensemble M, gate p 등 구현 디테일이 결과에 큰 영향을 미치는 만큼 코드 부재는 큰 약점.
7. **Online RL 미평가**: fully offline only. Online interaction 가능 환경에서의 sample efficiency 비교 없음.

### Attribution 문제
- LIBERO Long의 56→84 점프 중 **velocity matching의 기여 vs Cal-QL critic 품질의 기여** 분리가 명확치 않음. Critic이 동일한 PA-RL(74)과의 격차 10pp는 velocity matching의 효과로 볼 수 있으나, "동일 critic + non-velocity baseline의 최선" 비교만 있음.

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — VGG-Flow의 conditional VLA 인스턴스화 + look-forward projection + keep-best의 조합은 깔끔하고 독창적 |
| **Technical depth** | ★★★★★ — Optimal control 유도부터 Cal-QL, action injection, gate, anchor까지 체계적 ablation으로 각 요소의 필요성을 sharp하게 입증 |
| **Experimental rigor** | ★★★★☆ — Sim 2개(LIBERO/RoboTwin 2.0) + real-robot 통제 비교는 강점. 그러나 real-robot 규모/다양성은 작고 hyperparameter sensitivity 부족 |
| **Practical impact** | ★★★★☆ — Few-shot SFT → 자가 rollout만으로 +17.5pp(LIBERO)/+10.8pp(RoboTwin)/+27.5pp(real)는 실용적으로 매우 큰 향상. 추가 expert demo 없이도 자가 개선 가능한 점이 핵심 가치 |
| **Writing quality** | ★★★★☆ — 수식 전개와 알고리즘이 명료. Figure 의존도가 약간 높음 |

**강점**:
- VLA scale에서 작동 가능한, **chain backprop도 likelihood도 필요 없는** 최초 그룹의 value-gradient 방법.
- Same-SFT/same-critic 통제 비교로 "velocity-field supervision의 효과"를 깨끗하게 분리.
- Real-robot까지 일관된 향상.
- Ablation이 핵심 가정(look-forward를 t→0에서만 신뢰)을 정량적으로 뒷받침.

**약점**:
- Online RL 비교, hyperparameter sensitivity, backbone generalization 검증 부재.
- 코드/체크포인트 미공개로 재현 장벽.
- Real-robot scope이 작음.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | t→0에서만 신뢰하면 t≈1 부근(noisy)에서는 supervision이 0인 셈인데, 그쪽 velocity는 누가 학습하나? | Gate s(t)=(1-t)^p이 t→1에서 거의 0이지만 0은 아니어서 약한 supervision 존재. 또한 base v_base가 conditional flow matching으로 이미 학습된 상태이며, residual hθ만 학습 → t≈1에서는 hθ≈0으로 base에 가깝게 머무는 것이 자연스러움. Ablation에서 s(t)=1로 강제하면 오히려 -6.5pp로 악화. |
| 2 | Frozen base anchor vs current policy anchor의 trade-off — 정책이 발전하면 anchor도 갱신해야 하지 않나? | 논문은 fully offline 가정 하에 frozen이 critic 학습 분포 안에 머무는 데 유리하다고 입증(-6pp). 그러나 online setting에서 buffer가 갱신되면 anchor를 주기적으로 update하는 것이 자연스러움 — 논문은 이 확장을 다루지 않음. |
| 3 | Keep-best가 j=0(base) fallback을 한다면 결국 critic이 약하면 base와 동일해지는 것 아닌가? | 정확히 그 의도. Adaptive safeguard — critic이 신뢰 영역 안에서 개선 가능할 때만 update 적용. Conservative한 설계가 instability의 원천이 되는 잘못된 critic gradient를 차단. Ablation에서 keep-best 빼면 -4pp, 이는 ascent overshoot이 실제로 빈번함을 시사. |
| 4 | Cal-QL의 CQL penalty 강도 α_cql는 어떻게 설정? | 본문 명시 X. Appendix에 더 정보 있을 가능성. 일반적으로 Cal-QL은 calibration 덕에 α_cql에 덜 민감한 것으로 알려져 있으나, VLA scale critic에서의 sensitivity는 미검증. |
| 5 | Per-layer action injection이 효과적이라면 critic 구조를 더 크게 키우면 더 좋은가? | 논문은 2-layer(1024→512), ~3.9M/head에 머묾. Scaling law 분석 없음. 다만 ensemble M heads의 효과(-3pp)는 명시 — diverse head로 gradient smoothness 확보. |
| 6 | LIBERO Long 56→84의 성공이 단순히 SFT의 underfitting을 보완한 것 아닌가? | 부분적으로 맞을 수 있음. 그러나 동일 SFT에서 시작한 다른 baseline들의 최대치가 74(PA-RL) — 같은 underfitting을 동일하게 본 baseline들 중 Q-VGM이 +10pp 추가. 이는 단순 SFT 보완 이상의 효과. |
| 7 | RoboTwin 2.0 14-DoF에서 ∇_A Q의 차원성이 LIBERO 7-DoF의 2배인데 critic gradient의 noise도 늘지 않나? | 가능성 있음. 그러나 실제 RoboTwin Avg 향상은 +10.8pp로 여전히 큼 — gradient clipping과 keep-best가 noise robustness에 기여한다는 정황. Chunk H_a 차원이 곱해지면 더 큰 action vector가 되지만 per-layer injection으로 sensitivity 보존. |
| 8 | Off-policy 가정인데 SFT 정책으로 수집한 rollout만으로 충분한 coverage가 나오나? | Few-shot SFT 정책이 일정 성공률(LIBERO 75%, RoboTwin 76%)을 보이므로 reward signal이 있는 trajectory가 buffer에 존재. 그러나 더 어려운 task에서는 successful trajectory가 희소해질 우려 — Limitation에서 critic scaling 이슈로 언급. |
| 9 | Adjoint Matching([25,26])과 직접적 head-to-head 비교는 왜 없나? | 논문은 Q-Adjoint([26])를 "closest motivation but different formulation"으로 언급만 함. 동일 SFT/critic 셋업에서의 직접 비교가 없는 게 빈틈 — 후속 작업의 자연스러운 baseline. |
| 10 | Inference 시 vθ = v_base + hθ로 sampling한다면 별도 K Euler step 비용 외 추가 비용은? | 없음. critic guidance가 hθ로 amortize되었기 때문에 test-time critic 호출이나 multi-sample 검색이 없음. 이는 PA-RL과 동일한 efficiency 장점, Test-time Q Guidance 대비 강점. |

<!-- VERIFIED: pdf -->
