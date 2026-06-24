# WeaveLA: Event Driven Cross-Subtask Latent Memory Weaving

> **한 줄 요약**: π0.5 backbone을 frozen으로 두고, sub-goal completion event에서만 발화하는 Query-driven Memory Weaver가 직전 segment를 8개 latent token으로 압축해 다음 sub-task의 action expert(Gemma) AdaRMS에 직접 주입하는 cross-subtask memory 인터페이스. RoboMME의 SwingXtimes N=3에서 0% → 47.8%로 hardest repetition slice를 정확히 들어 올리면서 N=1은 ~100%로 유지, 메커니즘이 필요한 곳에만 작동함을 paired analysis로 입증.

## 1. 배경 및 동기 (§1)

- RT-2, OpenVLA, π0/π0.5, GR00T N1.x 등 VLA는 short-window 관측·proprioception·언어 instruction만으로 single-step grasp/press/place를 매우 잘 수행하지만, 현실 manipulation의 큰 부분은 **반복(assembly insertion, packaging N-items, swing N-times)** 작업이다.
- 반복 task의 본질은 "길이"가 아니라 **cross-subtask dependency** — 현재 stage의 올바른 행동이 직전 stage 완료 결과에 의존. Short-window VLA는 한 sub-task가 끝나도 다음으로 인계되는 compact summary 채널이 없어 collapse한다.
- 기존 memory-augmented VLA는 (1) per-frame write, (2) demo-stage retrieval, (3) sub-goal event write까지 일부 진보했지만 **sub-task → 다음 sub-task의 action expert로의 self-contained hand-off**를 누구도 제공하지 않았다.
- 저자의 핵심 명제: **sub-goal completion event가 cross-subtask 정보의 자연적 temporal unit**이며, 이 시점에서 다음 stage의 action 경로로 정보가 라우팅되어야 한다.

## 2. 방법론: 전체 구조 (§3.1, Fig. 2)

- Episode E = (τ₀, τ₁, …, τ_{K-1}). 표준 VLA가 o_t, s_t, ℓ로 action chunk a_t를 예측하는 데에 **하나의 cross-subtask memory channel**만 추가.
- τ_{k-1} 완료 event에서 latent m_{k-1}을 산출 → τ_k 내내 action expert를 조건화.
- 세 컴포넌트:
  - (A) Frozen perception backbone (π0.5의 PaliGemma + Gemma decoder)
  - (B) Sub-goal event에서만 발화하는 Query-driven Memory Weaver
  - (C) Action-side AdaRMS memory conditioning
- Base policy의 language prompt와 visual context는 **건드리지 않음** — interface가 action-side로 한정.

## 3. Sub-goal-event Trigger (§3.2)

- Memory write는 rollout 중 sub-goal completion event에서만 발생. Per-frame writer가 dense observation에서 boundary를 implicit하게 발견해야 하는 비용, demo-stage retrieval가 rollout-time progress에 key를 둘 수 없는 한계를 동시에 회피.
- Simulation 실험에서는 simulator가 boundary 제공 → memory mechanism과 boundary-detection 문제를 격리.
- e_k는 segment τ_{k-1}(frames between e_{k-1} and e_k)와 연결되며, event 사이에는 가장 최근 m_{k-1}을 계속 conditioning. Options framework(Sutton et al. 1999)의 termination condition과 같은 temporal abstraction.

## 4. Query-driven Memory Weaver (§3.3)

- 완료된 segment τ_{k-1}의 frame에서 frozen vision encoder visual token과 projected proprioceptive state token을 concat + LayerNorm → H ∈ ℝ^{B×L×d_h}, d_h=1024.
- **N=8 learnable query latent Q ∈ ℝ^{8×d_h}**가 H를 단 한 step의 attention pooling으로 요약:
  - A = softmax(QKᵀ/√d_h), m_{k-1} = W_out(AV + Q), K=V=H
  - Residual +Q로 attention pattern이 sharp할 때도 query에 gradient 유지
  - W_out: d_h=1024 → 출력 d=2048 linear projection
- Memory는 **segment-local**: m_{k-1}는 τ_{k-1}의 perceptual feature에서만 계산, m_{k-2}를 encoder에 다시 넣지 않음 — independent-per-hop 설계가 training stability에도 유리(§3.5, Appendix A.11).

## 5. Action-side Memory Consumption (§3.4)

- π0.5 instantiation: 각 action-expert Gemma block은 이미 두 timestep-conditioned AdaRMS(pre-self-attn / pre-FFN)를 가짐. WeaveLA는 둘을 그대로 두고 **세 번째 memory-conditioned AdaRMS**를 self-attn residual과 pre-FFN AdaRMS 사이에 삽입.
- Weaver token M ∈ ℝ^{B×8×d_m}을 action-expert width로 project: M̃ = MW_m ∈ ℝ^{B×8×d_act}.
- Layer ℓ의 action hidden state X_ℓ에 대해:
  - Q_ℓ = RoPE(RMSNorm(X_ℓ) W_ℓ^Q), K_ℓ = RoPE(RMSNorm(M̃) W_ℓ^K), V_ℓ = RMSNorm(M̃) W_ℓ^V
  - C_ℓ = ConcatHeads(softmax(Q_ℓ K_ℓᵀ/√d_head + 𝓜) V_ℓ) W_ℓ^O (𝓜는 padded slot mask)
  - [γ_ℓ, β_ℓ] = C_ℓ W_ℓ^Δ + b_ℓ^Δ, X_ℓ' = X_ℓ/√(mean(X_ℓ²)+ε) · (1+γ_ℓ) + β_ℓ
- Near-zero init으로 unmodulated RMSNorm 근방에서 시작 → 안전한 modulation.

## 6. Staged Training (§3.5)

- **Stage 0**: vision encoder frozen, flow-matching action loss만 활성 — target task family에서 competent short-window policy 확보.
- **Stage 1 (K=2 multi-subtask windows)**: Weaver + memory-conditioning을 도입하지만 action loss로만 학습 → memory pathway 안정화.
- **Stage 2 (K ∈ {2,3,4})**: semantic alignment + contrastive auxiliary로 full objective.
- 단일 merged stage로 시도하면 action/alignment loss는 안정적이지만 **weaver_latent_norm이 monotonically decay** (hidden representation collapse)하고 success가 near-zero로 붕괴. Action-grounded warm-up이 임의적 curriculum이 아니라 **필수 stabilisation**임을 진단(Fig. 7).

## 7. Training Objective & Computational Footprint (§3.6)

- Primary: π0.5의 flow-matching action loss. t ~ Beta(1.5, 1), a_t = tε + (1-t)a, v_θ(a_t, t | o_t, s_t, ℓ, m_{k-1})를 (ε - a)로 회귀:
  - L_action = E[‖v_θ(...) - (ε - a)‖²]
- Stage 2 보조 손실: L_align (Weaver latent ↔ sub-goal text encoder embedding, λ=0.05), L_ctr (같은 trajectory pull, 다른 trajectory push, λ=0.02). Sub-goal text는 **training-time only**, deployment에서는 episode-level instruction만 입력.
- Trainable: (i) PaliGemma LoRA in π0.5 backbone, (ii) Weaver query latent + projection, (iii) action-side cross-attention + AdaRMS scale/shift. 총 **~46M (≈1.4% of 3.4B frozen base)**. Memory write가 sub-goal boundary에서만 발생해 rollout cost가 dense per-frame writer 대비 무시 수준.

## 8. 실험: Aggregate (§4.1-4.2, Fig. 3)

- Benchmark: **RoboMME** (Dai et al. 2026), 16개 task × 50 episodes/task, Easy/Medium/Hard 3 pool, repetition task(PickXtimes, SwingXtimes)는 instruction이 sub-goal 수 N을 encoding.
- Backbone: π0.5 + Attention Pooling (π0.5 +AP). 6-task / 16-task 두 training scale.
- 6-task aggregate: Weaver-off **19.0%** → Weaver-on **24.7%**. 가장 큰 향상은 SwingXtimes 32→56, StopCube 8→22.
- 16-task aggregate: 17.3% → 23.3%. 동일 패턴으로 repetition / dependent-stop task가 주도.

## 9. Repetition / Difficulty / Paired 분석 (§4.3, Fig. 4)

- **N=1**(single sub-goal, dependency 없음): Weaver-on/off 모두 ~100% — mechanism이 **불필요한 곳에서는 켜지지 않음** 확인. Capacity boost 가설을 반증.
- **SwingXtimes N=3**: Weaver-off **0%** → Weaver-on **47.8%** (6-task). 16-task에서는 4% → 30%. 시각 scene이 swing 1과 swing 3에서 구분 불가하므로 cross-subtask channel만이 옳은 action을 결정.
- 6-task에서 N≥2 episode pooled: 7.2% → 24.6% (**3.4× relative**). 16-task: 5.8% → 17.4% (3.0×).
- Difficulty 계층화: gain이 **Hard episode에 집중** (1.4% → 12.5%), Easy/Medium은 marginal.
- Per-episode paired (50 matched episodes): SwingXtimes에서 Weaver-on 단독 성공 13 vs Weaver-off 1, StopCube 8 vs 1. 다른 task는 ±3 이내. **세 stratification 모두 동일한 mechanism-localisation 패턴.**

## 10. StopCube what/where/when Ablation & Extractor·Trigger 연구 (§4.3-4.5, Fig. 5, Table 1)

- StopCube는 RoboMME의 canonical count·time-critical task. WeaveLA 22.0%를 anchor로 한 1-축-한-번 ablation:
  - **What**: 모든 대체(recurrent TTT/RMT, sparse TokenDrop+Modul, non-oracle GroundSG+QwenVL, MemER, SAM2Act+)는 single digit으로 붕괴. Dense per-frame buffer **FrameSamp+Modul(42.0%)**만이 WeaveLA를 능가하는데, 이는 per-event vs dense per-frame의 trade-off.
  - **Where**: FrameSamp 계열에서 visual context 13.7% → separate expert 28.9% → action-side AdaLN 42.0% monotone. WeaveLA는 이미 action-side 정점.
  - **When**: no-write/per-step append/per-keyframe retrieval 모두 WeaveLA보다 낮음.
- **Extractor study (§4.4)**: Q-Former-style multi-layer decoder(π0.5 +QF)로 교체하면 16-task가 23.3% → **3.5%** 붕괴, 56.3% episode가 rollout timeout. 단일 step attention pooling이 rollout-time, action-grounded regime에서 더 안정.
- **Trigger source study (§4.5, Table 1 left)**: Stage 2 contrastive auxiliary(L_align + L_ctr)로 학습된 latent의 EMA cosine distance가 segment anchor에서 calibrated threshold를 초과하면 write — **deployment-time 외부 supervision 0**. 16-task aggregate latent-shift 17.4% vs oracle 17.0%로 indistinguishable. SwingXtimes N=3에서 4% → 21.7%로 oracle 30%의 68% gap 회복. 단 StopCube는 boundary가 순수 symbolic이라 latent-shift가 2%로 붕괴(oracle 12%).
- **Real-robot sanity (Table 1 right)**: 물리 PickXtimes(N=3, "put the blue block into the plate" × 3) n=20에서 Weaver-off **9/20** → Weaver-on **14/20**. Sim 패턴과 일관.

## 11. 한계 및 미해결 문제 (§5)

1. **StopCube에서 latent-shift trigger 붕괴**: symbol-counting task는 학습된 symbolic detector가 여전히 필요. Boundary가 perceptual shift로 표현되지 않을 때 본 trigger 방식이 무력.
2. **Q-Former extractor instability** (§4.4)는 미해명. 단일 step pooling이 안정한 이유의 mechanistic characterisation이 future work.
3. **Dense buffer와의 결합 부재**: FrameSamp(42.0%) > WeaveLA(22.0%) on StopCube. 두 mechanism이 what/when 축에서 orthogonal이므로 parallel modulation stream으로 합칠 가능성 — 미구현.
4. **Sub-goal annotation 의존성**: training-time에 sub-goal boundary와 sub-goal text가 필요. 실세계 적용 시 자동 annotation pipeline이 별도 과제.
5. **단일 backbone 의존**: π0.5에만 instantiate. OpenVLA/RT-2/GR00T N1.x 등 다른 architecture로의 일반화는 평가되지 않음.
6. **단일 benchmark family**: RoboMME 중심이고 LIBERO/CALVIN/SimplerEnv 등 표준 VLA benchmark에서의 비교가 없음 — repetition-bottlenecked task 외 일반 manipulation에서의 net cost 평가 부재.

## 12. 총평

- **Novelty: ★★★★☆** — "Sub-goal completion event가 cross-subtask hand-off의 자연 단위"라는 명제를 when/what/where 세 축으로 깔끔히 분해하고, MemGen(text-LLM)의 trigger–weaver factorisation을 embodied control에 retarget한 점이 명확. Per-event vs per-frame을 action-side AdaRMS 단일 modification으로 통합한 simplicity.
- **Mechanism-aligned evidence: ★★★★★** — Aggregate 숫자보다 **stratified analysis(N, difficulty, paired)** 가 mechanism이 필요한 slice에서만 켜짐을 일관되게 보여줌. SwingXtimes N=3 0→47.8% + N=1 unchanged + latent-shift trigger 17.4%로 oracle과 indistinguishable이 세 layer로 같은 신호.
- **Practical impact: ★★★☆☆** — π0.5의 ~1.4% trainable로 frozen backbone에 부착, sub-goal event에서만 write라 rollout cost negligible. 단 코드 미공개, 단일 benchmark, single backbone — VLA 분야 표준 메모리 인터페이스가 될지는 다른 backbone에서의 재현이 관건.

VLA가 "한 스텝씩 잘하지만 반복은 못한다"는 구조적 결함을, language prompt도 visual context도 건드리지 않고 **action-side AdaRMS 1줄 추가 + Weaver 8 query**로 푸는 minimal-surface intervention. Repetition-aware manipulation에서 mechanism-localised gain을 보여준 새로운 reference point.

<!-- VERIFIED: pdf -->
