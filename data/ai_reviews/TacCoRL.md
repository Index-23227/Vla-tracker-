# TacCoRL: Integrating Tactile Feedback into VLA via Simulation

> **한 줄 요약**: 사전학습된 Pi0.5-style VLA에 **tactile encoder + contact-aware binary gating**을 붙이고, **(i) sim-real co-training으로 warm-start → (ii) calibrated simulator에서 sparse-reward PPO + real-data anchor**로 post-train. 4종 bimanual contact-rich task 평균 real-world 성공률 vision-only RL **50.0% → visuo-tactile 72.5%** (§4.4, Tab. 2), 시뮬레이션에서는 **60.5% → 78.5%** (§4.3, Tab. 1).

---

## 1. 배경 및 동기 (§1)

- VLA는 강력한 vision/language/action prior를 주지만, **삽입·조립·in-hand**처럼 정렬·접촉 위치·압력 분포가 **시각적으로 가려지거나 모호한** contact-rich task에서는 한계.
- 선행 tactile 통합 연구는 (a) **대규모 tactile pretraining** (Sparsh, Touch100k) 혹은 tactile-language-action 풀스택 재학습, (b) inference-time tactile guidance 가 대부분 — 비용/데이터 측면에서 비효율.
- 실데모만으로는 **near-failure 상태(미세 정렬 불량, 잘못된 표면 접촉, 불안정한 grasp)** 가 희소·위험. 따라서 **safe·resettable·verifiable** simulator를 closed-loop tactile 학습 환경으로 사용.
- 핵심 질문: "**대규모 tactile pretraining 없이도 기존 VLA backbone에 tactile feedback을 통합할 수 있는가?**" (§1)

---

## 2. 방법론 (§3, Fig. 2)

### 2.1 표기 및 정책 (§3, Eq. 1)
- 시각 `o^v_t`, proprioception `q_t`, tactile `o^τ_t`, 언어 `ℓ` → 관측 `x_t=(ℓ, o^v_t, q_t, o^τ_t)`.
- 정책 `A_t = a_{t:t+H-1} ∈ R^{H×d_a} ~ π_θ(·|x_t)`. 실환경 `E_real`, calibrated simulator `E_sim(ψ)` (privileged state `s_t`는 reward·critic 용).

### 2.2 Simulation Environment Alignment (§3.1, Fig. 4)
- 정책 facing 인터페이스 3축 정렬:
  - **Scene/camera** (Appendix B.3)
  - **Controller response**: per-joint `K_p, K_d, T_ref` 식별 (J4 grav-comp 잔차가 가장 컸음, Fig. 4A).
  - **Tactile statistics**: matched contact rollout 후 정규화 분포 일치 (real mean 0.616 vs sim 0.658, Fig. 4B).

### 2.3 Tactile-Augmented VLA (§3.2, Eq. 2-5)
- Base tokenization: `Z_t^base = [Z_ℓ, Z_v_t, Z_q_t]`.
- Tactile path: 최근 `L`-스텝 taxel window `h^τ_t ∈ R^{L×K}` → encoder `E_τ` + projection `W_τ` → `Z^τ_t = W_τ E_τ(h^τ_t) ∈ R^{M×d}`.
- **Contact-aware binary gate** (Eq. 4): 어느 한 타임스텝에라도 `λ_f` 이상으로 활성화된 taxel 수가 `m` 이상이면 `c_t=1`, 아니면 `Z^τ_t`를 attention에서 제거.
- **두 갈래 conditioning** (Eq. 5):
  - VLM 문맥 갱신: `Z̄_t^base = CrossAttn(Z_t^base, Z̃^τ_t)`
  - Action expert 입력: `Z_t = [Z̄_t^base, Z̃^τ_t]` — flow-matching denoising 매 step마다 tactile 토큰을 직접 조건으로 사용.
- 효과: 비접촉 구간에서는 사전학습된 VLA 동작을 보존, 접촉 구간에서만 tactile이 action chunk를 reshape.

### 2.4 Sim-Real Co-Training (§3.3, Eq. 6-7)
- 데이터: `D_real`(50 real demo/task), `D_sim^teleop`(20 sim teleop) → **MimicGen [54]** 으로 200 traj `D_sim^Mimic` 합성, 성공 predicate filter 동일 적용.
- 손실: `L_co = α·E_{D_sim}[ℓ_flow] + (1-α)·E_{D_real}[ℓ_flow]`, default **α=0.5**.
- 목적: **tactile-conditioned action prior** 부여 + RL의 viable initialization.

### 2.5 Post-Training with Real-Data Anchor (§3.4, Eq. 8)
- `min_{θ,ω} L_RL = L_PPO(θ,ω; E_sim(ψ)) + β·E_{D_real}[ℓ_flow]`.
- 보상은 **sparse task-level success/failure predicate**, critic `ω`는 simulator only.
- Real-data anchor `β`가 simulator-specific 행동으로의 drift 방지. 배포 시 critic·reward·privileged state 제거.

---

## 3. 실험 결과

### 3.1 4종 Bimanual Contact-Rich 태스크 (§4.2, Fig. 3, 5)
- Test Tube Insertion, Do Puzzle (3-piece 장horizon), Assembly #1, Assembly #2.
- 각 task: 50 real demo + 20 sim teleop (→200 Mimic), 128 parallel sim env, 20 real trial.

### 3.2 Simulation Success Rates (§4.3, Tab. 1)
| Setting | Vision-Only Avg | Visuo-Tactile Avg |
|---|---|---|
| After Co-Training | 0.330 | 0.405 |
| RL Start w/ Exploration Noise | 0.225 | 0.293 |
| **RL from Base VLA (no co-train)** | **0.00 (all tasks)** | **0.00 (all tasks)** |
| **RL with Co-Training (final)** | **0.605** | **0.785** |

- 핵심: **sparse reward만으로는 base VLA에서 0% — co-training이 exploration을 reduction**.
- Visuo-tactile gain은 **Test Tube Insertion (0.50→0.72)**, **Assembly #2 (0.61→0.79)** 에서 두드러짐 (각도 오차·미세 잔류 misalignment가 시각적으로 가려지는 시나리오).

### 3.3 Real-World Success Rates (§4.4, Tab. 2)
| Stage | Vision Avg | **Visuo-Tactile Avg** |
|---|---|---|
| Real-Only Fine-Tuning | 0.213 | 0.338 |
| Sim-Real Co-Training | 0.300 | 0.438 |
| **RL Post-Training (full)** | **0.500** | **0.725** |

- Per-task RL post-training visuo-tactile: TestTube 0.70 / Do Puzzle 0.45 / Assembly #1 **0.95** / Assembly #2 0.80.
- **Do Puzzle** 의 vision-only 25% → visuo-tactile **45%**: 장horizon 3-piece 삽입은 부분 성공으로 멈추는 베이스라인 대비 RL이 완전한 trajectory를 reinforce.

### 3.4 Ablation: α (co-train) & β (real anchor) on Assembly #2 (§4.5, Fig. 7)
- **α 효과 (β=0 고정)**:
  - α=0.95 → best sim 42.9%, real 40%
  - α=0.5 → best sim 70.3%, real 45%
  - α=0 (sim-only init zero-shot) → sim 14.1%, real 25% → "**일부 real 비중이 sim rollout과 정책 분포의 정렬에 필수**".
- **β 효과 (α=0.5 고정)**:
  - β=0.1 또는 1.0 → real **80%**, sim **>92%**.
  - β=5.0 → supervised loss는 최저지만 RL refinement가 억제 → sim·real 모두 하락 ("over-anchoring").

### 3.5 정성 분석 (§4.4, Fig. 5)
- in-hand object/gripper가 접촉 region을 가리는 가운데 정책이 **tactile history 기반 incremental translation·reorientation**로 미세 정렬을 보정.

---

## 4. 한계 및 미해결 문제 (§6)

1. **여전히 real tactile demo 필요**: 소량이지만 anchor용. tactile representation learning + sim-only 방향으로 더 줄일 여지.
2. **Digital twin 구축 비용**: per-task asset reconstruction, camera·tactile alignment가 수동.
3. **Hard contact만 검증**: 변형체·유체·연성 객체는 simulator 신뢰도 부족 → 미검증.
4. **Sparse reward 의존**: success predicate 정의가 필요한 task에 한정.
5. **Bimanual에 한정**: cross-embodiment 일반화 미검증.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — tactile-VLA 통합 문제를 **post-training (co-train + sim-RL + real anchor)** 로 재정식화. tactile pretraining 없이도 강한 효과. |
| **Technical depth** | ★★★★☆ — controller SysID + tactile calibration + contact-gated cross-attention + PPO with BC anchor의 시스템적 통합. |
| **Experimental rigor** | ★★★★☆ — 4 task × (sim + real) × 4 stage + α/β 2D ablation. RL-from-base 0% 결과는 강한 negative control. |
| **Practical impact** | ★★★★☆ — Pi0.5 같은 기성 VLA에 즉시 적용 가능한 sim-to-real recipe, 50→72.5% 절대 개선. |

---

## 6. 예상 날카로운 질문

| # | 질문 | 핵심 답변 |
|---|------|---------|
| 1 | RL이 base VLA에서 직접 0%인 이유? | Tab. 1: sparse task reward + 풍부한 contact 정렬이 동시에 요구되어 random exploration로는 도달 불가. co-training이 viable init 제공. |
| 2 | β=5.0이 왜 더 안 좋은가? | §4.5: anchor가 너무 강하면 RL의 reward-driven contact correction을 억제 → imitation으로 회귀, sim·real 모두 하락. |
| 3 | α=0(real only init)이 실패하는 이유? | §4.5: sim policy distribution과 너무 멀어 simulator rollout이 학습 outside-support. balanced co-training이 핵심. |
| 4 | TacVLA·VT-Refine와의 차이? | TacVLA는 real-only LoRA fine-tuning(시뮬 부재), VT-Refine는 sim fine-tuning 중심. TacCoRL은 **co-train + sim PPO + real BC anchor**를 한 파이프라인에 결합. |
| 5 | tactile gating이 학습되는가? | Eq. 4: hard binary `c_t` (taxel threshold·count). 학습 가능한 gating은 future work로 직접 언급(§6 시사). |
| 6 | privileged state는 배포 시? | §3.4: critic·reward·privileged state 모두 deploy 시 제거, 정책만 real로 transfer. |
| 7 | Do Puzzle의 의의? | 장horizon (3-piece) 시나리오: real-only/co-train은 partial completion에서 멈춤. RL이 **completion을 reinforce** → visuo-tactile +20%p 추가 개선. |

<!-- VERIFIED: pdf -->
