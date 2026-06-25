# PolicyTrim: Boosting Intrinsic Policy Efficiency of Vision-Language-Action Models

> **한 줄 요약**: VLA 배포의 본질적 병목인 "정책 효율(policy efficiency)"을 처음으로 정식화하고, 두 단계 GRPO 후학습(신뢰 가능한 action chunk 확장 + 중복 step 제거)으로 LIBERO에서 성공률을 유지하면서 물리 step을 51.4% 줄이고 최대 5.83× end-to-end 가속을 달성한 ECCV 2026 논문.

## 1. 배경 및 동기

- VLA 모델의 실세계 배포 효율은 (i) per-step 추론 지연(compute-centric)과 (ii) 총 추론 호출 횟수(policy-centric)의 곱으로 결정된다. 기존 효율화 연구(token pruning, KV cache, quantization 등)는 (i)에만 집중하여 정책 자체의 비효율은 그대로 남는다 (§1, §2.2).
- 저자들은 두 가지 핵심 비효율을 관찰: (a) action chunk의 **꼬리 예측 신뢰도 저하**(tail degradation) — 긴 chunk를 실행시키면 SR이 감소하고 물리 step이 오히려 증가 (Fig. 1b), (b) **물리 step의 중복성** — 같은 태스크 반복 rollout에서 step 수 분산이 매우 커서 더 짧은 경로가 실제로 도달 가능함을 시사 (Fig. 1a).
- 이 두 축이 곱해져 총 추론 호출 수를 결정하므로, 정책 효율을 직접 최적화하면 compute-centric 가속과 곱셈적으로(orthogonally) 누적 가능하다는 가설.

## 2. 방법론

PolicyTrim은 GRPO 기반 2단계 RL 후학습 프레임워크다. 두 목표(긴 chunk 신뢰성 ↑, 총 step ↓)가 보상 신호를 얽히게 하기 때문에 **순차적**으로 분리한다 (§3.1, §3.3 말미).

### 2.1 Stage 1: Reliable Action Chunk Extension (§3.2)
- **Dynamic Execution Horizon Exploration**: 각 group의 trajectory τᵢ에 서로 다른 acceptance ratio γᵢ ∈ Γ = {γ₁,…,γ_M}을 할당하여 execution window hᵢ = ⌊γᵢH⌋를 다양화. 한 group 내에서 짧은/중간/긴 horizon을 동시에 탐사 → "reliability sweep".
- **Reliable Horizon Reward** (Eq. 1–2): `R_horizon(τᵢ) = β·γᵢ`. 단, group 내 성공 trajectory가 하나도 없으면 보상 신호가 horizon-biased가 되어 잘못된 긴 chunk를 강화할 수 있으므로, **성공 인디케이터 I_succ로 게이팅**: `R_ext(τᵢ) = I_succ^(i) · (R_succ + R_horizon)`.
- **GRPO 업데이트** (Eq. 3–4): group-normalized advantage `Aᵢ = (R_ext − μ_R) / (σ_R + ε)`, clipped surrogate + per-token KL penalty to π_ref. 동일 task instance 내에서 서로 다른 execution length를 직접 대조하므로 어떤 chunk 위치가 신뢰 가능한지 학습 가능.

### 2.2 Stage 2: Redundancy-Aware Step Reduction (§3.3)
- **Step-Saving Reward** (Eq. 5): `R_step(τᵢ) = max(0, S_base − S(τᵢ)) / S_base`. S_base는 초기 정책의 평균 성공 step의 약 1.3배로 설정.
- **Group-Anchored Stability Penalty** (Eq. 6): `P_stab = λ_stab · tanh(|S(τᵢ) − μ_group| / max(σ_group, σ_floor))`. group 내 성공 trajectory의 평균/표준편차를 anchor로 사용해, 우연히 발생한 짧지만 재현 불가능한 shortcut에 페널티. σ_floor로 분모 폭주를 방지(정책이 수렴해 σ_group이 작아져도 안정).
- **결합 보상** (Eq. 7): `R_eff = I_succ · (R_succ + R_step − P_stab)`. 실패 rollout은 step-saving 보상을 받지 못함.

### 2.3 학습 설정 (§4.1)
- RLinf 프레임워크 기반, group size G=8, M=3 (γ₁은 base 모델 default execution length에 맞춤, 나머지는 1까지 등간격).
- 8×A100 + 64 parallel envs.
- 적용 대상: π0.5 (flow-matching diffusion), OpenVLA-OFT (parallel decoding with placeholder tokens), GR00T (dual-system with diffusion head).

## 3. 실험 결과

### 3.1 LIBERO (Table 1)
| Model | Subset | Baseline SR/Steps/h | PolicyTrim SR/Steps/h | Speedup |
|---|---|---|---|---|
| π0.5 | Spatial | 97.8 / 108.3 / 5 | 97.8 / 59.8 / 15 | **5.43×** |
| π0.5 | Object | 99.1 / 125.0 / 5 | 98.5 / 64.3 / 15 | **5.83×** |
| π0.5 | Goal | 98.7 / 110.6 / 5 | 98.8 / 63.5 / 15 | 5.23× |
| π0.5 | Long | 93.0 / 249.8 / 5 | 93.3 / 171.8 / 10 | 2.91× |
| GR00T | Object | 95.0 / 71.3 / 5 | 95.3 / 65.5 / 10 | 2.18× |
| OpenVLA-OFT | Object | 98.5 / 135.2 / 8 | 98.5 / 68.8 / 8 | 1.97× |

OpenVLA-OFT는 parallel decoder의 고정 horizon h=8 특성상 chunk 확장이 어려워 Stage 2만 적용 → 그래도 약 2× 가속.

### 3.2 ManiSkill & Meta-World (Table 2)
- π0.5/ManiSkill: 88.1→89.8 SR, 45.2→38.3 steps, **2.36× speedup**.
- π0.5/Meta-World: 65.1→65.4 SR, 66.3→52.6 steps, **2.52× speedup**.

### 3.3 Cross-Architecture (Table 3)
- h=16으로 재사전학습된 OpenVLA-OFT + 전체 2-stage: 98.8 SR, h=14, **2.97×**.
- Autoregressive OpenVLA에 Stage 2만 적용: 84.7→87.0 SR, **1.41×** (SR 동시 향상).

### 3.4 실세계 배포 (Table 4)
- Agilex Piper + RealSense D435i ×2, FlipMug/HangMug/TapeBox 3개 task.
- Standard 설정 평균 wall-clock 15.9s → 8.6s (**1.86× 평균**), SR은 동등 또는 향상. Dynamic perturbation 하에서도 SR 유지(70→70 Flip, 65→70 Tape).

### 3.5 Ablation (Tables 5–6, Fig. 4)
- Chunk Extension 단독: hchunk 5→15, 2.86× — 그러나 물리 step은 108.3→113.8로 **증가**(tail 오류로 인한 보정 동작). Stage 2의 필요성 입증.
- Step-Saving Reward 단독: 108.3→81.7 steps이지만 SR 97.8→**93.7로 급락**(fragile shortcut 붕괴).
- + Group-Anchored Regularization: SR 93.7→97.5 회복, steps 81.7→61.6 추가 감소. Fig. 4에서 정규화 없으면 step≈125에서 보상 collapse, 있으면 단조 증가.
- 전체 결합 시 SR 98.3, 59.8 steps, **5.43×** (Spatial).
- Dynamic Horizon Exploration vs Fixed γ (Table 6, H=20): Fixed γ=1.0은 SR 94.4까지 떨어지지만 Dynamic은 SR 98.8 유지하며 hchunk=15 달성.

### 3.6 Compute-centric 방법과의 직교성 (Table 7)
- OpenVLA-OFT + VLA-Cache 단독: 1.26× → +PolicyTrim: **최대 2.48×**(Object). 두 축은 곱셈적으로 결합 가능함을 실증.

## 4. 한계 및 미해결 문제

1. **Parallel-decoder 한정 Stage 1**: OpenVLA-OFT 원본(h=8 고정, 전부 실행)에는 Stage 1이 무의미. h=16으로 재사전학습이 필요한 비용 발생.
2. **Autoregressive VLA의 chunk 확장 불가**: OpenVLA에는 Stage 2만 적용 가능(§4.2). Action chunk가 동일 fixed-horizon parallel decoding으로 생성되지 않아서.
3. **Sim→Real에 SFT 보정 필요**: 실세계 배포는 시뮬에서 RL 후 소량의 실세계 demo로 SFT 보정. 순수 real-world RL은 비검증.
4. **하이퍼파라미터 민감도**: S_base를 baseline 평균의 1.3배로 휴리스틱 설정. σ_floor, λ_stab, β의 robustness 분석은 부재.
5. **3개 base VLA(π0.5/OpenVLA-OFT/GR00T)로 한정**: RT-2, π0, CogACT 등 다른 아키텍처에서의 일반성은 미검증.
6. **코드 미공개**: 프로젝트 페이지만 존재(https://inceptionwang.github.io/PolicyTrim/), 재현성 검증 불가.

## 5. 총평

- **Novelty: ★★★★☆** — "policy efficiency"라는 축을 compute-efficiency로부터 명시적으로 분리·정식화한 점, 그리고 그것을 GRPO group-relative 구조에 horizon sweep과 group-anchored shortcut 방지로 자연스럽게 녹여낸 설계가 깔끔. 단일 reward로는 풀리지 않는 두 목표를 sequential stage로 분리한 결정도 합리적.
- **Practical impact: ★★★★★** — 추가 demonstration이나 아키텍처 변경 없이 후학습만으로 5.83× 가속 + SR 유지. 더욱이 VLA-Cache 같은 기존 가속과 곱셈적으로 결합 가능. 실제 로봇 1.86× 가속까지 보였다는 점에서 즉시 production 가치 있음.

VLA 효율화 연구가 거의 전적으로 per-step latency에 매달려 있던 흐름에 "총 호출 횟수"라는 직교 축을 개척한 작업. 특히 Group-Anchored Regularization이 단순 정규화 그 이상으로 SR과 step efficiency를 **동시에** 개선했다는 관찰은 GRPO 일반에서도 시사하는 바가 크다.

## 6. 예상 질문

| Q | A |
|---|---|
| 왜 chunk 확장과 step 감소를 동시에 학습하지 않고 sequential로? | 동시에 하면 두 reward가 conflate되어 최적화 지형이 복잡해지고, 긴 chunk만 강화되어 step이 오히려 증가(Table 5의 chunk-only 결과: 108.3→113.8 steps). 먼저 신뢰 가능한 horizon을 확립한 뒤 그 위에서 step 감축이 안전 (§3.3 말미). |
| Group-Anchored Regularization은 왜 tanh를 쓰나? | normalized deviation을 [0,1]에 bounded mapping하여, σ_group이 매우 작아져도 penalty가 폭주하지 않고 smooth하게 outlier 억제. σ_floor와 결합되어 수렴 후에도 well-conditioned 업데이트 보장 (§3.3 Eq. 6 설명). |
| 성공 trajectory가 group에 하나도 없으면? | Stage 1에서 `I_succ` indicator로 horizon reward를 게이팅하여 `R_ext = 0` (Eq. 2). 그렇지 않으면 horizon reward가 SR 신호 없이 긴 chunk만 강화해 unreliable behavior로 collapse. |
| OpenVLA-OFT 원본에 Stage 1을 적용 못하는 이유? | 원본 parallel decoder가 h=8 고정 horizon으로 8개 action을 모두 실행하므로 "γ를 작게 잡아 truncate"한다는 메커니즘 자체가 작동할 여지가 없음. 그래서 h=16으로 재사전학습한 변종에 한해 Stage 1+2 적용 (§4.2). |
| 정책 효율 향상이 compute-centric 가속(KV cache 등)과 정말 곱셈적으로 합쳐지나? | Table 7에서 OpenVLA-OFT/LIBERO-Object 기준 VLA-Cache 단독 1.26×, +PolicyTrim 2.48× — 약 두 배 누적. 두 축이 서로 다른 자원(per-step latency vs 총 호출 수)에 작용하므로 이론적으로도 곱셈적 결합이 자연스러움. |
| 실세계 RL 직접 수행했나? | 아니오. 시뮬레이션(LIBERO/ManiSkill/Meta-World)에서 RL 후 소량의 real demo로 SFT 보정해서 Agilex Piper에 배포. 그럼에도 wall-clock 1.86× 가속(Table 4). |

<!-- VERIFIED: pdf -->
