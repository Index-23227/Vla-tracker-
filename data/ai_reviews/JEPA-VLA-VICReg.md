# JEPA-VLA + VICReg: Output-Level Regularization Eliminates the Seed Lottery in Single-GPU VLA Fine-Tuning

> **한 줄 요약**: 단일 RTX 5090에서 VLA-JEPA를 LIBERO에 fine-tuning할 때, 동일 코드/데이터/아키텍처에 *seed만 다른* 13회 실행 중 12회는 91-94% Object SR을 달성하지만 1회는 65.2%로 조용히 붕괴하는 **"seed lottery"** 현상을 정량화하고, **frozen encoder가 만드는 trainable predictor Jacobian의 null-space**가 그 구조적 원인임을 보임. L2/EWC 같은 weight-level 정규화는 null-space 방향에 budget을 낭비하여 (EWC 파라독스: drift 7.8로 가장 작은데도 65.8% Object SR) lottery를 유지하는 반면, **output-level 정규화 3종(patch-level VICReg, Dropout p=0.1, LowLR=halved LR)**은 21회 결합 실행에서 0회 붕괴를 보임. VICReg 변형은 cross-seed std=1.4% (n=12)로 가장 타이트한 분포를 달성하며 이는 *평균 SR 향상이 아닌 tail-risk elimination*이 핵심 기여임.

---

## 1. 배경 및 동기

### Single-GPU VLA fine-tuning의 숨겨진 위험

- RT-2, OpenVLA, π0, VLA-JEPA, SpatialVLA, OpenVLA-OFT 등 대부분의 최신 VLA는 **frozen vision encoder + trainable action head** 구조를 차용한다. 이는 8xA100 같은 대형 클러스터에서 정상 작동한다고 *암묵적*으로 가정되며, single-GPU(소규모 학술 랩, 스타트업) 환경에서의 *재현성*은 체계적으로 연구된 적이 없다.
- 저자(Skoltech)는 RTX 5090 32GB 한 장으로 VLA-JEPA를 LIBERO에 fine-tuning하던 중, **동일한 코드/데이터/하이퍼파라미터로 seed만 바꿔서 3회 실행**했더니 LIBERO-Object SR이 65.2% / 93% / 94%로 나오는 충격적인 현상을 관찰했다 (Section 1, A의 "Experimental Journey" 1-4주차).
- 더 충격적인 점은: **첫 모델 실패는 어떤 error message나 training-curve 이상**으로도 사전에 알 수 없다는 것. 65.2% seed의 train loss/val loss는 정상 seed들과 *동일한 궤적*을 보이고, full closed-loop evaluation을 마쳐야만 비로소 실패가 드러난다.

### 핵심 질문

1. 이 분포가 정말로 bimodal인가, 아니면 일반적인 seed-noise인가? → 13 seed 체계적 실험.
2. 무엇이 catastrophic seed를 만드는가? Weight-space drift인가, output-space collapse인가?
3. 어떤 종류의 정규화가 lottery를 제거하는가? 그리고 *왜* 그런가?

### Lottery는 task suite에 따라 다르다

- LIBERO-**Object** (object identity 기반 pick-and-place): Baseline 65.2-94.2%, std 7.5% (n=13). **Lottery의 본진**.
- LIBERO-**Spatial** (positional precision): Baseline std 1.0% (n=9). 거의 lottery 없음.
- LIBERO-**Goal** (long-horizon language-guided): Baseline std 0.9%. Mild.

저자 가설: 정책이 *붕괴*하면 "평균 grasp"으로 default되는데, Spatial은 기하학적으로 평균적인 grasp이 통하지만 (kethup 98-100%), Object는 6.7cm 떨어진 cream cheese vs. butter를 구별해야 하므로 discriminative feature 없이는 실패한다 (Section E의 canary task 분석).

---

## 2. 방법론 심층 분석

### 2.1 아키텍처: JEPA-VLA 백본 (그대로 사용)

Figure 2의 구조:

```
[Frozen]                                  [Trainable]
V-JEPA2 ViT-L/16  ── z_t (P=256, D=2048)  ── ACPredictor (12-L Transformer, RoPE, d=1024) ─→ z_hat ─→ L_wm (L1 vs sg(z_T))
                                                                                              │
                                                                                              ↓
                                                                                          (VICReg ours)
                                                                                              │
Qwen3-VL-2B        ── h_act + h_emb       ── FlowMatching ActionHead (16-L DiT, H=7) ───────→ a_hat ─→ L_act (flow matching)
(eager attention)
```

- 입력: T=4-8 frame video + 현재 RGB + language instruction.
- V-JEPA2가 video를 패치 임베딩(B x P x D = B x 256 x 2048)으로 인코딩.
- Qwen3-VL-2B가 현재 image + language를 action token h_act, embodied token h_emb으로 인코딩.
- ACPredictor (12-layer Transformer, RoPE, d_model=1024)가 미래 패치 임베딩 z_T를 예측 → world-model L1 loss.
- FlowMatchingActionHead (16-layer DiT, H=7 step horizon, 7-DoF)이 action chunk 생성 → straight-path flow matching loss.

### 2.2 핵심 기여: Output/Weight Taxonomy

**Proposition 1**: 정규화는 *제약하는 공간*에 따라 두 부류로 나뉜다.
- **Output-level**: g_theta(phi(x))의 다양성을 제약 → VICReg (z_hat의 variance floor), Dropout (stochastic perturbation), LowLR (early-phase collapse basin 진입 회피).
- **Weight-level**: |Δθ|를 제약 → L2, EWC.

LIBERO-Object 결과: output-level 92-95%, weight-level ≤66%. **이 분할이 모든 7개 method를 cleanly separate한다.**

### 2.3 Observation 1: Jacobian Null-Space (구조적 설명)

성능은 |J Δθ| (output 변화)에 의존하지 *|Δθ|*에 의존하지 않는다. 여기서 J = ∂g_theta/∂θ.

Frozen encoder 하에서 φ(x)는 stationary distribution이며, J가 **rank-deficient**가 된다: dim(null(J)) >> rank(J).
- L2/EWC는 모든 방향에 동일하게 budget을 분배 → null-space 방향에 대부분 소비 → output-active 방향이 *unconstrained*.
- Output-level은 g_theta(phi(x))를 직접 제약 → null-space를 우회.

**Empirical evidence**:
- EWC drift = 7.8 (가장 작음) yet 65.8% Object SR.
- VICReg drift = 40.2 yet 92.7% Object SR.
- L2 drift = 140.4 (18x EWC) yet 62.0% Object SR.
- LowLR drift ≈ 10.1 (EWC와 VICReg 중간) yet 94.5% Object SR.

→ Weight proximity는 task performance를 예측하지 못한다 (Figure 1e).

### 2.4 Patch-Level VICReg (B=1 안정화의 트릭)

표준 VICReg은 B >> 1을 요구하지만 단일 GPU에서는 B=1이 강제됨. 저자는 **spatial-temporal patch dimension을 sample 축**으로 사용:

```
z_hat ∈ R^{B x (T-1) x P x D},  reshape → z_flat ∈ R^{N x D},  N = B(T-1)P = 768
```

768개 패치 예측을 자연스러운 "batch"로 본다.

**Variance loss** (dimensional collapse 방지):
```
L_var = (1/D) * Σ_d max(0, γ - σ_d(z_flat)),   γ = 1.0
```

**Covariance loss** (feature redundancy 방지):
```
L_cov = (1/D) * || C(z_flat) - diag(C(z_flat)) ||_F^2
```

**Total loss**:
```
L = L_act + λ_wm (L_wm + λ_var L_var + λ_cov L_cov)
```
with λ_var=1.0, λ_cov=0.04, γ=1.0. **VICReg invariance term은 의도적으로 생략** (predictions should vary with observations).

B=1 안정성은 cross-seed std=1.4% (n=12), drift 40.23±0.005 (n=3)로 경험적 검증됨.

### 2.5 Dropout과 LowLR (단순 대안)

- **Dropout p=0.1**: ACPredictor와 ActionHead에 적용. Stochastic perturbation으로 narrow output manifold로의 routing을 막음. 한 줄 코드.
- **LowLR**: 두 LR을 절반으로 (5e-6, 1.5e-5). **Zero code change** -- config 두 줄만 수정.
  - 메커니즘: weight를 anchoring하는 것이 아니라, action loss가 trajectory를 교정하기 *전에* collapse basin으로 빠르게 진입하는 것을 막음. Output-space mechanism이지만 weight-space implementation.
  - Drift ≈ 10.1로 EWC(7.8)와 비슷하지만 SR 94.5% (EWC 65.8%) → drift magnitude 자체가 원인이 아님을 확인.

---

## 3. 실험 셋업

### 3.1 Base / Compute / Data

- **Base**: VLA-JEPA (He et al., 2026) pretrained on LIBERO at 8xA100 (~30k steps), reaching 96.4% Spatial SR.
- **Compute**: 1x NVIDIA RTX 5090 32 GB. Gradient checkpointing 필수.
- **Frozen**: V-JEPA2 ViT-L/16, Qwen3-VL-2B (eager attention).
- **Training**: 4개 LIBERO suite 동시 학습 (~1693 episodes, LeRobot v2.1, AV1 256x256).
- **Optimizer**: AdamW. Default LR (Baseline/VICReg/Dropout/L2/EWC): 1e-5 ACPredictor, 3e-5 ActionHead. LowLR: 5e-6, 1.5e-5.
- **Steps**: VICReg 4k; 그 외 5k.

### 3.2 Evaluation

- LIBERO 3-suite closed-loop evaluation: Spatial / Object / Goal.
- 500 ep per suite (10 task x 50 trial).
- Multi-seed block은 mean ± population std로 보고.

### 3.3 Seed 수

- Baseline: n=9-13 (Object=13)
- VICReg: n=9-12 (Object=12)
- Dropout: n=3-4
- LowLR: n=5 (Object)
- EWC: n=4 Object, 1 elsewhere
- L2: n=1

---

## 4. 주요 결과

### 4.1 Table 1 (Multi-seed)

| Method | Class | n | Spatial | Object | Goal |
|--------|-------|---|---------|--------|------|
| VLA-JEPA pretrained | — | 1 | 96.4 ± 1.7 | 78.0 | 82.0 |
| Baseline | None | 9-13 | 95.1 ± 1.0 | **91.0 ± 7.5** ← lottery | 90.5 ± 0.9 |
| Dropout (p=0.1) | Output | 3-4 | **95.4 ± 0.4** | 91.8 ± 0.8 | 90.8 ± 2.2 |
| VICReg | Output | 9-12 | 94.9 ± 0.5 | 92.7 ± **1.4** | **93.6 ± 1.4** |

핵심 통계:
- Baseline Object **bimodal**: 1 seed at 65.2%, 12 seeds at 91-94%. *Collapse 제외* 시 Baseline 92.4%±1.3% (n=12) ≈ VICReg 92.7%±1.4% (t(22)=0.6, p=0.55) → **평균 동일, VICReg의 가치는 tail-risk 제거**.
- Variance ratio F(12,11) = **28.7, p<0.001** (Levene robust p<0.01).

### 4.2 Table 3 (Single-seed ablation, 500 ep per suite)

| Method | Class | Spatial | Object | Goal |
|--------|-------|---------|--------|------|
| VLA-JEPA pretrained | — | 96.4 ± 1.7 | 78.0 ± 8.0 | 82.0 ± 7.5 |
| **VICReg** | Output | 93.8 ± 2.1 | 91.6 | 90.0 |
| **Dropout** | Output | 96.0 ± 1.7 | 91.8 | 90.8 |
| **LowLR** | Output | 94.6 ± 2.0 | **94.5 ± 0.7** | 92.8 |
| L2 | Weight | 87.2 ± 2.9 | **62.0** | 66.0 |
| Baseline (collapsed seed) | None | 86.8 ± 3.0 | **65.2** | 90.4 |
| EWC (id. Fisher) | Weight | 86.8 ± 3.0 | **65.8** | 67.4 |

**The central dichotomy**: 모든 output-level은 pretrained Object 78%를 넘고; 모든 weight-level은 ≤68% Object/Goal로 붕괴.

### 4.3 VICReg λ sensitivity

| λ | Spatial | Object | Goal |
|---|---------|--------|------|
| 0.1 | 95.6 ± 1.8 | 92.4 | 94.0 |
| 0.5 | 94.6 ± 2.0 | 89.8 | 93.2 |
| **1.0** (default) | 93.8 ± 2.1 | 91.6 | 90.0 |
| 2.0 | 95.4 ± 1.9 | 92.2 | 94.6 |

20x 범위 sweep에서 ≤5 pp 분산. **VICReg는 tuning에 fragile하지 않다.**

### 4.4 Collapse diagnostics (Table 2)

- **Latent-level** ρ̄-lat ≈ 0.998 *모든* method에서 동일 → encoder-level collapse는 무정보.
- **Action-level** ρ̄-act에서 VICReg가 *qualitative outlier*: 0.571 (gap 0.073 to nearest). N=200, 500에서도 robust.
- **The discriminating signal is exclusively at the predictor output level.**

### 4.5 Canary task

LIBERO-Object Task 1 (cream cheese pick-and-place, Section E Table 6):
- Weight-level collapsed seeds: 0-2% (완전 실패).
- Output-level seeds: 82-100% (전부 성공).
- Task 4 (ketchup, 원통형): 모든 method 98-100% (geometry-trivial).

→ 정책이 *평균 grasp*으로 default되었는지 진단하는 single-task canary.

---

## 5. Ablation 및 추가 분석

### 5.1 The EWC paradox (sharpest evidence)

- EWC (identity Fisher): drift 7.8 (가장 nearest to pretrained), Object SR 65.8%.
- VICReg: drift 40.2, SR 92.7%.
- L2: drift 140.4, SR 62.0%.

**Weight proximity가 좋을수록 성능이 좋다는 직관과 정반대**. EWC는 초기 collapse basin 근처에 *training을 lock*해버린다 -- pretrained weights에 너무 충실해서 오히려 망한다는 결과.

### 5.2 LowLR의 메커니즘

Drift ≈ 10.1는 EWC와 VICReg 사이지만 SR은 best (94.5%). 이는 LowLR이 **weight를 anchoring하는 것이 아니라, 학습 초기 trajectory를 천천히 만들어 collapse basin 흡수 전에 action loss가 교정할 시간을 주는** 것임을 시사. Weight-space implementation이지만 *output-space mechanism*.

### 5.3 VICReg의 deterministic attractor

VICReg drift 40.23±0.005 (n=3)는 *seed에 거의 무관*. 이는 output constraint가 unique attractor를 강제한다는 fingerprint.

### 5.4 Falsifiable predictions

저자는 Jacobian null-space framework에서 3개의 testable prediction을 derive:
- **P1**: Encoder를 unfreeze하면 output/weight gap이 줄어든다 (null-space가 줄어들기 때문). 본 논문에서 encoder unfreeze 시도 → first 100 steps 내 gradient explosion on RTX 5090 → 검증 미완료.
- **P2** ✓: LowLR이 weight anchoring이 아니라 early collapse basin 회피로 작동 → drift ≈10.1로 EWC/VICReg 중간 ⇒ 확인됨.
- **P3**: VICReg + L2는 VICReg 단독을 능가하지 *못한다* (L2의 budget은 null-space에서 낭비). Future work.

---

## 6. 한계 및 미해결 문제

저자가 명시한 한계 (Section 6):

1. **Architecture coverage 부족**: VLA-JEPA + LIBERO에만 결과. OpenVLA, π0, SpatialVLA에서의 lottery rate는 미측정 (저자 주장: null-space structure는 frozen-encoder의 *수학적 귀결*이므로 정성적으로 동일할 것).
2. **Real-robot 부재**: Simulation only.
3. **Seed 수 부족**: Dropout n=3-4, LowLR n=5 → lottery elimination을 *individually* confirm하기에 통계력 부족. VICReg n=12만 strong.
4. **EWC는 identity-Fisher**: 진짜 Fisher information으로는 결과가 다를 수 있음.
5. **L2 n=1**: Single-seed.
6. **dim(null(J)) 미측정**: Observation 1의 정량적 검증이 부족.
7. **Data-mixture confound**: 4-suite joint training이 Object discrimination을 dilute할 가능성 배제 불가.

---

## 7. 강점 (개인 평가)

- **현상 자체의 가치**: Single-GPU 환경에서 1/13 silent failure rate는 *실제 배포 안전*과 직결되는 reproducibility crisis. 학계가 거의 보고하지 않은 측면을 명확히 드러냄.
- **Output/Weight taxonomy의 명료성**: 7개 method, 3개 suite에서 *clean dichotomy* (output 92-95%, weight ≤66%). 이런 sharpness는 드물다.
- **Jacobian null-space 가설의 설명력**: EWC paradox (drift 작은데 성능 최악)를 자연스럽게 설명.
- **LowLR의 실용성**: Config 두 줄만 바꿔서 best Object SR 94.5%. *Zero engineering overhead*는 비현실적으로 좋은 cost-benefit.
- **VICReg λ robustness**: 20x sweep에서 ≤5 pp → 실전 도입에 fragile하지 않음.
- **Canary task**: 단일 task 결과로 collapse를 사후-진단 가능 → field deployment 시 안전 검사로 활용 가능.
- **저자의 disclosure**: "AI usage disclosure: Claude Code was used for LaTeX/figures/grammar; all experimental results, analysis, and scientific conclusions were generated by the authors" -- modern reproducibility 모범.

---

## 8. 약점 / 비판

- **VLA-JEPA + LIBERO에 한정**: 가장 큰 한계. Frozen-encoder VLA family의 일반화는 *이론적 주장*에 머무름. OpenVLA/π0에서 lottery rate가 같은 1/13인지, 다른 mode를 보이는지 미검증.
- **Lottery rate의 통계적 정밀도**: 1/13 = 7.7% 추정에 대한 95% CI는 widely (0.2-36%). "1 in 13" 표현은 *오해의 소지*; 보다 많은 seed가 필요.
- **EWC를 identity-Fisher로 처리**: 실제 EWC는 task data의 Fisher information을 추정해야 함. Identity-Fisher EWC는 효과적으로 L2의 특수 경우. 진짜 EWC가 동일하게 실패하는지 미검증.
- **Weight-level method의 일반성에 대한 over-claim 가능성**: Surgical fine-tuning, BitFit, LoRA 같은 *구조적 parameter restriction*은 weight-level이지만 null-space와 다른 dynamics를 가질 수 있음. 본 논문은 L2/EWC 두 가지만 테스트.
- **Real-robot evidence 부재**: Sim-only 결과를 deployment safety crisis로 frame하는 것은 약간 leap.
- **Hyperparameter ablation 비대칭**: VICReg λ는 4개 sweep, Dropout p는 single value (0.1), LowLR은 single halving. Output-level 내 fairness 의문.
- **"Output collapse" 정의의 modal ambiguity**: Latent ρ̄-lat ≈ 0.998는 *모든* method에서 동일하다고 했지만 (Table 2), 이게 normal한지 abnormal한지 baseline 부재. Action-level ρ̄-act만 discriminative라는 결론은 약간의 post-hoc selection처럼 보임.
- **Spatial 결과 무차별**: Spatial에서는 모든 method가 86-96%로 비슷 → output-level의 advantage가 *Object/Goal에 집중*. 이 suite-dependence를 깊게 다루지 않음.

---

## 9. 관련 연구와의 비교

- **VLA-JEPA (He et al., 2026)**: 본 논문의 base. 8xA100에서 96.4% Spatial SR. 본 논문은 그 single-GPU 적응성 문제를 *드러냄*.
- **OpenVLA-OFT (Kim et al., 2025)**: Orthogonal fine-tuning으로 97.1% LIBERO 달성. 본 논문은 OpenVLA-OFT가 *parameter manifold*를 제약하므로 weight-level의 일종으로 분류될 가능성 지적 (미검증).
- **π0 / π0.5 (Black et al., 2024)**: Flow-matching action head 원조. 본 논문의 ActionHead가 π0와 유사한 16-layer DiT.
- **FiberTune (2606.08653)**: Action-fiber 잔차 보존이라는 *유사한 직관*에서 출발하지만, FiberTune은 action-orthogonal residual에 RADIO teacher 정렬 (representation level)이고, 본 논문은 predictor output 자체의 variance/covariance (output level). 두 연구가 *다른 trainable module에서 같은 collapse 가족*을 다룬다는 해석 가능.
- **VICReg (Bardes et al., 2022)**: SSL pretraining에서 invariance + variance + covariance. 본 논문은 *invariance를 빼고* variance + covariance만 fine-tuning에서 사용 -- predictions가 input에 따라 변해야 하므로 의도적 omission.
- **Mosbach et al. 2021** (BERT fine-tuning 안정성): 가장 가까운 방법론적 선례. 그러나 BERT는 *trainable encoder*이며 본 논문의 frozen-encoder structural argument가 적용되지 않음.
- **Henderson et al. 2018** (RL reproducibility): Seed가 2-5x performance variance를 만든다는 고전적 결과. 본 논문은 RL이 아닌 *supervised fine-tuning*에서, 그것도 *binary* failure mode를 보인다는 점에서 다름.
- **Mitigating catastrophic forgetting in VLAs (Grover & Zha, 2025)**: 본 논문은 catastrophic forgetting hypothesis를 *명시적으로 기각* (L2/EWC 모두 실패).

---

## 10. 향후 연구 방향

- **Cross-architecture replication**: OpenVLA, π0, SpatialVLA, OpenVLA-OFT, smolVLA에서 seed lottery rate 측정. 본 논문 framework가 universal한지 검증.
- **Encoder unfreezing의 정량화**: P1 (null-space shrinkage)를 측정 가능한 metric (effective rank of J)로 검증.
- **VICReg + Surgical/LoRA**: Output-level과 *structural* weight-level의 결합 (L2와는 다른 형태) 시너지 가능성.
- **Real-robot replication**: Sim2real lottery rate가 더 높은지 / 낮은지.
- **Online lottery detection**: Closed-loop evaluation 없이 *training time에* collapse-bound seed를 조기 탐지하는 indicator (action-level ρ̄-act 추정 등).
- **Stochastic Fisher EWC**: Identity Fisher 대신 진짜 EWC가 결과를 바꾸는지.
- **Task-dependent lottery intensity**: Why Object > Goal > Spatial? Action manifold의 *discriminative requirement*와 lottery rate의 정량적 관계.

---

## 11. 실용 가이드 (저자의 권고)

저자가 명시한 implementation cost별 정렬:

1. **LowLR (zero code change)**: optimizer config 두 줄 (LR halving). Best Object SR (94.5% ± 0.7%, n=5). *가장 추천*.
2. **Dropout p=0.1 (one line per module)**: ACPredictor + ActionHead에 dropout 추가. Best Spatial SR (95.4% ± 0.4%), strong Goal (90.8%).
3. **VICReg (~30 lines of PyTorch)**: Patch-level variance + covariance on z_hat. Best Goal SR (93.6% ± 1.4%), tightest cross-seed std (1.4%, n=12), mathematical variance floor → safety-critical settings.

세 방법 모두 *어떤 weight-level method보다 단순*하면서 더 안정적. 특히 LowLR은 L2/EWC를 *세팅하는 것보다도 쉬움*.

---

## 12. 종합 평가

본 연구는 **single-GPU VLA fine-tuning의 reproducibility crisis**라는 잘 보고되지 않은 현상을 (1) 정량화하고 (2) Jacobian null-space framework로 기계적으로 설명하며 (3) zero-to-30 line 수정만으로 *완전히 제거*하는 세 가지 recipe를 제안한 매우 *high-impact, low-cost* 논문이다.

가장 큰 가치는 **EWC paradox**의 명료성: 가장 작은 weight drift가 가장 나쁜 성능을 낳는다는 counterintuitive한 결과가 weight-space vs. output-space framework로 깔끔히 설명된다. 단 Single-architecture / single-benchmark 한계는 명백하며, frozen-encoder VLA family 전체에 대한 일반화는 향후 검증 과제로 남는다.

실용적으로는 **LowLR (config 두 줄)**이 academic-lab/스타트업 환경에서 *즉시 적용 가능한* 가장 안전한 default가 된다. VLA를 single-GPU로 fine-tuning하는 모든 실무자는 이 논문의 권고를 채택해야 할 정도로 cost가 거의 0이다.

CoRL 2026 venue (1차 검증 통과)와 함께, 향후 VLA fine-tuning protocol의 *de facto standard*로 자리 잡을 가능성이 높은 work.

<!-- VERIFIED: pdf -->
