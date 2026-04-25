# π*₀.₆ (pi-star-0.6): a VLA That Learns From Experience

> **한 줄 요약**: Physical Intelligence가 제안한 RECAP(RL with Experience and Corrections via Advantage-conditioned Policies)으로 학습된 ~5B 규모(4B Gemma 3 + 860M flow-matching action expert) VLA. Demonstration + autonomous rollout + expert correction을 advantage conditioning으로 통합 학습하여 실세계 laundry folding, espresso 만들기, box assembly에서 throughput 2× 증가, failure rate 1/2 감소.

## 1. 배경 및 동기

- VLA가 시연 데이터만으로는 compounding error에 취약하고 사람 수준 robustness/속도에 도달하지 못함 (§I, II). 진짜 mastery는 "연습"(autonomous experience + correction)이 필요.
- 기존 RL-VLA 시도들의 한계: PPO/REINFORCE는 flow matching 모델의 log-likelihood 부재로 적용 어려움. 최근 작업들은 discrete action 또는 단순 Gaussian 분포만 다룸. 본 논문은 expressive flow matching VLA를 end-to-end RL로 학습.
- Pre-trained π₀.₅ → π₀.₆ → π*₀.₆ 순서로 발전: π₀.₆은 (i) pre-training dataset 확장, (ii) Gemma 3 4B VLM 채택, (iii) action expert 860M 파라미터로 증가. π*₀.₆는 advantage indicator를 추가한 RL-ready 변종.
- 핵심 motivation: 시연(demonstrations) + autonomous rollouts + human teleoperated interventions(human-gated DAgger)를 단일 RL framework로 통합.

## 2. 방법론

### 2.1 RECAP 알고리즘 (§IV, Algorithm 1)
- 4단계 반복 가능 루프:
  1. **Data collection**: 정책 π_ℓ^(k-1)을 task ℓ에 배포 → autonomous rollout + 선택적 human correction. Episode-level success label 부여.
  2. **Value function training**: V^πref ← multi-task distributional value function을 모든 누적 데이터로 fine-tune.
  3. **Advantage conditioned policy training**: V^πref로부터 binarized advantage indicator I_t를 계산해 정책 학습.
  4. (반복)

### 2.2 Distributional Value Function (§IV-A, Eq. 1)
- p_φ(V|o_t, ℓ) ∈ Δ^B (B=201 bins)을 사용. Empirical return R_t(τ) = Σ_{t'=t}^T r_t'를 discretize → cross-entropy 학습 (Eq. 1).
- 작은 670M Gemma 3 VLM 백본. Multi-modal web data와 co-train으로 overfitting 방지.
- VLA 학습 중 on-the-fly inference로 advantage 추정. Reward (§V-C, Eq. 5): r_t = 0(success terminal) / -C_fail(failure terminal) / -1(otherwise) → value는 "성공까지 남은 step의 음수"로 해석.

### 2.3 Advantage-conditioned Policy Extraction (§IV-B, Eq. 2-3)
- Bayes rule로 π̂(a|o,ℓ) ∝ π_ref(a|o,ℓ) [π_ref(a|I,o,ℓ)/π_ref(a|o,ℓ)]^β 도출 (Eq. 2). β=1일 때 π̂ = π_ref(a|I=True, o, ℓ).
- Improvement indicator I_t = 𝟙[A^πref(o_t, a_t, ℓ) > ε_ℓ]는 task-dependent threshold ε_ℓ(value 분포의 30%ile)로 결정.
- 학습 목적함수 (Eq. 3): min_θ E[-log π_θ(a_t|o_t, ℓ) - α log π_θ(a_t|I_t, o_t, ℓ)]. 즉 conditional + unconditional 모델을 동시에 학습 → CFG inference 가능.
- Human correction은 모두 I_t=True로 강제하여 positive supervision으로 통합.

### 2.4 π*₀.₆ 모델 (§V-A, V-B, Fig. 3)
- π₀.₆에 advantage indicator 텍스트 입력("Advantage: positive" / "Advantage: negative") 추가.
- Knowledge Insulation (KI) 학습 [73]: 전체 모델은 next-token prediction(FAST 토큰 + sub-task ℓ̂)으로 학습, action expert(860M)는 stop gradient로 분리되어 flow matching loss로 학습.
- 입력: 다중 카메라(base + 양팔 wrist), joint/gripper position. 50 Hz joint position control, 6-DoF 양팔.
- Loss는 sub-task ℓ̂ next-token loss + discrete action FAST loss + flow matching MSE의 합 (Eq. 4 lower bound 유도).

### 2.5 학습 파이프라인 (§V-D, Algorithm 1)
1. Pre-training: 다양한 로봇 플랫폼 demonstrations + web VLM data → V_pre, π_pre.
2. Per-task initialization: V_ℓ^0, π_ℓ^0 ← V_pre, π_pre를 D_ℓ(demonstrations only)로 fine-tune. SFT 단계로 I_t=True 고정.
3. Iterative improvement: K번 반복하여 데이터 수집 → V_ℓ^k → π_ℓ^k.

## 3. 실험 결과

평가 task (§VI-A, Fig. 6): 3 카테고리 (Laundry t-shirts/shorts, diverse 11 items, targeted-failure-removal / Cafe double espresso / Box assembly).

- **LIBERO (YAML 기준)**: Spatial 96.4 / Object 97.2 / Goal 95.6 / Long 83.2 → libero_avg 93.10. (논문 본문은 실세계 task 중심이라 LIBERO 표는 model card에 위임.)
- **Throughput (Fig. 7)**: 모든 task에서 final π*₀.₆가 baseline π₀.₅, π₀.₆, RL pre-trained π*₀.₆, offline RL+SFT π*₀.₆ 대비 압도. **Diverse laundry와 espresso에서 2× 이상 throughput 증가**.
- **Success rate (Fig. 8)**: 모든 task에서 단계적 RECAP 적용으로 향상. Diverse laundry와 espresso는 failure rate **2× 감소**. T-shirts/shorts, espresso, box assembly는 90%+ 성공률 달성.
- **Box assembly subtask (Fig. 8 right)**: Pick sheet / Build box / Label / Place in crate 4단계 모두에서 π*₀.₆이 일관된 최고 성공률. Folding과 labeling 모두 600초 내 ~90%.
- **Iteration 효과 (Fig. 9, 10)**:
  - T-shirt (autonomous only, 4 robots × 300 traj/iter × 2 iter): 2 iter로 throughput +50%, success rate는 1 iter에 90%+ 도달.
  - Box assembly (600 autonomous + 360 intervention/iter × 2): 2nd iteration에서 throughput **2×** 증가.
- **Policy extraction 비교 (Fig. 11, T-shirt+shorts)**: RECAP advantage conditioning이 AWR(slower policies)와 PPO(η=0.01 trust-region)를 모두 큰 차이로 능가. PPO는 off-policy setting에서 안정성을 위해 trust region을 좁혔지만 성능 저하.
- **Failure mode removal (Fig. 12)**: Strict success(collar facing up) 변형 task에서 2 iter × 600 traj로 **97% 성공률 + 빠른 속도** 달성. 적은 데이터로 specific 실패 모드 제거 가능.
- **실세계 deployment**: 13시간 연속 espresso 제작, 새 가정에서 2시간 이상 다양 laundry 접기, 공장 box assembly 운용 성공.

## 4. 한계 및 미해결 문제

1. **사람 의존**: Reward labeling, intervention 제공, episode reset이 모두 human in the loop (§VII). 완전 자율 RL이 아니며, scale에 한계.
2. **Greedy exploration**: 정책 stochasticity와 human intervention에만 의존하는 단순한 exploration 전략. 본문도 "더 정교한 exploration 필요"라고 인정.
3. **Iterated offline RL**: 데이터 batch 수집 → 모델 retrain → 반복하는 offline-style. Fully concurrent online RL이 아니라 sample efficiency와 update latency에 trade-off.
4. **Closed-source 데이터/평가**: 실세계 task evaluation이 Physical Intelligence 내부 robot setup에 의존하여 외부 재현이 어렵다. 코드(openpi)는 공개되어 있으나 RECAP 데이터와 specific task setup은 비공개.

## 5. 총평

- **Novelty: ★★★★☆** — 개별 기법(advantage conditioning [Frans et al., CFGRL], distributional value function, KI training, FAST tokenizer, human-gated DAgger)은 기존 작업의 연장선이지만, 이를 large flow-matching VLA에 통합하여 실세계 long-horizon task에서 작동시킨 첫 사례. PPO/AWR 비교(Fig. 11)에서 advantage conditioning의 우위를 명확히 입증.
- **Practical impact: ★★★★★** — 13시간 espresso, 2시간 laundry, 공장 box assembly 등 **실세계 production-grade deployment**가 가능한 수준. 대부분의 VLA 논문이 simulation이나 controlled lab task에 머무는 것과 차원이 다른 실용성. Throughput 2×, failure rate 1/2 감소는 비즈니스 임팩트가 큰 수치.

VLA를 "데모 모방"에서 "경험을 통한 mastery"로 끌어올린 마일스톤. RECAP의 advantage conditioning은 flow matching VLA의 RL 학습 표준이 될 가능성이 높다. 다만 reward labeling을 자동화하지 못한 점은 향후 연구 과제.

## 6. 예상 질문

| Q | A |
|---|---|
| Advantage conditioning이 PPO/AWR보다 우월한 근본 이유는? | (1) Flow matching 모델은 log-likelihood가 tractable하지 않아 PPO 적용이 어려움. (2) AWR은 데이터를 importance weight로 down-weighting하여 일부를 사실상 폐기. RECAP은 모든 데이터를 supervised + indicator로 활용. Fig. 11에서 실증. |
| Human correction이 없는 task에서도 효과 있는가? | T-shirt task는 autonomous-only로 2 iter에 throughput +50% 달성(§VI-C-2, Fig. 9). Correction이 없어도 RECAP은 작동하지만, 어려운 long-horizon task(box assembly)에서는 correction이 sample efficiency를 크게 높임. |
| Reward는 episode success label(0/1)인데 어떻게 fine-grained advantage가 나오나? | Sparse terminal reward를 distributional value function으로 학습하면 V(o_t, ℓ)가 "success까지 남은 step 수"를 예측하게 되어 dense advantage signal로 변환됨 (Fig. 4 visualization, Eq. 5). |
| ε_ℓ을 30%ile로 설정한 근거는? | §V-D에서 task별 value 분포의 30%ile을 threshold로 사용. 너무 엄격하면 positive 데이터 부족, 너무 느슨하면 improvement signal 약화. 30%로 trade-off를 잡고 β=1로 sampling. |

<!-- VERIFIED: pdf -->
