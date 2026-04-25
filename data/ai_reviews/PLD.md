# PLD: Self-Improving VLA Models with Data Generation via Residual RL

> **한 줄 요약**: NVIDIA GEAR/CMU의 PLD는 Probe(잔차 RL specialist 학습) → Learn(VLA prior로 base policy probing 후 specialist takeover) → Distill(SFT)의 3단계 post-training pipeline으로, π0/OpenVLA 모두에서 LIBERO 평균 99% 이상을 달성하고 SimplerEnv에서 +24.9%의 절대 개선을 기록한 self-improving 데이터 flywheel.

---

## 1. 배경 및 동기 (Section 1)

- **SFT 한계**: VLA의 표준 post-training인 SFT는 비싼 teleoperation demonstration에 의존하며, "operator가 미리 실패 모드를 예측해야" 한다는 근본적 한계 (Section 1, p.2). 결과적으로 in-distribution은 개선되어도 OOD/unseen task로 transfer가 보장되지 않음.
- **direct RL fine-tuning 한계**: OpenVLA-OFT의 LIBERO RL은 batch size 8에서 GPU당 ~62.5GB가 필요해 multi-task scaling이 어렵다 (Section 3, p.4 인용 of Kim et al. 2025). Sparse reward 환경에서 expressive flow head를 직접 Q-maximize 하는 것은 매우 unstable.
- **핵심 통찰**: "data collection should not be agnostic to the base policy" (Section 1). data collection 정책과 generalist가 상호작용해야 base의 trajectory distribution과 정렬된 회복 데이터를 얻을 수 있음.

---

## 2. 방법론 (Section 3, Algorithm 1, Figure 3)

### Stage 1 — Probe (Section 3.1, Eq. 2)
- VLA backbone π_b를 freeze하고, lightweight residual Gaussian policy π_δ(·|s, a_b)를 off-policy actor-critic으로 학습.
- Combined action: ā = a_b + a_δ, where a_δ ∈ [−ξ, ξ], ξ는 scheduler로 조절 (Section 3.1).
- Buffer는 offline B_offline (base policy 성공 rollout)과 online을 50:50으로 mini-batch 구성 → "value function이 항상 high-value state-action pair에 노출"되도록 보장.
- Q-function은 conservative objective (Cal-QL, Nakamoto 2024)로 warm-up; behavior constraint를 policy loss에 부과하지 않아 π̄가 base policy 품질에 덜 영향받음.

### Stage 2 — Learn / Hybrid Rollout (Section 3.2, Figure 4)
- "Base policy probing": 처음 T_base ~ U[0, αT] 스텝은 π_b로 굴리고 이후 specialist takeover.
- 결과 trajectory τ_demo = {(s_1, a_b,1), …, (s_t−1, a_b,t−1)} ∪ {(s_t, a_b,t + ā_t), …} → 회복 행동을 포함.
- α 민감도 ablation (Figure 11): α=0.6에서 성능 plateau (+4.3% over α=0.0). α를 더 늘리면 episode 길이가 비효율적으로 길어져 성능이 다시 하락.

### Stage 3 — Distill
- Hybrid trajectory를 그대로 base VLA에 SFT. flow-matching head (π0)와 autoregressive head (OpenVLA) 모두에 architecture-agnostic하게 적용 가능 (Section 3, "agnostic to VLA architectures").

---

## 3. 실험 결과

### LIBERO In-Distribution (Table 1)
| Model | Spatial | Object | Goal | Avg |
|-------|---------|--------|------|-----|
| π0 baseline | 95.2 | 97.6 | 87.4 | 93.4 |
| π0 + PLD | 97.7 | 98.5 | 95.3 | **97.2** (+3.8) |
| OpenVLA baseline | 92.9 | 99.1 | 83.25 | 91.8 |
| OpenVLA + PLD | 99.5 | 99.1 | 98.9 | **99.2** (+7.4) |

특히 LIBERO-Goal에서 OpenVLA +15.7%p의 큰 폭 향상. 이는 Goal task가 "long-horizon recovery"가 중요한 suite임을 시사.

### SimplerEnv (Table 2)
Octo-SFT 대비 평균 71.8 → **96.6** (+24.9% absolute). 가장 어려운 WidowX Pick Carrot에서 43.3 → 93.9 (+50.6%p)로 abstract에서 주장한 ">50% gain"의 근거.

### Generalization (Figure 2, Section 4.3)
- LIBERO-90의 10% task subset만으로 학습해도 unseen task에 24.4% SR로 zero-shot transfer (Figure 2 caption).
- 동일 budget의 self-bootstrap (0/1 REINFORCE)는 in-dist도 underperform하고 OOD에 generalization 실패.
- Long-horizon: LIBERO-90 → LIBERO-10 one-shot (Figure 6)에서 base-policy rollout 대비 향상하나 human demo에는 못 미침.

### 실세계 (Section 4.4, Figure 8)
- Franka peg insertion: PLD/RLPD/Human 모두 30/30 성공.
- Franka cube pick-up (randomized): **+D_PLD 30/30**, +D_RLPD 16/30, +D_Human 10/30. 실패 모드는 cube를 corner에 밀어넣어 gripper가 stuck되는 것 — RLPD/Human data는 corner state 미방문, PLD만 회복 행동을 명시적으로 probe.
- YAM bi-manual GPU insertion (Section 4.4): 4-stage state machine으로 분해, **1시간 무인 사이클** 성공.

---

## 4. 한계 및 비판적 검토

- **Specialist 학습 비용**: per-task로 250k online interaction (Figure 5)이 필요. multi-task scale에서 simulator 의존도가 높음.
- **Reward classifier 의존**: YAM 실험에서 stage 전환에 reward classifier 학습 필요 — 새 task마다 reward shaping 비용.
- **Long-horizon 한계 인정**: LIBERO-10 long-horizon에서는 여전히 human expert demo에 못 미침 (Figure 6).
- **Residual magnitude scheduler**: ξ scheduler가 manual hyperparameter — robust한 자동 조절 미해결.

---

## 5. 총평

PLD의 핵심 기여는 "RL data를 base policy의 deployment distribution과 정렬"한 점이다. Figure 10의 trajectory 시각화가 이를 가장 잘 보여준다 — RL expert만으로 수집한 데이터는 좁고 base와 멀리 떨어져 있으나, PLD data는 base 주변에 cluster를 형성하면서 다양한 회복 경로를 포함한다. 저자가 "KL-divergence as forgetting indicator" (Section 4.5, Shenfeld et al. 2025 인용)를 통해 LLM SFT의 분석과 연결시킨 점은 흥미롭다. LIBERO 99%는 거의 saturation이지만, **SimplerEnv +24.9%**가 더 의미있는 결과로 보인다.

---

## 6. 예상 질문 (세미나 Q&A)

- **Q1**: Residual policy를 별도로 학습하는 것보다 base VLA를 직접 RL fine-tune하면 안 되나?
  - A: Section 3에서 "expressive foundation policy를 직접 Q-maximize하는 것이 매우 어렵다"고 논증. 또한 OpenVLA-OFT 62.5GB/GPU 메모리 비용 인용.
- **Q2**: α=0.6이 최적인 이유?
  - A: Figure 11 — α<0.6에서는 data diversity 부족, α>0.8에서는 trajectory가 너무 길어져 optimality 저하 (+150 step). diversity-optimality trade-off의 sweet spot.
- **Q3**: Human demo 대비 진정으로 우월한가?
  - A: Table 2 SimplerEnv은 명확히 우월. 다만 Figure 6 LIBERO-10 long-horizon에서는 여전히 human이 best. PLD는 in-distribution과 short-horizon recovery에 강점.
- **Q4**: SFT/OFT distill 후 catastrophic forgetting은?
  - A: PLD data가 base policy 분포에 가까워 forgetting이 적음 (Section 4.5에서 KL divergence 관점 논의).

<!-- VERIFIED: pdf -->
