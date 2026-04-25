# SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning

> **한 줄 요약**: DeepSeek-R1 스타일의 단순한 결과 기반(outcome-based) GRPO 강화학습을 OpenVLA-OFT에 적용하여 LIBERO 평균 99.1%(SoTA)를 달성하고, 데이터 부족과 일반화 한계를 동시에 해결한 효율적 VLA 온라인 RL 프레임워크.

## 1. 배경 및 동기

- VLA 모델은 대규모 사전학습 + SFT 패러다임으로 발전했지만, (i) 사람 시연 궤적 수집의 비용 문제(데이터 희소성), (ii) 분포 변화 상황에서 일반화 실패라는 두 근본적 한계를 가진다 (§1).
- DeepSeek-R1이 단순 결과 보상만으로 LLM의 step-by-step reasoning 능력을 극대화한 것에서 영감을 받아, 같은 패러다임이 VLA의 step-by-step action planning에도 적용 가능한지 질문한다.
- 기존 로봇 RL은 hand-crafted process reward에 의존하여 확장성에 제한이 있고, VLA rollout은 LLM과 달리 환경과의 다중 상호작용을 요구하여 비용이 매우 높다.
- veRL (Volcano Engine RL for LLMs)을 기반으로 VLA-specific trajectory sampling, 병렬 multi-environment rendering, 분산 학습-추론-렌더링 통합 구조를 구현하여 확장성 문제를 해결.
- "Pushcut" 현상(시연에 없는 push 동작이 RL을 통해 자발적으로 발현)을 발견하여 RL의 SFT 대비 본질적 우위를 시연.

## 2. 방법론

### 2.1 Interactive VLA Rollout (§3.1)
- VLA에 RL을 적용할 때 핵심은 다양성 있는 trajectory 생성이다. 세 가지 action decoding 방식 중 (1) action token distribution이 PPO-like RL에 가장 적합하다고 판단해 OpenVLA-OFT의 token-based decoding을 채택.
- LLM rollout과 달리 VLA는 환경과 closed-loop 상호작용이 필요. Listing 1에서 LLM `policy.generate(...)` 한 번 호출과 비교해 VLA는 매 timestep마다 `envs.step(actions)` 호출이 필요하며, 이를 multi-process 병렬 envs로 처리한다.

### 2.2 Outcome Reward Modeling (§3.2, Eq. 9)
- DeepSeek-R1을 따라 binary reward만 사용. 성공 시 trajectory 전체에 1, 실패 시 0을 부여하고 모든 action token에 균일하게 propagate.
- 복잡한 process reward 설계가 필요 없어 환경 간 전이성 확보.

### 2.3 Exploration Enhancements (§3.3)
세 가지 핵심 수정으로 vanishing gradient와 좁은 solution space 문제를 해결:
1. **Dynamic Sampling (Eq. 10)**: 모든 trajectory의 reward가 동일한 group을 제외하여 advantage가 0이 되는 상황 방지.
2. **Clip Higher**: GRPO clipping 범위를 [0.8, 1.2] → [0.8, 1.28]로 확장(DAPO 방식). 저확률 토큰의 확률 상승 여지 확보.
3. **Higher Rollout Temperature**: T=1.0 → 1.6으로 증가시켜 다양한 trajectory 생성. Figure 3에서 LIBERO-Long SR이 각각 +15%, +10%, +15% 향상됨을 보고.

### 2.4 Training Objective (§3.4, Eq. 11)
- KL divergence regularization을 제거(DAPO 방식)하여 reference model을 학습 중 유지할 필요가 없게 함 → 메모리 절감, 탐색 폭 확대.
- Hyperparameters: lr=5×10⁻⁶, batch=64, samples per group=8, ε_low=0.2, ε_high=0.28, T=1.6, action chunk=8(LIBERO)/25(RoboTwin), 256 action tokens.

## 3. 실험 결과

- **LIBERO (Table 2)**: OpenVLA-OFT base 91.0% → +RL 99.1% (Δ+8.1). Spatial 99.4 / Object 99.1 / Goal 99.2 / Long 98.5. π₀ (94.2), UniVLA (95.2)를 모두 능가.
- **RoboTwin1.0 (Table 3)**: 4개 dual-arm 태스크 평균 39.8% → 70.4% (Δ+30.6). DP3 (58.1)도 능가.
- **RoboTwin2.0 (Table 4)**: 12개 태스크 평균 38.3% → 68.8% (Δ+30.5). π₀ (49.2), RDT (33.3) 대비 압도적. Short 64.9 / Medium 72.5 / Long 69.0.
- **데이터 효율성 (Table 5)**: One-Trajectory SFT 기준 LIBERO 평균 48.9% → +RL 96.9% (Δ+48.0). LIBERO-Long의 경우 17.3% → 91.7% (Δ+74.4)로 +430.1% 상대 향상.
- **실세계 sim2real (Table 6)**: AgileX Piper 양팔 로봇 4개 태스크 평균 17.5% → 38.5% (Δ+21.0). Stack Bowls 38→70%, Click Bell 30→60%로 RDT (23.5%)도 능가.
- **Pushcut 현상 (Figure 5, §6.1)**: "move can pot"과 "place a2b" 태스크에서 시연 데이터에는 없던 push 동작을 자발적으로 발견.
- **실패 모드 (Table 7)**: 0-trajectory SFT 시 RL은 0%에서 향상되지 않음. 초기 모델 prior가 있어야 RL이 효과적임을 보임 (100 traj +18.1%, 1000 traj +22.2%).

## 4. 한계 및 미해결 문제

1. **Cold-start 한계 (§6.2, Table 7)**: 초기 SFT 능력이 0%에 가까우면 RL이 전혀 작동하지 않음. Outcome-only reward의 본질적 약점으로, 매우 어려운 새 태스크에는 여전히 SFT용 시연이 필요.
2. **Token-based action decoder에 한정**: Diffusion expert나 deterministic MLP regression 기반 VLA(예: π₀ 일부 변종)에는 직접 적용 불가. 일반화된 RL 프레임워크 부재.
3. **연산 비용**: 8×A800 80GB 풀 파라미터 학습 + 병렬 환경 렌더링은 여전히 무겁고, "low-cost"는 데이터 측면에 한정될 뿐 compute 측면 cost는 크다.

## 5. 총평

- **Novelty: ★★★★☆** — DeepSeek-R1 패러다임을 VLA에 가장 깔끔하게 이식. 이론적 신규성보다는 시스템적 완성도와 "outcome-only면 충분하다"는 강력한 실증.
- **Practical impact: ★★★★★** — LIBERO 99% SoTA + sim2real 효과 + 1-trajectory SFT만으로도 동작하여 데이터 부족 환경에 즉시 유용. 코드 공개(PRIME-RL/SimpleVLA-RL)로 재현성 우수.

VLA에서 "RL은 비용 대비 효율이 낮다"는 통념을 정면으로 반박한 작업. 특히 "pushcut" 현상은 SFT가 도달할 수 없는 emergent 영역을 RL이 열어준다는 직관적 증거로, 향후 VLA 학습 파이프라인에서 RL 단계가 필수가 될 가능성을 시사한다.

## 6. 예상 질문

| Q | A |
|---|---|
| 왜 KL regularization을 제거했는가? | DAPO를 따라 reference policy로부터의 분기를 허용해야 새로운 행동 패턴(pushcut 등)을 발견할 수 있고, reference model 보존 비용도 절감(§3.4). |
| Diffusion-based VLA(π₀, RDT)에는 왜 직접 적용 못하는가? | Action token distribution이 없어 PPO-like importance ratio 계산이 어렵기 때문. 본 논문은 token-based OpenVLA-OFT에 한정(§3.1). |
| Outcome-only reward로 long-horizon에서 신뢰할 만한 신호가 나오는가? | Table 4에서 Extra-Long(450-650 steps) Blocks Rank Rgb는 +11.1, Put Bottles Dustbin은 +18.7%p 향상으로 long-horizon에도 효과적. 다만 초기 SR이 0%인 경우엔 학습 불가(Table 7). |
| 실제 로봇으로 직접 RL을 했나? | 아니오. RoboTwin2.0 시뮬레이션에서만 RL 후, 실세계 평가만 수행. 그럼에도 sim2real로 +21%p 평균 향상(Table 6). |

<!-- VERIFIED: pdf -->
