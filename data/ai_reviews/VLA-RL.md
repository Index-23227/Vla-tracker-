# VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable RL

> **한 줄 요약**: OpenVLA-7B를 baseline으로 **online RL**을 적용하여 LIBERO 40 tasks에서 **base 모델 대비 +4.5%p**, π0-FAST 수준에 도달. **Robotic Process Reward Model (PRM)** + multi-modal multi-turn formulation + critic warmup + curriculum이 핵심.

---

## 1. 배경 및 동기

- 기존 VLA는 imitation learning (SFT) 중심 → offline demo의 supervised exploitation에 한정.
- Online RL이 자연스러운 확장이지만 (1) action space (multi-modal/multi-turn), (2) sparse reward, (3) value-policy mismatch 등의 문제로 미해결.
- **목표**: OpenVLA-7B를 RL로 fine-tune하여 일반화 + masterful 성능 동시 달성.

---

## 2. 방법론

### Auto-Regressive RL Formulation
- Policy: OpenVLA-7B (Llama-2-7B + two-stream visual encoder).
- Multi-modal multi-turn POMDP로 reformulation: token-level RL이 아닌 action-step level로 PPO 변형 적용.

### Robotic Process Reward Model (PRM)
- VLM (separate)을 fine-tune하여 robot trajectory의 progress reward를 dense하게 제공.
- Pseudo reward labels 자동 추출 → human reward annotation 비용 제거.
- Sec 3.4: dense reward를 RL training의 sparse-reward problem 해결 수단으로 사용.

### Systematic Implementation Details
1. **Curriculum selection**: 쉬운 task부터 점진적 노출.
2. **Vectorized environments**: parallel rollout으로 throughput 향상.
3. **Batch decoding**: VLA의 long sequence를 batch로 처리.
4. **Critic warmup**: value model을 RL 시작 전 supervised로 warmup → bootstrap 안정성.

### Algorithm 1 (VLA-RL pipeline)
- Trajectory rollout → PRM dense reward → advantage estimation → PPO-style update.
- Multi-task curriculum.

---

## 3. 실험 결과

### Fig. 4 — LIBERO Test-time Scaling (4 suites)

| Suite | SFT Baseline | **VLA-RL (best)** |
|-------|-------------:|---------------:|
| LIBERO-Spatial | 84.7 | **90.2** |
| LIBERO-Object | 88.4 | **91.8** |
| LIBERO-Goal | 79.2 | **82.2** |
| LIBERO-Long | 53.7 | **59.8** |
| **Avg** | **76.5** | **81.0 (+4.5)** |

→ 4 suites 모두 일관된 향상, π0-FAST와 비등.

### Test-time Scaling Curve (Fig. 4)
- Training step 0 → 10K 진행에 따라 monotonic improvement.
- LIBERO-Spatial: 84.0 → 90.2 (10K steps)

---

## 4. 한계 및 미해결 문제

1. **PRM 의존**: Process Reward Model 품질이 전체 성능 결정. Reward hacking 가능성 존재.
2. **OpenVLA-7B base에 한정**: Flow-matching VLA (pi0, GR00T) 위에서 동일 RL 적용 시 효과 미검증.
3. **LIBERO-only**: CALVIN/SimplerEnv/RoboTwin 등 다른 benchmark 검증 부족.
4. **Sample efficiency**: 10K step training이 10K rollout episodes로 수렴 — real-robot에는 비현실적 비용.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — VLA의 online RL을 systematic하게 다룬 첫 framework |
| **Practical impact** | ★★★★☆ — OpenVLA-7B를 π0-FAST 수준으로 끌어올린 실질 결과. SFT-only의 한계 극복 |

**강점**: 4 suites 일관된 +4.5%p, dense PRM reward + curriculum + critic warmup의 시너지. ByteDance.
**약점**: AR-VLA에 specialized, flow-matching 시대로 가는 흐름과 별개 트랙. LIBERO 외 검증 부족.

---

## 6. 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | SimpleVLA-RL과 차이? | SimpleVLA-RL은 GRPO outcome-based + SFT 우회, VLA-RL은 PPO + dense process reward (PRM). VLA-RL이 reward 풍부, SimpleVLA-RL이 더 단순. |
| 2 | Why critic warmup 필요? | RL 시작 시 random value estimate → bootstrap noise. SFT로 warmup하면 policy update가 안정. |
| 3 | PRM은 어떻게 학습하나? | VLM을 trajectory 진행도(progress label)로 fine-tune. Auto-extracted pseudo labels (Sec 3.4) 사용. |
| 4 | Flow-matching VLA에 적용 가능? | Theoretically yes. AR token-level → flow-matching velocity field로 reformulation 필요. Future work. |

<!-- VERIFIED: pdf -->
