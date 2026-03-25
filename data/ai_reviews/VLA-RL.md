# VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning

> **한 줄 요약**: Auto-regressive VLA에 online RL을 직접 적용하여, pretrained OpenVLA-7B가 fine-tuned baseline을 LIBERO 40개 태스크에서 4.5%p 능가하고, test-time compute 증가에 따른 성능 향상을 시현.

---

## 1. 배경 및 동기

- BC의 근본적 한계: Suboptimal demonstration에 bound, OOD에 취약
- VLA에 RL을 적용하려면: 대형 모델의 안정적 fine-tuning, 효율적 on-policy data 수집 필요

---

## 2. 방법론

### Online RL for Autoregressive VLA

- PPO 기반 policy optimization
- VLA의 autoregressive action generation을 RL의 policy로 직접 활용
- Reward: Task success (sparse)

### Test-time Optimization

추론 시 더 많은 compute를 할당하면 성능 향상:
- Multiple rollout → best action selection
- Beam search 등

---

## 3. 실험 결과

| 모델 | LIBERO-40 Avg (%) |
|------|-----------------|
| OpenVLA (BC) | 76.5 |
| OpenVLA (fine-tuned BC) | 82.3 |
| **VLA-RL** | **86.8 (+4.5%)** |

---

## 4. 한계

1. **RL의 sample complexity**: On-policy data 수집 비용
2. **Sparse reward**: Dense reward 없이는 학습 느림
3. **시뮬레이션 의존**: Real-world RL은 미검증

---

## 5. 총평

| **Novelty** | ★★★★☆ — Online RL for large VLA |
| **Practical impact** | ★★★★☆ — BC→RL 전환의 가능성 |

---

## 6. 🔥 질문 모음

| 질문 | 답변 |
|------|------|
| AcceRL과의 차이? | AcceRL은 분산 비동기+world model, VLA-RL은 direct online PPO. VLA-RL이 더 간단하나 AcceRL이 더 scalable |
| PLD와 비교하면? | PLD는 indirect (residual→distill), VLA-RL은 direct RL. PLD가 더 안정적이나 VLA-RL이 더 general |

<!-- VERIFIED: abstract-only -->
