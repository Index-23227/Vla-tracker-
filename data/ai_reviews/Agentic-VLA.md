# Agentic-VLA: Efficient Online Adaptation for Vision-Language-Action Models

> **한 줄 요약**: OpenVLA-OFT 위에 **세 모듈**(Adaptive Reward Synthesis, Language-Guided Exploration, Experience Memory)을 얹어 GRPO로 online RL adaptation을 수행하는 프레임워크. LIBERO 4-suite 평균 **97.8%**(OpenVLA-OFT 89.2% 대비 +8.6%), Long suite에서 +12.3%, one-shot 70.5%, EVOLVE-VLA 대비 **2.4× 빠른 수렴**.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 대형 VLA의 **online RL fine-tuning**은 매력적이나 두 가지 본질적 문제:
  1. **Sparse reward**: manipulation은 자연히 sparse — episode 끝에 성공/실패 한 비트.
  2. **High-dim action space**: 7-DoF chunk를 random exploration으로 발견하긴 사실상 불가능.
- 기존 접근:
  - **EVOLVE-VLA** 등은 dense reward를 manually shaping → 새 task마다 reward engineering 필요.
  - **Naive GRPO/PPO** on VLA → 수렴 매우 느리고 catastrophic forgetting.
  - **Adapter MoE**: task-specific weight를 저장하나 online adaptation 자체엔 도움 안 됨.

### 핵심 질문
- VLA가 **새 task에 online으로 적응**할 때, (1) 어떻게 reward를 자동 dense화하고, (2) 어떻게 exploration을 의미 있게 유도하고, (3) 어떻게 과거 task의 학습을 재사용할 수 있는가?

---

## 2. 방법론 심층 분석

### 2.1 전체 아키텍처

기본 정책은 **OpenVLA-OFT** (autoregressive VLA with parallel decoding + action chunking). 그 주변에 세 agentic 모듈이 작동:

1. **ARS** (Adaptive Reward Synthesis) — Llama-3-8B로 task를 sub-goal로 분해, VLAC critic으로 progress 측정.
2. **LGE** (Language-Guided Exploration) — Qwen3-VL-8B-Instruct이 scene + task를 보고 actionable suggestion을 instruction에 주입.
3. **EM** (Experience Memory) — 학습된 정책 파라미터를 task embedding으로 indexing해 warm-start.

전체 최적화는 **GRPO** (Group Relative Policy Optimization, Shao et al. 2024).

### 2.2 Adaptive Reward Synthesis (ARS)

- Llama-3-8B가 high-level task("put the bowl in the cabinet")를 sub-goal $\{g_1, ..., g_K\}$로 자동 분해.
- 각 sub-goal에 대해 VLAC critic이 progress score $p_k(s_t) \in [0,1]$을 출력.
- 정책의 capability estimate $\hat{c}_k$를 EMA로 추적:
$$\hat{c}_k^{(t+1)} = \alpha \cdot \hat{c}_k^{(t)} + (1-\alpha) \cdot \mathbb{I}[\text{success of } g_k]$$
- Reward 가중치는 **부족한** sub-goal에 더 무겁게:
$$r_t = \sum_k w_k \cdot \Delta p_k(s_t),\quad w_k = 1 - \hat{c}_k$$

→ **self-paced curriculum**: 잘하는 sub-goal은 자동으로 무시, 못하는 부분에 학습 신호 집중.

> **예상 질문**: VLAC critic이 잘못된 progress를 주면?  
> **답변**: VLAC는 대규모 manipulation pre-training으로 학습된 foundation critic. 다만 OOD scene에서 hallucinate 가능하며, 이는 ARS의 가장 큰 잠재적 약점.

### 2.3 Language-Guided Exploration (LGE)

- 각 time step에 일정 확률로 Qwen3-VL-8B가 현재 scene을 보고 actionable suggestion 생성("try grasping the cup from the rim").
- 이 suggestion을 task instruction에 concat → 정책이 새로운 행동을 시도하도록 유도.
- Suggestion 주입 빈도는 **현재 reward에 따라 적응적**으로 감소:
$$p_{\text{suggest}}(t) = p_{\max} \cdot \exp(-\lambda \cdot \bar{R}^{(t)})$$
→ 성공률이 오를수록 외부 suggestion은 줄어듦.

### 2.4 Experience Memory (EM)

- 학습 종료 후 task embedding $e_T$와 정책 파라미터 $\theta_T$를 함께 저장.
- 새 task $T'$ 도착 시:
1. embedding $e_{T'}$ 계산.
2. cosine similarity로 top-k 유사 task 검색.
3. softmax (τ=0.1, peaked) 가중 보간으로 warm-start parameter:
$$\theta_0 = \sum_{i \in \text{top-}k} \frac{\exp(\text{sim}(e_{T'}, e_i)/\tau)}{\sum_j \exp(\text{sim}(e_{T'}, e_j)/\tau)} \theta_i$$

VLA-Pro의 procedural memory와 유사하지만 여기는 **full policy parameter** 융합(LoRA가 아님)이라 비용이 더 큼.

---

## 3. 데이터셋

| 데이터 | 용도 | 규모 |
|--------|------|------|
| LIBERO (4 suites) | 평가 | 각 suite 50 task, task당 50 demo |
| RoboTwin 2.0 | 평가 | 50 task, dual-arm, 50 demo/task |

**중요**: Agentic-VLA의 핵심은 demo 데이터 자체보다 **online rollout**으로 만들어지는 RL data. LIBERO suite당 약 **22.4k rollout**으로 90% 도달.

---

## 4. 학습 세부사항

| 항목 | 값 |
|------|-----|
| Hardware | 4× A100 80GB |
| Algorithm | GRPO |
| Learning rate | 1e-5 |
| Batch size | 32, group size 8 |
| Max rollout horizon | 500 |
| Compute per LIBERO suite | ~8 hours |
| Seeds | 5 (논문은 mean±std 보고) |

5-seed mean±std는 robust한 보고 방식 — 단일 seed 결과로 결론짓지 않음.

---

## 5. 실험 결과

### 5.1 LIBERO Main (Table 1, mean±std over 5 seeds)

| Suite | Agentic-VLA | OpenVLA-OFT | EVOLVE-VLA |
|-------|-------------|-------------|------------|
| Spatial | 97.2±0.6 | 91.3±1.0 | 95.4±0.9 |
| Object | 98.6±0.5 | 90.1±1.2 | 97.4±0.8 |
| Goal | 97.4±0.6 | 89.8±1.3 | 95.8±0.9 |
| Long | **98.1±0.8** | 85.8±1.8 | 94.4±1.2 |
| **Avg** | **97.8±0.4** | 89.2±0.9 | 95.8±0.7 |

가장 큰 폭의 향상은 **Long suite +12.3%** — long-horizon에서 ARS의 sub-goal reward가 크게 기여.

### 5.2 One-Shot (Table 2)

| 모델 | One-shot LIBERO 평균 |
|------|---------------------|
| OpenVLA-OFT | 43.6% |
| EVOLVE-VLA | 61.3% |
| **Agentic-VLA** | **70.5%** |

EM의 warm-start와 LGE의 exploration이 시너지를 내는 영역.

### 5.3 Cross-Task Transfer (Table 3, LIBERO-Long → LIBERO-Object)

| 모델 | Success | Progress |
|------|---------|---------|
| Direct SFT transfer | 0.0% | – |
| EVOLVE-VLA | 20.8±2.7 | 54.2±3.9 |
| **Agentic-VLA** | **31.2±2.3** | **68.7±3.1** |

새 suite에 demo 없이 EM warm-start + online RL만으로 31%. 0%에서 출발하는 SFT 대비 큰 격차.

### 5.4 Training Efficiency (Table 4)

| 모델 | 90% 도달까지 iter | rollouts | speedup |
|------|------------------|----------|---------|
| EVOLVE-VLA | 1680 | 53.8k | 1× |
| **Agentic-VLA** | **700** | **22.4k** | **2.4×** |

### 5.5 RoboTwin 2.0 Subset (Table 8)

| 모델 | Easy avg | Hard avg |
|------|----------|---------|
| RDT | 34.5 | 13.7 |
| π0 | 46.4 | 16.3 |
| **Agentic-VLA** | **62.5** | **34.7** |

Hard randomized setting에서도 큰 우위.

---

## 6. 어블레이션 (Table 5, LIBERO-Long)

| 구성 | Success | 비고 |
|------|---------|------|
| OpenVLA-OFT base | 85.8 | |
| + ARS만 | 94.6 (+8.8) | sub-goal reward 단독 효과가 가장 큼 |
| + ARS + LGE | 96.2 (+1.6) | 320 iter 절약 |
| + ARS + LGE + EM (Full) | **98.1** (+1.9) | 180 iter 추가 절약 |
| Full − ARS | 95.4 (−2.7) | ARS 제거가 가장 큰 손실 |
| Full − LGE | 96.8 (−1.3) | |
| Full − EM | 96.4 (−1.7) | +350 iter |

→ **ARS > EM > LGE** 순으로 기여. 세 모듈이 모두 필요.

---

## 7. 한계

1. **추가 모델 비용**: Llama-3-8B + Qwen3-VL-8B + VLAC critic + OpenVLA-OFT 7B → 추론/학습 모두 무거움. "Efficient"는 **수렴 iter** 기준이지 **wall-clock**은 아닐 수 있음.
2. **VLAC critic 의존**: VLAC가 OOD scene에서 reward를 잘못 주면 ARS 전체가 흔들림.
3. **EM full-policy interpolation**: parameter 평균이 LoRA보다 위험 — 두 정책이 다른 mode면 평균이 무의미할 수 있음.
4. **Real-world 실험 부재**: 모든 평가가 시뮬레이션(LIBERO, RoboTwin) — 실제 로봇에서의 online RL은 안전/지연 문제로 훨씬 어려움.
5. **5 seed mean이 약점도 드러냄**: Long suite std 0.8은 작지만 Spatial 0.6과 비교하면 long-horizon 안정성이 상대적으로 낮음.

---

## 8. 예상 질문

| # | 질문 | 핵심 답변 |
|---|------|----------|
| 1 | Sub-goal 분해를 Llama-3-8B에 맡기면 잘못된 분해는? | 잘못된 분해는 ARS 신호를 noisy하게 만들지만, capability EMA가 동작하지 않는 sub-goal을 자동 down-weight해 어느 정도 보상. 다만 핵심 sub-goal을 빠뜨리는 경우는 회복 어려움. |
| 2 | GRPO를 manipulation에 쓰는 이유? PPO 대비? | GRPO는 critic 없이 group-relative advantage로 학습 → VLA에서 별도 value head를 둘 필요 없음. 단, group size(=8) 만큼 rollout이 필요해 sample cost는 높음. |
| 3 | One-shot 70.5%가 정말 one-shot인가? EM이 사전학습된 task와 매우 유사한 demo를 봤다면? | EM은 학습된 다른 task의 weight만 보유 — 새 task의 demo는 1개만 본다는 의미. 그러나 EM에 있는 task와 새 task가 매우 유사하면 사실상 transfer learning. 논문은 task split을 명확히 하지만 "semantic distance" 분석은 부족. |
| 4 | 2.4× speedup의 baseline EVOLVE-VLA가 약하면 의미가 줄어들지 않나? | EVOLVE-VLA는 manual reward shaping이라 강한 baseline. 그래도 ARS의 자동 분해가 그만큼의 효과를 내는 건 의미 있음. |
| 5 | Real-world 적용 시 가장 큰 장애물은? | (1) 8B LM 두 개 + VLAC + VLA를 실시간 inference 가능한 hardware, (2) online rollout에서 hardware reset/safety, (3) VLAC가 sim과 다른 real scene에서 progress를 정확히 estimate하는지. |

<!-- VERIFIED: pdf -->
