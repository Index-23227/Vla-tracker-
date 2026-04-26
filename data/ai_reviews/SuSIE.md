# SuSIE: Zero-Shot Robotic Manipulation with Pretrained Image-Editing Diffusion Models

> **한 줄 요약**: InstructPix2Pix를 finetune한 image-editing diffusion이 텍스트 명령에 대응하는 **subgoal 이미지**를 생성하고, goal-conditioned diffusion policy가 그 subgoal을 추종하는 2-단계 hierarchical 구조. CALVIN ABC→D zero-shot에서 5-task 평균 0.26으로 SOTA를 달성하고, 실제 WidowX에서 RT-2-X(55B)를 능가 (Black & Nakamoto 등, UC Berkeley/Stanford/Google DeepMind).

---

## 1. 배경 및 동기 (Sec. 1, 2)

- 일반화된 manipulation에는 **새로운 객체/장면에 대한 semantic 이해**가 필수.
- 기존 VLM-init policy(예: RT-2)는 semantic 일반화는 가능하지만 **저수준 정밀도(localization, grasping)** 가 약함 (Fig. 3).
- 저자의 통찰: "task 수행 = 픽셀 편집"으로 보고, image-editing model이 가진 internet-scale 지식을 활용해 **subgoal 이미지를 합성**하면 high-level planning과 low-level control을 분리할 수 있음.

---

## 2. 방법론 (Sec. 4, Algorithm 1)

### Phase I — Subgoal Synthesis
- 사전학습 InstructPix2Pix $p_\theta(s_{\text{edited}}|s_{\text{orig}}, l)$을 video data $\mathcal{D}_l \cup \mathcal{D}_{l,a}$로 finetune.
- 학습 objective (Eq. 1): $q(j|i) = U(j; [i+k_{\min}, i+k_{\max}))$로 미래 timestep을 sampling. $k_{\min}, k_{\max}$는 dataset-dependent.
- Classifier-free guidance를 **language와 image에 각각 별도 적용** (Sec. 4.3).

### Phase II — Goal-Conditioned Policy
- $\pi_\phi(a|s_i, s_j)$, GCBC objective (Eq. 2). 4-action chunk를 출력하는 **diffusion policy**, test-time temporal averaging.
- 입력 해상도 256×256.

### Test-time loop (Algorithm 1)
1. 새 subgoal $\hat{s}^+ \sim p_\theta(s^+|s_t^{\text{test}}, l^{\text{test}})$ 생성.
2. $k_{\text{test}}$ step 동안 $\pi_\phi$로 그 subgoal을 추종.
3. 반복.

---

## 3. 실험 결과 (PDF 표 인용)

### CALVIN ABC→D zero-shot (Table 1, success rate of chained instructions)
| Method | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| HULC | 0.43 | 0.14 | 0.04 | 0.01 | 0.00 |
| MdetrLC | 0.69 | 0.38 | 0.20 | 0.07 | 0.04 |
| AugLC | 0.69 | 0.43 | 0.22 | 0.09 | 0.05 |
| LCBC | 0.67 | 0.31 | 0.17 | 0.10 | 0.06 |
| UniPi (HiP) | 0.08 | 0.04 | 0.00 | 0.00 | 0.00 |
| UniPi (Ours impl.) | 0.56 | 0.16 | 0.08 | 0.08 | 0.04 |
| **SuSIE** | **0.87** | **0.69** | **0.49** | **0.38** | **0.26** |

→ 5-chained 26%로 prior best 5%를 큰 폭으로 상회. **avg.length 환산 시 약 2.69**.

### Real-world WidowX (Table 2): SuSIE는 모든 Scene에서 1위.
- Scene A 평균: SuSIE **0.87** vs LCBC 0.63 vs MOO 0.47 vs UniPi 0.0 vs RT-2-X 0.43.
- Scene B 평균 (bell pepper, novel container): SuSIE **0.50** vs RT-2-X 0.00 (RT-2-X은 가벼운 bell pepper grasp 실패).
- Scene C 평균 (novel table texture): SuSIE **0.88** vs RT-2-X 0.75.

### Oracle-goal 비교 (Table 3): SuSIE의 합성 subgoal이 **privileged final goal image**보다 높은 성능 (Scene B 0.50 vs 0.05; CALVIN 평균 0.95 vs 0.66) → "중간 subgoal이 final goal보다 학습 신호로 우월".

### Co-training 효과 (Table 본문, Sec. 5.4): Bridge + Something-Something 추가 시 Scene B 평균 0.30 → 0.50 → **human video co-training의 명확한 이득**.

---

## 4. 한계 및 비판적 고찰

1. **Latency**: Image diffusion이 frame당 수백 ms; 실제 inference에서는 $k_{\text{test}}$만큼 streaming하지만 reactive control 한계.
2. **Inconsistent video models**: open-source video diffusion(UniPi 재구현)에서 frame inconsistency 관찰 → SuSIE는 single-step subgoal만 생성하여 회피 (Sec. 2, Appendix C).
3. **Subgoal scheduling**: $k_{\min}, k_{\max}, k_{\text{test}}$가 dataset-dependent hyperparameter. 자동 선택 미해결.
4. **Bell pepper grasp 등 contact-rich**: 여전히 0.5 — perfect와 거리.

### YAML 정합성
- YAML `dataset: Bridge V2`만 명시 → PDF는 BridgeData V2 (60K trajectories) **+ Something-Something** (~75K filtered video clips, Sec. 5.1)을 함께 사용. Subgoal model 학습에 human video가 필수임을 감안하면 **dataset 항목에 Something-Something 추가가 정확함**.
- YAML `venue: ICRA 2024` 표기. PDF 자체에는 conference 라벨이 없으나 ICLR 2024로 게재된 것으로 알려짐 → **검증 필요**.

---

## 5. 총평

**강점**: Internet-scale image-editing prior를 robot planner로 직접 재사용한 최초 사례. Diffusion policy(low-level)와 image diffusion(high-level)을 **fully decoupled**로 학습 → 모듈화 우수. CALVIN zero-shot SOTA, real-world에서 RT-2-X(55B)를 작은 모델로 outperform. **약점**: Latency, hyperparameter 다수, subgoal hallucination에 fragile.

---

## 6. 예상 날카로운 질문

| # | 질문 | 답 (논문 근거) |
|---|------|----------------|
| 1 | Subgoal 이미지가 부정확하면 policy가 추종해도 실패하지 않나? | Sec. 5.3 / Table 3에서 oracle-goal보다 SuSIE의 imperfect subgoal이 더 높은 성능. 약간의 inaccuracy가 오히려 policy에게 attention 신호 역할. |
| 2 | UniPi 대비 핵심 차별점은? | 전체 video를 생성하지 않고 **single subgoal frame**만 생성. 이로써 frame consistency 부담을 제거 (Sec. 2). UniPi(HiP) 0.08 → SuSIE 0.87 (CALVIN 1-step). |
| 3 | RT-2-X(55B)보다 작은 모델로 이긴 비결? | RT-2-X의 약점이 **저수준 grasp 정밀도**임을 진단(Fig. 3). SuSIE는 subgoal이 robot arm pose까지 명시 → grasp pre-shape를 안내. |

<!-- VERIFIED: pdf -->
