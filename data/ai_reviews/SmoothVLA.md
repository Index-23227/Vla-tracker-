# SmoothVLA: Aligning VLA Models with Physical Constraints via Intrinsic Smoothness Optimization

> **한 줄 요약**: OpenVLA에 physics-informed hybrid reward (binary task reward + jerk-based smoothness penalty, λ=0.2)를 적용한 GRPO RL로, 기존 RL 대비 **13.8% smoothness 향상** + LIBERO-Plus robustness **46.2%** (SFT 22.0% 대비 +24.2%p) 달성.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- RL fine-tuning (GRPO 등)이 task success를 높이지만 **jittery, erratic trajectories** 생성
- 물리적으로 실현 불가능하거나 위험한 동작 → real deployment에서 문제
- "Exploration-Stability Paradox": RL exploration이 trajectory quality를 해침

### 핵심 질문
- **Task success와 motion smoothness를 동시에 최적화할 수 있는가?**

---

## 2. 방법론 심층 분석

### 2.1 Hybrid Reward

$$R(\tau) = I_{\text{success}} \cdot \left(1 - \lambda \cdot \frac{1}{T}\sum|Jerk(t)|_2\right)$$

- $I_{\text{success}}$: Binary task completion
- $\lambda = 0.2$: Grid search로 결정
- Jerk: 3차 미분 (가가속도), inverse kinematics로 joint space에서 계산

### 2.2 GRPO (Group Relative Policy Optimization)

$$\hat{A}_i = \frac{R_i - \text{mean}(R)}{\text{std}(R)}$$

Critic network 없이, 같은 instruction에서 생성된 trajectory 그룹 내에서 상대적 advantage 계산.

### 2.3 Training

| 단계 | LR | Batch |
|------|-----|-------|
| SFT | 4e-5 | 16 |
| GRPO | 2e-5 | 16 |

Base: OpenVLA with LoRA

---

## 3. 실험 결과 심층 분석

### LIBERO In-Distribution

| Model | Avg SR (%) |
|-------|-----------|
| Octo | 73.9 |
| OpenVLA-SFT | 82.1 |
| **SmoothVLA-GRPO** | **85.6** |

### Smoothness (Average Jerk)

| Model | Avg Jerk |
|-------|---------|
| OpenVLA-SFT | 0.374 |
| OpenVLA-RL (standard) | 0.402 (worse) |
| **SmoothVLA** | **0.322 (-13.8%)** |

- Standard RL은 오히려 smoothness를 **악화** (0.374→0.402)
- SmoothVLA는 **개선** (0.374→0.322)

### LIBERO-Plus Robustness (핵심 결과)

| Method | Original | Language | Light | Background | Layout | **Avg** |
|--------|---------|---------|------|-----------|--------|---------|
| OpenVLA-SFT | 82.1 | 26.8 | 4.4 | 25.3 | 31.6 | **22.0** |
| SmoothVLA-GRPO | 85.6 | 43.3 | 16.1 | 61.2 | 64.4 | **46.2** |

- **+24.2%p** robustness 향상 → smoothness optimization이 generalization에 도움

### Ablation (Reward Function)

| Reward | In-Domain | Robustness Avg |
|--------|----------|---------------|
| R_binary (sparse only) | 84.5 | 48.2 |
| R_random (noise) | 83.2 | 46.5 |
| **R_smooth (full)** | **85.6** | **54.1** |

---

## 4. 한계 및 미해결 문제

1. **OpenVLA base only**: 다른 VLA (pi0, FLOWER 등)에서의 효과 미검증
2. **LIBERO 80.5%**: 최근 SOTA (97%+) 대비 낮음. Base model의 한계
3. **λ=0.2의 일반성**: Task마다 최적 λ가 다를 수 있음
4. **Compute 미공개**

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Physics-informed reward for VLA RL |
| **Technical depth** | ★★★★☆ — Jerk computation, GRPO integration |
| **Experimental rigor** | ★★★★☆ — LIBERO + LIBERO-Plus robustness |
| **Practical impact** | ★★★★★ — Smooth trajectories는 real deployment에 필수 |

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Standard RL이 smoothness를 악화시키는 이유는? | Exploration이 diverse but erratic trajectories를 생성. Reward에 smoothness가 없으면 jittery한 성공도 보상 |
| 2 | Smoothness가 robustness를 올리는 메커니즘은? | Smooth trajectories → perturbation에 덜 민감한 policy. Overfitting이 아닌 generalizable motor primitives 학습 |
| 3 | pi0 + SmoothVLA reward이면? | 매우 유망. Flow matching base에 smoothness reward 추가 시 더 좋을 가능성 |

<!-- VERIFIED: pdf -->
