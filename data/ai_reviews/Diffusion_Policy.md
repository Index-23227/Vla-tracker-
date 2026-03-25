# Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

> **한 줄 요약**: 로봇 visuomotor policy를 조건부 denoising diffusion process로 정의하여, CNN(1D temporal) 및 Transformer(minGPT) 두 구현을 제공하고, 12개 태스크에서 position control로 기존 대비 평균 46.9% 성능 향상, real-world에서 Push-T 95%, 6DoF mug flip 90% 달성.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- **Explicit policy** (LSTM-GMM): Action distribution의 mode 수를 사전 지정 → 복잡한 multimodal distribution에 부적합
- **Implicit policy** (IBC/Energy-based): 이론적으로 유연하나 **학습 불안정성과 high-dim에서 failure** (Can, Square, Transport에서 0% 성공)
- **Deterministic BC**: 단일 mode만 학습 → multimodal task에서 averaging artifact

### 핵심 질문
- **Denoising diffusion이 로봇 action distribution을 표현하기에 적합한가?**
- **Action space는 image보다 저차원인데, diffusion의 이점이 정당화되는가?**

---

## 2. 방법론 심층 분석

### 2.1 두 가지 구현

| | **CNN (1D Temporal)** | **Transformer (minGPT)** |
|---|---|---|
| Conditioning | FiLM (channel-wise) | Cross-attention |
| Visual encoder | ResNet-18 + spatial softmax + GroupNorm | Same |
| Strength | 속도, 안정적 학습 | 복잡한 multi-stage task |
| Weakness | High-frequency action에 약함 (smoothing bias) | Hyperparameter 민감 |
| Layers | - | **8** (최적; 증가 시 성능 하락) |

### 2.2 Closed-Loop Receding Horizon

$$\text{Predict: } \mathbf{a}_{t:t+T_p}, \quad \text{Execute: } \mathbf{a}_{t:t+T_a}, \quad T_a < T_p$$

- Observation horizon $T_o$, Prediction horizon $T_p$, Execution horizon $T_a$
- **Vision-based optimal: $T_o = 2$**

### 2.3 Training & Inference

| 항목 | CNN | Transformer |
|------|-----|-----------|
| Batch size | 256 (state) / 64 (vision) | 256 / 64 |
| LR | 1e-4, cosine, 500-step warmup | 1e-4, cosine, 1000-step warmup |
| WD | 1e-6 | 1e-3 |
| Train denoising steps | **100** | 100 |
| Inference steps (sim) | 100 | 100 |
| Inference steps (real) | **16 (DDIM)** | 16 |
| Noise schedule | Squared cosine (iDDPM) | Same |
| Epochs | 4500 (state) / 3000 (vision) | Same |

**Real-time inference**: **0.1s on 3080 GPU** (DDIM 10 steps), **10Hz** closed-loop 가능

> ❓ **예상 질문**: 100 training steps인데 16 inference steps면 성능 하락이 없는가?
> **답변**: DDIM acceleration으로 100→10~16 step 압축 시 성능 하락 최소. Squared cosine schedule이 이 압축에 특히 적합함을 empirically 발견.

---

## 3. 실험 결과 심층 분석

### State-Based BC — Robomimic (Table I, 10 tasks)

| Task | LSTM-GMM | IBC | BET | **DP-CNN** | **DP-Trans** |
|------|---------|-----|-----|----------|------------|
| Lift (ph) | 1.00/0.96 | 0.79/0.41 | 1.00/0.96 | 1.00/0.98 | **1.00/1.00** |
| Can (ph) | 1.00/0.91 | 0.00/0.00 | 1.00/0.89 | 1.00/0.96 | **1.00/1.00** |
| Square (ph) | 0.95/0.73 | 0.00/0.00 | 0.76/0.52 | **1.00/0.93** | 1.00/0.89 |
| Transport (ph) | 0.76/0.47 | 0.00/0.00 | 0.38/0.14 | **0.94/0.82** | 1.00/0.84 |
| ToolHang (ph) | 0.67/0.31 | 0.00/0.00 | 0.58/0.20 | 0.50/0.30 | **1.00/0.87** |
| Push-T (ph) | 0.67/0.61 | **0.90/0.84** | 0.79/0.70 | **0.95/0.91** | 0.95/0.79 |

형식: best/avg_last_10, 3 seeds × 150 evaluations

### Vision-Based BC (Table II)

Transformer가 ToolHang에서 0.76/0.47, CNN이 **0.95/0.73** → **Vision에서는 CNN이 ToolHang 우위**

### Multi-Stage Tasks (Table IV, State)

| Task | LSTM-GMM | IBC | BET | **DP-CNN** | **DP-Trans** |
|------|---------|-----|-----|----------|------------|
| BlockPush (p1/p2) | 0.03/0.01 | 0.01/0.00 | 0.96/0.71 | 0.36/0.11 | **0.99/0.94** |
| Kitchen (p1/p2/p3/p4) | 1.00/0.90/0.74/0.34 | 0.99/0.87/0.61/0.24 | 0.99/0.93/0.71/0.44 | **1.00/1.00/1.00/0.99** | 1.00/0.99/0.99/0.96 |

- **Kitchen 4-subtask**: CNN **0.99**, Transformer 0.96 — CNN 우위
- **BlockPush**: Transformer **0.94** >> CNN 0.11 — 극적 차이 (multimodal task에서 Transformer 우위)

### Real-World Results

| Task | Robot | DP | LSTM-GMM | IBC |
|------|-------|-----|---------|-----|
| Push-T | UR5 | **95%**, IoU 0.80 | 0% | 0% |
| 6DoF Mug Flip | Franka | **90%** (20 trials) | 0% | 0% |
| Sauce Pouring | Franka | **79%**, IoU 0.74 | 0% | 0% |
| Periodic Spreading | Franka | **100%**, 0.77 coverage | 0% | 0% |

- **LSTM-GMM과 IBC가 모든 real task에서 0%** → Diffusion Policy의 real-world 우위가 극명

---

## 4. Ablation 분석

### Action Horizon (Figure 6)
- **최적 $T_a = 8$ steps** for most tasks
- 짧으면: temporal consistency 부족
- 길면: 환경 변화에 대한 responsiveness 하락

### Observation Horizon
- State-based: 민감하지 않음
- **Vision-based: $T_o = 2$ optimal**

### Latency Robustness (Figure 6)
- **4 step latency까지 peak performance 유지**
- Position control이 velocity control 대비 latency에 더 robust

### Position vs Velocity Control (Figure 5)
- Position control로 전환 시 **평균 46.9% 향상** (이것이 "46.9%" 주장의 출처)

> ⚠️ **주의**: "46.9% 향상"은 **velocity→position control 전환**의 효과이지, "기존 방법 대비 46.9%"가 아님. Position control이 diffusion의 temporal consistency와 시너지.

---

## 5. 관련 연구 비교

| 방법 | Distribution | Multimodal | Temporal | Real-world |
|------|-------------|-----------|---------|-----------|
| LSTM-GMM | Explicit mixture | △ (K fixed) | ✗ | 0% (all tasks) |
| IBC (Energy) | Implicit | ✓ | ✗ | 0% (all tasks) |
| BET (VQ) | Discrete | ✓ | ✗ | - |
| **Diffusion Policy** | **Implicit (diffusion)** | **✓** | **✓ (chunk)** | **79-100%** |

---

## 6. 한계 및 미해결 문제

### 방법론적 미비점
1. **추론 latency**: 100 denoising step → DDIM 10~16으로 축소 가능하나 여전히 단일 forward 대비 10-16배
2. **CNN의 smoothing bias**: High-frequency action에서 CNN이 약함 (BlockPush p2: 0.11 vs Transformer 0.94)
3. **Transformer의 hyperparameter 민감성**: Task별 최적 설정이 다름 (embedding dim 256-768, dropout 0.01-0.3)
4. **Language conditioning 부재**: 초기 논문이므로 task specification이 demonstration에 의존
5. **Large-scale 미검증**: 수백 demonstrations 규모. 대규모 cross-embodiment 데이터에서의 scaling 미검증

### Attribution 문제
- "46.9% 향상"이 **diffusion formulation** 때문인지 **position control 전환** 때문인지 — 논문의 주장은 position control이 diffusion의 이점을 살린다는 것이며, 이 둘이 결합된 효과

---

## 7. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — 로봇 학습에 diffusion 적용의 선구적 연구 |
| **Technical depth** | ★★★★★ — 두 구현(CNN/Transformer), 포괄적 분석 |
| **Experimental rigor** | ★★★★★ — 12 tasks (state+vision+multi-stage+real), 150 eval × 3 seeds |
| **Practical impact** | ★★★★★ — 이후 모든 VLA 연구의 기반 |
| **Writing quality** | ★★★★★ — 교과서적 명확성 |

**강점**: Real-world에서 기존 방법이 0%인 곳에서 79-100% 달성. CNN/Transformer 두 구현의 trade-off 분석이 후속 연구에 큰 영향. **약점**: Latency, language 부재는 시대적 한계이며 후속 연구에서 해결.

---

## 8. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | "46.9%"가 position control 전환 효과면, diffusion 자체의 기여는? | Position control + diffusion의 결합 효과. Position만으로도 LSTM-GMM이 개선되겠지만, diffusion의 multimodal 처리가 핵심 차별화 (IBC가 multimodal인데도 0%) |
| 2 | CNN vs Transformer: 어떤 것을 선택해야 하는가? | Simple task: CNN (안정적, 빠름). Complex multimodal task: Transformer. BlockPush가 결정적 증거 (CNN 0.11 vs Trans 0.94) |
| 3 | Real-world 10Hz면 빠른 manipulation에 부족하지 않은가? | Consistency model, flow matching으로 1-step까지 가속 가능. 이후 pi0 등이 이를 실현 |
| 4 | 이 방법이 VLM과 결합되면? (→ CogACT, pi0) | 정확히 이 방향이 VLA 시대를 열었음. Diffusion Policy가 VLA의 action head 표준이 됨 |
| 5 | Squared cosine schedule이 왜 최적인가? | Empirical selection. Cosine이 시작/끝에서 gentle하여 action space의 boundary에서 안정적 |

<!-- VERIFIED: pdf -->
