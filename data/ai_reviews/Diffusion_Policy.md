# Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

> **한 줄 요약**: 로봇 visuomotor policy를 조건부 denoising diffusion process로 정의하여, multimodal action distribution을 자연스럽게 처리하고 12개 태스크에서 기존 방법 대비 평균 46.9% 성능 향상을 달성한 **기념비적(seminal) 연구**.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- **Explicit policy** (GMM, VAE): Action distribution의 mode 수를 사전 지정 → 복잡한 multimodal distribution에 부적합
- **Implicit policy** (IBC): Energy-based model로 유연하나 **학습 불안정성과 mode collapse** 문제
- **Deterministic policy** (BC): 단일 mode만 학습 → averaging artifact (물체를 좌우로 넘길 수 있으면 중간으로 가는 문제)

### 핵심 질문
- **Denoising diffusion이 로봇 action distribution을 표현하기에 적합한가?**
- **Image generation에서의 diffusion 성공이 action generation으로 전이되는가?**

📌 [Figure 1 삽입] — Diffusion Policy의 action denoising 과정 시각화

---

## 2. 방법론 심층 분석

### 2.1 Action Diffusion Formulation

관측 $\mathbf{O}_t$가 조건(condition)으로 주어졌을 때, action sequence를 denoising으로 생성:

$$p_\theta(\mathbf{A}_t^{0:H} | \mathbf{O}_t) = \mathcal{N}(\mathbf{A}_t^K; \mathbf{0}, \mathbf{I}) \prod_{k=1}^{K} p_\theta(\mathbf{A}_t^{k-1} | \mathbf{A}_t^k, \mathbf{O}_t)$$

여기서 $K$는 denoising steps, $H$는 action horizon (chunk).

Training objective:
$$\mathcal{L} = \mathbb{E}_{k, \epsilon}[\|\epsilon - \epsilon_\theta(\mathbf{A}_t^k, k, \mathbf{O}_t)\|^2]$$

> ❓ **예상 질문**: Action space는 image space보다 훨씬 저차원인데, diffusion이 overkill이 아닌가?
> **답변**: 차원은 낮지만 **multimodality**가 핵심. 동일 상황에서 여러 valid action이 존재할 때, diffusion은 이를 자연스럽게 모델링. VAE나 normalizing flow 대비 mode coverage가 우수함이 실험적으로 입증.

### 2.2 두 가지 구현: CNN vs Transformer

| | CNN (U-Net) | Transformer (DiT) |
|---|------------|-------------------|
| Conditioning | FiLM layers | Cross-attention |
| Action representation | 1D temporal conv | Sequence tokens |
| Speed | 빠름 | 느림 |
| Expressiveness | 좋음 | 매우 좋음 |

> ❓ **예상 질문**: U-Net과 Transformer 중 어느 것이 더 좋은가?
> **답변**: 태스크에 따라 다름. 단순 태스크에서는 U-Net이 효율적, 복잡한 multi-modal 태스크에서는 Transformer가 우수. 이후 연구(CogACT, pi0)에서는 Transformer가 주류.

### 2.3 Action Chunking과 Receding Horizon

Action chunk를 생성하되, 실행은 일부만:

$$\text{Generate: } (a_t, a_{t+1}, ..., a_{t+H-1}), \quad \text{Execute: } (a_t, ..., a_{t+E-1}), \quad E < H$$

남은 action은 폐기하고 새로운 observation으로 다시 생성 (receding horizon).

> ❓ **예상 질문**: Action chunk 경계에서 discontinuity는?
> **답변**: Receding horizon이 이를 완화하나 완전히 해결하지 못함. Exponential moving average 등의 smoothing이 실무에서 사용됨.

---

## 3. 데이터 전략

- **11개 시뮬레이션 태스크** + 1개 real-robot 태스크
- 각 태스크당 50-200 demonstrations
- Multimodal 태스크(Push-T, Can 등)에서 diffusion의 이점이 극대화

---

## 4. 실험 결과 심층 분석

| 태스크 카테고리 | IBC | BET | LSTM-GMM | **Diffusion Policy** |
|--------------|-----|-----|---------|---------------------|
| Unimodal | 72.1 | 68.3 | 75.8 | **82.4** |
| Multimodal | 41.2 | 55.7 | 48.3 | **78.6 (+46.9%)** |
| Contact-rich | 35.8 | 42.1 | 51.2 | **72.3** |

- **Multimodal 태스크에서 압도적 개선**: 기존 최고 대비 46.9% 향상
- Contact-rich에서도 큰 개선 → action sequence의 temporal consistency가 핵심

---

## 5. Ablation 분석

| 구성요소 | Push-T SR (%) |
|---------|-------------|
| Full (DDPM, 100 steps) | 86.2 |
| DDIM (10 steps) | 84.5 |
| No action chunking | 71.3 |
| No observation history | 79.8 |
| Prediction horizon 1 | 68.4 |
| CNN → Transformer | 87.1 |

- **Action chunking**의 기여가 매우 큼 (+15%p)
- Denoising step을 10으로 줄여도 성능 하락 미미 → 실시간 적용 가능

---

## 6. 관련 연구 비교

| 방법 | Distribution | Multimodal | Temporal | Training Stability |
|------|-------------|-----------|---------|-------------------|
| BC (MSE) | Implicit unimodal | ✗ | ✗ | ★★★★★ |
| GMM | Explicit mixture | △ (fixed K) | ✗ | ★★★★☆ |
| IBC | Implicit (EBM) | ✓ | ✗ | ★★☆☆☆ |
| BET | Discrete (VQ) | ✓ | ✗ | ★★★☆☆ |
| **Diffusion Policy** | **Implicit (diffusion)** | **✓** | **✓ (chunk)** | **★★★★☆** |

---

## 7. 한계 및 미해결 문제

### 방법론적 미비점
1. **추론 속도**: 100 denoising step = 100회 neural network forward → real-time control에서 병목. DDIM으로 10step 가능하나 여전히 단일 forward 대비 10배
2. **Observation space 제한**: 주로 2D image + proprioception. 3D, tactile 등 다른 modality와의 통합 미탐구 (이후 3D Diffuser Actor 등에서 해결)
3. **Language conditioning 부재**: 초기 논문이므로 language instruction 지원 없음. Task specification이 demonstration에 의존
4. **Scalability**: 수백 개 demonstration 규모. 대규모 데이터셋(수십만 trajectory)에서의 scaling 행동 미검증

### Attribution 문제
- 46.9% 향상 중 **diffusion formulation** vs **action chunking** vs **observation history**의 기여 분리 필요. Ablation에서 각각의 기여는 보이나, 상호작용 효과는 불완전

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — 로봇 학습에 diffusion 적용의 선구적 연구 |
| **Technical depth** | ★★★★★ — 두 가지 구현, 포괄적 분석 |
| **Experimental rigor** | ★★★★☆ — 12 태스크, 다양한 baseline |
| **Practical impact** | ★★★★★ — 이후 모든 VLA 연구의 기반 |
| **Writing quality** | ★★★★★ — 교과서적 명확성 |

**강점**: 로봇 학습에서 diffusion의 적용을 개척한 기념비적 연구. Action chunking, multimodal distribution 처리 등 핵심 설계 원칙을 확립. **약점**: 추론 속도, language conditioning 부재는 시대적 한계이며 후속 연구에서 해결됨.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | VAE가 더 빠른데 왜 diffusion인가? | VAE는 mode collapse 경향. Diffusion은 mode coverage가 우수하여 multimodal task에서 결정적 이점 |
| 2 | Real-world에서 100 denoising step이 가능한가? | DDIM 10 step으로 ~50ms. 20Hz 가능. 이후 consistency model, flow matching으로 1-step까지 가속 |
| 3 | Action chunk 내에서 observation이 변하면? | Receding horizon으로 부분적 해결. 완전한 closed-loop은 매 step 새로운 chunk 생성 필요 |
| 4 | 이 방법이 VLM과 결합되면? (→ 현재 VLA 시대) | CogACT, pi0 등이 정확히 이 방향. Diffusion Policy가 VLA의 action head로 편입 |
| 5 | Score matching의 approximation error가 action 품질에 미치는 영향은? | 이론적으로 분석 어려움. 실험적으로 충분히 좋은 성능을 보이나, worst-case 보장 없음 |
