# Action Head Taxonomy

## Classification Principle

> **"모델이 내부 표현(representation)에서 로봇 action을 어떤 메커니즘으로 생성하는가?"**

Action head category는 **action 생성의 핵심 메커니즘**을 기준으로 분류합니다.
부가적인 요소(reasoning, world model, MoE routing 등)는 `tags`로 표현하며 category에 영향을 주지 않습니다.

---

## Categories

### 1. `autoregressive`
**정의**: Action을 discrete token sequence로 변환하고, **왼→오른 순차적으로 하나씩 예측**.

**판별 기준**:
- Next-token prediction (cross-entropy loss)
- Causal attention mask on action tokens
- Action을 256/1024 bin으로 discretize

**대표 모델**: OpenVLA, RT-2-X, SpatialVLA, GR-1, GR-2

**주의**: Autoregressive generation 후에 RL이나 CoT가 추가되더라도, **action 생성 자체가 AR이면** 이 카테고리.
- OpenVLA + ECoT (reasoning 추가) → 여전히 `autoregressive`
- SimpleVLA-RL (RL fine-tuning) → 여전히 `autoregressive`

---

### 2. `diffusion`
**정의**: Gaussian noise에서 시작하여 **반복적 denoising**으로 continuous action을 생성.

**판별 기준**:
- DDPM/DDIM noise schedule
- Epsilon prediction 또는 score matching loss
- 여러 denoising step (10~100)을 거쳐 action 복원

**대표 모델**: Diffusion Policy, CogACT, DexVLA, Octo, RDT-1B

**주의**: Action head의 architecture가 U-Net이든 DiT(Diffusion Transformer)이든, **denoising 메커니즘이 DDPM/DDIM이면** 모두 이 카테고리.
- CogACT (DiT로 구현) → `diffusion` (DiT는 architecture, diffusion은 generation mechanism)
- GR00T-N1 (Diffusion Transformer) → `diffusion`

---

### 3. `flow_matching`
**정의**: Noise→action을 **직선 경로(optimal transport)**로 변환하는 ODE solver 기반 생성.

**판별 기준**:
- Flow matching / rectified flow loss
- Velocity field prediction ($v_\theta$)
- ODE integration (vs diffusion의 SDE)
- 일반적으로 diffusion보다 적은 step으로 고품질 생성

**대표 모델**: pi0, pi0.5, FLOWER, SmolVLA, InstructVLA

**Diffusion vs Flow Matching 구분**:
- Loss가 epsilon prediction → `diffusion`
- Loss가 velocity/flow field prediction → `flow_matching`
- 논문에서 "flow matching" 또는 "rectified flow"를 명시적으로 사용 → `flow_matching`

---

### 4. `discrete_diffusion`
**정의**: **Discrete token 위에서 masking/noise 기반 diffusion**을 수행하여 action token을 iterative하게 생성.

**판별 기준**:
- 마스킹 + iterative unmasking (BERT-style이지만 multi-step)
- Discrete token에 noise를 적용 (continuous embedding 상의 noise 포함)
- Parallel decoding이 가능하지만 iterative refinement 포함

**대표 모델**: DD-VLA, E0, dVLA, UD-VLA

**Autoregressive vs Discrete Diffusion 구분**:
- 왼→오른 순차 생성 → `autoregressive`
- Mask→unmask iterative 생성 (순서 무관 또는 confidence 기반) → `discrete_diffusion`

---

### 5. `regression`
**정의**: **Single forward pass**로 action을 직접 예측. Iterative refinement 없음.

**판별 기준**:
- L1/L2/MSE loss로 continuous action을 직접 regression
- 또는 single-pass parallel decoding (1 NFE)
- Stochastic sampling이나 iterative denoising 없음

**대표 모델**: OpenVLA-OFT (parallel decoding), HPT, CrossFormer, RoboMamba

**주의**: "Parallel decoding"은 한번에 모든 action을 예측하므로 `regression`.
OpenVLA-OFT는 기존 AR을 parallel로 바꾸고 L1 loss로 전환 → `regression`.

---

### 6. `inverse_dynamics`
**정의**: 미래 상태(이미지, point cloud 등)를 먼저 예측하고, **상태 변화에서 action을 역추론**.

**판별 기준**:
- World model / video prediction이 action prediction에 **선행**
- Action = f(current_state, predicted_future_state)
- Inverse dynamics model이 핵심 action 생성 모듈

**대표 모델**: UniPi, AVDC, DreamVLA, SuSIE

**주의**: World model을 **auxiliary task**로만 사용하고 action은 별도 head로 생성하면, 해당 head의 카테고리를 따름.
- GR-1 (video prediction + action head) → action head가 regression이므로 `regression`이 아니라... 실제로는 action head가 linear layer → `regression`
- Fast-WAM (video co-training but action is flow matching) → `flow_matching`

---

### 7. `hybrid`
**정의**: **두 가지 이상의 생성 메커니즘을 명시적으로 결합**하여 action을 생성.

**판별 기준**:
- 논문이 명시적으로 두 메커니즘의 결합을 주장
- 단일 모델 내에서 AR reasoning → diffusion action 등의 **cascaded 사용**
- 단순히 "RL로 fine-tune했다"는 hybrid가 아님 (RL은 학습 방법, 생성 메커니즘이 아님)

**대표 모델**: HybridVLA (AR+diffusion), Diffusion-VLA (AR reasoning + diffusion action), DeepThinkVLA (causal CoT + bidirectional action)

**주의**: "CoT reasoning 후 action 생성"이 두 가지 **다른 generation mechanism**을 사용하면 hybrid.
- ECoT (reasoning은 AR, action도 AR) → `autoregressive` (같은 메커니즘)
- Diffusion-VLA (reasoning은 AR, action은 diffusion) → `hybrid` (다른 메커니즘)

---

### 8. `other`
**정의**: 위 7개 카테고리에 명확히 속하지 않는 특수한 아키텍처.

**사용 기준**: 가능한 한 사용을 **최소화**. 위 카테고리로 분류 가능하면 반드시 그렇게 할 것.

**해당 모델**: Gemini Robotics (비공개), NanoVLA (세부 미공개), Humanoid-VLA (특수 아키텍처)

---

## Edge Cases & Decision Rules

| 상황 | 결정 | 근거 |
|------|------|------|
| RL로 fine-tune된 AR 모델 | `autoregressive` | RL은 학습 방법이지 생성 메커니즘이 아님 |
| MoE routing + flow matching | `flow_matching` | MoE는 capacity 분배 방식이지 생성 메커니즘이 아님 |
| DiT architecture + DDPM loss | `diffusion` | DiT는 architecture, DDPM은 generation mechanism |
| DiT architecture + flow matching loss | `flow_matching` | Loss가 결정 |
| Early exit + base model | base model의 카테고리 | Early exit은 효율화 기법 |
| Video prediction (auxiliary) + action head | action head의 카테고리 | Auxiliary task는 분류에 영향 없음 |
| Video prediction → inverse dynamics | `inverse_dynamics` | Action이 상태 변화에서 추론됨 |
| AR reasoning → diffusion action | `hybrid` | 두 다른 메커니즘의 결합 |
| AR reasoning → AR action | `autoregressive` | 같은 메커니즘 |
| FAST/DCT tokenization + AR | `autoregressive` | FAST는 tokenization 방식이지 생성 메커니즘이 아님 |

---

## Summary Table

| Category | Core Mechanism | Loss Function | Iterative? | Continuous? |
|----------|---------------|--------------|-----------|------------|
| `autoregressive` | Sequential token prediction | Cross-entropy | No (sequential) | No (discrete) |
| `diffusion` | Gaussian denoising | Epsilon/score matching | Yes (10-100 steps) | Yes |
| `flow_matching` | ODE velocity field | Flow matching | Yes (4-10 steps) | Yes |
| `discrete_diffusion` | Masked token denoising | Masked cross-entropy | Yes (8-12 steps) | No (discrete) |
| `regression` | Direct prediction | L1/L2/MSE | No (single pass) | Yes |
| `inverse_dynamics` | State-change → action | Varies | Depends | Varies |
| `hybrid` | Multiple combined | Multiple | Varies | Varies |
| `other` | Specialized | Varies | Varies | Varies |
