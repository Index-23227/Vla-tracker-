# KineVLA: Towards Kinematics-Aware VLA Models with Bi-Level Action Decomposition

> **한 줄 요약**: OpenVLA 위에 bi-level RVQ-VAE (goal codebook + kinematics codebook)를 구축하여, goal-invariant 행동과 kinematics-variant 행동을 분리하고, kinematics-sensitive task에서 **76.5%** kinematics SR (baseline 35%) + 96ms inference 달성.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 기존 VLA는 **goal completion만 평가** → "어떻게" 수행하는지 (kinematics: 방향, 궤적, 회전)는 무시
- 동일 goal을 달성하는 여러 kinematics 경로 중 **특정 경로를 지정**할 수 없음
- Action tokenization이 goal과 kinematics를 entangle → fine-grained control 불가

### 핵심 질문
- **Goal-invariant한 coarse action과 kinematics-variant한 fine action을 분리하면, 지정된 동작 방식으로 task를 수행할 수 있는가?**

---

## 2. 방법론 심층 분석

### 2.1 Bi-Level RVQ-VAE

두 개의 codebook으로 action을 계층적으로 표현:

**Goal codebook ($E_l$)**: Stage I에서 Open X-Embodiment + ManiSkill 데이터로 학습. Task의 semantic goal을 인코딩.

**Kinematics codebook ($E_h$)**: Stage II에서 kinematics-annotated 데이터로 학습 (goal codebook freeze). 방향, 궤적, 회전 등 세밀한 kinematic parameter를 인코딩.

### 2.2 Bi-Level Reasoning

Chain-of-thought 스타일로 coarse + fine reasoning text와 discrete action token을 생성:
- Coarse reasoning: "pick up the cup" (goal)
- Fine reasoning: "approach from the left, rotate 90°" (kinematics)
- 동일 vocabulary space에서 text + action token을 unified하게 생성

### 2.3 Mutual Information Regularization

$$\mathcal{L}_{MI} = -\frac{1}{2N}\sum\left[\log\frac{e^{s_{ii}}}{\sum e^{s_{ij}}} + \log\frac{e^{s_{ii}}}{\sum e^{s_{ji}}}\right]$$

InfoNCE objective로 reasoning과 action의 일관성을 강제.

### 2.4 Architecture

- **Base**: OpenVLA (Prismatic-7B)
- **Fine-tuning**: LoRA rank=32, all linear layers
- **Action chunk**: H=5 timesteps
- **Inference**: 96ms

---

## 3. 실험 결과 심층 분석

### LIBERO-Goal-Relabeled

| Model | Goal SR (%) | Kinematics SR (%) |
|-------|-----------|------------------|
| OpenVLA | ~70 | ~35 |
| π₀.₅ | ~69 | ~38 |
| VQ-VLA | ~71 | ~40 |
| **KineVLA** | **~70** | **76.5** |

- Goal SR은 유사하나 **kinematics SR에서 36%p+ 향상**

### Ablation (LIBERO-Goal-Relabeled)

| Component | Kinematics SR (%) |
|-----------|------------------|
| Baseline (OpenVLA) | ~35 |
| + Bi-Rep (bi-level representation) | ~57 (+22) |
| + Bi-Rea (bi-level reasoning) | ~65 (+8) |
| **+ MI (mutual information)** | **76.5 (+11.5)** |

### Real Robot (Kine-Realman-75)

| Model | Kinematics SR (%) |
|-------|------------------|
| OpenVLA | ~25 |
| **KineVLA** | **65.0** |

### Inference Speed

| Model | Latency (ms) |
|-------|-------------|
| OpenVLA | 237 |
| π₀.₅ | 86 |
| VQ-VLA | 84 |
| **KineVLA** | **96** |

---

## 4. Training

- RVQ-VAE: 1× A100, batch 1024
- Fine-tuning: 4× A100, batch 64, 50K steps
- Datasets: LIBERO-Goal-Relabeled (63K), Kine-LIBERO (266K), Kine-Realman-75 (49K)

---

## 5. 한계 및 미해결 문제

1. **Goal SR이 향상되지 않음**: Kinematics에 집중하면서 goal 성능은 baseline과 동일. Trade-off 존재
2. **Kinematics annotation 비용**: 사람이 방향, 궤적 등을 명시적으로 annotation해야 함
3. **표준 LIBERO 4-suite 결과 없음**: Custom relabeled 데이터셋에서만 평가

---

## 6. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — "어떻게 수행하는가"를 VLA에 도입 |
| **Technical depth** | ★★★★★ — RVQ-VAE, MI regularization, 체계적 ablation |
| **Experimental rigor** | ★★★★☆ — 3 datasets + real robot. 표준 벤치마크 부재 |
| **Practical impact** | ★★★★☆ — Kinematics-aware control의 새로운 방향 |

---

## 7. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Goal SR이 안 오르는데, kinematics를 신경쓸 필요가 있는가? | 산업 현장에서 "어떻게"가 중요 (안전, 효율). 같은 goal이라도 경로에 따라 충돌 가능 |
| 2 | Codebook 크기가 성능에 미치는 영향은? | 미보고. 작으면 expressiveness 부족, 크면 학습 어려움 |
| 3 | FAST tokenizer와의 결합은? | FAST는 frequency-space, KineVLA는 semantic-space 분해. 상보적 가능성 |

<!-- VERIFIED: pdf -->
