# 3D Diffuser Actor: Policy Diffusion with 3D Scene Representations

> **한 줄 요약**: 3D scene representation과 diffusion policy를 통합하여, language-conditioned manipulation에서 3D denoising transformer 기반의 keypose 예측으로 기존 2D diffusion policy 및 3D 방법론을 동시에 능가.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- **Diffusion Policy** (Chi et al., 2023): 강력한 action generation이지만 2D observation에 의존 → 3D spatial reasoning 부재
- **PerAct/Act3D**: 3D voxel 기반 정책이지만 deterministic prediction → multimodal action distribution 처리 불가
- 두 계열의 장점(3D 이해 + diffusion의 multimodal generation)을 결합하는 시도 부재

### 핵심 질문
- **3D scene feature를 diffusion process의 conditioning으로 자연스럽게 통합할 수 있는가?**
- **3D-aware denoising이 2D observation 기반 diffusion 대비 실질적 이점이 있는가?**

📌 [Figure 1 삽입] — 3D Diffuser Actor 아키텍처 overview

---

## 2. 방법론 심층 분석

### 2.1 3D Scene Representation

Multi-view RGB-D 이미지에서 3D feature cloud를 구성:

$$\mathbf{F}_{3D} = \text{Unproject}(\{(\mathbf{I}_i, \mathbf{D}_i, \mathbf{K}_i)\}_{i=1}^{V})$$

각 3D point에 2D feature(사전학습된 visual encoder에서 추출)를 backproject하여 per-point feature를 부여.

> ❓ **예상 질문**: Feature backprojection에서 depth 오차가 3D feature 위치에 미치는 영향은?
> **답변**: Depth 오차는 feature의 공간적 mis-alignment를 유발. 시뮬레이션에서는 정확한 depth를 사용하므로 문제없으나, 실제 환경에서는 depth sensor 노이즈가 누적됨. 저자들은 이 문제를 직접 다루지 않음.

### 2.2 3D Denoising Transformer

핵심 기여: denoising process에서 **3D scene token과 noisy action token 간의 cross-attention**을 수행하는 transformer:

$$\hat{\mathbf{a}}_0 = D_\theta(\mathbf{a}_t, t, \mathbf{F}_{3D}, \mathbf{l})$$

여기서:
- $\mathbf{a}_t$: timestep $t$에서의 noisy action (6-DoF keypose)
- $\mathbf{F}_{3D}$: 3D scene features
- $\mathbf{l}$: language instruction embedding

> ❓ **예상 질문**: 왜 Diffusion Transformer이고 U-Net이 아닌가?
> **답변**: 3D point cloud는 불규칙 구조(irregular structure)이므로 grid-based U-Net보다 transformer의 set-to-set attention이 자연스러움. 또한 language와 proprioception을 다중 modality로 융합하기에 transformer 구조가 유리.

### 2.3 Keypose Prediction

연속 trajectory가 아닌 **keypose** (task-critical waypoint)를 예측:

$$\mathbf{a} = (\mathbf{p}, \mathbf{q}, g) \in \mathbb{R}^3 \times SO(3) \times \{0, 1\}$$

Position $\mathbf{p}$, orientation quaternion $\mathbf{q}$, gripper state $g$.

> ❓ **예상 질문**: Keypose 예측은 waypoint 간 motion planning을 별도로 필요로 하는데, 이 overhead는?
> **답변**: Keypose 방식은 motion planner(RRT, linear interpolation 등)에 의존. 이는 장점(높은 수준 계획에 집중)이자 단점(충돌 회피 등 low-level 문제를 별도 처리). Diffusion Policy의 dense trajectory 대비 trade-off.

---

## 3. 데이터 전략

| 벤치마크 | 데이터 형태 | View 수 | 3D 정보 |
|---------|-----------|---------|---------|
| RLBench | 시뮬레이션 demo | 4 views | Ground-truth depth |
| CALVIN | 시뮬레이션 demo | 1-3 views | Ground-truth depth |

- 시뮬레이션 환경의 **완벽한 depth** 사용 → real-world 적용 시 depth estimation 품질이 핵심 bottleneck
- 데이터 증강 전략에 대한 언급 미비

---

## 4. 실험 결과 심층 분석

### RLBench (18 Tasks)

| 모델 | Multi-view SR (%) | Single-view SR (%) |
|------|------------------|-------------------|
| PerAct | 49.4 | - |
| Act3D | 65.0 | - |
| Diffusion Policy (2D) | 54.2 | 48.1 |
| **3D Diffuser Actor** | **76.3 (+18.1%)** | **61.2 (+13.1%)** |

### CALVIN (ABC→D)

| 모델 | Avg. Len (5 tasks) |
|------|-------------------|
| HULC | 2.64 |
| 3D Diffuser Actor | 2.88 (+9% rel.) |

- RLBench에서의 개선폭이 인상적이나, **CALVIN에서의 개선이 상대적으로 작음** → language grounding보다는 3D precision이 요구되는 task에서 강점

---

## 5. Ablation 분석

| Component | RLBench 변화 |
|-----------|-------------|
| 3D features → 2D features | -15%p |
| Denoising transformer → MLP | -8%p |
| Multi-view → Single-view | -12%p |
| Language conditioning 제거 | -6%p |

- 3D feature의 기여가 압도적 → **모델의 성능이 3D representation quality에 크게 의존**

---

## 6. 관련 연구 비교

| 모델 | Representation | Policy Type | Multimodal Action | Language |
|------|---------------|-------------|-------------------|----------|
| PerAct | Voxel grid | Deterministic | ✗ | ✓ |
| Act3D | Point cloud | Deterministic | ✗ | ✓ |
| Diffusion Policy | 2D image | Diffusion | ✓ | △ |
| **3D Diffuser Actor** | **3D feature cloud** | **Diffusion** | **✓** | **✓** |

핵심 차이: 유일하게 **3D + diffusion + language**를 모두 결합. PerAct/Act3D는 deterministic이므로 bimodal action distribution에 취약, Diffusion Policy는 2D이므로 spatial precision 부족.

---

## 7. 한계 및 미해결 문제

### 방법론적 미비점
1. **Multi-view 의존성**: 대부분의 성능 향상이 multi-view setup에서 나옴. 실제 deployment에서 multi-camera 설치는 비용·보정 문제 수반
2. **Keypose의 한계**: 접촉 역학이 중요한 태스크(삽입, 비틀기 등)에서 keypose만으로는 불충분. Dense trajectory가 필요한 경우의 성능 미검증
3. **추론 속도**: Diffusion의 iterative denoising → latency 문제. 논문에서 denoising step 수 대비 성능 trade-off 분석 부재
4. **Real-world 실험 부재**: 전적으로 시뮬레이션 기반 평가. 실제 depth sensor 노이즈, calibration error 하에서의 robustness 미검증

### Attribution 문제
- 성능 향상이 **3D representation** vs **diffusion modeling** vs **transformer architecture** 중 어디서 주로 오는지 완전히 분리되지 않음
- PerAct에 diffusion을 추가한 variant, 또는 Diffusion Policy에 depth를 추가한 variant와의 직접 비교 부족

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — 3D + Diffusion 결합의 자연스러운 통합 |
| **Technical depth** | ★★★★☆ — 3D denoising transformer 설계가 잘 구성됨 |
| **Experimental rigor** | ★★★☆☆ — Sim-only, real-world 부재 |
| **Practical impact** | ★★★☆☆ — Multi-view 의존성이 deployment 제약 |
| **Writing quality** | ★★★★☆ — 깔끔한 presentation |

**강점**: 3D와 diffusion이라는 두 강력한 도구를 자연스럽게 결합한 설계가 우아함. RLBench에서의 큰 폭 성능 향상이 설득력 있음. **약점**: 시뮬레이션 전용 결과로 real-world applicability가 검증되지 않았고, multi-view 필수라는 점이 실용성을 제한.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Single camera + depth estimation으로도 작동하는가? | Single-view 결과가 있으나 multi-view 대비 큰 성능 하락. Monocular depth estimation의 metric scale 불확실성이 추가적 문제 |
| 2 | Diffusion step 수에 따른 latency와 성능 trade-off는? | 논문에서 미제공. DDIM acceleration 적용 가능하나 3D cross-attention의 overhead가 병목 |
| 3 | PerAct에 diffusion head를 추가하면 어떻게 되는가? | 이 비교가 없어 3D representation vs diffusion의 독립적 기여를 분리 불가 |
| 4 | Contact-rich task(peg insertion 등)에서의 성능은? | Keypose 방식의 구조적 한계로, 접촉 과정의 fine-grained force/position control이 필요한 task에서는 한계 예상 |
| 5 | Point cloud density는 성능에 얼마나 민감한가? | Ablation 미제공. Sparse point cloud에서의 degradation 양상이 real-world 배포에 중요 |
| 6 | Language instruction 없이 goal image만으로도 작동하는가? | 구조적으로 가능하나 실험 미수행. Goal-conditioned variant의 성능 비교 필요 |
| 7 | 왜 CALVIN 개선폭이 RLBench보다 작은가? | CALVIN은 상대적으로 simple geometry의 task가 많아 3D 이점이 덜 드러남. Language grounding 능력이 더 중요한 벤치마크 |
