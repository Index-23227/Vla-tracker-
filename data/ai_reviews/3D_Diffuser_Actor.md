# 3D Diffuser Actor: Policy Diffusion with 3D Scene Representations

> **한 줄 요약**: CLIP ResNet50로 추출한 2D feature를 depth로 3D로 lifting하고, 3D relative position attention을 가진 denoising transformer로 keypose를 생성하여, RLBench multi-view에서 81.3% (+18.1%p SOTA), single-view 78.4% (+13.1%p), CALVIN 3.27 avg len 달성.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- **PerAct/Act3D**: 3D voxel 또는 point 기반 정책이지만 **deterministic prediction** → multimodal action distribution 처리 불가
- **Diffusion Policy** (Chi et al., 2023): 강력한 multimodal action 생성이지만 **2D observation에 의존** → 3D spatial reasoning 부재
- 두 계열의 장점(3D spatial understanding + diffusion의 multimodal generation)을 결합하는 시도 부재

### 핵심 질문
- **3D scene feature를 diffusion process의 conditioning으로 자연스럽게 통합하면, 두 계열 모두를 능가할 수 있는가?**
- **3D relative position attention이 spatial generalization에 결정적 이점을 제공하는가?**

📌 [Figure 1 삽입] — 3D Diffuser Actor 아키텍처 overview

---

## 2. 방법론 심층 분석

### 2.1 3D Scene Representation

CLIP ResNet50으로 2D feature 추출 후, depth + camera intrinsics로 3D lifting:

$$\mathbf{F}_{3D} = \text{Lift}(\text{CLIP-ResNet50}(\mathbf{I}), \mathbf{D}, \mathbf{K})$$

- 각 2D feature token이 corresponding 3D position을 가짐
- Multi-view의 경우 여러 카메라의 features를 3D 공간에서 합침

### 2.2 3D Relative Position Attention (핵심 기여)

핵심 설계: Attention이 **절대 위치가 아닌 상대 위치**에만 의존:

$$\text{PE}(\mathbf{p}_i, \mathbf{x}_i)^T \text{PE}(\mathbf{p}_j, \mathbf{x}_j) = \mathbf{x}_i^T M(\mathbf{p}_j - \mathbf{p}_i) \mathbf{x}_j$$

이를 통해 **translation equivariance** 달성: 물체 위치가 변해도 동일한 relative 관계를 인식.

Rotary positional embedding으로 구현.

> ❓ **예상 질문**: Translation equivariance가 실질적으로 얼마나 중요한가?
> **답변**: Ablation에서 relative attention 제거 시 81.3% → 71.3% (**-10%p**). 이는 3D generalization에서의 결정적 기여를 입증. 물체가 workspace의 다른 위치에 놓여도 동일한 파지 전략을 적용할 수 있게 함.

### 2.3 Denoising Transformer

10-dimensional action (3D position + 6D rotation + gripper) 예측:

$$\hat{\boldsymbol{\epsilon}} = D_\theta(\mathbf{a}_t, t, \mathbf{F}_{3D}, \mathbf{l})$$

- **Position과 rotation에 별도 noise schedule**: Position은 scaled-linear ($\beta_{max}=0.02$, $\beta_{min}=0.0001$), rotation은 squared cosine
- **별도 MLP head**: position error, rotation error, gripper state 각각 예측

> ❓ **예상 질문**: Position과 rotation에 다른 noise schedule을 쓰는 이유는?
> **답변**: Position (Euclidean space)과 rotation (SO(3))의 geometry가 근본적으로 다름. 동일 schedule은 한쪽에 과소/과다 noise를 줄 수 있음. 분리된 schedule이 각 공간의 특성에 맞는 denoising을 가능하게 함.

### 2.4 Keypose Prediction

Dense trajectory가 아닌 **keypose** (task-critical waypoint) 예측:

$$\mathbf{a} = (\mathbf{p} \in \mathbb{R}^3, \mathbf{r} \in \mathbb{R}^6, g \in \{0,1\})$$

CALVIN에서 평균 10 keypose / 60 actions per task → keypose 방식의 효율성.

> ❓ **예상 질문**: Keypose 방식은 contact-rich task에서 한계가 있지 않은가?
> **답변**: 맞음. Keypose 간 motion planner (linear interpolation 등)에 의존하므로, 접촉 과정의 세밀한 force control이 필요한 task에는 부적합. Dense trajectory 예측 방식(Diffusion Policy)이 이런 task에 유리.

---

## 3. 데이터 전략

| 벤치마크 | 시연 수 | View 수 | 3D 정보 |
|---------|--------|---------|---------|
| RLBench PerAct setup | 100 demos/task | 4 views | GT depth |
| RLBench GNFactor setup | 20 demos/task | 1 view | GT depth |
| CALVIN | ~24h play data의 35% | Wrist + static | GT depth |
| Real-world | 15 demos/task | Multi-view | Calibrated depth |

---

## 4. 실험 결과 심층 분석

### RLBench Multi-view (18 Tasks, PerAct Setup)

| 모델 | Average SR (%) |
|------|---------------|
| C2F-ARM-BC | 20.1 |
| PerAct | 49.4 |
| RVT | 62.9 |
| Act3D | 63.2 |
| **3D Diffuser Actor** | **81.3 (+18.1%p)** |

**주목할 태스크별 결과**:
- Stack blocks: **+39.5%p** 절대 향상 (Act3D 대비)
- Insert peg: **+20.8%p** 절대 향상
- 이 태스크들은 3D spatial precision이 결정적 → 3D representation의 가치 입증

### RLBench Single-view (GNFactor Setup)

| 모델 | Average SR (%) |
|------|---------------|
| GNFactor | 31.7 |
| Act3D | 65.3 |
| **3D Diffuser Actor** | **78.4 (+13.1%p)** |

### CALVIN (ABC→D, Zero-shot Long-horizon)

| 모델 | 1 task | 2 tasks | 3 tasks | 4 tasks | 5 tasks | **Avg Len** |
|------|--------|---------|---------|---------|---------|------------|
| GR-1 | 85.4% | 71.2% | 59.6% | 49.7% | 40.1% | 3.06 |
| **3D Diffuser Actor** (60 keyposes) | **92.2%** | **78.7%** | **63.9%** | **51.2%** | **41.2%** | **3.27** |

- GR-1 대비 +0.21 avg len (**7% relative improvement**)

### 추론 속도

- **600ms on NVIDIA GeForce 2080 Ti** (CALVIN)
- 평균 10 keypose/task → 매 keypose 당 600ms

---

## 5. Ablation 분석

### PerAct Setup (Table III)

| Configuration | Average SR (%) |
|--------------|---------------|
| **3D Diffuser Actor (full)** | **81.3** |
| w/o Relative Attention (absolute position) | 71.3 (**-10.0%p**) |
| 2D Diffuser Actor (no 3D) | 47.0 (**-34.3%p**) |

- **3D representation의 기여**: +34.3%p (2D→3D)
- **Relative attention의 기여**: +10.0%p
- 두 요소 모두 결정적이나, **3D lifting이 더 큰 영향**

---

## 6. 관련 연구 비교

| 모델 | Representation | Policy Type | Multimodal | Language | Avg SR (RLBench) |
|------|---------------|-------------|-----------|---------|-----------------|
| PerAct | Voxel grid | Deterministic | ✗ | ✓ | 49.4% |
| Act3D | Point cloud | Deterministic | ✗ | ✓ | 63.2% |
| Diffusion Policy | 2D image | Diffusion | ✓ | △ | N/A (2D only) |
| **3D Diffuser Actor** | **3D feature cloud** | **Diffusion** | **✓** | **✓** | **81.3%** |

핵심 차이: 유일하게 **3D + diffusion + language**를 모두 결합. Deterministic 방법 대비 multimodal action 처리 우수, 2D diffusion 대비 spatial precision 우수.

---

## 7. 한계 및 미해결 문제

### 방법론적 미비점 (저자 명시 포함)
1. **Camera calibration + depth 필수**: 저자들이 직접 명시한 한계. 실제 배포에서 calibration 비용과 depth sensor 의존성
2. **Quasi-static task 가정**: RLBench, CALVIN 모두 정적 환경. **Dynamic task와 velocity control로의 확장**은 저자들이 future work로 명시
3. **Keypose의 한계**: Contact-rich task (삽입, 비틀기)에서 keypose 사이의 dynamics가 중요하나, linear interpolation에 의존
4. **추론 latency 600ms**: Real-time closed-loop에는 느림. 2080 Ti 기준이므로 최신 GPU에서 더 빠르겠으나, dense control (>10Hz)에는 부적합
5. **Real-world 결과의 깊이**: 12개 real task에서의 정량적 결과가 보고되나, 시연 수(15 demos/task)가 매우 적어 통계적 유의성 우려

### Attribution 문제
- Ablation에서 2D→3D의 기여(+34.3%p)가 diffusion의 기여보다 훨씬 큼 → **"3D가 핵심, diffusion은 보조"라는 해석**이 가능
- PerAct에 diffusion을 추가한 variant (3D + diffusion의 interaction effect 분리)가 없음

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — 3D + Diffusion 결합의 자연스러운 통합, relative position attention |
| **Technical depth** | ★★★★★ — 별도 noise schedule, rotary PE, position/rotation 분리 등 세심한 설계 |
| **Experimental rigor** | ★★★★☆ — RLBench 18 tasks + CALVIN + Real-world. 다만 real-world 규모 작음 |
| **Practical impact** | ★★★★☆ — 600ms latency, depth+calibration 의존이 배포 제약 |
| **Writing quality** | ★★★★★ — 매우 깔끔한 presentation |

**강점**: RLBench에서 +18.1%p라는 결정적 성능 차이. Relative attention의 translation equivariance가 이론적으로도 실험적으로도 잘 뒷받침됨. 3D+diffusion 결합의 레퍼런스 연구. **약점**: Quasi-static 가정, depth 의존성, keypose 방식의 한계는 후속 연구(dense trajectory VLA)에서 극복됨.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | PerAct에 diffusion head를 추가하면 어떻게 되는가? | 이 비교 없음. 3D representation vs diffusion의 독립적 기여 분리가 불완전. Ablation에서 3D가 +34.3%p, diffusion의 기여는 indirect |
| 2 | Relative attention 없이 absolute PE만 써도 71.3%이면 충분하지 않은가? | Task에 따라 다름. Stack blocks처럼 위치 변화가 큰 task에서 relative의 이점이 극대화. 단순 task에서는 차이 작을 수 있음 |
| 3 | 600ms latency를 어떻게 줄일 수 있는가? | DDIM acceleration (fewer steps), model distillation, keypose caching. 논문에서 denoising step 수 분석 부재 |
| 4 | CALVIN avg len 3.27이면 GR-1(3.06) 대비 modest한 향상인데? | +7% relative. CALVIN에서는 language grounding이 3D precision보다 중요하여 3D의 이점이 덜 드러남 |
| 5 | Real-world에서 depth estimation (GT depth 대신)을 쓰면? | 논문에서 실제 로봇 실험은 calibrated depth sensor 사용. Monocular estimation으로 전환 시 성능 하락 예상되나 정량화 없음 |
| 6 | Dense trajectory 예측으로 전환하면? | Keypose 방식의 구조적 한계를 극복할 수 있으나, denoising 대상의 차원이 증가 (10D × 1 → 10D × H). Trade-off 미분석 |
| 7 | 왜 CLIP ResNet50인가? DINOv2나 SigLIP은? | 실험 당시 CLIP이 주류. 이후 SpatialVLA 등에서 DINOv2가 spatial feature에 더 유리함 보임 |

<!-- VERIFIED: pdf -->
