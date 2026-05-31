# ConsisVLA-4D: Advancing Spatiotemporal Consistency in Efficient 3D-Perception and 4D-Reasoning for Robotic Manipulation

> **한 줄 요약**: OpenVLA 백본 위에 CV-Aligner(cross-view 의미 일관성, FiLM + Top-32 token), CO-Fuser(cross-object 기하 일관성, 64 aggregation token + block-wise causal attention), CS-Thinker(cross-scene 시공간 일관성, 1 dynamic + 3 depth decoder)의 3단 모듈을 얹어 LIBERO를 OpenVLA 76.5 → **98.1**(+21.6%p) 으로 끌어올리며 동시에 2.3배 빠른 추론(72.7 Hz)을 달성한 CVPR 2026 논문.

---

## 1. 배경 및 동기

### 기존 VLA의 시공간 인식 한계
- 대부분의 VLA(OpenVLA, RT-2 등)는 **single-view 2D 이미지 + 언어**만 입력으로 사용. 카메라 시점이 바뀌거나 occlusion이 생기면 객체 정체성이 흔들림.
- 객체 간 **기하학적 관계**(상대 위치, 접촉 가능성)는 학습 데이터에서 implicit하게만 얻어짐.
- Trajectory가 진행되며 장면이 변하는 **4D(공간 + 시간)** 관점에서의 명시적 일관성 제약은 거의 없음.

### 핵심 질문
- **3D 인식과 4D 추론을 효율적으로(parameter / latency 적게) VLA에 통합할 수 있는가?**
- **이 통합이 LIBERO 같은 표준 벤치마크에서 정량적으로 의미있는 향상을 가져오는가?**

📌 [Figure 1 삽입] — ConsisVLA-4D 전체 파이프라인: CV-Aligner → CO-Fuser → CS-Thinker

---

## 2. 방법론 심층 분석

### 2.1 세 가지 일관성 모듈 — 직관

| 모듈 | 일관성 종류 | 핵심 메커니즘 |
|------|------------|--------------|
| **CV-Aligner** | Cross-View 의미 일관성 | 다른 시점에서 같은 객체에 동일 의미 token 할당 |
| **CO-Fuser** | Cross-Object 기하 일관성 | 객체 간 상대 기하 관계를 명시 token으로 |
| **CS-Thinker** | Cross-Scene 시공간 일관성 | 시간 흐름에 따른 dynamic + depth 예측 |

### 2.2 CV-Aligner (Cross-View Aligner)

- 입력: 다중 카메라 view(예: 3rd-person + wrist)에서 추출한 SigLIP semantic token + DINOv2 geometric token
- 처리:
  1. **Layer-wise FiLM modulation**: 각 view의 semantic token을 모든 layer에 걸쳐 modulation
  2. **Explicit Semantic Object Selection**: Top-K = 32 token만 유지 (원본 256 → 32, 즉 1/8)
  3. **Single-Fusion**: VGGT 기반 3D feature와 4-layer Transformer (1152-dim hidden) 로 융합

> ❓ **예상 질문**: 왜 Top-32만 유지하는가? 정보 손실 위험?
> **답변**: 256 token 중 manipulation에 관여하는 객체는 보통 2~5개. 그 객체별 6~8개 token이면 충분. 잔여 token은 배경 잡음으로 작용. Latency를 0.204s → 0.110s로 거의 절반 단축 (Table 3 ablation).

> ❓ **예상 질문**: FiLM이 아닌 cross-attention을 쓰지 않은 이유?
> **답변**: FiLM은 affine modulation으로 cheaper. Cross-attention은 quadratic cost. Efficient 3D-perception 목표상 FiLM이 자연스러운 선택.

### 2.3 CO-Fuser (Cross-Object Fuser)

- **64 aggregation tokens**를 추가 — 객체별이 아닌 "관계"를 담는 slot.
- **Cosine-decayed weights** (α₀ = 0.2, α_L' = 0.01) 로 layer 깊이에 따라 영향 점진 감소.
- **Block-wise causal self-attention** — aggregation token이 객체 token을 보되, 객체 token은 다른 객체를 통해 간접 정보만 얻음 → "관계는 explicit하지만 객체끼리는 noisy하게 섞이지 않게".
- 결과: **Implicit Geometric Relation Aggregation** — 명시 좌표 없이 token space에서 객체 간 기하 관계 학습.

> ❓ **예상 질문**: 64라는 숫자는?
> **답변**: 일반 manipulation scene의 object pair 수에 대한 upper bound. 객체 5개 → pair C(5,2) = 10, ternary relation 포함 시 ~60. 64는 약간의 여유.

### 2.4 CS-Thinker (Cross-Scene Thinker)

- **Spatiotemporal Consistency Attention (SC-Attn)** 로 시간 축 처리.
- **1 dynamic decoder** + **3 independent depth decoder** (각각 8 transformer block, 1024 hidden, 16 attention heads).
- 동적 객체의 미래 위치와 global depth를 동시에 예측 — 단순 "현재 인식" 이 아니라 "action이 진행됨에 따른 변화"를 일관되게.
- 즉 모델이 **action을 출력하는 동시에 그 action이 일으킬 scene 변화를 internal하게 예측**, 이 두 head 사이의 일관성으로 강한 학습 신호.

### 2.5 Action Decoding

- **Action chunking + parallel decoding**, **L1 loss**.
- Single-arm chunk size **K=8**, dual-arm **K=25**.
- 즉 action head는 **regression 계열** — diffusion이나 flow matching이 아닌, parallel L1 regression. CV/CO/CS의 representation 강화 + chunking이 핵심.

> ❓ **예상 질문**: 왜 diffusion이 아닌 단순 L1 regression?
> **답변**: 핵심 가설이 "representation이 충분히 좋으면 action은 단순 regression으로 충분". 그래서 inference도 빠름 (72.7 Hz). 만일 diffusion이었으면 chunk K=25 dual-arm은 훨씬 느렸을 것.

---

## 3. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Backbone | OpenVLA (7B) |
| Vision encoders | SigLIP (semantic), DINOv2 (geometric), VGGT (3D) |
| Action chunk (single-arm) | K=8 |
| Action chunk (dual-arm) | K=25 |
| Training steps | 80K |
| Batch size (single-arm) | 64 |
| Batch size (dual-arm) | 32 |

---

## 4. 실험 결과 심층 분석

### 4.1 LIBERO (Table 1)

| Suite | OpenVLA (baseline) | **ConsisVLA-4D** | Δ |
|-------|-------------------|------------------|----|
| Spatial | 84.7 | **98.8** | +14.1 |
| Object | 88.4 | **99.8** | +11.4 |
| Goal | 79.2 | **98.0** | +18.8 |
| Long | 83.7 | **95.6** | +11.9 |
| **Avg** | **76.5** | **98.1** | **+21.6** |

- **Goal suite에서 가장 큰 향상(+18.8)** — Goal은 객체 정체성이 task에 결정적, CV-Aligner 효과가 자연스럽게 드러남.
- **Long(+11.9)에서도 큰 향상** — CS-Thinker의 시간 일관성이 long-horizon 누적 오류를 억제.

### 4.2 추론 효율

| 환경 | Inference Hz |
|------|-------------|
| Simulation | **72.7 Hz** |
| Real-world bimanual | **108.2 Hz** |

OpenVLA 대비 **2.3x speedup** (simulation), **2.4x speedup** (real-world).

> ❓ **예상 질문**: 추가 모듈이 3개 들어갔는데 어떻게 더 빠른가?
> **답변**: Top-32 token selection이 LLM block들이 다뤄야 할 token 수를 8배 줄임. FiLM이 cross-attention보다 cheap. Action chunking으로 step당 inference call 빈도가 K배 줄어듦. 종합하면 추가 모듈 cost를 크게 상쇄.

### 4.3 Ablation (Table 3)

| 설정 | Latency | FLOPs |
|------|---------|-------|
| Full ConsisVLA-4D | 0.110 s | 4.59 T |
| w/o E3D | 0.204 s | 16.83 T |

E3D(Efficient 3D-Perception, 즉 Top-32 token selection)가 latency를 약 절반, FLOPs를 약 3.7배 절감.

### 4.4 RoboTwin 2.0 & Real-World

- **RoboTwin 2.0**: 7개 dual-arm task × 100 trial. 본 표에 절대 평균치는 집계 안되어 있고 Figure 5에서 시각적 비교.
- **Real-world**: Galaxea R1 Lite, AgileX Cobot Magic. 4개 long-horizon task (microwave, banana peeling, drawer, T-shirt folding). 15-trial 평균치 보고 (정량 표 4).
- Real-world에서 OpenVLA 대비 **+41.5%** 절대 향상 보고.

---

## 5. 관련 연구 비교

| 모델 | 3D | 4D(시간) | 다중 view | 추론 속도 | LIBERO Avg |
|------|----|---------|-----------|----------|-----------|
| OpenVLA | ✗ | ✗ | △ | 1x | 76.5 |
| 3D-VLA | ✓ (estimated depth) | △ (goal generation) | ✗ | 매우 느림 | low(CALVIN 약함) |
| RDT-1B | ✗ | △ | ✓ | 1x | ~96 |
| Pi0 / Pi0.5 | ✗ | ✗ | ✓ | 1.5x | ~94 |
| **ConsisVLA-4D** | **✓ (VGGT)** | **✓ (CS-Thinker)** | **✓ (CV-Aligner)** | **2.3x** | **98.1** |

### 핵심 차별점
- "Efficient" 3D-perception (Top-K token) + 명시적 4D consistency가 동시에 들어간 거의 유일한 디자인.
- 속도까지 빨라진 점이 가장 강한 selling point.

---

## 6. 한계 및 미해결 문제

### 방법론적
1. **VGGT 의존**: 3D feature extractor로 VGGT가 fix. 다른 3D encoder로 swap 시 성능 보장 없음.
2. **Top-32가 어떤 task에서나 적합한가**: clutter가 매우 심한 scene에서는 32개로 부족할 수 있음.
3. **Action head는 단순 L1 regression**: 멀티모달 action 분포(예: pick from either side)는 표현하기 어려움. RDT/Pi0 의 diffusion/FM head 대비 표현력 부족.

### 평가
1. **RoboTwin 2.0 수치가 본문 표로 명시 안됨** — Figure로만 비교, 재현/비교 어려움.
2. **OpenVLA 76.5 baseline은 standard FT 기준** — 다른 강한 baseline (Pi0 94, RDT 96) 대비 비교는 본문 핵심 표에 부족.
3. **Real-world Table 4의 subtask 단위 보고**: 전체 task 성공률보다 stage 성공률 위주 — failure mode 분석은 좋으나 absolute success rate 비교가 직관적이지 않음.

### 해석
- **+21.6%p LIBERO 향상의 attribution**: SigLIP+DINOv2+VGGT 3개 encoder 추가 effect 자체가 크다. CV/CO/CS 모듈의 marginal contribution을 분리한 ablation이 본문에 명시적으로 부족.
- 이는 "세 가지 일관성 모듈이 핵심이다"라는 주장과, "더 강한 visual encoder를 그냥 더했다"라는 회의적 해석을 구분하기 어렵게 만든다.

---

## 7. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — 3D + 4D consistency를 명시 모듈로 분리한 디자인. CO-Fuser의 64 aggregation token 아이디어 특히 신선. |
| **Technical depth** | ★★★★☆ — 세 모듈 각각의 hyperparameter (Top-32, 64 agg, α₀=0.2, 4-layer/8-layer)가 정밀하게 정의됨. |
| **Experimental rigor** | ★★★☆☆ — LIBERO에서 강함. RoboTwin/real-world 정량 비교는 부족. Module-level ablation도 더 필요. |
| **Practical impact** | ★★★★★ — 72.7 Hz 시뮬, 108.2 Hz 실세계 — production 가능한 latency. |
| **Writing/Clarity** | ★★★★☆ — 모듈 명명(CV / CO / CS)이 직관적. |

**강점**: 표현력(98.1 LIBERO Avg)과 속도(2.3x)를 동시에 잡은 보기 드문 사례. CVPR 2026 채택될 만한 완성도. **약점**: action head가 단순 L1 — 멀티모달 action 분포가 중요한 long-horizon task에서 한계 가능.

---

## 8. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | LIBERO 98.1은 이미 ceiling에 가까운데, 진짜 가치는 어디서? | 동일 ceiling을 2.3x speed로 달성. 그리고 OpenVLA 76.5 baseline 대비 +21.6은 base가 약한 곳에서의 ceiling 도달이 핵심. |
| 2 | CV/CO/CS 각 모듈 단독 contribution은? | 본문 ablation에서 "E3D 유무"는 보고하나 세 모듈 각각 분리 ablation은 명시적으로 부족. 약점. |
| 3 | Top-32 token selection이 어떤 객체 token을 고르는가는 학습되는가? | Top-K는 attention score 기반 select. Score 자체는 task-driven 학습. |
| 4 | VGGT가 freeze인가 fine-tune인가? | 본 fetch 결과에 명시 안됨. 일반적으로 cost와 안정성 위해 freeze 추정. |
| 5 | 64 aggregation token이 dual-arm 25 chunk에서 충분한가? | dual-arm clutter는 single-arm보다 객체 수 많음. 본문에서 수치 비교 부족. |
| 6 | RoboTwin 2.0에서 절대 수치가 왜 본문 표에 없는가? | Figure 5로만 제시 — peer review 단계에서 reviewer 지적 가능 지점. |
| 7 | Real-world +41.5%는 어떤 metric인가? | Stage-aware partial success rate 평균으로 보임 (Table 4). Absolute task completion보다 더 generous한 metric일 수 있음. |
| 8 | Goal suite +18.8 향상의 source는? | CV-Aligner의 cross-view semantic identity 일관성 — Goal task는 "어떤 객체를 옮길까"가 핵심이므로 자연스러운 게인. |

<!-- VERIFIED: pdf -->
