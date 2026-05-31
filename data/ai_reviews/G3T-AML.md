# G3T + AML: Learning Action Manifold with Multi-view Latent Priors for Robotic Manipulation

> **한 줄 요약**: 단안 입력의 깊이 모호성을 해소하기 위해 LongCat-Image-Edit(6B) 잠재공간 multi-view diffusion으로 합성 novel view를 만들고, VGGT의 단안 geometric prior와 함께 **Geometry-Guided Gated Transformer (G3T)** 가 occlusion noise를 게이팅하며 정렬한다. Action head는 noise/velocity 회귀 대신 **Action Manifold Learning (AML)** 로 valid action manifold 위에서 직접 action을 예측 (속도-일관성 reweighted MSE w(τ)=1/(1-τ)²). Qwen3-VL(4B) backbone에 16-layer DiT action expert. **LIBERO 98.6%**, **LIBERO-Plus zero-shot 85.7%**, **RoboTwin 2.0 Clean/Randomized 85.18%/86.06%**.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 대부분의 VLA는 **단안 RGB** 입력으로 spatial reasoning을 수행 → 깊이 모호성, occlusion에서 취약.
- 명시적 3D backbone(point cloud, RGB-D)을 쓰면 데이터 부족 + 일반화 한계.
- Diffusion VLA의 action head는 **noise/velocity 회귀**로 학습 → action space의 구조(physical reachability, joint limit)를 명시적으로 활용하지 못함 → 학습 비효율.

### 핵심 질문
- **단안 입력만으로 multi-view geometric prior를 합성해 spatial reasoning을 강화할 수 있는가?**
- **Action을 unstructured target(noise/velocity)이 아니라 valid action manifold 위에서 직접 예측하면 학습 효율과 성능이 모두 향상되는가?**

📌 [Figure 1 삽입] — 전체 파이프라인: Qwen3-VL → (VGGT monocular prior, LongCat multi-view synthesis) → G3T fusion → 16-layer DiT (AML) → action.

---

## 2. 방법론 심층 분석

### 2.1 Multi-view Latent Prior (LongCat-Image-Edit)
- **LongCat-Image-Edit (6B)** 는 latent-space에서 multi-view diffusion으로 novel view를 합성하는 사전학습 모델.
- 단안 입력 I 가 들어오면, latent z = E(I) → multi-view diffusion으로 별도 viewpoint의 latent z_v 들을 생성.
- Pixel space 디코딩 없이 **latent space에서 직접** G3T에 공급 → 효율적.

### 2.2 VGGT Geometric Prior
- VGGT (Visual Geometry Grounded Transformer) 는 단안 image로부터 dense geometric feature(implicit depth/normal/camera)를 추출하는 foundation model.
- 단안 prior가 정확한 metric depth는 아니지만 **상대적 geometric 구조** 를 제공.

### 2.3 Geometry-Guided Gated Transformer (G3T)
G3T는 두 가지 입력을 정렬·융합:
1. Multi-view latent features (LongCat에서 합성된 view들)
2. VGGT의 geometric feature (단안 prior)

**핵심 메커니즘**:
- Cross-attention 기반 view fusion.
- **Geometry-guided gating**: VGGT geometric feature가 attention weight를 조절 — occlusion / 비일관 view를 down-weight.
- 출력: spatially-grounded multi-view-aware token sequence.

> ❓ **예상 질문**: LongCat이 합성한 novel view가 부정확하면 노이즈를 G3T에 주입하는 셈 아닌가?
> **답변**: 정확히 그 우려를 G3T의 gating이 해결. VGGT geometric prior와 어긋나는 view feature는 게이팅으로 down-weight되어 occlusion noise가 자동 필터링. Ablation에서 gating 제거 시 LIBERO-Plus 성능 큰 폭 하락.

### 2.4 Action Manifold Learning (AML)

기존 diffusion VLA는 noise ε 또는 velocity v 를 예측. AML은 직접 **action a** 를 예측 (a-prediction):
- 손실: reweighted MSE
  ```
  L_AML = w(τ) · ||a_pred − a_gt||²,  w(τ) = 1/(1−τ)²
  ```
- τ는 flow time(0→1). w(τ)는 τ→1(action 도착점 근처)에서 가중치 증가 → final-action accuracy 강조.

> ❓ **예상 질문**: a-prediction이 noise-prediction보다 본질적으로 우월한 이유는?
> **답변**: noise prediction은 sample-quality 측면에서 우수하지만, action은 **저차원 valid manifold** 에 분포 → unstructured noise를 거치지 않고 직접 매니폴드로 가는 것이 sample-efficient. Reweighted loss는 마지막 step의 정확도(실제 로봇 action quality)와 직결.

📌 [Figure 2/3 삽입] — G3T attention map (geometry-guided gating 시각화), AML loss reweighting curve.

---

## 3. 데이터 전략

| 데이터셋 | 용도 |
|---------|------|
| LIBERO (Spatial/Object/Goal/Long, 모두) | 메인 평가 |
| LIBERO-Plus (perturbation suite) | zero-shot robustness |
| RoboTwin 2.0 (Clean / Randomized) | 멀티-arm 평가 |
| Franka Emika Panda 실로봇 (4 task) | real-world |

- 추가적인 multi-view paired data 없이 LongCat이 **사전학습된 novel-view synthesis** 만으로 합성 view를 제공 — 데이터 효율적.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| VLM backbone | Qwen3-VL 4B |
| Geometric encoder | VGGT |
| View synthesis | LongCat-Image-Edit 6B (latent space) |
| Action expert | 16-layer DiT |
| Hardware | 4 × NVIDIA H20 GPU |
| Training | 30K steps, ~27 hours, batch size 16, bfloat16 |
| Action horizon | 8 steps |
| Inference denoising | 4 steps |

---

## 5. 실험 결과 심층 분석

### 5.1 LIBERO 메인 결과 (Table II)

| Suite | Score |
|-------|-------|
| Spatial | 98.8 |
| Object | 99.8 |
| Goal | 99.0 |
| Long | 96.6 |
| **Avg** | **98.6** |

비교: π0 94.4, GR00T-N1.6 97.0, Spatial Forcing 98.5 → SOTA 갱신.

### 5.2 LIBERO-Plus Zero-shot Robustness (Table I)

| Perturbation | Score |
|-------------|-------|
| Camera | 89.6 |
| Robot | 60.1 |
| Language | 86.9 |
| Light | 98.0 |
| Background | 95.7 |
| Noise | 97.2 |
| Layout | 78.2 |
| **Overall** | **85.7** |

→ **Robot perturbation 60.1%** 가 약점 (proprio 변경에 취약). 그 외엔 매우 강건.

### 5.3 RoboTwin 2.0 (Table III)

| Split | Avg SR |
|-------|--------|
| Clean | 85.18 |
| Randomized | **86.06** (!) |

X-VLA 72.80 / 72.84, π0.5 42.98 / 43.84 대비 큰 폭 우위. **Randomized가 Clean보다 높은** 비정상적 현상은 noise injection이 일부 task에서 over-fit을 깨주는 효과로 추정.

### 5.4 실로봇 (Table XI, Franka Panda 4 task)

| Task | Success |
|------|---------|
| Stack Block | 70% |
| Insert Cube | 60% |
| Place Cylinder | 60% |
| Place Cup | 70% |
| **Avg** | **65%** |

### 5.5 Ablation (LIBERO-Plus 기준)

| 구성 | LIBERO-Plus |
|------|-------------|
| VGGT only | 66.4 |
| + LongCat (2 views) | 72.4 |
| + G3T integration | 77.9 |
| **+ AML (full)** | **85.7** |

→ 각 컴포넌트가 명확히 누적 기여. **AML alone +7.8%p** 로 가장 큰 단일 기여.

---

## 6. 관련 연구 비교

| 모델 | 3D source | Action head | LIBERO avg |
|------|----------|------------|-----------|
| π0 | mono | flow-matching (noise) | 94.4 |
| π0.5 | mono | flow-matching | 96.7 |
| GR00T-N1.6 | mono | DiT | 97.0 |
| Spatial Forcing | depth-aux | DiT | 98.5 |
| **G3T+AML** | **synth multi-view + VGGT** | **AML (a-pred)** | **98.6** |

핵심 차별점:
- **명시적 3D 입력 없이** novel-view synthesis로 multi-view geometric reasoning 확보.
- **a-prediction + reweighted loss** 로 action manifold 활용.

---

## 7. 한계 및 미해결 문제

1. **계산 비용**: VLM 4B + view-synthesis 6B + DiT 16-layer → 추론 시 multi-stage 파이프라인. Inference latency / FPS 보고 없음 — real-time deployment 적합성 불명.
2. **Robot perturbation 60.1%**: proprioception 변경(robot arm 다른 모델, 다른 calibration)에 약함. 실로봇 일반화에서 가장 큰 위협.
3. **합성 view의 사실성 의존**: LongCat이 비현실적 view를 생성하면 G3T가 noise만 보게 됨. Out-of-distribution scene에서의 view-synthesis 품질 검증 부재.
4. **AML 우월성의 일반화**: AML이 모든 manifold에서 작동하는지, 아니면 LIBERO/RoboTwin 같은 정형 manipulation task에 특화된 결과인지 불명. Dexterous / contact-rich task에서의 검증 필요.
5. **단일 acronym 부재**: 논문이 G3T와 AML을 두 독립 contribution으로 제시 — 후속 연구 / 인용 시 명명이 혼란스러움.
6. **Code 미공개**: 프로젝트 페이지(`junjxiao.github.io/Multi-view-VLA.github.io`)만 존재. 실제 GitHub repo는 미공개.
7. **Randomized > Clean (RoboTwin)**: 정상적 표 흐름과 반대 — 통계적 노이즈인지, 실제 robustness 효과인지 추가 분석 부재.

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — multi-view latent synthesis + action manifold direct prediction의 결합은 신선함 |
| **Technical depth** | ★★★★☆ — G3T gating + AML reweighting 모두 명확한 design rationale |
| **Experimental rigor** | ★★★★★ — LIBERO + LIBERO-Plus + RoboTwin 2.0 + real-world + 풍부한 ablation |
| **Practical impact** | ★★★☆☆ — 성능은 SOTA이지만 10B+ 컴포넌트 파이프라인은 deployment 부담 |
| **Writing quality** | ★★★★☆ — 두 main contribution이 잘 분리되어 있으나 unified name 부재 |

**강점**: LIBERO 98.6%, LIBERO-Plus 85.7% zero-shot, RoboTwin 2.0 85%+로 다중 벤치마크 SOTA. Multi-view synthesis를 latent space에서 처리하는 효율성과, AML의 +7.8%p ablation gain은 강력한 contribution. **약점**: 6B view synthesizer + 4B VLM의 무거운 stack, robot perturbation에서 60%로 떨어지는 일반화, 그리고 명확한 acronym 부재.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 왜 명시적 multi-view input 대신 latent synthesis인가? | 실제 multi-view rig은 데이터 수집 비용 큼. LongCat은 사전학습된 view synthesizer라 단안 input만으로 multi-view를 무료로 얻음. |
| 2 | VGGT의 prior가 정확한 metric depth가 아닌데, geometric guidance가 신뢰할 수 있나? | 절대 metric은 아니지만 relative geometric structure(어느 view가 일관된가)는 충분. Gating은 absolute depth가 아닌 consistency check로 작동. |
| 3 | AML의 w(τ)=1/(1-τ)² 는 어디서 영감? | flow-matching에서 marginal density가 (1-τ)² 비례하는 점에서 정규화. τ→1에서 action error를 강조하는 효과. |
| 4 | a-prediction이 noise-prediction 대비 항상 우월한가? | LIBERO에서는 명백히 우월(+7.8%p). 다만 action manifold가 매우 복잡한(dexterous, multi-modal) task에서는 noise-prediction의 multi-modal capture가 유리할 수 있음. |
| 5 | RoboTwin 2.0에서 Randomized > Clean인 이유? | 노이즈가 일부 over-fit을 깨주는 regularization 효과로 추정. 그러나 표 전반에서 일관된 패턴은 아니므로 통계적 노이즈 가능성도 있음. |
| 6 | Robot perturbation 60%인데 진정한 generalization이라 할 수 있나? | Proprio 변경에 약하다는 한계 인정. Camera/light/background는 90%+로 강건. Robot일반화는 별도 train-time augmentation 필요. |
| 7 | 4 H20 GPU 27시간 training은 적절한 비용인가? | LIBERO 단일 학습으로는 합리적. 그러나 6B LongCat은 frozen이지만 inference 시 매번 호출 — deployment 비용이 학습 비용보다 큰 문제. |
| 8 | 실로봇 4 task 65%는 SOTA 대비 어떠한가? | π0.5 실로봇 평균과 비교 가능 수준(60-70%). 단, task 수와 환경이 paper-specific이라 직접 비교 어려움. |
| 9 | View 수(2개)는 충분한가? Ablation에서 view 수 sweep은? | LongCat 2-view ablation만 보고. 4/8 view 효과는 미검증. Diminishing return 가능성. |
| 10 | LongCat을 fine-tune하지 않는데 task-specific objects(LIBERO의 단순 cube/box)에서 잘 작동하는가? | LongCat은 generic image-edit 모델이라 toy-scene에서도 reasonable view 생성. 다만 매우 OOD scene(투명/거울)에서는 검증 부재. |

<!-- VERIFIED: pdf -->
