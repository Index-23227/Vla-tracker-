# CogACT: A Foundational VLA Model for Synergizing Cognition and Action in Robotic Manipulation

> **한 줄 요약**: DINOv2+SigLIP vision encoder와 LLaMA-2 backbone으로 "cognition feature"를 추출하고, 별도의 Diffusion Transformer (DiT) action module로 action을 생성하는 componentized VLA로, RT-2-X (55B) 대비 18%p 우위를 7.6B로 달성하고 action head scaling law를 실증.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- **Monolithic VLA** (RT-2, OpenVLA): VLM이 action token을 직접 생성 → action space의 continuous 특성을 discrete token으로 강제 변환, 정밀도 손실
- Action head 설계에 대한 체계적 탐색 부재: 대부분 단순 token prediction이나 MLP regression에 의존
- VLM의 vision-language 능력과 fine-grained action generation 능력이 하나의 decoder에서 **상충(conflict)** 가능

### 핵심 질문
- **VLM의 인지 출력을 specialized action module의 conditioning으로 사용하면, action 품질이 얼마나 향상되는가?**
- **Action module의 크기를 scaling하면 성능이 예측 가능하게 향상되는가? (Scaling law)**

---

## 2. 방법론 심층 분석

### 2.1 Componentized Architecture (3-Module)

```
[DINOv2 + SigLIP] → 256 visual tokens → [LLaMA-2] → cognition feature → [DiT Action Module] → 7-DoF actions
                                             ↑
                                     Language instruction
```

- **Vision Module**: DINOv2 (spatial) + SigLIP (semantic) → 256 visual tokens
- **Cognition Module**: LLaMA-2 backbone, **learnable cognition token**이 VLM의 마지막 hidden state를 집약
- **Action Module**: Diffusion Transformer (DiT), cognition feature를 conditioning으로 받아 action chunk 생성

### 2.2 Diffusion Action Transformer

7-DoF action: (Δx, Δy, Δz, Δφ, Δθ, Δψ, gripper)

- **Action chunking**: 현재 + N 미래 step 동시 예측 (default N=15, 총 16 steps)
- **Diffusion**: MSE loss on noise prediction, 학습 시 8 diffusion steps/sample
- **Inference**: DDIM 10 steps, classifier-free guidance coefficient = 1.5

> ❓ **예상 질문**: Classifier-free guidance가 action generation에서 왜 필요한가?
> **답변**: CFG는 conditioning (cognition feature)에 대한 adherence를 강화. Action이 "instruction에 더 충실하게" 생성되도록 함. 계수 1.5는 empirically 선택되었으며, 너무 크면 diversity 상실, 너무 작으면 conditioning 효과 약화.

### 2.3 Adaptive Action Ensemble (AAE)

Temporal fusion으로 consecutive action chunk 간 discontinuity 방지:

$$\mathbf{a}_t = \sum_k w_k \cdot \hat{\mathbf{a}}_t^{(k)}, \quad w_k \propto \cos(\hat{\mathbf{a}}_t^{(k)}, \hat{\mathbf{a}}_t^{(k-1)})^{1/\alpha}$$

- $\alpha = 0.1$ (temperature parameter)
- Cosine similarity 기반 → 이전 예측과 유사한 현재 예측에 높은 가중치
- Standard temporal ensemble과 action chunking 모두를 능가

> ❓ **예상 질문**: AAE에서 cross-modal action blending이란?
> **답변**: 서로 다른 intention의 action chunk가 평균화되면 무의미한 action이 나올 수 있음. Cosine similarity 기반 weighting이 "같은 방향의 예측만" 결합하여 이를 방지.

---

## 3. 데이터 전략

### Pre-training

| 항목 | 값 |
|------|-----|
| Dataset | Open X-Embodiment (25 datasets) |
| Frames | 22.5M |
| Trajectories | ~400K |
| Batch size | 256 |
| LR | 2e-5 (constant) |
| Iterations | 135K |
| GPU | 16× NVIDIA A100, ~5 days |

### Fine-tuning (Real Robot)

| 로봇 | Demonstrations |
|------|--------------|
| Realman | 391 (48 Pick + 67 Stack + 79 Place + 197 others) |
| Franka | 400 (100/task × 4 tasks) |

---

## 4. 실험 결과 심층 분석

### SIMPLER Benchmark (Simulation)

| 설정 | OpenVLA-7B | RT-2-X (55B) | **CogACT-7.6B** |
|------|-----------|-------------|-----------------|
| Google Robot Visual Matching | ~40% | ~57% | **74.8% (+35%p vs OpenVLA)** |
| Google Robot Variant Agg. | - | - | **61.3%** |
| WidowX | - | - | **51.3%** |

- RT-2-X (55B) 대비 **18%p 절대 우위**, **7x 적은 파라미터**

### Real Robot — Realman

| 태스크 | OpenVLA | **CogACT** |
|-------|---------|-----------|
| Pick | 11.7% | **70.8%** |
| Stack | 23.2% | **82.3%** |
| Place | 1.2% | **60.4%** |
| **Overall** | **12.1%** | **71.2% (+59.1%p)** |

### Real Robot — Franka

4 tasks (open/close oven, pick bowl/brush): **평균 61.4%**

### Generalization

| 조건 | SR (%) |
|------|--------|
| Unseen colors | **87.5** |
| Unseen shapes | **81.3** |
| Unseen categories | 25.0 |

- 색상/형상 변화에 강건하나, **완전히 새로운 물체 카테고리**에서는 약함 (25%)

---

## 5. Ablation 분석

### Action Head Architecture Comparison (Table 7)

| Action Head | Params | Avg SR (%) |
|------------|--------|-----------|
| MLP-7Layer | 89M | 52.5 |
| DiT-Small | 13M | 58.5 |
| DiT-Base | 89M | 62.5 |
| DiT-Large | **308M** | **64.8** |

**핵심 발견**: "성능과 log(model size) 사이에 근사적 선형 관계" → **Action head의 scaling law 최초 실증**

> ❓ **예상 질문**: MLP 89M이 DiT-Small 13M보다 파라미터가 많은데 성능이 낮은 이유는?
> **답변**: DiT의 attention 메커니즘이 temporal correlation과 cross-modal conditioning을 더 효과적으로 처리. MLP는 각 action dimension을 독립적으로 처리하는 경향이 있어 coordination이 약함.

### Multi-step Prediction (Table 8)

| Future Steps (N) | Avg SR (%) |
|------------------|-----------|
| 0 (current only) | 42.8 |
| 3 | 55.5 |
| **15** | **62.5** |
| 31 | 51.2 (degradation) |

- **N=15가 sweet spot**: 너무 적으면 temporal consistency 부족, 너무 많으면 prediction accuracy 하락

### Action Ensemble Comparison (Table 9)

| 방법 | Avg SR (%) |
|------|-----------|
| Action Chunking (FIFO) | 50.7 |
| Temporal Ensemble (uniform) | 58.9 |
| **Adaptive Ensemble (cosine)** | **62.5** |

- AAE가 standard temporal ensemble 대비 **+3.6%p**

---

## 6. 관련 연구 비교

| 모델 | VLM | Action Head | Params | Cross-Embodiment | Avg SR (SIMPLER) |
|------|-----|-------------|--------|-----------------|-----------------|
| RT-2 | PaLI-X | Token pred | 55B | ✗ | ~57% |
| OpenVLA | Prismatic | Token pred | 7B | ✓ | ~40% |
| Octo | - | Diffusion | 93M | ✓ | - |
| **CogACT** | **LLaMA-2 + DINOv2/SigLIP** | **DiT (308M)** | **7.6B** | **✓** | **~63%** |

---

## 7. 한계 및 미해결 문제

### 방법론적 미비점
1. **Unseen category 25%**: 완전히 새로운 물체에서 일반화가 약함 → VLM의 open-vocabulary 능력이 action으로 전이되지 않는 한계
2. **추론 latency**: DDIM 10 steps의 DiT inference + VLM forward → total latency 미보고. 10Hz 제어에는 충분할 것으로 추정되나 정확한 수치 부재
3. **Cognition token bottleneck**: VLM의 모든 정보가 하나의 learnable token으로 압축 → information bottleneck 가능. 이 token의 dimension이 충분한지 분석 부재
4. **LIBERO/CALVIN 미평가**: SIMPLER + real robot 위주. 표준 벤치마크에서의 직접 비교 부족

### Attribution 문제
- CogACT의 성능이 **DiT action head** 때문인지, **pre-training data (22.5M frames)** 때문인지, **dual visual encoder** 때문인지 완전 분리 어려움
- OpenVLA와 동일한 학습 데이터·epoch에서의 fair comparison이 더 설득력 있을 것

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Componentized 설계 + 체계적 action head scaling 분석 |
| **Technical depth** | ★★★★★ — DiT scaling law, AAE, CFG 등 세심한 설계 |
| **Experimental rigor** | ★★★★☆ — SIMPLER + 2 real robots + ablation 풍부 |
| **Practical impact** | ★★★★★ — 오픈소스, 즉시 적용 가능 |
| **Writing quality** | ★★★★☆ — 체계적 presentation |

**강점**: Action head scaling law의 최초 실증이 매우 가치 있음. "VLM을 크게 키우는 대신 action head를 키우라"는 practical insight. RT-2-X를 7x 적은 파라미터로 능가. **약점**: 표준 벤치마크(LIBERO) 결과 부재, unseen category 일반화 약함, latency 미보고.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Action head 308M을 1B로 키우면 성능이 더 오르는가? | Scaling law가 log-linear이므로 diminishing returns 예상되나 미검증. DexVLA가 3B expert로 이 방향 탐구 |
| 2 | LIBERO에서 성능은? | 미보고. OpenVLA-OFT가 LIBERO 97.1%인데, CogACT가 이를 능가하는지 불명확 |
| 3 | Cognition token 1개 대신 여러 개를 쓰면? | Information bottleneck 완화 가능하나 action module의 cross-attention 비용 증가. Trade-off 미탐구 |
| 4 | 왜 Flow matching이 아닌 DDPM/DDIM인가? | 발표 시점 기준 DDPM이 주류. 이후 pi0가 flow matching 채택. 직접 비교 없으나 flow가 더 빠른 추론 가능 |
| 5 | OpenVLA 대비 +59.1%p (real robot)가 공정한가? | 학습 데이터(22.5M vs OpenVLA의 970K?)와 fine-tuning 데이터가 다를 수 있음. 동일 조건 비교 필요 |
| 6 | CFG coefficient 1.5의 sensitivity는? | Ablation 미제공. 1.0 (no guidance)과 2.0에서의 성능 비교가 있으면 더 설득력 있을 것 |

<!-- VERIFIED: pdf -->
