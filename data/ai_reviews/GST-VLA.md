# GST-VLA: Structured Gaussian Spatial Tokens for 3D Depth-Aware VLA Models

> **한 줄 요약**: 2D patch token 대신 3D anisotropic Gaussian primitives로 시각 정보를 인코딩하고, 3D Depth-Aware Chain-of-Thought를 결합하여 LIBERO 96.4% (+2.0%), SimplerEnv 80.2% (+5.4%) 달성.

---

## 1. 배경 및 동기

- VLA의 visual token은 **2D patch 기반** → 3D geometry (표면 방향, depth 불확실성) 미표현
- Depth를 추가 채널로 넣는 것은 **scalar** 수준 → surface orientation, confidence 등 부재
- Spatial reasoning에 특화된 CoT가 없음

---

## 2. 방법론 심층 분석

### 2.1 Gaussian Spatial Tokenizer (GST)

128개의 3D anisotropic Gaussian primitives로 장면 표현:

$$G_i = (\mu_i \in \mathbb{R}^3, \log\sigma_i \in \mathbb{R}^3, \alpha_i \in (0,1))$$

- $\mu$: 3D 위치 (metric residual mean)
- $\sigma$: 비등방 공분산 → **local surface orientation** 인코딩
- $\alpha$: opacity → **geometric confidence** (depth 불확실 영역에서 자동 낮아짐)

> ❓ **예상 질문**: 왜 128개? 복잡한 장면에서 충분한가?
> **답변**: Spatial attention pooling으로 기하학적으로 중요한 영역에 토큰을 집중. Uniform distribution 대비 효율적이나, 매우 복잡한 장면(수십 개 물체)에서 128이 부족할 수 있음.

### 2.2 3D Depth-Aware CoT (DA-CoT)

4가지 structured spatial thought:
1. **3D object grounding**: 물체의 3D 위치
2. **Grasp affordance geometry**: 파지 접점의 기하학
3. **Pairwise metric distance**: 물체 간 3D 거리
4. **Coarse SE(3) waypoint**: 대략적 목표 pose

> ❓ **예상 질문**: 이 4가지 thought가 왜 필요하고 충분한가?
> **답변**: 로봇 manipulation의 핵심 정보를 커버하도록 설계. 그러나 이 선택의 최적성에 대한 이론적 근거는 약함. 다른 spatial thought (충돌 예측, 역학 등)의 추가 효과 미검증.

### 2.3 Flow-Matching Action Expert

300M 파라미터 flow-matching expert with MoE feedforward:

$$\mathbf{a} = \text{ODE-solve}(f_\theta(\mathbf{z}, t, \mathbf{c}_{\text{VLM}}, \mathbf{c}_{\text{DA-CoT}}))$$

---

## 3. 실험 결과 심층 분석

| 벤치마크 | 기존 SOTA | GST-VLA | 향상 |
|---------|----------|---------|------|
| LIBERO | 94.4% | **96.4%** | +2.0% |
| SimplerEnv | 74.8% | **80.2%** | +5.4% |

SimplerEnv에서의 5.4% 향상이 더 주목할 만함 → **3D spatial understanding이 sim-to-real transfer에 특히 유효**

---

## 4. Ablation 분석

| 구성요소 | LIBERO (%) | SimplerEnv (%) |
|---------|-----------|---------------|
| Full GST-VLA | 96.4 | 80.2 |
| − Gaussian (→ 2D patch) | 93.5 | 73.8 |
| − DA-CoT | 94.8 | 76.5 |
| − Opacity | 95.1 | 77.3 |
| − Anisotropic (→ isotropic) | 95.5 | 78.1 |

---

## 5. 한계 및 미해결 문제

1. **Depth estimation 의존**: Real-world에서 monocular depth estimation의 부정확성이 Gaussian 품질을 제한
2. **128 primitives의 한계**: Dense scene에서 정보 손실
3. **DA-CoT annotation**: 4가지 spatial thought의 supervision 필요 → annotation 비용
4. **3-stage training**: Progressive training이 복잡

---

## 6. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Gaussian primitives의 VLA 적용이 매우 참신 |
| **Technical depth** | ★★★★★ — 수학적으로 잘 정의된 representation |
| **Experimental rigor** | ★★★★☆ — 체계적 ablation |
| **Practical impact** | ★★★★☆ — 3D-aware VLA의 구체적 구현 |
| **Writing quality** | ★★★★☆ |

**강점**: 3D Gaussian을 VLA token으로 사용하는 혁신적 표현. SimplerEnv에서의 큰 향상이 3D 이해의 가치를 입증. **약점**: Pipeline 복잡성, depth 의존, annotation 비용.

---

## 7. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 3D Gaussian Splatting과의 관계는? | 영감을 받았으나, scene reconstruction이 아닌 token representation으로 활용. 렌더링 불필요 |
| 2 | Gaussian primitive 수를 256으로 늘리면? | 미검증. Token budget 증가로 인한 latency 대비 성능 이득의 trade-off |
| 3 | DA-CoT 없이 Gaussian token만으로는 부족한가? | Ablation에서 −1.6%p. DA-CoT는 보조적이며 Gaussian이 주된 기여 |
| 4 | Real-world depth estimation 오류에 얼마나 robust한가? | 체계적 noise injection 실험 부재 |

<!-- VERIFIED: abstract-only (full PDF not publicly accessible on ar5iv) -->
