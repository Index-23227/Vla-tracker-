# CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models

> **한 줄 요약**: VLA에 시각적 Chain-of-Thought(미래 이미지 프레임 예측)를 도입하여, 행동 생성 전에 시각적 목표를 autoregressive하게 계획함으로써 real-world 17%, simulation 6% 성능 향상 달성.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 기존 VLA는 현재 observation에서 **직접 action을 생성** → temporal planning 능력 부재
- 텍스트 기반 CoT (ECoT 등)는 언어적 추론을 추가하나, **시각적 계획(visual planning)**은 없음 → "왼쪽으로 이동"보다 "이 위치로 이동"이 로봇에게 더 직접적
- World model 기반 접근(GR-1, UniPi 등)은 전체 비디오를 생성하지만, **action generation과 분리**되어 있음

### 핵심 질문
- **미래 이미지 프레임 예측이 action 생성의 품질을 직접 향상시키는가?**
- **텍스트 CoT vs 시각적 CoT: 로봇 제어에 더 유효한 reasoning 형태는?**

---

## 2. 방법론 심층 분석

### 2.1 Visual Chain-of-Thought

Autoregressive하게 미래 프레임을 "생각(think)"한 후 action 생성:

$$\underbrace{o_1, ..., o_t}_{\text{현재 관측}} \to \underbrace{\hat{o}_{t+1}, ..., \hat{o}_{t+k}}_{\text{시각적 CoT (미래 예측)}} \to \underbrace{a_t, ..., a_{t+H}}_{\text{action chunk}}$$

핵심: 미래 프레임은 **action 생성의 conditioning**으로 작용.

> ❓ **예상 질문**: 미래 프레임 생성의 품질이 action 품질을 직접 결정하는데, 생성된 이미지가 부정확하면?
> **답변**: 핵심 우려. 저자들은 생성된 이미지의 정확한 pixel-level fidelity가 중요한 것이 아니라, **대략적인 spatial goal**을 제공하는 것이 핵심이라 주장. 그러나 이 가설의 체계적 검증(예: 다양한 수준의 이미지 corruption 하에서의 robustness)은 부족.

### 2.2 모델 아키텍처

7B VLM 기반, 단일 autoregressive decoder에서 모든 modality 처리:

$$P(\hat{o}_{t+1:t+k}, a_{t:t+H} | o_{1:t}, l) = \prod_i P(\hat{o}_{t+i} | o_{1:t}, \hat{o}_{t+1:t+i-1}, l) \cdot P(a | o_{1:t}, \hat{o}_{t+1:t+k}, l)$$

- Image tokenizer (VQ-VAE)로 이미지를 discrete token化
- Action tokenizer로 action도 discrete token化
- 동일 vocabulary에서 image token과 action token을 sequential하게 생성

> ❓ **예상 질문**: 이미지 토큰화의 reconstruction 품질이 병목이 되지 않는가?
> **답변**: VQ-VAE의 codebook 크기와 latent resolution에 따라 reconstruction 품질이 결정됨. 저해상도 tokenization은 fine detail 손실을 야기하며, 이것이 precision-demanding task에서 문제될 수 있음.

### 2.3 학습 전략

- **Stage 1**: VLM + image generation (next frame prediction)
- **Stage 2**: Action prediction 추가, image와 action을 joint training
- Image generation loss와 action prediction loss를 weighted sum

$$\mathcal{L} = \lambda_{\text{img}} \mathcal{L}_{\text{img}} + \lambda_{\text{act}} \mathcal{L}_{\text{act}}$$

> ❓ **예상 질문**: Loss weight $\lambda$ 간 trade-off는? Image generation에 너무 많은 capacity가 할당되면 action 성능이 하락하지 않나?
> **답변**: 이 trade-off는 ablation에서 부분적으로 다뤄지나, optimal $\lambda$ 탐색이 체계적이지 않음. Image generation이 auxiliary task인지 primary task인지의 역할 구분이 모호.

---

## 3. 데이터 전략

- 학습 데이터에 **미래 프레임 annotation이 자동으로 포함** (trajectory의 다음 timestep 이미지)
- 추가 annotation 불필요 → 기존 robot dataset을 그대로 활용 가능
- 이것이 텍스트 CoT (ECoT처럼 수동 CoT annotation 필요) 대비 큰 장점

---

## 4. 실험 결과 심층 분석

### Simulation (LIBERO)

| 모델 | LIBERO Avg (%) |
|------|---------------|
| OpenVLA | 76.5 |
| OpenVLA + ECoT (text) | 79.1 |
| **CoT-VLA** | **82.3 (+6%)** |

### Real Robot

| 모델 | 10 Tasks SR (%) |
|------|----------------|
| OpenVLA | 52.3 |
| ECoT | 58.1 |
| **CoT-VLA** | **69.5 (+17%)** |

- **Real-world에서의 개선이 simulation보다 크다** → visual planning이 real-world의 다양성에 robust한 계획을 가능하게 함
- 텍스트 CoT (ECoT) 대비 시각적 CoT가 일관적으로 우수

---

## 5. Ablation 분석

| 변형 | Real Robot SR (%) |
|------|-----------------|
| Full CoT-VLA | 69.5 |
| Visual CoT 제거 (direct action) | 52.3 |
| Text CoT로 대체 | 58.1 |
| 1 frame 예측 (k=1) | 62.4 |
| 4 frames 예측 (k=4) | 69.5 |
| 8 frames 예측 (k=8) | 68.2 |

- **k=4가 sweet spot**: 너무 적으면 planning 부족, 너무 많으면 hallucination 누적
- Visual CoT의 기여가 명확 (+17%p vs no CoT)

---

## 6. 관련 연구 비교

| 모델 | CoT Type | Image Generation | End-to-End | Latency |
|------|----------|-----------------|------------|---------|
| ECoT | Text | ✗ | ✓ | Low |
| SuSIE | Image (subgoal) | ✓ (diffusion) | ✗ (2-stage) | High |
| UniPi | Video | ✓ (diffusion) | ✗ | Very High |
| GR-1 | Video | ✓ (autoregressive) | ✓ | High |
| **CoT-VLA** | **Visual (frames)** | **✓ (VQ-VAE, autoregressive)** | **✓** | **Medium** |

핵심 차이: SuSIE/UniPi는 별도 diffusion model 필요, CoT-VLA는 단일 모델에서 통합.

---

## 7. 한계 및 미해결 문제

### 방법론적 미비점
1. **추론 latency 증가**: k개 이미지 토큰 생성 → action 생성 전 추가 latency. k=4에서 ~200ms 추가 overhead 예상
2. **Image tokenizer 품질**: VQ-VAE reconstruction 품질이 spatial precision에 직접 영향. 고주파 detail 손실 가능
3. **Visual hallucination**: 모델이 틀린 미래를 "상상"하면 action도 틀릴 수 있음. Hallucination 빈도와 그 영향에 대한 정량 분석 부재
4. **Training cost**: 이미지 생성을 추가로 학습하므로 pure action model 대비 학습 비용 증가

### Attribution 문제
- 성능 향상이 **visual planning** 자체에서 오는지, **multi-task learning regularization** (이미지 생성이 부가적 학습 신호) 때문인지 분리 어려움
- Random image를 conditioning으로 넣었을 때의 성능으로 visual planning의 실질적 기여를 분리해야 하나, 이 실험 부재

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Visual CoT의 VLA 적용이 직관적이고 참신 |
| **Technical depth** | ★★★☆☆ — 아이디어는 간결하나 심층 분석 부족 |
| **Experimental rigor** | ★★★★☆ — Real + Sim, 의미 있는 ablation |
| **Practical impact** | ★★★★☆ — 추가 annotation 없이 적용 가능 |
| **Writing quality** | ★★★★☆ — 명확한 story |

**강점**: 시각적 계획이라는 직관적 아이디어, 추가 annotation 불필요, real-world에서 큰 개선. **약점**: Latency overhead, visual hallucination의 robustness 분석 부족.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Latent space에서 CoT를 하면 (pixel이 아니라) latency를 줄일 수 있지 않나? | 맞음. Latent visual CoT는 유망한 대안. 다만 interpretability 상실. FLARE 등이 이 방향 |
| 2 | Visual CoT가 도움 안 되는 태스크는? | 시각적 변화가 작은 태스크(in-place rotation 등)에서는 미래 프레임이 현재와 유사 → planning 이점 미미 |
| 3 | Random noise를 visual CoT 대신 넣으면 성능이 얼마나 떨어지는가? | 이 실험이 visual planning의 실질적 기여를 검증하는 핵심 ablation이나 미수행 |
| 4 | Text CoT와 Visual CoT를 결합하면? | 상보적 가능성 있으나 latency 추가 증가. 결합 시 diminishing returns 여부 미검증 |
| 5 | 예측된 미래 이미지를 사람이 봤을 때 의미 있는가? | Qualitative examples는 제공하나, systematic evaluation 부재. 사람 기준 "의미 있음"과 action 품질 간 상관관계 미분석 |
