# CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models

> **한 줄 요약**: VILA-U 기반 7B VLA에 시각적 CoT(미래 subgoal image 예측)를 autoregressive하게 통합하고, hybrid attention + action chunking으로 LIBERO 81.1%, real-world 17% 향상 달성. 단, 이미지 생성으로 **7배 추론 slowdown**.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 기존 VLA는 현재 observation에서 **직접 action을 생성** → temporal planning 능력 부재
- 텍스트 기반 CoT (ECoT 등)는 언어적 추론을 추가하나, **시각적 계획(visual planning)**은 없음
- World model 기반 접근(GR-1, UniPi)은 전체 비디오를 생성하지만 **action generation과 분리**

### 핵심 질문
- **미래 이미지 프레임 예측(visual CoT)이 action 생성의 품질을 직접 향상시키는가?**
- **텍스트 CoT vs 시각적 CoT: 로봇 제어에 더 유효한 reasoning 형태는?**

---

## 2. 방법론 심층 분석

### 2.1 아키텍처

**VILA-U 기반 7B VLA**:
- 이미지를 256×256 → **16×16×4 = 1024 visual tokens** (residual quantization depth=4)
- Residual depth transformer로 D개 residual token 예측

### 2.2 Visual Chain-of-Thought

2단계 추론:

**Step 1**: Subgoal image $\hat{s}^{(t+n)}$ 생성 (autoregressive token generation)
**Step 2**: Action chunk 생성 (subgoal conditioned)

$$[o_1, ..., o_t, \text{instruction}] \xrightarrow{\text{causal attn}} \hat{s}^{(t+n)} \xrightarrow{\text{full attn}} [a_t, ..., a_{t+9}]$$

- **Causal attention**: Image/text token 생성 시
- **Full (bidirectional) attention**: Action token 생성 시 — 미래 이미지와 현재 관측 모두 참조 가능

> ❓ **예상 질문**: 이 hybrid attention 설계의 핵심 이유는?
> **답변**: Image 생성은 autoregressive (순차적)이므로 causal이 자연스러움. Action은 모든 context를 동시에 참조해야 최적이므로 bidirectional. 이 설계가 단순 causal-only 대비 성능 향상 (ablation 확인).

### 2.3 Action Tokenization

- 7-DoF action, 각 dimension **256 bins**으로 독립 이산화
- Bin width: 학습 데이터의 **1st-99th percentile** 기준
- **Action chunk size: 10 timesteps**
- Total: 7 tokens × 10 steps = 70 action tokens per prediction

### 2.4 Subgoal Prediction Horizon

$n$을 데이터셋별 범위 $[n_l, n_u]$에서 uniform 샘플링:
- 짧은 horizon: 단기 계획만
- 긴 horizon: 장기 계획이지만 prediction 어려움

---

## 3. 데이터 전략

### Pre-training 데이터

| 소스 | 유형 | 규모 |
|------|------|------|
| Open X-Embodiment (7 sources) | Robot demonstrations | ~수십만 trajectories |
| EPIC-KITCHENS-100 | Action-less video | Human kitchen |
| Something-Something V2 | Action-less video | Object manipulation |
| Bridge dataset | Robot + language | 45K trajectories |

### Fine-tuning

| 벤치마크 | 데이터 |
|---------|--------|
| LIBERO | 50 demos/task × 10 tasks × 4 suites |
| Franka-Tabletop | 10-150 demos/task |

### 학습 설정

| 단계 | LR | Batch | Epochs | Compute |
|------|-----|-------|--------|---------|
| **Pre-training** | 1e-4 (cosine) | **2048** | 10 | **11K A100 GPU-hours** (96 GPUs) |
| **Fine-tuning** | 1e-5 (constant) | - | 150 | 10-24h (single A100 node) |

---

## 4. 실험 결과 심층 분석

### LIBERO Benchmark (Table 1)

| 모델 | Spatial | Object | Goal | Long | **Avg** |
|------|---------|--------|------|------|---------|
| Diffusion Policy | 78.3±1.1 | 92.5±0.7 | 68.3±1.2 | 50.5±1.3 | 72.4±0.7 |
| Octo (FT) | 78.9±1.0 | 85.7±0.9 | 84.6±0.9 | 51.1±1.3 | 75.1±0.6 |
| OpenVLA (FT) | 84.7±0.9 | 88.4±0.8 | 79.2±1.0 | 53.7±1.3 | 76.5±0.6 |
| **CoT-VLA-7B** | **87.5±1.4** | **91.6±0.5** | **87.6±0.6** | **69.0±0.8** | **81.1±0.6** |

- **Avg 81.1%**: OpenVLA 대비 **+4.6%p**, LIBERO-Long에서 **+15.3%p**
- **500 trials/suite, 3 seeds**로 통계적 유의성 확보

### Bridge-V2 (Table 2, 10 trials/category)

| 카테고리 | SUSIE | Octo | OpenVLA | **CoT-VLA** |
|---------|-------|------|---------|------------|
| Visual | 30 | 35 | **75** | 65 |
| Motion | 10 | 10 | 45 | **60** |
| Semantic | 20 | 0 | 40 | **50** |
| Language | 40 | 40 | **75** | 70 |

- **Motion, Semantic에서 OpenVLA 능가**, Visual/Language에서 약간 열위
- Visual/Language에서의 하락은 "action chunking에 의한 grasping failure"로 설명

### Pre-training Ablation (Franka-Tabletop)

| 설정 | SR (%) |
|------|--------|
| Direct fine-tune (no pretrain) | 53.7 |
| **With OpenX pretraining** | **78.8 (+46.7% relative)** |

### Out-of-Distribution (Table 3)

| 조건 | Task 1 | Task 2 |
|------|--------|--------|
| Generated goal images | 20% | 0% |
| Ground-truth goal images | 60% | 40% |

- **Generated vs GT goal: 40%p gap** → visual CoT의 OOD 한계가 명확

---

## 5. Ablation 분석 (Figure 6)

LIBERO-Spatial & LIBERO-Goal에서의 점진적 구성요소 추가:

| 구성 | 효과 |
|------|------|
| Baseline VLA (no CoT, no chunk) | 기준 |
| + Action chunking | 소폭 향상 |
| + Hybrid attention | 추가 향상 |
| **+ Visual CoT (full)** | **최고 성능** |

---

## 6. 추론 비용

> ⚠️ **핵심 한계**: 이미지 생성으로 인해 **평균 7배 추론 slowdown**

- 256 image tokens을 autoregressive하게 생성해야 action 예측 가능
- Action chunking (10 steps)이 일부 상쇄: 10 step마다 1회 image generation
- 그래도 chunk당 total latency = image gen + action gen → 실시간 제어에 상당한 부담

---

## 7. 관련 연구 비교

| 모델 | CoT Type | End-to-End | Image Quality | Latency | LIBERO Avg |
|------|----------|-----------|--------------|---------|-----------|
| ECoT | Text 4-layer | ✓ | N/A | Low | 79.1 (추정) |
| SuSIE | Image (diffusion) | ✗ (2-stage) | High | Very High | ~3.42 (CALVIN) |
| GR-1 | Video (AR) | ✓ | Medium | High | N/A |
| **CoT-VLA** | **Image (VQ-VAE, AR)** | **✓** | **Medium** | **Medium-High (7x)** | **81.1** |

---

## 8. 한계 및 미해결 문제

### 방법론적 미비점 (저자 명시 포함)
1. **7x 추론 slowdown**: 저자 명시. 256 image tokens의 autoregressive 생성이 병목. "Action chunking이 부분적으로 완화하나 근본적 해결 아님"
2. **이미지 품질**: 저자 명시 — "autoregressive image generation produces lower visual quality compared to state-of-the-art diffusion-based models." VQ-VAE의 reconstruction 한계
3. **Action chunking discontinuity**: 저자 명시 — "discontinuous actions between chunks and lacks high-frequency feedback"
4. **OOD 한계**: Generated goal image vs GT goal 사이 40%p gap → visual reasoning이 새로운 task에서 unreliable
5. **11K A100 GPU-hours**: Pre-training 비용이 상당

### Attribution 문제
- 81.1%의 기여가 **visual CoT** 자체에서 오는지, **VILA-U backbone의 강력한 image generation** 능력에서 오는지, **hybrid attention** 설계에서 오는지 분리 불완전
- Random image를 CoT로 주입한 실험이 없어 visual planning의 실질적 기여 정량화 불가

---

## 9. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Visual CoT의 VLA 통합이 직관적이고 참신 |
| **Technical depth** | ★★★★☆ — Hybrid attention, residual VQ, action chunking 등 세심 |
| **Experimental rigor** | ★★★★★ — 3 benchmarks, 500 trials/suite, ablation 포괄적 |
| **Practical impact** | ★★★☆☆ — 7x slowdown이 실용성 크게 제한 |
| **Writing quality** | ★★★★☆ — 명확한 story |

**강점**: Visual CoT가 특히 LIBERO-Long (+15.3%p)에서 큰 향상을 보여 temporal planning의 가치를 입증. Pre-training ablation (+46.7%)이 pre-training의 중요성을 강력히 보여줌. **약점**: 7x slowdown, OOD 한계 (40%p gap), 이미지 품질.

---

## 10. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Latent space에서 CoT를 하면 (FLARE처럼) 7x slowdown을 줄일 수 있지 않나? | 맞음. 가장 유망한 개선 방향. 256 pixel tokens 대신 16-64 latent tokens이면 4-16x 가속 가능 |
| 2 | 7x slowdown의 실제 wall-clock은? | Action chunk 10 steps 기준, 이미지 생성이 ~2-3초 추정 (VQ-VAE 256 tokens). 0.3Hz 제어 |
| 3 | Random image를 CoT 대신 넣으면? | 이 실험이 visual planning의 실질적 기여를 검증하는 핵심이나 **미수행** |
| 4 | OOD에서 generated goal (20%) vs GT (60%)의 40%p gap은 치명적이지 않은가? | 맞음. Visual CoT의 OOD 일반화가 핵심 과제. VLM의 world knowledge가 부족하면 잘못된 미래를 상상 |
| 5 | ECoT (text) + CoT-VLA (visual) 결합이 가능한가? | 이론적으로 가능하나 latency 추가 증가. Text CoT는 semantic, visual CoT는 spatial에 강점 → 상보적 |
| 6 | 11K A100 GPU-hours는 이 수준의 성능 향상(+4.6%p)에 정당화되는가? | OpenVLA (76.5%) → CoT-VLA (81.1%)의 +4.6%p가 11K GPU-hrs의 ROI를 정당화하려면, 실용적 임팩트가 더 명확해야 함 |

<!-- VERIFIED: pdf -->
