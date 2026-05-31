# QuoVLA: Quotient Space for Vision-Language-Action Models

> **한 줄 요약**: pi0.5 backbone에 **8-bit symmetric uniform quantization**을 prefix token (VLM 출력)에 적용한 1-layer Transformer 모듈을 삽입, **dual-branch (quantized + raw stop-gradient reference)** 구조와 **relative temporal-complexity regularization**으로 학습 — *"VLM 표현은 action-sufficient하지만 overcomplete하다"*는 Quotient Theory를 구현, LIBERO 평균 **99.6%** (pi0.5의 96.9% 대비 +2.7), real-robot 4-task 평균 **88.0%** (pi0.5의 64.5% 대비 +23.5).

---

## 1. 배경 및 동기

### 기존 연구의 통념
- 기존 VLA 연구는 *"pretrained VLM 표현이 action에 부족하다 (action-insufficient)"*고 가정 → 추가 학습으로 action 정보를 *주입*해야 함
- 예: pi0의 dual-stream, RDT의 DiT, OpenVLA의 discrete action tokens — 모두 VLM 위에 더 많은 학습 모듈을 *덧붙임*

### 핵심 가설 전환
**Quotient Theory**: pretrained VLM 표현은 *이미 action에 충분하다 (action-sufficient)*. 그러나 **overcomplete**하다 — 즉, 동일한 optimal action을 만드는 서로 다른 prompt-level variation들이 latent space에서 redundant하게 표현됨.

- 정확한 표현: "pretrained VLM latents are action-sufficient yet overcomplete"
- 해결책: **압축 (compression)**으로 prompt-irrelevant variation을 제거 → quotient space로 mapping

📌 [Figure 1 삽입] — pi0.5 baseline vs QuoVLA: VLM과 action expert 사이에 quantized Transformer prefix 삽입

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

```
[Image + Text] → PaliGemma-2.6B VLM → [Prefix Tokens]
                                          ↓
                            ┌─────────────┴─────────────┐
                            ↓                            ↓
            Quantized Transformer Block         Raw prefix (stop-grad)
            (8-bit symm. uniform quant)         ↓ (reference branch)
            ↓                                   ↓
            Quantized prefix                    Reference prefix
            ↓
            π0.5 Action Expert (300M, flow matching)
            ↓
            Action chunks (50-step)
```

### 2.2 8-bit Symmetric Uniform Quantization

수식:
$$Q_b(z) = s \cdot \text{clip}\left(\text{round}\left(\frac{z}{s}\right), -q_{\max}, q_{\max}\right)$$

- $b = 8$ (8-bit), $q_{\max} = 2^{b-1} - 1 = 127$
- 위치: **MHA와 MLP sublayer 사이**, 1-layer Transformer block 내부
- **Adaptive Straight-Through Estimator (STE)**: backward pass에서 gradient를 quantizer 통과시킴 — adaptive하게 scale 조정

> ❓ **예상 질문**: 왜 prefix token에만 quantization을 적용하나? Action expert 자체는 왜 안 하나?
> **답변**: Quotient theory는 *VLM의 prompt-induced variation* 제거가 핵심. Action expert는 이미 task-specific으로 학습되므로 redundancy가 적음. Prefix token이 VLM과 action 사이의 *bridge*이므로 여기서 quotient를 형성하는 것이 합리적.

### 2.3 Dual-Branch Design

- **Main branch**: quantized prefix → action expert (학습 대상)
- **Reference branch**: raw prefix (stop-gradient) → 같은 action expert (gradient 차단)
- 두 branch의 출력 trajectory의 *temporal complexity*를 비교

### 2.4 Relative Temporal-Complexity Regularization

$$L_{tc} = [C(N(v^q)) - C(N(v^r))]_+$$

- $v^q$: quantized branch trajectory, $v^r$: reference branch trajectory
- $C(v)$: temporal complexity — first-order + second-order differences
- $N(\cdot)$: 정규화
- $[\cdot]_+$: ReLU (raw reference보다 quantized가 *덜* 매끄러우면 penalize)

전체 loss:
$$L = L_q + \lambda_{tc} \cdot L_{tc}$$

> ❓ **예상 질문**: Quantization이 trajectory smoothness를 해치는데 그것을 다시 regularize하면 quantization의 이점을 상쇄하지 않나?
> **답변**: Quotient theory의 핵심은 *prompt-irrelevant variation 제거*. Temporal smoothness는 *task-relevant signal*에 가까움. 두 가지가 다른 축이므로 동시 만족 가능. 결과적으로 L_tc는 quantization의 noise를 task-relevant 방향으로 정렬.

---

## 3. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Backbone | pi0.5 (PaliGemma-2.6B + 300M flow-matching expert) |
| Total params | ~2.9B |
| Hardware | 8× A800 80GB |
| Epochs | ~10 (LIBERO), 5 (RoboTwin 2.0) |
| Optimizer | AdamW (lr 2.5e-5, wd 0.01) |
| Image res | 224×224 |
| Action chunk | 50 step (32-dim padding) |
| Real-robot | 100 samples, 5000 steps per task, single A800 |

---

## 4. 실험 결과 심층 분석

### 4.1 LIBERO (Table 1a)

| 모델 | Spatial | Object | Goal | Long | **Avg** |
|------|---------|--------|------|------|---------|
| OpenVLA | - | - | - | - | 76.5 |
| pi0.5 | - | - | - | - | 96.9 |
| **QuoVLA** | **99.8** | **99.9** | **100.0** | **98.7** | **99.6** |

- **LIBERO Goal 100%** — 사실상 ceiling 도달
- LIBERO-Long 98.7%는 가장 어려운 long-horizon suite에서도 ceiling 근접

### 4.2 LIBERO-PRO Robustness (Table 1b, Spatial 발췌)

| 변형 | Original | Object change | Position | Semantic | Task change |
|------|----------|--------------|----------|----------|------------|
| Spatial | 100% | 99% | **40%** | 99% | **10%** |
| Object | 100% | 87% | **26%** | 100% | **26%** |

- **Position perturbation에 매우 취약** (Spatial 40%, Object 26%)
- **Task change에 더 취약** (Spatial 10%, Object 26%) — 새 task로의 zero-shot transfer는 여전히 hard
- "Quotient"가 prompt variation을 제거하더라도 *physical state* 변화에는 robustness 한계

### 4.3 LIBERO-Plus (Table 2)

| 평가 | Avg |
|------|-----|
| Zero-shot transfer | **90.3%** |
| Supervised fine-tuning | **90.9%** |

- Zero-shot과 SFT의 격차가 매우 작음 (0.6%p) — 압축된 표현이 transferable함을 시사

### 4.4 RoboTwin 2.0 (Table 3)

| Setting | pi0.5 | **QuoVLA** |
|---------|-------|-----------|
| Easy | - | 45.1 |
| **Hard** | 43.84 | **58.6** (+14.8) |

- Dual-arm bimanual에서도 pi0.5 대비 명확한 향상
- 그러나 절대값 58.6%는 RoboTwin 2.0의 hard 난이도를 감안하면 *완성형*은 아님

### 4.5 Real-Robot (Table 4, 4 tasks)

| 메트릭 | pi0.5 | **QuoVLA** |
|--------|-------|-----------|
| Average SR | 64.5% | **88.0%** (+23.5) |
| Put apple on yellow plate | 31% | **83%** (+52) |

- **+23.5 평균 향상은 시뮬레이션 이상의 sim-to-real gap을 좁히는 강력한 증거**
- 특히 specific task (apple → yellow plate)에서 +52 absolute gain은 prompt 변동에 강한 robustness를 시사

---

## 5. Ablation 분석 (Table 5)

| 구성 | LIBERO Avg |
|------|-----------|
| **Default (8-bit, depth=1)** | **99.6** |
| Without quantization | 96.85 (-2.75) |
| Quantization depth=2 | 95.7 (-3.9) |
| Quantization depth=6 | 92.2 (-7.4) |
| 16-bit quantization | 97.1 (-2.5) |
| Without adaptive STE | 98.45 (-1.15) |
| Without dual-branch | 98.65 (-0.95) |
| Without constraints | 98.78 (-0.82) |

### 핵심 관찰
1. **Quantization 자체의 기여가 가장 큼** (-2.75)
2. **Depth=1이 최적** — 더 깊게 쌓으면 오히려 성능 저하 (depth=6에서 -7.4)
3. **8-bit이 16-bit보다 우수** — 더 강한 압축이 quotient effect를 극대화
4. **Adaptive STE / dual-branch / constraints**는 각각 작지만 누적적 기여

> ❓ **예상 질문**: 16-bit이 8-bit보다 못한 결과는 직관에 반하는데?
> **답변**: 정확한 quotient theory의 예측. 16-bit은 precision이 높아 redundant variation을 *덜* 제거 → overcomplete 상태가 유지되어 prompt noise에 더 민감. 8-bit의 적절한 information bottleneck이 quotient 효과를 극대화.

---

## 6. 관련 연구 비교

| 모델 | Backbone | Action Head | LIBERO Avg | Real-robot |
|------|----------|-------------|------------|-----------|
| OpenVLA | Llama-2 7B | AR discrete | 76.5 | - |
| RDT | DiT | DiT diffusion | - | - |
| pi0 | PaliGemma 3B | flow-matching | 94.2 | - |
| pi0.5 | PaliGemma 2.6B | flow-matching | 96.9 | 64.5 |
| **QuoVLA** | **pi0.5 + 8-bit quant prefix** | **flow-matching** | **99.6** | **88.0** |

### 핵심 차이
- 기존 VLA는 모두 *capacity 증가* 방향 (more params, more layers, more data)
- QuoVLA는 *capacity 압축* 방향 — quantization으로 information bottleneck 형성
- "Less is more"의 명확한 사례

---

## 7. 한계 및 미해결 문제

### 방법론적 미비점
1. **Inference latency 미보고**: Quantization이 *학습* 시에만 적용되는지, *추론* 시에도 적용되어 속도 이득이 있는지 불명. 1-layer Transformer 추가는 latency를 *증가*시킬 가능성 있음
2. **LIBERO-PRO Position/Task에서 큰 drop**: Position perturbation 40%/26%, Task change 10%/26%는 robustness 주장에 큰 약점. Quotient가 *prompt*는 처리하지만 *physical state*는 처리하지 못함을 시사
3. **CALVIN / SimplerEnv 미평가**: long-horizon (CALVIN) / cross-embodiment (SimplerEnv) 표준 benchmark 부재 → robustness 일반화 검증 부족
4. **Code release pending**: "Code will be made publicly available" — v1에서 repo URL 없음
5. **Quotient theory의 formal 증명 부재**: "VLM은 action-sufficient하지만 overcomplete하다"는 강한 주장이나, information-theoretic 증명이나 mutual information 측정 등이 부재. 실험적 검증만 존재
6. **RoboTwin 2.0 Hard 58.6%는 absolute 기준 강하지 않음**: pi0.5 대비 +14.8 향상은 인상적이나, 절대값은 dual-arm 표준으로는 중간 정도

### Attribution 문제
- 99.6%는 LIBERO에 거의 ceiling — 추가 향상의 marginal value가 작음
- Real-robot +23.5는 인상적이나, pi0.5 자체의 baseline (64.5%)가 낮음을 감안하면 *어디서부터 향상이 시작되는지*에 따라 의미가 달라짐

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Quotient theory는 VLA 통념에 정면 도전 |
| **Technical depth** | ★★★★☆ — 8-bit quant + dual-branch + L_tc의 조합이 깔끔 |
| **Experimental rigor** | ★★★★☆ — LIBERO + LIBERO-PRO + LIBERO-Plus + RoboTwin 2.0 + 실로봇 (광범위) |
| **Practical impact** | ★★★★☆ — 실로봇 +23.5는 실용성 직접 입증 |
| **Writing quality** | ★★★★☆ — clear formulation |

**강점**: "VLM은 action-insufficient하다"는 VLA 분야의 dominant assumption을 명시적으로 반박하고, **압축이 곧 향상**이라는 정량적 증거 제시. LIBERO 99.6%는 사실상 ceiling, real-robot +23.5는 sim-to-real gap을 좁히는 강력한 결과.

**약점**: LIBERO-PRO Position/Task에서의 큰 drop은 "quotient가 모든 redundancy를 제거하지 않는다"는 한계 노출. Inference latency 미공개로 실시간성 평가 불가. Quotient theory가 formal하기보다 실험적 관찰에 머무름.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 8-bit quantization이 16-bit보다 우수한 이유는? | Quotient theory의 정확한 예측 — 강한 information bottleneck이 prompt-irrelevant variation 제거를 극대화 |
| 2 | LIBERO-PRO Task change 10%는 robustness 주장에 치명적 아닌가? | 맞음. Quotient는 prompt-level redundancy만 제거, task-level zero-shot transfer는 별개 문제 |
| 3 | Inference latency는? | 미보고. 1-layer Transformer 추가는 *느려지는* 방향. 실용성 평가의 큰 공백 |
| 4 | Quotient theory의 formal 증명은 있나? | 실험적 관찰에 머무름. Mutual information이나 information-theoretic argument 부재 |
| 5 | Real-robot +23.5는 pi0.5의 약한 baseline 때문 아닌가? | 가능. pi0.5의 4-task real-robot 64.5%는 절대 기준 강하지 않음. 더 강한 baseline (예: pi0.5 + extensive fine-tuning)과 비교했다면 differential이 다를 수 있음 |
| 6 | RoboTwin 2.0에서 Easy 45.1% / Hard 58.6%의 *역전*은 어떻게 설명하나? | 직관에 반함. Easy/Hard 정의가 task semantic 다양성에 따라 정해지므로, hard task가 오히려 quotient가 유효한 large variation 공간에 분포 |
| 7 | CALVIN / SimplerEnv 평가는 왜 없는가? | 미보고. long-horizon (CALVIN ABC→D) / cross-embodiment (SimplerEnv) 검증 부재로 일반화 주장 약함 |
| 8 | Depth=1 이상의 quant Transformer가 왜 성능을 떨어뜨리나? | Quotient theory의 자연스러운 결과 — 한 번의 8-bit bottleneck으로 충분, 추가 bottleneck은 task-relevant signal까지 잃음 |

<!-- VERIFIED: pdf -->
