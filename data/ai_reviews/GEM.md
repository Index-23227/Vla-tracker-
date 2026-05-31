# GEM: Generative Supervision Helps Embodied Intelligence

> **한 줄 요약**: Qwen3-VL backbone에 *depth-map 생성*을 VLM pretraining 단계의 보조 감독 신호로 직접 주입하고, 이를 RDT2-style DiT flow-matching action expert와 결합한 GEM-VLA — LIBERO 96.1%, SimplerEnv WidowX 67.0%로 pi0/OpenVLA를 큰 격차로 추월한 generative-pretraining + VLA 통합 프레임워크.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 기존 VLA (OpenVLA, pi0, RDT)는 표준 VLM (PaliGemma, Llama-3 등)을 그대로 backbone으로 사용 → **VLM이 이미지 → 텍스트** 정렬만 학습했기 때문에 *기하학적/공간적* 표현이 빈약
- 그 결과 long-horizon LIBERO-Long, SimplerEnv WidowX와 같이 **3D 공간 추론**이 필요한 시나리오에서 성능 한계
- "Embodied"라는 단어가 무색하게도, 현재 VLA는 *깊이 (depth)*를 명시적으로 학습하지 않음

### 핵심 질문
- **VLM pretraining 단계에서 depth-map을 생성하도록 강제하면, downstream VLA action 학습이 향상되는가?**
- **Generative supervision (텍스트가 아닌 픽셀/깊이 출력)이 VLA의 spatial grounding을 개선하는가?**

📌 [Figure 1 삽입] — GEM 전체 파이프라인: GEM (depth-aware VLM) + DiT action expert → GEM-VLA

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

GEM-VLA는 두 단계로 학습:
1. **GEM (VLM)**: Qwen3-VL backbone (2B / 8B 변형) — 텍스트 next-token 예측에 **depth map generation**을 보조 task로 결합
2. **GEM-VLA**: GEM 위에 **RDT2-style Diffusion Transformer (DiT) action expert**를 flow-matching objective로 fine-tune

### 2.2 Depth-Map Generative Supervision

- VLM pretraining 중 입력 RGB 이미지에 대해 **monocular depth map**을 함께 출력하도록 학습
- 텍스트 generation loss + depth generation loss의 multi-task setup
- 핵심 가설: depth 예측은 모델로 하여금 *기하학적 / 공간적 표현*을 잠재 공간에 압축하도록 강제 → 후속 action 학습에 유리

> ❓ **예상 질문**: Depth는 monocular estimation으로 supervision한 것인가, GT인가?
> **답변**: 논문은 GEM-4M 데이터셋이 "grounding, reasoning, and depth/planning data"의 4M 샘플로 구성된다고 명시. 대규모 데이터의 다수는 monocular depth estimator (ZoeDepth 등)로 만든 pseudo-label일 가능성이 높음 — 3D-VLA에서도 동일한 trade-off가 존재 (95% estimated depth).

### 2.3 DiT Action Expert (Flow Matching)

- **Action chunk size = 32**, 3-camera observation (top + 2 wrist)
- RDT2의 DiT 구조를 채택 — pi0의 flow-matching action head와 유사한 family
- Global batch size 256으로 8 GPU에서 fine-tune

> ❓ **예상 질문**: 왜 RDT2-style DiT인가? pi0의 dual-stream PaliGemma + action expert가 아닌 이유는?
> **답변**: GEM은 VLM 자체에 spatial supervision을 주입했기 때문에 dual-stream 구조가 불필요. 단일 DiT가 image+text+state token 모두를 attention으로 처리하면 충분 — 단순성과 generative pretraining 효과를 살림.

---

## 3. 데이터 전략

### GEM-4M Dataset

**규모**: 4M 샘플, 세 가지 축으로 구성
- **Grounding**: 객체-언어 alignment
- **Reasoning**: chain-of-thought style 캡션 / QA
- **Depth / Planning**: monocular depth 생성 + action planning context

### Action Fine-tuning Data
- LIBERO (4 suites)
- SimplerEnv (Bridge / Fractal)
- Real-world UR5 demonstrations (3 long-horizon tasks)

---

## 4. 시스템/학습 세부사항

| 단계 | Hardware | Batch | Steps |
|------|----------|-------|-------|
| GEM pretraining | 32× A800 | - | (논문에서 정확한 step 수 미보고) |
| GEM-VLA fine-tuning | 8 GPUs / task | 256 | 50,000 |

- Action chunk: 32
- 카메라: 1 top + 2 wrist

---

## 5. 실험 결과 심층 분석

### 5.1 LIBERO (Table 3)

| 모델 | Spatial | Object | Goal | Long | **Avg** |
|------|---------|--------|------|------|---------|
| OpenVLA | - | - | - | - | 76.5 |
| pi0 | - | - | - | - | 94.2 |
| **GEM-VLA (8B)** | **99.0** | **98.8** | **97.1** | **89.3** | **96.1** |

- 가장 어려운 LIBERO-Long에서 89.3%로 pi0 대비 명확한 향상
- Spatial / Object / Goal에서는 사실상 상한선 수준

### 5.2 SimplerEnv WidowX (Table 5)

| 태스크 | OpenVLA | pi0 | **GEM-VLA** |
|--------|---------|-----|-------------|
| Put Spoon on Towel | - | - | **82.0** |
| Put Carrot on Plate | - | - | 58.0 |
| Stack Blocks | - | - | 44.0 |
| Put Eggplant in Basket | - | - | **84.0** |
| **Average** | 1.0 | 54.9 | **67.0** |

- OpenVLA의 1.0%는 catastrophic underfit (Bridge에서 fine-tune 안 한 결과)
- pi0 (54.9%) 대비 **+12.1%** 의 일관된 우위
- Stack Blocks (44%)에서 여전히 약함 — 정밀 6-DoF placement

### 5.3 Real-world UR5 (3 tasks)

| 평가 | Prior SOTA | **GEM-VLA** |
|------|-----------|-------------|
| Avg success (table bussing, cloth folding, unzip backpack) | 28.7% | **43%** |

- Long-horizon + deformable 조합 — VLA에게 가장 어려운 setting
- 그러나 정확한 per-task 수치는 공개되지 않음 (논문 한계)

---

## 6. Ablation 분석

### ⚠️ Ablation Coverage

논문에서 가장 핵심 가설인 "depth supervision이 도움이 되는가?"에 대한 head-to-head ablation (depth-pretraining ON vs OFF, 동일 backbone)이 명시적으로 보고되었는지 검증이 필요. v1 HTML에서는 ablation table이 명시적이지 않음 → **이 부분이 reviewer에게 가장 큰 challenge**가 될 가능성.

---

## 7. 관련 연구 비교

| 모델 | VLM 추가 감독 | Action Head | LIBERO Avg | SimplerEnv |
|------|---------------|-------------|------------|------------|
| OpenVLA | 없음 | AR discrete | 76.5 | 1.0 |
| pi0 | 없음 | flow-matching | 94.2 | 54.9 |
| RDT | 없음 | DiT diffusion | - | - |
| **GEM-VLA** | **Depth generation** | **DiT flow-matching** | **96.1** | **67.0** |

### 핵심 차이
- "VLM에 generative spatial supervision을 주입"은 3D-VLA와 동일한 방향성이지만, GEM은 **embodied diffusion model을 별도로 두지 않고** depth task를 VLM에 직접 흡수
- pi0 대비의 향상이 *VLM의 depth-aware backbone* 덕분인지, *RDT2-style action expert* 덕분인지 분리하려면 ablation이 필수

---

## 8. 한계 및 미해결 문제

### 방법론적 미비점
1. **Depth supervision의 단독 기여 검증 부재**: GEM-4M 데이터 자체가 4M 규모 (대형) — depth task 없이 동일 데이터로 학습한 baseline과의 비교가 핵심이나, v1 HTML에서 이 ablation이 명확히 보고되지 않음
2. **Real-world per-task 수치 미공개**: 43% average만 보고. Table bussing이 잘되는지 cloth folding이 잘되는지 불명
3. **CALVIN 미평가**: long-horizon benchmark의 다른 표준인 CALVIN ABC→D를 평가하지 않음 — long-horizon 주장의 robustness 약화
4. **VLM 파라미터 vs Action expert 파라미터 분리 미공개**: 정확한 모델 크기 (전체 합 또는 action expert 단독)를 공개하지 않아 efficiency 비교 어려움
5. **Pseudo-depth label 가능성**: GEM-4M의 depth가 GT인지 monocular estimation인지 명시 부족 → 3D-VLA가 겪었던 "95% estimated depth"의 동일 한계 우려

### Attribution 문제
- LIBERO +2-point 향상이 (a) Qwen3-VL의 강력함, (b) DiT action expert, (c) depth generative supervision 중 무엇 덕분인지 분리 불가
- SimplerEnv 67%는 pi0 (54.9%) 대비 큰 격차이지만, pi0의 backbone (PaliGemma 3B) vs GEM-VLA의 backbone (Qwen3-VL 8B) 차이가 일부 기여 가능

---

## 9. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Depth supervision을 VLM pretraining에 직접 결합한 첫 large-scale 시도 |
| **Technical depth** | ★★★★☆ — 4M 데이터 + 32 A800 학습은 인상적 |
| **Experimental rigor** | ★★★☆☆ — Ablation/CALVIN/per-task real-world 미공개 |
| **Practical impact** | ★★★★☆ — LIBERO 96.1, SimplerEnv 67.0은 강력한 numbers |
| **Writing quality** | ★★★★☆ — 명확한 contribution |

**강점**: VLM의 spatial grounding을 generative depth task로 보강한다는 clean한 아이디어. pi0/OpenVLA를 일관되게 outperform하고, real-world UR5에서도 prior SOTA (28.7%)를 큰 폭으로 갱신 (43%).

**약점**: 핵심 가설인 "depth supervision이 도움이 된다"에 대한 controlled ablation의 부재. backbone (Qwen3-VL 8B)이 워낙 강력해서 향상이 backbone에서 오는 것인지 supervision에서 오는 것인지 분리되지 않음.

---

## 10. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Depth supervision을 빼면 LIBERO/SimplerEnv 점수가 얼마나 떨어지나? | v1 HTML에서 명확한 controlled ablation 부재. Reviewer가 반드시 요구할 ablation |
| 2 | 4M GEM-4M 데이터의 depth는 GT인가 ZoeDepth-style pseudo-label인가? | 명확히 공개되지 않음. pseudo-label이라면 noise가 supervision quality 제한 |
| 3 | pi0와 비교 시 backbone (PaliGemma 3B vs Qwen3-VL 8B) 차이가 결과의 주된 원인 아닌가? | Backbone 동등 비교 (Qwen3-VL 8B + flow-matching, depth supervision 없음) 부재로 분리 불가 |
| 4 | CALVIN ABC→D를 왜 보고하지 않았나? | 미보고. long-horizon 능력 주장의 robustness가 LIBERO-Long 89.3% 단독에 의존 |
| 5 | DiT action expert 파라미터 수와 추론 latency는? | 미공개. 8B VLM + DiT라면 step latency가 OpenVLA보다 더 무거울 가능성 |
| 6 | Real-world 43% — 3-task 각각의 success rate은? | 공개되지 않음. Cloth folding과 unzip backpack은 극단적으로 어려운 task, 평균만 보고하는 것은 위험 |
| 7 | Stack Blocks 44%가 다른 SimplerEnv task 대비 크게 낮은 이유는? | Block stacking은 precise 6-DoF placement 필요. Flow-matching의 chunk 32가 fine control에서 약점 |
| 8 | GEM-4M 데이터의 4M 샘플은 어떻게 collection / curation됐나? | 논문이 데이터 출처를 명확히 공개해야 reproducibility 확보 가능 |

<!-- VERIFIED: pdf -->
