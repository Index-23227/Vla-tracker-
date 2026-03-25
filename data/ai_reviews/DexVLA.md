# DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control

> **한 줄 요약**: Qwen2-VL 2B에 **1B 파라미터 diffusion expert** (32-layer transformer)를 plug-in하고, 3-stage curriculum (cross-embodiment→embodiment-specific→task-specific)으로 학습하여, 셔츠 접기 0.92, LIBERO 97.3%, **60Hz inference on A6000** 달성.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 기존 VLA의 action head는 소형(수백만~수억 파라미터) → 고차원 dexterous action 표현력 부족
- Cross-embodiment 학습 시 서로 다른 action space를 **하나의 작은 action head로 통합**하면 capacity bottleneck
- Sub-step reasoning 없이 long-horizon task를 직접 prompting으로 수행하기 어려움

### 핵심 질문
- **Action expert를 1B 규모로 scaling하면 dexterous/long-horizon control이 가능한가?**
- **3-stage curriculum이 direct fine-tuning 대비 얼마나 효과적인가?**

---

## 2. 방법론 심층 분석

### 2.1 Architecture

**VLM**: Qwen2-VL 2B
**Diffusion Expert**: **32 layers, hidden 1280, 16 heads → ~1B parameters**
- Scale Diffusion Policy (ScaleDP) 기반 transformer
- Multi-head output: 각 head가 단일 robot configuration 담당 → cross-embodiment
- **FiLM conditioning**: VLM의 reasoning tokens이 projection layer를 scale/shift

$$\mathcal{L} = \mathcal{L}_{\text{diff}} + \alpha \mathcal{L}_{\text{ntp}}, \quad \alpha = 1$$

### 2.2 Three-Stage Curriculum

| Stage | 학습 대상 | 데이터 | LR | Scheduler |
|-------|---------|--------|-----|----------|
| **1. Cross-embodiment** | Diffusion expert only | 100h multi-robot, 91 tasks | 1e-4 | Constant |
| **2. Embodiment-specific** | VLM + projection + expert | Single-embodiment | 2e-5 | Cosine |
| **3. Task-specific** | Full | Sub-step annotated (5초 간격) | 2e-5 | Cosine |

- All stages: AdamW (β₁=0.9, β₂=0.95), 5 epochs, WD=0.0
- Stage 1에서 ResNet-50 image encoder + DistilBERT 사용 (VLM 미사용)
- Stage 2에서 VLM visual encoder **freeze**, 나머지 학습

> ❓ **예상 질문**: Stage 1에서 왜 VLM을 쓰지 않는가?
> **답변**: Cross-embodiment 데이터가 매우 diverse하여 VLM을 동시에 학습하면 불안정. Stage 1은 **motor primitive만** 학습에 집중. Stage 2에서 VLM을 통합.

### 2.3 Sub-step Reasoning

Long-horizon task에서 **5초마다 sub-step annotation** 제공:
- "Grab the shirt" → "Fold left side" → "Fold right side" → "Final fold"
- 이 reasoning이 FiLM으로 diffusion expert에 주입

> ❓ **예상 질문**: SayCan과 어떻게 다른가?
> **답변**: SayCan은 external LLM이 discrete skill을 선택하는 modular 방식. DexVLA는 sub-step reasoning이 FiLM으로 continuous action에 직접 conditioning되는 **end-to-end** 방식. Table 6에서 DexVLA (0.70) > SayCan (0.58).

---

## 3. 데이터 전략

| Stage | 데이터 | 규모 |
|-------|--------|------|
| Stage 1 | Multi-robot (Agile X 42.7%, Franka 34.7%, 기타) | **100 hours, 91 tasks** |
| Stage 2 | Single-embodiment subset | Target robot에 맞춤 |
| Stage 3 | Sub-step annotated demonstrations | Task-specific |

---

## 4. 실험 결과 심층 분석

### Diffusion Expert Size Ablation (Figure 10)

| Model | Params | Shirt Folding Score |
|-------|--------|-------------------|
| UNet | 93M | 0.17 |
| Diffusion Expert | 410M | 0.63 |
| **Diffusion Expert** | **1B** | **0.92** |

- **1B가 93M 대비 5.4x 높은 score** → expert scaling의 결정적 중요성

### LIBERO Benchmark (Table 3)

| 모델 | Spatial | Object | Goal | **Avg** |
|------|---------|--------|------|---------|
| Diffusion Policy | 78.3 | 92.5 | 68.3 | 79.7 |
| OpenVLA | 84.7 | 88.4 | 79.2 | 84.1 |
| π₀-FAST | 96.4 | 96.8 | 88.6 | 93.9 |
| π₀ | 96.8 | 98.8 | 95.8 | 97.1 |
| **DexVLA** | **97.2** | **99.1** | **95.6** | **97.3** |

- π₀ 대비 **+0.2%p** (97.3 vs 97.1) — competitive

### 3-Stage Curriculum Ablation (Table 1)

| Configuration | Shirt Folding | Laundry Folding |
|--------------|-------------|----------------|
| Stage 1 only | 0.0 | 0.0 |
| Stage 2 only | 0.0 | 0.0 |
| Stage 1+2 | **0.92** | 0.0 |
| **Stage 1+2+3** | **0.92** | **0.4** |

- Stage 1 or 2 alone: **완전 실패** (0.0) → 각 stage가 필수
- Stage 3 (sub-step reasoning): Laundry folding에서 0.0→0.4

### Sub-step Reasoning Ablation (Table 5)

| 설정 | Score |
|------|-------|
| Stage 1 only | 0.07 |
| Stage 2 only | 0.0 |
| Stage 1+2 | **0.92** |

### Visual Generalization (Table 2)

| Task | Novel Object | Novel Scene | Both |
|------|-------------|-------------|------|
| Shirt folding (Bimanual) | 0.78 | 0.78 | 0.56 |
| Drink pouring (Dexterous) | 0.83 | 0.67 | 0.67 |

### Long-horizon Direct Prompting

| Task | π₀ | DexVLA |
|------|-----|--------|
| Laundry folding | 0.2 | **0.4** |
| Table bussing (hard) | ~0.5 | **~0.58+0.08** |
| Dryer unloading | 0.0 (Octo/OpenVLA) | **0.8** |

---

## 5. 추론 속도

> **60Hz on single NVIDIA A6000 GPU** — real-time control 가능

---

## 6. 관련 연구 비교

| 모델 | VLM | Action Expert | Expert Params | Curriculum | Inference |
|------|-----|-------------|-------------|-----------|----------|
| OpenVLA | 7B | Token pred | ~10M | ✗ | ~5Hz |
| CogACT | 7B | DiT | 308M | ✗ | ~10Hz |
| π₀ | ~3B | Flow matching | ~300M | ✗ | ~20Hz |
| **DexVLA** | **Qwen2-VL 2B** | **ScaleDP** | **1B** | **✓ (3-stage)** | **60Hz** |

---

## 7. 한계 및 미해결 문제

### 방법론적 미비점
1. **Sub-step annotation 비용**: 5초 간격 manual annotation이 필요 → scalability 제약. "Acquiring new tasks still requires collecting substantial human demonstration data"
2. **Laundry folding 0.4**: Long-horizon에서 여전히 낮은 성공률. Stage 3 추가해도 제한적
3. **Visual generalization (Both 0.56)**: Novel object + novel scene 동시에서 44% 실패
4. **Training cost**: 100h 데이터 + 3-stage pipeline의 복잡성. 재현 비용 높음
5. **LIBERO-Long 미보고**: Table 3에 Long suite 결과 없음

### Attribution 문제
- 성능이 **1B expert** 때문인지, **3-stage curriculum** 때문인지, **sub-step reasoning** 때문인지 분리 필요. Expert size ablation (93M→1B)과 curriculum ablation 각각 존재하나, interaction effect 미분석

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — 1B action expert + 3-stage curriculum |
| **Technical depth** | ★★★★★ — Expert scaling, curriculum, sub-step reasoning 체계적 |
| **Experimental rigor** | ★★★★☆ — LIBERO + real-world + ablation. LIBERO-Long 부재 |
| **Practical impact** | ★★★★★ — 60Hz, dexterous task 최초 실용 수준 |
| **Writing quality** | ★★★★☆ |

**강점**: 1B expert scaling이 dexterous task에서 결정적임을 실증 (93M: 0.17 → 1B: 0.92). 60Hz real-time. 3-stage 각 단계의 필요성이 ablation으로 입증. **약점**: Sub-step annotation 비용, laundry folding 0.4의 한계, curriculum 복잡성.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 60Hz면 VLM (Qwen2-VL)도 매 step 실행하는가? | 아니오. VLM은 sub-step 단위(~5초)로만 실행, diffusion expert만 60Hz. 이것이 plug-in 설계의 핵심 이점 |
| 2 | π₀ 대비 LIBERO +0.2%p면 saturation 아닌가? | 맞음. DexVLA의 진짜 차별화는 LIBERO가 아니라 **dexterous task (셔츠 접기 0.92)와 60Hz** |
| 3 | Sub-step annotation 없이 가능한가? | Table 1에서 Stage 1+2만으로 shirt folding 0.92 달성. 그러나 laundry folding은 Stage 3 필수 (0.0→0.4) |
| 4 | 1B expert를 3B로 더 키우면? | 미검증. 410M→1B에서 0.63→0.92이므로 diminishing returns 예상되나 확인 필요 |
| 5 | VLM을 2B에서 7B로 키우면? | Expert가 아닌 VLM scaling의 효과 미검증. RoboVLMs에서는 VLM 선택이 중요함을 보임 |
| 6 | Training cost (3-stage × 100h data)의 재현성은? | 산업 연구 수준. 학계에서 100h 데이터 수집은 상당한 투자 |

<!-- VERIFIED: pdf -->
