# ALAM: Algebraically Consistent Latent Action Model for Vision-Language-Action Models

> **한 줄 요약**: Pi0 (PaliGemma-2B + Gemma-300M) 백본 위에서, 1111개 action-free video로 latent transition encoder를 사전학습하되 **composition consistency** ($z^c_a \approx z^b_a + z^c_b$) 와 **reversal consistency** ($z^a_b \approx -z^b_a$) 두 가지 대수적 정규화를 부과하여 latent 공간을 "locally additive"하게 만들고, transfer 단계에서 frozen encoder와 robot action을 **joint K-step flow matching**으로 결합. MetaWorld MT50을 47.9% → **85.0%**, LIBERO Avg를 94.1% → **98.1%** 로 끌어올림.

---

## 1. 배경 및 동기

### Action-Label Scarcity 문제
- VLA 학습에는 (vision, language, **action**) 트리플이 필요한데, action label은 가장 수집이 어렵고 비싼 modality.
- 반면 (vision, language)만 있는 **action-free video**는 인터넷, Open-X-Embodiment, EgoExo 등에서 방대하게 존재.
- 이를 어떻게 action을 갖는 정책에 효과적으로 transfer할 것인가가 LAPA, GR-1 등 연속 연구의 핵심 질문.

### 기존 Latent Action 접근의 약점
- LAPA: VQ-VAE 기반 discrete latent action — discretization이 정밀 manipulation에 한계.
- 일반 latent action: token간 관계가 unstructured — 같은 transition을 다양한 latent로 표현해버려 generalization 약함.

### 핵심 질문
- **Latent transition 공간에 어떤 algebraic structure를 부과하면, action-free pretraining이 downstream policy로 잘 transfer되는가?**

📌 [Figure 1 삽입] — Pretraining(algebraic regularizers) → Transfer(joint flow matching) 두 단계

---

## 2. 방법론 심층 분석

### 2.1 핵심 아이디어: "Locally Additive Transition Space"

연속 두 시점 $t_a < t_b$ 사이의 transition을 latent $z^b_a$ 로 표현. ALAM은 이 latent에 두 가지 **algebraic axiom**을 부과한다:

| Axiom | 수식 | 의미 |
|-------|------|------|
| **Composition (additivity)** | $z^c_a \approx z^b_a + z^c_b$ | 두 연속 transition의 합이 전체 transition |
| **Reversal** | $z^a_b \approx -z^b_a$ | 시간을 뒤집은 transition은 부호만 반대 |

두 규제는 **L2 soft loss**로 부과됨 (hard constraint가 아님).

> ❓ **예상 질문**: 왜 이런 algebraic 구조가 도움이 되는가?
> **답변**: Locally additive 공간은 **Lie algebra / tangent space**와 유사. 행동(action)이 본질적으로 differential operation이라는 점을 활용. Policy가 latent transition을 예측하는 것이 곧 "infinitesimal action을 sum하여 finite trajectory를 구성"하는 자연스러운 framework가 됨.

> ❓ **예상 질문**: Hard constraint(equality)가 아닌 soft L2 인 이유?
> **답변**: 비디오의 noise (occlusion, blur, scene change)로 hard constraint는 over-restrictive. Soft penalty로 average에서 additivity를 유도하면서 정상적 noise 허용.

### 2.2 Pretraining Stage

- **Encoder**: relational encoder가 두 프레임 $(o_a, o_b)$를 받아 transition latent $z^b_a$ 생성.
- **Objective**: 
  1. **Future-frame reconstruction**: $z^b_a$ + $o_a$ 로 $o_b$ 재구성
  2. **Composition L2**: $|z^c_a - (z^b_a + z^c_b)|_2^2$
  3. **Reversal L2**: $|z^a_b - (-z^b_a)|_2^2$
- **Data**: 1111개 action-free video source, 주로 Open-X-Embodiment + CALVIN 혼합.

### 2.3 Transfer Stage

- Encoder freeze.
- Pi0 backbone (PaliGemma-2B vision-language) 위에 Gemma-300M action expert.
- **Joint K-step flow matching**: latent transition $z$와 robot action $a$를 같은 flow trajectory에서 함께 예측.
- 즉 학습 시 두 modality (transition, action) 모두에 flow loss를 부과 → encoder의 algebraic structure가 action expert로 자연스럽게 흘러들어감.

> ❓ **예상 질문**: K-step의 K는?
> **답변**: Action chunk size H=16 framework 내에서 flow matching step 수. 추론 시는 보통 H 보다 적은 step으로 빠른 generation 가능.

> ❓ **예상 질문**: Joint flow에서 latent transition은 실제로 행동되는가?
> **답변**: 아니오 — latent transition은 학습 시 auxiliary, **decoder가 없고 robot에서 실행되지 않음**. 추론 시 이 부분은 cost가 추가되지만 robot 입장에서는 invisible. 학습 시 representation regularizer 역할.

### 2.4 Architecture 요약

| 구성 | 값 |
|------|----|
| Vision-Language Backbone | PaliGemma-2B |
| Action Expert | Gemma-300M |
| Action chunk H | 16 |
| Total parameters | ~2.3B |
| Action head category | Flow Matching (joint) |

---

## 3. 데이터 및 학습 인프라

### 3.1 Pretraining Data

- **1111개 action-free video source** (대부분 real-robot manipulation video, Open-X-Embodiment subset + CALVIN)
- Table 7에 11개 핵심 dataset 혼합 비중 명시:
  - fractal20220817: weight 150
  - CALVIN: weight 200
  - (나머지 9개 dataset)

### 3.2 Hardware

| 항목 | 값 |
|------|-----|
| GPU | 128 × H20 (90 GB) |
| Epochs | 3939 |
| Wall-clock | ~44 days |

> ❓ **예상 질문**: 3939 epoch은 비정상적으로 많은데?
> **답변**: 1111 dataset 혼합이므로 effective epoch 정의가 mixture-weighted. 실 batch 관점에서는 일반 large-scale pretraining 수준.

---

## 4. 실험 결과 심층 분석

### 4.1 LIBERO (Table 2)

| Suite | Pi0 baseline | **ALAM** | Δ |
|-------|-------------|---------|----|
| Spatial | - | **99.2** | - |
| Object | - | **99.6** | - |
| Goal | - | **99.0** | - |
| Long | - | **94.4** | - |
| **Avg** | **94.1** | **98.1** | **+4.0** |

(suite-by-suite baseline는 Pi0 paper 참조; ALAM 본문은 Avg 비교에 집중)

### 4.2 MetaWorld MT50 (Table 1)

| Difficulty | **ALAM** |
|-----------|---------|
| Easy | 89.3 |
| Medium | 83.6 |
| Hard | 85.0 |
| Very Hard | 82.0 |
| **Avg** | **85.0** |

| 비교 baseline | MT50 Avg |
|--------------|---------|
| Pi0 (Flow Matching) | 47.9 |
| Pi0.5 (FM) | 31.5 |
| OpenVLA (AR) | 76.5 (LIBERO only) |
| LAPA (Latent Action) | 83.5 (LIBERO only) |
| GR-1 (Video Action) | 50.5 |
| **ALAM** | **85.0** |

- **MetaWorld에서 Pi0 대비 +37.1%p** — 본 논문의 가장 인상적 수치.
- Pi0.5가 오히려 Pi0보다 낮은 31.5인 점이 특이 — Pi0.5의 학습 setup이 MT50에 그대로 transfer 안되는 한계.

### 4.3 Algebraic Error Reduction

- ALAM은 unstructured baseline 대비:
  - **Additivity error** 25-85배 감소
  - **Reversibility error** 25배 감소 (t=5k checkpoint 기준 reversibility ratio 25x, additivity ratio 46x)
- 즉 regularizer가 latent 공간을 실제로 algebraic하게 만들고 있음을 직접 검증.

### 4.4 Ablation (MetaWorld MT50)

| 설정 | MT50 Avg |
|------|---------|
| 두 loss 모두 제거 | 58.3 |
| Reversal만 제거 | 63.6 |
| Additivity만 제거 | 67.2 |
| **Full ALAM** (epoch-44) | **78.0** |
| Full ALAM (final) | 85.0 |

- **Additivity가 reversal보다 더 큰 기여** (가중치 차이: removing additivity → 67.2 / removing reversal → 63.6, base 58.3 대비).
- 둘 다 있는 full이 single loss보다 훨씬 좋음 — 두 axiom의 시너지가 명확.

### 4.5 Real-World (Piper 6-DoF)

- 4개 task: insert cylinder, insert cube, stack cup, fold towel.
- ALAM이 Pi0, Pi0.5 대비 4개 모두 향상, 최대 **+45%p** 개선.
- Camera: 3rd-person D435 + wrist D405, 224x224.

---

## 5. 관련 연구 비교

| 모델 | Latent Action | Algebraic Structure | Backbone | LIBERO Avg | MT50 Avg |
|------|--------------|--------------------|---------:|-----------:|--------:|
| OpenVLA | ✗ | - | Llama2-7B | 76.5 | - |
| Pi0 | ✗ | - | PaliGemma-2B | 94.1 | 47.9 |
| LAPA | Discrete (VQ) | ✗ | various | 83.5 | - |
| GR-1 | Implicit | ✗ | GPT2-medium | - | 50.5 |
| **ALAM** | **Continuous, joint FM** | **Composition + Reversal** | **PaliGemma-2B** | **98.1** | **85.0** |

### 핵심 차별점
- "Latent action을 학습한다"는 라인의 연속이지만, **latent에 explicit algebraic axiom을 부과한 최초**.
- LAPA의 discrete tokenization, GR-1의 video prediction과 명백히 구분되는 디자인 축.

---

## 6. 한계 및 미해결 문제

### 방법론적
1. **128 × H20 90G × 44일 = 매우 비싼 pretraining**: 재현이 사실상 academic에서 불가. 코드 공개도 본 fetch 시점에 확인되지 않음.
2. **Algebraic axiom의 valid range**: composition은 local에서만 성립. Trajectory 양 끝 (시작/끝 occlusion)에서는 위배 가능. 본문에 boundary 처리 명시 부족.
3. **Encoder freeze가 좋은 선택인가**: transfer 시 frozen이 표현 한계를 가질 수 있음. Partial finetune ablation 부재.

### 평가
1. **LIBERO suite별 OpenVLA/Pi0 baseline 명시 부족**: ALAM 본문의 비교 표가 average 위주.
2. **MetaWorld vs LIBERO 차이**: MT50에서 Pi0 47.9 → ALAM 85.0 (+37) 의 거대한 게인이 LIBERO에서는 +4.0에 그침. LIBERO ceiling에 가까운 것을 감안해도 차이가 큰데, 이게 "ALAM이 어려운 task에서 빛난다"는 해석인지, "easy MetaWorld가 representation 향상에 더 민감하다"인지 불명확.

### 해석
- **Composition L2가 reversibility보다 contribution이 큰 이유**: composition은 모든 triple (a,b,c)에서 검증되므로 dataset당 supervision sample 수가 reversibility보다 훨씬 많음. 더 많은 supervision = 더 큰 효과.
- **Joint flow matching의 비용**: latent transition을 함께 generate하는 K-step flow는 inference 시에도 cost 추가 (논문은 명시적 latency 미보고). Real-world Piper에서 작동했다는 점은 cost가 prohibitive하지 않음을 시사하나, 정확한 Hz 미공개.

---

## 7. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Latent action에 algebraic axiom을 부과한 최초. Composition + Reversal 조합의 산뜻함. |
| **Technical depth** | ★★★★★ — Lie-algebra-inspired 디자인이 단순 heuristic을 넘어선 원리적 접근. Joint flow matching framework도 정교. |
| **Experimental rigor** | ★★★★☆ — MT50 47.9→85.0, LIBERO 94.1→98.1, real-world 45%p — 일관된 게인. 그러나 LIBERO suite 비교가 baseline 측 부족. |
| **Practical impact** | ★★★☆☆ — 결과는 훌륭하지만 128×H20×44일 학습 비용이 reproducibility를 크게 제한. 코드 미공개. |
| **Writing/Clarity** | ★★★★☆ — algebraic axiom 동기 부여가 직관적. |

**강점**: VLA latent space에 명시적 수학적 구조를 도입한 원리적 디자인. MT50 +37%p는 같은 backbone에서 거의 두 배의 성능. **약점**: 막대한 학습 비용, 코드 미공개, frozen encoder 가정.

---

## 8. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | "Locally additive"의 "locally"는 어느 범위? | 인접 프레임 (보통 1-step) 단위에서 enforcing. Long-range composition은 ratio로 decay 가능. |
| 2 | Composition $z^c_a \approx z^b_a + z^c_b$는 어떤 dataset triple에서 sampling? | (a, b, c) 시점 triple을 trajectory 안에서 random sample. video 길이가 길수록 triple 수 quadratic 증가. |
| 3 | Reversal axiom은 video data가 시간 가역적이라는 가정이 필요한데 deformable object에서도 성립하는가? | 엄밀하게는 아님 — 천 접기 등은 비가역. 그래서 hard constraint 대신 soft L2. |
| 4 | LAPA(VQ-VAE 기반)와의 head-to-head 비교는? | LIBERO Avg에서 ALAM 98.1 vs LAPA 83.5 — 큰 차이. 단, 정확히 같은 fine-tuning protocol인지는 본문 확인 필요. |
| 5 | Pi0.5가 Pi0보다 MT50에서 낮은 이유는? | Pi0.5의 학습 mixture에 MetaWorld가 약하게 들어가 있어 transfer가 잘 안 됨. ALAM은 algebraic regularizer로 transfer를 보강. |
| 6 | Joint flow matching의 K-step 수는? | Action chunk H=16 framework 내. Inference 시 step reduction 가능 (논문 미명시). |
| 7 | Frozen encoder가 다양한 robot embodiment로 transfer 잘 되는가? | Pretraining data가 multi-embodiment (Open-X)이므로 base가 어느 정도 일반화. 그러나 한 frozen encoder가 모든 embodiment를 cover한다는 강한 주장은 부재. |
| 8 | 1111개 dataset에서 weighting은 어떻게? | Table 7. fractal=150, CALVIN=200 등 dataset 별 sampling weight 명시. |

<!-- VERIFIED: pdf -->
