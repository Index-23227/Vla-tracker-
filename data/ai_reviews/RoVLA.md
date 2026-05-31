# RoVLA: Multi-Consistency Constraints for Robust Vision-Language-Action Models

> **한 줄 요약**: InternVL3.5-2B + 32-layer Diffusion Transformer (flow-matching)에 세 가지 일관성 제약 — **Instructional Consistency** (paraphrase invariance), **Evolutionary Consistency** (flow-matching timestep 간 velocity field 일관성), **Observational Consistency** (관측 perturbation에 대한 adversarial 강건성) — 을 추가해 LIBERO-Plus overall **82.0%** (zero-shot 74.3%), RoboTwin 2.0 randomized 50.0%를 달성한 **robustness-focused** VLA.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 최신 VLA(OpenVLA-OFT, π0.5, GR00T-N1.6)는 **fixed instruction phrasing**과 **clean visual condition**에 과적합:
  - 같은 task를 "pick up the red cup" → "grab that crimson mug"로 바꾸면 성공률이 크게 떨어짐.
  - 조명/카메라 위치/배경 텍스처를 흔들면 성공률이 절반 이하로 떨어지는 경우도.
- **LIBERO-Plus** 벤치마크가 등장하며 이 문제(language/camera/light/background perturbation)를 정량화.

### 핵심 질문
- VLA의 brittleness는 **superficial correlation** 때문 — 모델이 "어떤 정확한 단어" 또는 "어떤 정확한 픽셀 패턴"에 action을 묶어버림.
- 이를 **세 가지 일관성**으로 깨뜨릴 수 있는가? (1) instruction 표현, (2) flow-matching 시간 진화, (3) 관측 perturbation에 대한 invariance.

---

## 2. 방법론 심층 분석

### 2.1 전체 아키텍처 (Dual-System)

1. **Semantic Encoder**: InternVL3.5-2B의 **첫 16 decoder layer**만 사용. 마지막 4 layer만 unfreeze하여 fine-tune. → 강력한 사전학습 시맨틱은 보존, downstream alignment만 학습.
2. **Action Generator**: **32-layer Diffusion Transformer (DiT)** with **conditional flow-matching**.
   - AdaLN으로 timestep $\tau$ conditioning.
   - Action chunk 길이 16.
   - 추론은 **8-step forward Euler integration** (8 denoising step).

### 2.2 Instructional Consistency (IC)

학습 데이터에 **Qwen3-8B로 task당 ~15개 paraphrase** 생성 (7 prompt template):
- 원문: "open the top drawer"
- Paraphrase 예: "slide the upper drawer out", "pull the topmost drawer open", ...

각 학습 iteration에서 **하나를 uniform sample** → 정책이 표면적 phrasing이 아닌 의미에 grounding하도록 implicit regularization. **명시적 loss term 없음** — 데이터 augmentation 기반.

### 2.3 Evolutionary Consistency (EC)

Flow-matching에서 정책 $v_\theta(s_t, \tau)$는 timestep $\tau$에 의존. **같은 trajectory의 다른 두 timestep $\tau_1, \tau_2$에서 예측한 clean-velocity field가 일관**되어야 한다는 제약:

$$\mathcal{L}_{\text{EC}} = \|\hat{v}^{\tau_1}_{\text{clean}}(s_t) - \hat{v}^{\tau_2}_{\text{clean}}(s_t)\|_2^2$$

직관: 노이즈가 많이 섞인 단계든 거의 깨끗한 단계든, "최종적으로 어디로 갈 것인가"는 task semantic이 결정 — 그게 timestep 따라 흔들리면 안 됨.

> **예상 질문**: 이게 단순히 high-noise step에서의 학습을 강제하는 것과 다른가?  
> **답변**: 일반 flow-matching loss는 각 step의 conditional velocity를 fit하지만, EC는 **다른 step들끼리의 합의**를 강제 — 시간축 전체에 걸쳐 same target action에 수렴하도록 함.

### 2.4 Observational Consistency (OC)

**Gradient-based adversarial perturbation**을 vision feature와 robot state에 적용:
- $\delta^* = \arg\max_{\|\delta\| \le \epsilon} \mathcal{L}(s + \delta, a)$ (PGD-style).
- 정책 예측이 perturb 입력과 clean 입력에서 일관되도록:
$$\mathcal{L}_{\text{OC}} = \|\hat{v}(s + \delta^*) - \text{stop\_grad}(\hat{v}(s))\|_2^2$$

stop-gradient 사용은 perturb branch만 학습 — clean branch가 본 모델로 collapse하지 않도록.

### 2.5 총 손실

$$\mathcal{L} = \mathcal{L}_{\text{FM}} + \lambda_{\text{EC}} \mathcal{L}_{\text{EC}} + \lambda_{\text{OC}} \mathcal{L}_{\text{OC}}$$

IC는 데이터 sampling으로 적용되어 loss에 없음.

---

## 3. 데이터셋

| 데이터 | 규모 |
|--------|------|
| LIBERO-Plus | 15,874 demo (4 suites 병합) |
| RoboTwin 2.0 | 2,500 clean + 25,000 randomized demo |
| 실제 로봇 | 125 trajectory (25 × 5 task) |
| Instruction paraphrase | ~15/trajectory via Qwen3-8B |

**LIBERO-Plus는 perturbation-augmented LIBERO** — language/camera/light/background variation이 포함되어 robustness 평가에 적합.

---

## 4. 학습 세부사항

| 항목 | 값 |
|------|-----|
| Optimizer | AdamW |
| Peak LR | 1e-4 (5% warmup, cosine decay) |
| Steps | 60K (LIBERO-Plus, real-world) / 120K (RoboTwin 2.0) |
| Hardware | NVIDIA H20 GPUs (수량 미명시) |
| Inference | 8 denoising step, forward Euler |

NVIDIA H20는 중국 시장용 GPU — Sun Yat-sen University 환경과 일치.

---

## 5. 실험 결과

### 5.1 LIBERO-Plus Zero-Shot (perturbation은 학습 미포함)

| 모델 | Overall | Language | Camera | Light | Background |
|------|---------|----------|--------|-------|-----------|
| OpenVLA-OFT | 69.6 | 79.5 | 56.4 | 88.7 | 93.3 |
| **RoVLA** | **74.3** | **92.9** | 58.4 | **95.6** | **95.0** |

Language perturbation에서 **+13.4%p** — IC의 직접 효과. Camera만 baseline과 비슷(58.4 vs 56.4) — 시점 변화는 IC/EC/OC만으로는 충분치 않음.

### 5.2 LIBERO-Plus Fine-Tuned on Perturbation

| 모델 | Overall | Language | Camera | Light | Background |
|------|---------|----------|--------|-------|-----------|
| GR00T-N1.6 | 79.4 | 80.1 | 92.6 | 93.6 | 95.4 |
| **RoVLA** | **82.0** | **91.5** | **96.6** | **95.9** | **96.1** |

Perturbation 데이터로 fine-tune하면 Camera도 96.6%까지 회복. Language는 zero-shot에서도 92.9%로 강해 fine-tune의 추가 마진이 작음.

### 5.3 RoboTwin 2.0

| Environment | π0.5 | InternVL3.5+DiT (no consistency) | **RoVLA** |
|-------------|------|---------------------------------|----------|
| Clean | 43.0 | 44.9 | **48.2** |
| Randomized | 43.8 | 45.4 | **50.0** |

흥미로운 점: **randomized > clean** for RoVLA — perturbation에 강해지면서 오히려 randomized에서 더 잘함. **반과적합 효과**.

### 5.4 실제 로봇 (5 task, 10 trial)

| Task | RoVLA | GR00T-N1.6 |
|------|-------|-----------|
| Pick Up Banana | 80 | 80 |
| Pick Up Apple | 70 | 50 |
| Put Banana in Bowl | 80 | 70 |
| Put Apple in Bowl | 50 | 40 |
| Put Apple in Drawer | 20 | 10 |
| **Overall** | **60** | 50 |

저-data regime(task당 25 demo)에서 +10%p — paraphrase robustness가 real-world에서도 유효함을 시사.

---

## 6. 어블레이션 (LIBERO-Plus)

| 구성 | Overall | 핵심 효과 |
|------|---------|----------|
| Baseline (no constraint) | 77.1 | |
| + EC only | 78.2 | flow-matching 시간 일관성 단독 |
| + IC + EC | 80.5 | **Language 68.4 → 89.5** |
| + EC + OC | 79.0 | |
| **+ IC + EC + OC (Full)** | **82.0** | balanced robustness |

- **IC의 marginal 효과가 가장 큼** (특히 Language split).
- EC alone은 약하지만 IC와 결합 시 시너지.
- OC는 background/light에서 효과 — adversarial perturb가 visual augmentation 역할.

---

## 7. 한계

1. **Camera robustness가 zero-shot에서 약함** (58.4%) — IC/EC/OC는 view-invariance를 명시적으로 다루지 않음. View augmentation 또는 multi-view input이 필요할 듯.
2. **OC의 ε hyperparameter 분석 부재** — 너무 크면 clean 성능 저하, 너무 작으면 효과 없음. 논문은 단일 값만 보고.
3. **Paraphrase quality는 Qwen3-8B에 의존** — 잘못된 paraphrase("pick the apple" → "throw the apple")는 학습을 망가뜨림. Filtering 절차가 명시되지 않음.
4. **Contact-heavy task에서 약점**: 정밀한 양손 협조나 contact-rich 조작은 consistency 제약만으로 부족 — 논문도 인정.
5. **GPU 수량 미공개** — H20 클러스터 규모를 알 수 없어 reproducibility 비용 추정 어려움.
6. **CALVIN, SimplerEnv 미평가** — 전반적인 generalization 입증에 제한.

---

## 8. 예상 질문

| # | 질문 | 핵심 답변 |
|---|------|----------|
| 1 | IC를 단순 paraphrase data augmentation으로 환원할 수 있지 않나? | 정확히 그렇게 구현됨 (explicit loss term이 없음). 그러나 ~15 paraphrase × 60K step × Qwen3 quality라는 결합이 핵심 — 단순 synonym replacement보다 풍부함. |
| 2 | EC의 timestep $\tau_1, \tau_2$는 어떻게 sample? | 논문은 uniform sampling을 시사하나 detail은 한정적. 두 timestep이 너무 가까우면 trivially satisfied, 너무 멀면 학습이 불안정 — 실용적으로 stratified sampling이 합리적. |
| 3 | OC의 adversarial perturbation cost는? | 매 step 추가 forward+backward (PGD-style) → 학습 비용 거의 **2×**. 60K step이 사실상 120K equivalent compute. |
| 4 | randomized > clean (RoboTwin)은 우연 아닌가? | 1.8%p 차이는 noise일 수 있으나, baseline InternVL3.5+DiT에서도 같은 경향(44.9 vs 45.4) — RoboTwin 2.0 randomized data의 양(25K vs 2.5K)이 더 많은 것이 큰 요인. 즉 robustness 효과 + data 효과의 합. |
| 5 | LIBERO-Plus camera 58.4%는 왜 zero-shot에서 약한가? | OC는 feature-level perturbation을 가하지만 카메라 view shift는 **geometric** 변화 — feature-level adversarial은 이를 표현하지 못함. 본격적인 view robustness엔 multi-view encoder나 NeRF-style augmentation이 필요. |
| 6 | InternVL3.5-2B는 왜 첫 16층만 쓰나? | Decoder의 깊은 층은 highly task-specific해 manipulation에 noise. 첫 16층은 visual-language alignment에 집중 → action에 더 유용한 representation. Layer-pruning은 inference cost 절감 효과도. |

<!-- VERIFIED: pdf -->
