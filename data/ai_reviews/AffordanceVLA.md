# AffordanceVLA: A Vision-Language-Action Model Empowering Action Generation through Affordance-Aware Understanding

> **한 줄 요약**: PKU/HKUST(GZ)/CUHK 공동연구로, PaliGemma backbone 위에 **Mixture-of-Transformer (MoT) 3-expert (Understanding / Affordance Generation / Action)** 구조를 얹고, affordance를 *Which2Act (object grounding) + Where2Act (2D map) + How2Act (3D shape + layout)* 세 가지 sub-task로 분해해 **task-oriented intermediate supervision**으로 internalize. 3-stage progressive curriculum (referential grounding → InternData-A1 co-training → target-task post-training)으로 학습하여 LIBERO 95.8%, CALVIN ABC→D Avg.Len 4.33, real-world basic 88.3% / complex 82.9% 달성.

> ⚠️ **혼동 주의**: VLA-Tracker에는 비슷한 이름의 두 모델이 이미 존재함.
> - **AffordVLA** (Zhejiang Grasp Lab, arXiv 2605.17517): π0.5 base + frozen teacher → cosine alignment.
> - **Afford-VLA** (Fudan/KAUST, arXiv 2605.24203): Qwen3-VL base + learnable `<AFF>` query → mask decoder.
> - **AffordanceVLA** (본 review, PKU/HKUST(GZ)/CUHK, arXiv 2606.06155): PaliGemma + MoT 3-expert + Which/Where/How2Act.
> 세 연구 모두 affordance를 다루지만 architecture와 supervision 방식이 다른 독립 paper.

---

## 1. 배경 및 동기

### 기존 VLA의 구조적 gap
- VLM은 vision-language를 **2D semantic space**에 정렬하도록 사전학습되어 있음.
- 그러나 robot action은 본질적으로 **3D physical space**의 표현.
- 직접적인 end-to-end mapping은 이 gap을 좁히기 어려움.

### 기존 intermediate representation의 한계
| 계열 | 예시 | 문제점 |
|------|------|--------|
| Video / visual foresight | UniPi, CLOVER, VPP, Seer | 픽셀-수준 dense signal → 정보 redundancy + rollout 비용 |
| 텍스트 rationale / CoT | CoT-VLA, ThinkAct | 너무 coarse → "어디"와 "어떻게"를 못 잡음 |
| Sparse contact points + external planner | RAM, Robo-ABC | open-loop, long-horizon에 brittle |

### 본 논문의 가설
**Affordance**(객체의 actionable possibility)는 "**무엇을(which) / 어디서(where) / 어떻게(how)**" 상호작용할지를 자연스럽게 표현하므로:
- vision의 spatial grounding,
- language의 semantic conditioning,
- action의 execution guidance
를 **동시에** coupling하는 ideal intermediate representation.

📌 [Figure 1 삽입] — 3-expert MoT 구조와 3-stage curriculum의 진행 과정.

---

## 2. 방법론 심층 분석

### 2.1 MoT 아키텍처: 세 명의 전문가

```
Observation O_t, instruction l, proprio s_t
              ↓
   ┌────────────────────┐
   │ Understanding Expert │   M_und(O_t, l) → h_und_t
   │  (SigLIP + PaliGemma)│
   └─────────┬──────────┘
             │  (UAA causal attention)
             ↓
   ┌────────────────────┐
   │  Affordance Gen.    │   M_gen(h_und_t) → Â_t
   │  Expert (queries)   │   = {Which, Where, How}
   └─────────┬──────────┘
             │
             ↓
   ┌────────────────────┐
   │   Action Expert     │   M_act(h_und_t, Â_t, s_t)
   │  (diffusion / flow) │   → a_{t:t+k}
   └────────────────────┘
```

핵심 설계:
- **MoT** [Liang et al. 2024]: 각 expert가 독립적인 transformer block을 가지되 attention은 공유.
- **UAA Progressive Attention**: intra-expert는 bidirectional, **inter-expert는 strict causal** (Affordance → Understanding만, Action → 둘 다). Action 정보가 Affordance로 leakage되는 것을 차단.

### 2.2 Which2Act — object-centric grounding

언어 의도를 **특정 시각 entity**에 anchor:

1. Target bounding box를 GT로 crop.
2. Frozen pre-trained encoder (**Flux VAE** [Black Forest Labs 2025])로 visual latent `z_q ∈ R^{C×H×W}` 추출.
3. Which2Act query가 latent `ẑ`를 reconstruct.
4. MSE loss:

```
L_which = (1 / (C·H·W)) · Σ ||ẑ_{c,h,w} − z_{q,c,h,w}||²
```

> ❓ **왜 mask가 아니라 latent reconstruction인가?**
> Mask는 binary supervisory signal이지만 latent는 **continuous & rich** → 더 informative한 학습 signal. 또한 background 정보를 자연스럽게 억제하면서 target의 visual identity 보존.

### 2.3 Where2Act — 2D interaction localization

상호작용 hotspot을 pixel 단위로 예측:

1. 1D query token을 lightweight Transformer decoder가 받아 spatial position embedding을 query로 사용 → cross-attention.
2. Spatial logits `ŷ ∈ R^{H_t × W_t}`를 GT mask `M ∈ [0,1]^{H_t × W_t}`와 정렬.
3. Pixel-wise BCE:

```
L_where = -(1/(H_t·W_t)) · Σ [M_i·log σ(ŷ_i) + (1−M_i)·log(1−σ(ŷ_i))]
```

> 💡 **핵심 통찰**: 1D compressed query를 "**unfold**"해서 2D hotspot으로 decode → semantic intent를 explicit contact point guidance로 변환.

### 2.4 How2Act — 3D geometric reasoning

두 갈래로 분기:

**(a) Shape generation** (conditional diffusion):
```
L_shape = E_{t~U(0,T), ε~N(0,I)} [|| ε − ε̂_θ(x_t, t, h̄_shape) ||²]
```
- Iterative Transformer denoiser가 target의 3D shape latent 생성.
- SAM-3D [Ravi et al. 2025] 라벨을 GT로 사용.

**(b) Layout regression** (10-DoF: rotation + scale + translation):
```
L_layout = (1/10) · Σ SmoothL1(ŷ_layout^{(j)}, y_layout^{(j)})
```
- MLP head가 직접 regression.

→ Shape는 **what the object looks like in 3D**, layout은 **where & how oriented**. 둘이 합쳐 6-DoF 이상의 manipulation에 필요한 spatial prior 공급.

### 2.5 학습 signal의 시너지

세 head는 **shared learnable query**와 bidirectional intra-expert attention으로 jointly refine. 이것이 단순 multi-task augmentation과 결정적으로 다른 점 (Block-wise ablation에서 입증, §4).

> ❓ **예상 질문**: 왜 cascade(Which→Where→How)가 아닌 parallel인가?
> **답변**: Cascade는 upstream 오류가 downstream으로 그대로 전파(error compounding). Parallel + shared attention은 각 dimension이 서로를 보강하면서도 한 head가 실패해도 **graceful degradation** (ablation §4.2.2: 어느 한 head 제거 시 catastrophic 아님).

---

## 3. 데이터 전략 & 3-Stage Curriculum

| Stage | 데이터 | Trainable | Loss |
|-------|--------|-----------|------|
| **I. Pre-training** | AGD20K + RefSpatial (referential grounding); PRISM (interaction-aware scene) | Affordance Gen. Expert + queries + decoders | 0.1·L_which + 0.1·L_where + 0.1·L_shape + 0.04·L_layout |
| **II. Co-training** | InternData-A1 (large-scale synthetic robotic, +100k auto-annotated affordance labels) | Understanding + Affordance Gen + Action (Vision Encoder는 LR ↓ fine-tune) | 1.0·L_act + 0.5·L_afd |
| **III. Post-training** | LIBERO / CALVIN / DROID + in-house | 동일 | 1.0·L_act + 0.15·L_afd |

### 3.1 자동화된 affordance annotation pipeline

InternData-A1에 부족한 dense affordance label을 채우기 위해 설계:
1. Rule-based keyframe 추출 (action sequence 기반).
2. **Claude Opus 4.5**가 global instruction을 per-keyframe sub-instruction으로 decompose.
3. **Qwen3-VL**이 각 keyframe → (detection category, spatial affordance query) 변환.
4. **RexOmni** (PRISM에서 fine-tuned) + **SAM** + **SAM-3D**가 최종 affordance annotation 생성.
5. → 100k+ affordance label 자동 생산.

> 💡 **의의**: VLA가 LLM/VLM tool을 활용해 자기 학습 신호를 합성하는 **self-bootstrapping** 패턴. Affordance label의 scarcity가 더 이상 병목이 아님.

### 3.2 Loss weight annealing

- Stage II에서 `λ_afd = 0.5` → affordance와 action을 거의 동등하게.
- Stage III에서 `λ_afd = 0.15` → action에 무게 이동. 단, **0으로 보내지 않음**: affordance objective가 backbone의 vision-language 능력을 erode되지 않도록 anchor 역할 유지.

---

## 4. 실험 결과

### 4.1 LIBERO (Table 1, 50 rollouts/suite)

| Method | Spatial | Object | Goal | Long | **Avg** |
|--------|---------|--------|------|------|---------|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| SpatialVLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| CoT-VLA | 87.5 | 91.6 | 87.6 | 69.0 | 83.9 |
| ThinkAct | 88.3 | 91.4 | 87.1 | 70.9 | 84.4 |
| π0 | 98.0 | 96.8 | 94.4 | 88.4 | 94.4 |
| GR00T-N1 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| F1-VLA | 98.2 | 97.8 | 95.4 | 91.3 | 95.7 |
| **AffordanceVLA (w/o S2)** | 88.5 | 91.7 | 91.3 | 73.3 | 86.2 |
| **AffordanceVLA (full)** | **98.6** | **98.4** | **96.2** | 89.8 | **95.8** |

- 비교 대상 중 1위, F1-VLA(95.7%)와 박빙.
- LIBERO-Long에서만 π0(88.4)/F1-VLA(91.3) 대비 약간 부족(89.8) — long-horizon은 explicit memory가 필요하다는 paper의 자체 인정.

### 4.2 CALVIN ABC→D (Table 2, 1000 rollouts, zero-shot OOD)

| Method | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | **Avg.Len** |
|--------|-----|-----|-----|-----|-----|-------------|
| π0 | 93.8 | 85.0 | 76.7 | 68.6 | 60.1 | 3.84 |
| Seer-Large | 96.3 | 91.6 | 86.1 | 80.3 | 74.0 | 4.28 |
| **AffordanceVLA (w/o S2)** | 93.4 | 84.7 | 75.4 | 68.1 | 58.9 | 3.81 |
| **AffordanceVLA (full)** | **96.8** | **92.0** | **87.5** | **80.8** | **75.9** | **4.33** |

- ABC→D는 학습 환경(ABC)에서 본 적 없는 환경 D로 zero-shot transfer → **OOD generalization** 지표.
- Stage II 유무가 결정적: w/o S2 (3.81) → full (4.33), +0.52 길이.

### 4.3 Real-world (Table 5, 15 trials/task)

| Task 유형 | π0 | AffordanceVLA |
|-----------|-----|---------------|
| **Basic Avg** (close microwave/safe + pick by color/shape/object) | 70.8% | **88.3%** |
| **Complex Avg** (Drawer pick/close, Toaster pick/toast, Pick all rubbish) | 44.8% | **82.9%** |

특히:
- **Drawer (close)**: π0 40.0% → AffordanceVLA **100.0%**.
- **Toaster (toast)**: π0 26.7% → AffordanceVLA **86.7%**. π0가 "press button" 명령을 무시하고 pick-and-place처럼 행동하는 instruction-following 약점을, affordance grounding이 해결.
- **Pick all rubbish (continuous)**: 3rd execution에서 π0 6.7% → AffordanceVLA 46.7%, empty pick 33회 → 11회로 절감 → emergent long-horizon.

### 4.4 Ablation (Table 3)

#### Architecture & Training (Q2, Q3)

| Variant | LIBERO Avg | CALVIN Avg.Len |
|---------|-----------|----------------|
| **No-Afd (π0 Arch on same Stage II data)** | 92.4 | 3.93 |
| **Frozen-Afd** (Stage I 후 affordance expert 동결) | 67.1 | 2.83 |
| **w/o Stage II** | 86.2 | 3.81 |
| **Full** | **95.8** | **4.33** |

- **No-Afd**가 vanilla π0(94.4 / 3.84) 대비 marginal 개선만 → 데이터 양 자체가 핵심이 아님.
- **Frozen-Afd**가 catastrophic 붕괴 (67.1, 2.83) → affordance는 **co-optimize**되어야 함. 외부 prior로 주입되면 control space와 mismatch.
- → **MoT decoupled co-optimization**이 진짜 ingredient.

#### Affordance Representation (Q1)

| Variant | LIBERO Avg | CALVIN Avg.Len |
|---------|-----------|----------------|
| w/o Which2Act | 94.6 | 4.20 |
| w/o Where2Act | 93.2 | 4.13 |
| w/o How2Act | 93.7 | 4.01 |
| Full | **95.8** | **4.33** |

- 어떤 head를 제거해도 catastrophic 아니고 graceful degradation → cascade가 아니라 jointly refined evidence.
- How2Act는 LIBERO/CALVIN(simple tabletop)에선 modest하지만 real-world 6-DoF에선 결정적 (Table 5 complex tasks).

#### Same-Density Control: Block-wise Tokens

| Variant | LIBERO Avg | CALVIN Avg.Len |
|---------|-----------|----------------|
| **Block-wise** (intra-affordance attention block 차단) | 90.3 | 3.89 |
| Full | **95.8** | **4.33** |

- **Loss term 수 & 데이터는 동일**, 오직 cross-attention만 차단 → multi-task supervision density 가설을 배제.
- → 진짜 동인은 **structured jointly-refined representation**.

### 4.5 Data efficiency (Figure 3)

- 40% downstream data만으로도 π0의 100% data 천장(LIBERO ~94, CALVIN 3.84) 돌파 (LIBERO ~92%, CALVIN >4.0).
- Affordance가 perception-action mapping을 sub-problem으로 decomposition → 샘플당 학습 signal 다중화.

---

## 5. 강점

1. **Structured intermediate supervision의 가장 명료한 형태**: "which / where / how"는 직관적이고 이론적으로 깔끔.
2. **MoT 3-expert decoupling**: 각 책임을 분리하면서 UAA causal attention으로 정보 흐름 통제 — 표현 collapse 방지.
3. **Block-wise ablation**: supervision density 가설을 정면으로 배제한 **same-density control**로 representation의 인과성을 강하게 입증.
4. **Stage II auto-annotation pipeline**: 100k+ affordance label을 LLM/VLM tool stack으로 생성. Affordance label scarcity를 정면 돌파.
5. **Real-world에서 instruction-following 입증**: 동일 시각 입력 + 다른 instruction (Drawer pick vs close) 비교로 V-L 능력 erosion 방지 가설을 isolation.
6. **Train-only intermediate**: 추론 시 affordance head decode 안 함(π0.5/π0.7 트렌드와 동기화) → deployment 비용 무관.

## 6. 약점·한계

1. **LIBERO-Long에서 F1-VLA(91.3) 대비 89.8** → long-horizon은 explicit temporal/memory module이 따로 필요. Paper도 future work로 명시.
2. **모델 파라미터·hardware·optimizer 등 학습 디테일이 main paper 본문에선 부족**. Supplementary 의존도가 높음.
3. **Affordance teacher annotation의 noise**: Claude Opus 4.5 + Qwen3-VL이 만든 sub-instruction과 spatial query의 quality 추정/검증 절차가 main paper에 미수록.
4. **Inference latency / Hz 미보고**: π0 대비 expert 3개로 늘었으니 throughput 손실이 있을 텐데, AffordVLA 같이 Hz를 명시한 비교가 없음.
5. **Affordance "anchor" 가설은 mechanistic 증거 없음**: 저자도 명시적으로 "interpretive hypothesis"라고 밝힘. Probing/representation similarity 분석은 아직 미흡.
6. **Bimanual/deformable로의 확장 미검증**: 현재는 tabletop two-finger 위주.

## 7. 후속 연구를 위한 시사점

| 방향 | 구체화 |
|------|--------|
| Explicit long-term memory | MemoryVLA류 RNN/state-space + affordance subgoals 결합 |
| Affordance teacher quality | LLM/VLM 합성 라벨의 신뢰도 calibration + active relabeling |
| Bimanual / deformable | How2Act를 multi-target 또는 SE(3) field로 확장 (A3D, GarmentPile++ 참고) |
| Mechanism verification | Affordance head의 attention probe / VLM erosion 정량 측정 (Drawer/Toaster 결과를 mechanistic하게 해석) |
| Cross-embodiment | UniVLA latent action과의 결합 — affordance가 cross-embodiment에서 더 보편 |

## 8. 다른 affordance VLA와의 비교

| 모델 | Base VLA | Affordance 표현 | Supervision 방식 | 대표 metric |
|------|----------|-----------------|------------------|-------------|
| **AffordVLA** (Zhejiang) | π0.5 | Frozen teacher representation | Cosine alignment (λ=0.5) on layer-12 feature | RoboTwin2.0 Easy 61.2 / Hard 28.8 |
| **Afford-VLA** (Fudan/KAUST) | Qwen3-VL-4B | Learnable `<AFF>` query → mask decoder | Mask BCE + action loss | Bench별로 별도 |
| **AffordanceVLA** (PKU/HKUST) | PaliGemma + MoT 3-expert | Which (latent) + Where (mask) + How (3D shape + layout) | 3-stage + UAA causal attention | LIBERO 95.8 / CALVIN 4.33 |
| CoA-VLA | OpenVLA류 | Visual-textual chain-of-affordance | External cue로 consume | — |

→ AffordanceVLA는 **세 가지 affordance dimension(2D 객체 anchor + 2D map + 3D geometry)을 jointly refine**한다는 점에서 가장 구조화된 형태.

## 9. 재현성 & 공개

- **Code**: https://github.com/Skywalker-yqz/AffordanceVLA/ — public.
- **Project page**: https://skywalker-yqz.github.io/AffordanceVLA/
- 모델 weight 공개 여부는 main paper에서 단정 안 함; repo 확인 필요.
- Stage II의 자동 annotation pipeline은 supplementary에 기술; LLM(Claude Opus 4.5) 의존 → 재현 시 다른 LLM 대체 가능 여부가 관건.

## 10. 평가 — 시뮬에서 실세계로

- LIBERO/CALVIN 같은 simple tabletop 벤치에서는 π0/F1-VLA류와 박빙이지만, **real-world Drawer/Toaster/Pick-all-rubbish의 격차**(π0 44.8 vs 82.9)가 본 paper의 진짜 결과.
- 핵심 메시지: "data scaling이 아니라 **representation engineering**이 ceiling을 깬다." Pi0가 40% data 천장에 갇히는 동안 AffordanceVLA는 같은 양으로 천장 돌파.

## 11. 결론

AffordanceVLA는 **affordance를 "외부 prior로 consume"하지 않고 VLA 내부에서 jointly co-optimize**한 가장 구조화된 시도. Which2Act + Where2Act + How2Act의 분해는 단순한 multi-task auxiliary가 아니라, **shared attention으로 서로를 promote하는 organic representation**임이 same-density control로 입증됨. 3-stage curriculum + 100k+ 자동 affordance label로 scarcity 병목 돌파. Real-world instruction-sensitivity (Drawer/Toaster)는 affordance가 VLM의 vision-language 능력을 anchor 한다는 paper의 가설에 강한 경험적 증거 제공.

남은 과제는 (1) explicit long-horizon memory, (2) annotation pipeline의 quality calibration, (3) "anchor hypothesis"의 mechanistic 검증, (4) bimanual/deformable 확장. **affordance-based VLA의 강력한 reference point**가 될 것.

## 12. Discussion Seeds (세미나 토론용 질문)

1. **Affordance vs. 3D scene representation**: How2Act의 SAM-3D shape token이 정말 manipulation을 돕는가, 아니면 explicit 3D scene reconstruction(예: NeRF/Gaussian Splatting)을 통합하는 게 더 자연스러운가?
2. **LLM-generated affordance label의 한계**: Claude Opus 4.5의 sub-instruction decomposition이 잘못된 affordance를 생성하면? Self-bootstrapping의 hallucination 누적 risk.
3. **UAA causal attention vs. fully bidirectional**: Action expert가 Affordance에 영향을 주면 안 되는가? Closed-loop reactive control에서는 action 결과가 affordance update에 도움될 수도.
4. **Block-wise ablation의 해석**: 90.3 → 95.8 (+5.5) gap이 정말 "structured jointly-refined representation"의 증거인가, 아니면 단순히 attention bandwidth 증가의 효과인가?
5. **π0.5/π0.7의 train-only intermediate supervision과의 관계**: Affordance가 정말 더 자연스러운 anchor인가, 아니면 bounding box / discrete action token처럼 다른 임의의 structured signal로도 같은 효과가 나올까?
6. **Inference Hz 트레이드오프**: 3-expert MoT가 단일 expert 대비 얼마나 느려지는가? Train-only라고 해도 affordance generation expert가 inference 경로에 있으니 latency 증가 불가피.

---

<!-- VERIFIED: pdf -->
