# SceneDiver: Breaking the Perceptual Bottleneck in Vision-Language Decision Making via Focus Plan Generation

> **한 줄 요약**: VLM/VLA의 perceptual bottleneck(object hallucination, attention drift, mis-grounding)을 **focus plan 생성 문제**로 재정의. OvSGTR scene graph로 coarse-stage 추론 → fine-stage agentic exploration(multi-scale verify / semantic zoom / local spatial search) → focus score map으로 입력 영상 modulation. 실시간 VLA용으로는 **SceneDiver Adapter**(Slot Attention + Scene-Aware MaskNet)를 OpenVLA-OFT projector 뒤에 삽입, Hungarian matching 기반 Structure/Mask Loss로 distillation. LIBERO-plus에서 최대 **+9.6% SR**, 단 **2.64% latency** 오버헤드. ICML 2026.

---

## 1. 배경 및 동기

### Perceptual Bottleneck
VLM(고수준 planner)과 VLA(reactive executor)는 역할은 다르지만 **공통의 perceptual bottleneck**을 공유:

1. **Object hallucination**: 존재하지 않는 객체 예측
2. **Object omission**: 존재하는 객체 누락
3. **Erroneous attribute binding**: 색·형태 등 속성 잘못 묶기
4. **Inaccurate instance counting**: 동일 카테고리 수량 오류

### 기존 접근의 한계
- **One-step focus** (직접 essential object에 attend): 깊은 scene 이해 없이 작동 불가
- **SoM/SAM tagging**: 마커가 시각적 noise 추가 → 오히려 reasoning 방해
- **Multi-Resolution (Monkey/LLaVA/Sphinx)**: spatial topology 손실
- **VCD (output logit 보정)**: 근본적 perceptual 부족 해결 못 함
- **Thinking-enhanced models** (doubao/gemini thinking): 추론 확장만으로는 perception 결손 보완 불가

### 핵심 아이디어
- Scene 이해는 **coarse-to-fine, agentic exploration**이 필요
- VLM의 long-term planning 능력으로 **focus plan을 자율 생성**
- 그 plan을 가벼운 adapter로 VLA에 distill → 실시간 가능

---

## 2. 방법론 심층 분석

### 2.1 Pipeline 개요

```
Input image
   → (1) OvSGTR scene graph 구축
   → (2) Coarse stage: graph reasoning으로 sub-scene 분해
   → (3) Fine stage: 3가지 agentic primitive로 verify/zoom/search
   → (4) Focus score map s 생성 + 영상 modulation
   → VLM 의사결정
```

### 2.2 Coarse Stage: Graph Reasoning

- **OvSGTR**(Chen et al. 2024b)로 그래프 생성. 노드=object, 엣지=spatial relation
- 텍스트 포맷으로 VLM에 주입: `<ref>` (객체 ID), `<pred>` (관계), `<box>` (좌표)
- VLM이 task-relevant sub-scene 자율 선택
- VLM이 **single source of truth**, graph는 **reasoning guide** — 불일치 시 VLM이 자유롭게 폐기/유지

### 2.3 Fine Stage: 3가지 Agentic Primitive

| 동작 | 트리거 | 행동 |
|---|---|---|
| **Multi-scale verification** | local view에 target 발견 | 후보 set에 추가 |
| **Semantic zoom** | 시각 증거 모호 | FoV 좁혀 fine-grained 확인 |
| **Local spatial search** | graph-image misalignment로 target 부재 | 이웃 영역 스캔으로 재획득 |

→ Hard-coded rule 없이 VLM 자체 능력만으로 verified set C 구성.

### 2.4 Focus Score Map + Soft Modulation

후보 박스 `{b_k}_{k∈C}` 를 픽셀맵 `s ∈ [0,1]^{H×W}` 로 래스터화. visibility floor `β` 사용:

```
I_dim = I ⊙ (β + (1-β) s)            (1)
I_out = s ⊙ I_dim + (1-s) ⊙ B_σ(I_dim)   (2)
```

→ Foreground는 sharp/bright, background는 dim/blur. Context 보존하되 distractor 억제.

### 2.5 SceneDiver Adapter (VLA용 distillation)

**위치**: visual projector 뒤, LLM 앞 — cross-modal alignment 공간.

#### Task-Guided Slot Attention
- 시각 토큰 `F ∈ R^{L×D}` (D=4096) → object-centric slot `S ∈ R^{K×d_s}` (d_s=256)
- **Task-Conditional Init**: `v_task = Σ w_i h_i` (학습된 attention pooling)
- `S0_k ~ N(Proj(v_task) + δ_k, σ_global)` — task별 의미적 초기화
- T=5 iteration, temperature τ=0.4 (sharpening), GRUCell update

#### Scene-Aware MaskNet (coarse-to-fine)
1. **Slot-level relevance** `r_k = MLP([LN(s_k); Proj(m_k); Proj(v_task)])` (slot semantic + slot mass m_k + task)
2. **Patch-level residual** `Δ_l` (slot back-projection f_l = Σ A_kl s_k + task context)
3. **Final mask**: `M_l = Σ r_k A_kl + α · Δ_l` (α 초기 ≈ 0; 학습 진행에 따라 fine refinement 증가)
4. Learnable scale `exp(β)` (β_init=1.1 → ~3) — sigmoid 0.5 부근 모호성 방지

#### Distillation 손실
- **Structure Loss**: Hungarian matching으로 slot ↔ scene-graph object alignment
- **Mask Loss**: pixel-level supervision
- **Entropy-based dynamic gating**: ambiguous-patch uncertainty 초과 시 masking 우회 → raw observation을 VLA에 직접 공급 (graceful degradation)

---

## 3. 데이터/시스템

| 항목 | 값 |
|---|---|
| VLA base | OpenVLA-OFT |
| Scene graph generator | OvSGTR (Chen et al. 2024b) |
| Train data | LIBERO joint (spatial+object+goal+long, 클린) |
| OvSGTR train data | AI2-THOR / MuJoCo synth RGB + instance mask + <s,p,o> triples (image-topology + 3D-geometry) |
| Latency 측정 | NVIDIA RTX 4090, 10,000 action 평균 |
| VLM frontends 평가 | Qwen2.5-VL-7B/32B-AWQ, InternVL2.5-8B, gpt-4o-mini, gemini-2.5-flash, doubao-seed-1.6-flash |

---

## 4. 실험 결과 (Paper Tables 1-10 직접 확인)

### 4.1 LIBERO-plus (Table 3, 3 seeds, **zero-shot from clean LIBERO**)

5축 perturbation: Objects Layout / Camera Viewpoints / Robot Initial States / Background Textures / Light Conditions

| Suite | Layout | Camera | Initial | Background | Light |
|---|---|---|---|---|---|
| **Spatial** | **94.43 (+2.04)** | 54.35 (+5.19) | 29.49 (+9.58) | 87.41 (+4.74) | 97.37 (+3.51) |
| **Object** | 76.82 (+0.81) | 64.56 (+2.06) | 15.72 (+1.36) | 90.61 (+0.87) | **99.50 (+0.51)** |
| **Goal** | 52.53 (+3.17) | 52.74 (+1.72) | 18.85 (+6.29) | 91.09 (+9.31) | 97.35 (+3.41) |
| **Long** | 74.52 (**+9.24**) | 42.97 (+2.84) | 37.73 (+5.41) | 90.00 (+8.93) | 92.42 (+7.95) |

→ 모든 20개 cell에서 OpenVLA-OFT 대비 향상. **Long-horizon에서 가장 큰 향상** (Layout +9.24, Background +8.93, Light +7.95) — error accumulation을 attention 안정화로 완화.

### 4.2 MuJoCo Robotic Manipulation (Table 1)

decoy board + decoy cube distractor 환경, 30 scenes × 5 seeds:

| VLM | Base | + SceneDiver |
|---|---|---|
| Qwen2.5-VL-7B-AWQ | 14.7 | **28.7** (+14.0) |
| Qwen2.5-VL-32B-AWQ | 21.3 | **31.3** (+10.0) |
| gpt-4o-mini | 28.7 | **34.0** (+5.3) |
| gemini-2.5-flash | 38.7 | **46.7** (+8.0) |

### 4.3 Modified Room Navigation (Table 2)

4 sub-tasks: Base / Common-Sense / Complex-Instruction / Visual-Appearance.

| Model | Base | CS | CI | VA |
|---|---|---|---|---|
| Qwen2.5-VL-7B Base | 32.7 | 30.7 | 32.0 | 27.3 |
| Qwen2.5-VL-7B + Ours | **44.0** | **36.0** | **37.3** | **35.3** |
| gemini-2.5-flash Base | 68.0 | 62.0 | 60.0 | 55.3 |
| gemini-2.5-flash + Ours | **74.7** | **65.3** | **66.0** | **62.0** |

SoM, Multi-Resolution, VCD, thinking-enhanced baseline 모두 능가.

### 4.4 Scene Graph Stress Test (Table 5, 500 scenes)

수동으로 노드 일부 drop:

| Drop Ratio | Recovery (↑) | Hallucination (↓) | Miss (↓) |
|---|---|---|---|
| 10% | 96.78 | 1.52 | 1.70 |
| 20% | 98.01 | 1.23 | 0.76 |
| 30% | 97.16 | 1.89 | 0.95 |
| 50% | 97.69 | 1.02 | 1.29 |

→ Scene graph가 50%까지 손실되어도 fine-stage exploration이 안정적으로 복구.

### 4.5 Latency Profiling (Table 10, RTX 4090)

| Module | Mean | P95 | P99 |
|---|---|---|---|
| Action (Total) | 114.45 ms | 116.50 | 121.78 |
| **Slot Attention** | **2.18 ms** | 2.30 | 2.33 |
| **MaskNet** | **0.83 ms** | 0.86 | 0.87 |
| Adapter 합계 | 3.01 ms = **2.64%** | — | — |

→ 실시간 deployment 가능.

---

## 5. Ablation 핵심 (Table 4)

| Ablation | 내용 | 결과 |
|---|---|---|
| **A1: Text-only scene graph** | graph를 텍스트 prompt로만 주입 | 미미한 향상 (시각 attention 분산) |
| **A2: Direct focus (no coarse-to-fine)** | scene graph 주고 target 직접 예측 | distractor 혼동에 취약 |
| **A3: Coarse only (no fine verify)** | fine stage 제거 | hallucination 발생 — fine verification이 critical |
| **End-to-end baseline (Appendix Table 6)** | image→focus 직접 예측 | 20개 LIBERO-plus 셀 모두 SceneDiver 이하 |

→ **Two-stage agentic exploration이 필수**. Coarse alone은 grounding 부정확, fine alone은 global plan 결손.

---

## 6. Related Work 비교

| 접근 | Perception 보완 | 한계 |
|---|---|---|
| SoM (Yang et al. 2023) | SAM tagging | 마커 noise로 reasoning 방해 |
| Multi-Resolution (Monkey/LLaVA/Sphinx) | 다해상도 crop | spatial topology 손실 |
| VCD (Leng et al. 2023) | output logit 보정 | 입력 시각 부족 미해결 |
| Thinking models (doubao/gemini) | 확장 추론 | perception 결손 ≠ reasoning 결손 |
| **SceneDiver** | **input image modulation + agentic explore** | spatial 보존 + selective zoom |

---

## 7. Limitations

1. **빠른 동적 scene**: 빠르게 움직이는 객체 환경에서 scene graph 생성 모듈 추가 최적화 필요
2. **계층적 planning 미통합**: 더 복잡한 multi-stage manipulation에서는 hierarchical planner와의 결합이 향후 과제
3. **VLA backbone 제한**: 본문 검증은 OpenVLA-OFT 중심 — 다른 VLA 일반화는 미확인
4. **OvSGTR 의존**: scene graph 품질이 generator 성능에 결박. 다만 stress test로 50% drop까지 견디는 robustness 입증

---

## 8. 종합 평가

| 항목 | 평점 (5점) |
|---|---|
| 혁신성 | 4.5 (focus plan 자율 생성 + agentic exploration + distillation 패러다임) |
| 재현성 | 4.0 (code/data 공개: future-item.github.io/SceneDiver) |
| 실험 폭 | 4.5 (LIBERO-plus 20 cell + MuJoCo + modified EmbodiedBench + vanilla EmbodiedBench + stress test + latency) |
| 이론적 깊이 | 4.0 (Hungarian matching, slot attention 수식 명료) |
| 실용성 | 4.5 (2.64% latency, OpenVLA-OFT plug-in) |

**총평**: VLA의 perception 한계를 representation 개선이 아닌 **focus plan 생성** 으로 우회. LIBERO-plus zero-shot 전 셀 향상 + 2.64% latency라는 강한 cost-benefit. ICML 2026 채택.

---

## 9. 예상 세미나 질문

> ❓ **OvSGTR scene graph가 틀리면 catastrophic 실패 아닌가?**
> Stress test (Table 5)에서 50% node drop에도 RR 97.69%, HR 1.02%. fine-stage local spatial search가 graph-image misalignment를 복구하도록 설계 — graph를 reasoning guide로만 다루고 VLM이 single source of truth.

> ❓ **SceneDiver Adapter는 OpenVLA-OFT에만 적용 가능한가?**
> 위치가 "projector 뒤, LLM 앞" alignment 공간이라 일반 VLA 구조에 이식 가능. 단 본 논문 실험은 OpenVLA-OFT 중심.

> ❓ **2.64% overhead라지만 K슬롯 + T=5 iteration이면 mobile robot에서는?**
> 절대 시간 3.01 ms (RTX 4090). slot K와 iteration T를 작은 backbone에 맞춰 줄이는 ablation은 미수록 — open question.

> ❓ **LIBERO-plus Long-horizon에서 +9.24가 가장 큰데 왜?**
> Long-horizon은 error accumulation에 취약 → attention drift가 시간에 따라 증폭. SceneDiver의 latent instantiation이 단계마다 attention을 task-relevant entity에 재고정 → 누적 오류 차단.

> ❓ **Entropy-based gating의 threshold는?**
> 논문 본문에서 정량값 미공개 (ambiguous-patch 기반 dynamic). graceful degradation 의도이므로 보수적 설정으로 추정.

---

## 10. 코드 & 재현

- **Code / data**: https://future-item.github.io/SceneDiver
- **Base VLA**: OpenVLA-OFT (clean LIBERO joint training)
- **Scene graph backend**: OvSGTR (Chen et al. 2024b)
- **Hardware**: NVIDIA RTX 4090 (latency profiling 기준)
- **Hyperparameters**: slot d_s=256, K slots, T=5 iter, τ=0.4, α≈0 (init), exp(β) scale (β_init=1.1)

---

## 11. 데이터셋 / 후속 연구

- **Train**: 원본 LIBERO 4 suites joint (perturbation 없는 clean 데이터만)
- **Eval**: LIBERO-plus (zero-shot), modified EmbodiedBench navigation, MuJoCo brick assembly with decoys
- **OvSGTR train**: AI2-THOR + MuJoCo synthetic RGB + instance mask + <subject, predicate, object> triple (image-topology + 3D-geometry)
- **후속 방향**: hierarchical planner 통합, dynamic scene 대응, 다른 VLA backbone 일반화, slot/iteration scaling laws

---

## 12. 결론

SceneDiver는 (1) **focus plan 생성**이라는 새로운 문제 framing, (2) **coarse-to-fine agentic exploration** (3-primitive: verify/zoom/search), (3) **Slot Attention + MaskNet adapter**를 통한 VLA distillation의 세 기여를 가진다. **LIBERO-plus 20/20 cell 향상 + 2.64% latency**의 강한 cost-benefit은 perception bottleneck을 representation 학습이 아닌 **시각 입력 modulation** 으로 해결할 수 있음을 보였다. ICML 2026.

---

<!-- VERIFIED: pdf -->
