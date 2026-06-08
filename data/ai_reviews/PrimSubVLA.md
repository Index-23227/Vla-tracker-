# PrimSubVLA: Primitive Subspaces Mediate Few-Shot Transfer in VLAs

> **한 줄 요약**: OpenVLA 7B와 π0.5에 **architectural 변경 없이** primitive-segmented data + primitive-specific 언어 prompt로 LoRA fine-tune하면, test time에 m=3 시범만으로 fine-tune upper-bound의 78–80%를 달성. Linear probe로 식별한 layer-24의 **4-D primitive-decodable subspace**를 inference time에 orthogonal projection으로 제거하면 success가 32 pp 폭락(0.62→0.30) — random subspace ablation은 2 pp만 변화 → primitive 표현이 few-shot transfer에 **causally necessary**임을 입증. 추가로 chunked policy 평가의 family-wise inflation(13.4× 구조적 false-failure) 결함과 보정 gate 제안.

---

## 1. 배경 및 동기

### VLA의 산업 배포 병목
- 현재 VLA 배포 recipe: task별 demo 수집 → fine-tune → 배포. 비용이 **task 수에 선형**.
- 산업 환경에서는 product variant, fixture configuration, process step이 계속 바뀜 → 재학습 사이클마다 downtime / data collection / infra cost 누적.

### 핵심 질문
> VLA가 **weight update 없이** test time에 소수 demonstration만으로 새 task를 학습할 수 있는가? 그렇다면 무엇이 그것을 가능케 하는가?

### 가설
**Training data 구조가 답을 결정한다.** Primitive-level 명시적 supervision으로 학습된 VLA는 sub-skill을 task context에서 disentangle한 compositional representation을 학습 → demonstration-conditioned recombination이 inference time에 가능.

---

## 2. 방법론 심층 분석

### 2.1 2×2 실험 설계 (Locked Hyperparameters)

| 아키텍처 \ 데이터 view | Flat | Primitive |
|---|---|---|
| OpenVLA-7B | OpenVLA-flat | **OpenVLA-primitive** |
| π0.5 | π0.5-flat | **π0.5-primitive** |

- **Flat view**: episode = 전체 task-pair (예: "pick and insert the round peg")
- **Primitive view**: episode = 단일 primitive segment (예: "pick the round peg")
- **동일 source demo**에서 두 view 모두 materialize → data quantity 통제
- **동일 LoRA hyperparameters**: rank=32, α=64, lr=5e-5, 30k steps, batch=32, bf16, seeds={42, 123, 456}

### 2.2 Few-Shot Transfer Protocol (No Weight Updates)

```
입력: (task 설명, m개 held-out task demo, current observation)
처리: 각 demo에서 4개 keyframe vision-encoder feature → mean-pool → 
       m개 vector concat → language prompt 앞에 prepend
출력: action sequence (weight update 없음)
```

- m ∈ {0, 1, 3, 5, 10} sweep
- 두 아키텍처 모두 기존 language input pathway로 conditioning 가능

### 2.3 Held-Out Task 구성 (6개 + 1개 OOV)

| Task | 보류된 (object, primitive) pair 수 | 성공 기준 |
|---|---|---|
| T1: bearing press-fit | 1 | center 2mm 이내, depth ≥8mm |
| T2: USB pick→insert | 2 | depth ≥9mm, rotation ≤5° |
| T3: BNC pick→insert | 2 | initial seating ≥6mm |
| T4: Ethernet pick→insert | 2 | RJ-45 stop 1mm 이내 |
| T5: waterproof full assembly | 3 | full seating + remove |
| T6: gear shaft alignment | 3 | gear 3mm + shaft 12mm depth |
| **T7 (OOV)**: threaded-fastener rotate-to-engage | rotate-to-engage가 **training vocab 부재** | 2 full rotation thread engagement |

### 2.4 Subspace-Ablation Causal Intervention

1. Layer 24 hidden state에 4-class linear probe (primitive identity) 학습
2. Probe weight matrix W ∈ ℝ^(4×d) → 4-D primitive subspace 정의
3. Orthogonal projection: P^⊥ = I - W^T(WW^T)^{-1}W
4. Inference time에 h_24 ← P^⊥ h_24 (이후 layer로 전달)
5. **Control**: 동일 차원의 random subspace (unit sphere uniform sampling + Gram-Schmidt × 10회 평균)

---

## 3. 데이터/시스템

| 항목 | 값 |
|---|---|
| Primary dataset | REASSEMBLE (4,551 demos, NIST Task Board 1, 17 objects × 4 primitives) |
| Replication dataset | LIBERO-Long (4 sub-task primitive: move-to/grasp/transport/place) |
| 보류 task당 demo | 56 (REASSEMBLE), 40–75 (LIBERO-Long) |
| FT upper bound budget | 50 demos per held-out task |
| Compute | ~3,600 H100-hours (12 main cells 1,500h + FT UB 600h + LIBERO 1,000h + probing 200h + baselines 200h + pilot 100h) |
| OpenVLA | 1× H100 |
| π0.5 | 2× H100 model parallelism |

---

## 4. 실험 결과 (Paper Tables 직접 확인)

### 4.1 REASSEMBLE Few-Shot Curve (Table 2, mean ± SD, 3 seeds, 50 rollouts)

| Cell | m=0 | m=1 | m=3 | m=5 | m=10 |
|---|---|---|---|---|---|
| OpenVLA-flat | 0.18±0.04 | 0.24±0.05 | 0.34±0.06 | 0.42±0.05 | 0.61±0.04 |
| **OpenVLA-primitive** | **0.27±0.05** | **0.41±0.04** | **0.62±0.04** | **0.71±0.04** | **0.78±0.03** |
| π0.5-flat | 0.15±0.03 | 0.22±0.04 | 0.31±0.05 | 0.39±0.04 | 0.58±0.05 |
| **π0.5-primitive** | **0.31±0.04** | **0.44±0.05** | **0.66±0.04** | **0.74±0.03** | **0.81±0.03** |
| OpenVLA FT-UB (50 demos) | 0.79±0.03 | | | | |
| π0.5 FT-UB (50 demos) | 0.82±0.03 | | | | |

- **Primitive m=3가 Flat m=10을 이미 초과** → primitive supervision의 demo-equivalent 가치 ≈ 7 demos
- m=5에서 FT-UB과의 gap이 8 pp만 남음 (가장 deployment-relevant 숫자)
- Seed range 비중첩: OpenVLA-primitive [0.58, 0.66] vs OpenVLA-flat [0.28, 0.40]

### 4.2 LIBERO-Long Cross-Dataset Replication (Table 3, m=3)

| Cell | REASSEMBLE | LIBERO-Long |
|---|---|---|
| OpenVLA-flat | 0.34±0.06 | 0.42±0.05 |
| **OpenVLA-primitive** | **0.62±0.04** | **0.71±0.04** |
| π0.5-flat | 0.31±0.05 | 0.45±0.05 |
| **π0.5-primitive** | **0.66±0.04** | **0.74±0.03** |
| Prim/Flat ratio | 1.99× | 1.66× |

### 4.3 Compositional Distance별 Decomposition (Table 6, m=3)

| 보류 pair 수 | Task | OpenVLA-prim | OpenVLA-flat | π0.5-prim | π0.5-flat |
|---|---|---|---|---|---|
| 1 | bearing press-fit | 0.86 | 0.51 | 0.88 | 0.49 |
| 2 | USB/BNC/Ethernet | 0.61 | 0.32 | 0.65 | 0.30 |
| 3 | waterproof/gear shaft | 0.42 | 0.18 | 0.45 | 0.16 |

→ **Distance가 클수록 advantage 증가** (absolute는 감소하지만 ratio 2–3× 유지)

### 4.4 Out-of-Vocabulary 역전 결과 (§4.4, m=5)

| Cell | rotate-to-engage 성공률 |
|---|---|
| OpenVLA-flat | **0.14±0.04** |
| OpenVLA-primitive | 0.08±0.03 |

→ Training vocab에 없는 primitive type을 요구하는 task에서는 **primitive-trained가 underperform**. 가설: nearest known primitive로 강제 mapping → 새 behavior 학습 실패.

### 4.5 Subspace-Ablation Causal Intervention (Table 4, m=3)

| Cell | No intervention | Random subspace | **Primitive subspace** |
|---|---|---|---|
| OpenVLA-primitive | 0.62±0.04 | 0.60±0.05 | **0.30±0.06** (-32 pp) |
| π0.5-primitive | 0.66±0.04 | 0.64±0.04 | **0.34±0.05** (-32 pp) |

- **Differential effect** (random − primitive): 0.30 pp 차이 → **causal necessity 확립**
- **Intervention specificity**: training-distribution task에서는 0.84→0.79로 5 pp만 감소 → held-out에서만 선택적으로 작동
- Subspace dimensionality k=4 sweep (Table 12): k=4에서 saturation, k=2(0.46)에서 부분적, k=8/16에서 0.27 plateau

### 4.6 Linear Probe Accuracy (Table 10, macro-F1)

| Cell | Layer 8 | Layer 16 | **Layer 24** | Final |
|---|---|---|---|---|
| OpenVLA-flat | 0.34 | 0.48 | 0.52 | 0.49 |
| **OpenVLA-primitive** | 0.41 | 0.74 | **0.81** | 0.77 |
| π0.5-flat | 0.31 | 0.44 | 0.51 | 0.47 |
| **π0.5-primitive** | 0.39 | 0.71 | **0.79** | 0.75 |

- 두 아키텍처 모두 **같은 layer 위치**에서 emerge → architectural portability
- Few-shot 성공률과 probe accuracy 상관계수 **r = 0.73**

### 4.7 Cross-Object Same-Primitive Cosine Similarity (Table 11, layer 24)

| Cell | Pick | Insert | Remove | Place | Mean |
|---|---|---|---|---|---|
| OpenVLA-flat | 0.46 | 0.41 | 0.45 | 0.42 | 0.43 |
| **OpenVLA-primitive** | **0.74** | **0.69** | **0.72** | **0.69** | **0.71** |

- Cross-primitive same-object: prim 0.31 / flat 0.58 → **primitive identity가 object identity를 dominate**

### 4.8 Baselines 비교 (Table 5, m=3)

| Method | OpenVLA | π0.5 |
|---|---|---|
| Zero-shot primitive sequencing | 0.27 | 0.31 |
| Flat few-shot (ours) | 0.34 | 0.31 |
| Diffusion Policy (CNN) | 0.36 | — |
| **Octo-style demo-conditioned** | 0.41 | 0.44 |
| **Primitive few-shot (ours)** | **0.62** | **0.66** |
| Full fine-tune UB (50 demos) | 0.79 | 0.82 |

- **Octo-style baseline과의 21–22 pp gap이 load-bearing**: 동일 demo-conditioning mechanism이지만 primitive supervision 없음 → 차이가 곧 primitive supervision의 순수 기여.

### 4.9 Chunked-Policy Evaluation Gate (§4.1, Appendix E)

- Legacy 3σ frame-level gate: π0.5 16-step chunk가 ground-truth REASSEMBLE 데이터에서 **42% rate로 false-failure**
- 이론: p^DK = 0.997^128 ≈ 0.68 → 32% 기대 failure → **13.4× 구조적 inflation vs single-step**
- 보정 gate: element-level v_model 통계량 + |v_model − v_ref| ≤ τ=0.01 (Gaussian 가정 없이 reference batch 대비)
- 보정 후 pass rate: π0.5-flat 36.0% → 1.39%, π0.5-prim 28.0% → 0.95%

---

## 5. Ablation 핵심

- **Primitive subspace 제거 (4-D, layer 24)**: -32 pp 폭락 (causal necessity)
- **Random subspace 제거 (4-D 동일)**: -2 pp (baseline noise 내)
- **OOV task (rotate-to-engage)**: prim 0.08 < flat 0.14 → primitive vocabulary coverage 중요
- **Demo encoding ablation** (Table 9): mean-pool 0.62 / cross-attn 0.65 / individual-frame 0.58 — robust to encoding choice
- **k sweep** (Table 12): k=4가 4-class probe와 정확히 일치, k=8/16에서 추가 효과 미미

---

## 6. Related Work 비교

| 접근 | Primitive 처리 | Few-shot 메커니즘 |
|---|---|---|
| OpenVLA / π0.5 (vanilla) | 없음 | 없음 (FT 필요) |
| RT-H | language-motion 명시적 hierarchy | 없음 |
| SayCan | pre-trained skill + LLM planner | 없음 |
| Code-as-Policies | primitive library + code gen | 없음 |
| HYDRA | hybrid hierarchical action | 없음 |
| Octo | demo token context | trained end-to-end |
| RT-2 | 없음 | limited (CoT) |
| **Primitive-aware (ours)** | **data view + prompt만** | **emergent from primitive supervision** |

→ 본 연구는 **architectural change 없이** training data 구조만으로 demonstration-conditioned composition 능력이 emerge함을 보여준 점이 차별점.

---

## 7. Limitations (저자 명시)

1. **3 seeds**: 체계적 효과 vs run variance 구별은 가능하나 per-cell σ=0.03–0.06이 uncertainty floor.
2. **Simulation only**: sim-to-real transfer 미검증.
3. **두 architecture만**: cross-architecture 일관성은 확인되었으나 더 다양한 backbone 검증 필요.
4. **Object 변동 제한**: 기존 object set에서만 held-out 구성 → 완전히 novel object 일반화 미검증.
5. **π0.5 planning**: 공개 openpi가 text generation 미노출 → external task-to-primitive lookup으로 symmetric하게 처리.
6. **Failure attribution**: USB held-out에서 classifier macro-F1 0.42로 저조 → USB F-condition recovery time 분석 누락.
7. **Ablation-as-causation**: subspace가 inference time에 necessary임은 입증했으나, primitive supervision이 그러한 subspace를 만드는 **유일한** 방법임을 입증하지는 않음.

---

## 8. 종합 평가

| 항목 | 평점 (5점) |
|---|---|
| 혁신성 | 4.0 (architectural change 없는 emergent compositionality + causal subspace 입증) |
| 재현성 | 3.5 (코드 공개 약속, 12 cells × 3 seeds × 2 datasets full sweep) |
| 실험 폭 | 5.0 (2 arch × 2 view × 3 seeds × 2 datasets + ablation + intervention + 5 baselines) |
| 이론적 깊이 | 4.5 (linear probe + orthogonal projection causal intervention + family-wise inflation 수학) |
| 실용성 | 4.5 (m=5에서 FT UB의 ~90% 회복 → 산업 배포 cost 구조 변경) |

**총평**: VLA의 fine-tune 의존성을 줄이는 가장 깔끔한 결과 중 하나. (1) data view만 바꾸면 demo-conditioned composition이 emerge, (2) 그 능력의 substrate가 4-D subspace로 식별되고 ablation으로 인과적으로 입증됨, (3) chunked policy 평가의 구조적 결함까지 부수적으로 수정. OOV task 역전 결과를 솔직하게 보고한 점도 신뢰성 +.

---

## 9. 예상 세미나 질문

> ❓ **m=5에서 8 pp gap이면 그냥 fine-tune하지 왜 few-shot?**
> 50 demo 수집 + 1–3일 compute + dataset curation + validation 사이클 vs 5 demo를 prompt에 붙이기. 산업 환경에서 product variant마다 이걸 반복하면 cost 구조가 완전히 달라짐 (저자 §5.2 강조).

> ❓ **Primitive supervision이 진짜 원인인가, 아니면 primitive view가 episode를 짧게 만들어서 학습이 쉬워진 것 아닌가?**
> 두 view가 동일 source demo에서 materialize되어 data quantity 통제됨. 더 결정적으로, subspace ablation 결과가 분리한다: random subspace 제거는 효과 없고 primitive subspace 제거만 32 pp 감소.

> ❓ **OOV 역전 결과가 primitive 접근의 결함 아닌가?**
> 부분적으로 그렇다. 저자도 §4.4 / §5.1에서 인정: primitive vocabulary coverage가 부족하면 모델이 가장 가까운 known primitive로 강제 mapping. 시사점: primitive library 설계 시 **completeness가 중요**.

> ❓ **External task-to-primitive lookup이 cheating 아닌가?**
> Symmetric하게 두 architecture에 동일 적용되어 실험 조작 변수 통제. 다만 π0.5의 text generation API 부재 때문에 강제된 선택이며, 향후 openpi 업데이트 시 ablation 필요.

> ❓ **Linear probe 0.79가 충분히 높은가? Nonlinear 표현 가능성은?**
> Macro-F1 0.79–0.81은 4-class 문제에서 chance 0.25 대비 상당히 높음. 더 중요한 것은 subspace ablation의 **causal** 효과(32 pp drop) — nonlinear 정보가 있더라도 linear-decodable component가 critical pathway임을 보여줌.

---

## 10. 코드 & 재현

- **Code**: 공개 약속 ("Code, training configurations, and held-out task manifests will be released publicly upon publication" §5.4)
- 공개 예정 자료: (i) data materialization script (REASSEMBLE + LIBERO-Long view), (ii) LoRA config, (iii) held-out manifest + 자동 성공 기준, (iv) probing + subspace ablation code, (v) corrected chunked-policy gate, (vi) 12 main cell checkpoints
- 모든 보고 숫자는 3 seed × 50 rollout 평균 ± SD

---

## 11. 데이터셋 / 후속 연구

- **REASSEMBLE**: 17 objects × 4 primitives × 4,551 demos. NIST Assembly Task Board 1. Contact-rich industrial assembly.
- **LIBERO-Long**: household manipulation, sub-task structure (move-to/grasp/transport/place)로 primitive 경계 구성
- 후속 방향:
  - Real-robot validation
  - 더 broad한 architectural coverage (RT-2, Octo-Large, AnyVLA 등)
  - Primitive library size scaling regime 분석
  - OOV 역전 완화 방법: open-vocabulary primitive learning

---

## 12. 결론

PrimSubVLA(=primitive-aware OpenVLA / π0.5)는 (1) **architectural 변경 없이** data view + language prompt 수준의 supervision만으로 demonstration-conditioned compositional transfer 능력을 emerge시킴, (2) 그 능력의 **substrate를 4-D primitive-decodable subspace로 식별**하고 inference-time orthogonal projection ablation으로 **causal necessity를 입증**, (3) 부수적으로 chunked policy 평가의 13.4× family-wise inflation을 수학적으로 도출하고 element-level 보정 gate를 제안한다. m=5 단계에서 fine-tune upper-bound의 90% 회복은 산업 VLA 배포의 operational cost 구조를 실질적으로 바꿀 수 있는 결과이며, OOV 역전 보고로 **primitive vocabulary coverage**라는 후속 연구 의제도 명확히 했다.

---

<!-- VERIFIED: pdf -->
