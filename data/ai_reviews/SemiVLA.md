# SemiVLA: Semi-Supervised Vision-Language-Action Model

> **한 줄 요약**: 10% action-labeled + 90% action-unlabeled vision-language trajectory만으로 pretrained VLA를 적응시키는 self-distilled teacher-student framework. **VLA-specific reliability controller** (vision-language alignment × action feasibility × temporal transition consistency)로 pseudo-action 신뢰도를 평가하고, **Bottleneck-Projected Alignment Update (BPA)**로 teacher를 EMA가 아닌 alignment-필터링된 student displacement로 갱신. OpenVLA backbone에서 LIBERO **89.0%** (supervised LoRA 대비 +8.0pt), CALVIN 2.58, SIMPLER-ENV 62.4% 달성.

---

## 1. 배경 및 동기

- **VLA fine-tuning의 비용 문제**: 새로운 robot/환경에 적응하려면 teleoperation 기반 action-labeled demonstration이 필수 → 수집 비용 폭증.
- **얻기 쉬운 데이터**: 로봇 비디오·egocentric 조작 영상·task description은 풍부하나 **low-level action label은 희소** [9, 14, 18].
- **표준 SSL의 한계**: 이미지 분류에서의 pseudo-label은 discrete class이지만, VLA의 pseudo-label은 **continuous action vector / action token sequence**로 (i) 언어와 일관, (ii) 시각적으로 grounded, (iii) 물리적으로 feasible, (iv) 시간적으로 안정해야 함 — confidence-only pseudo-labeling 부적합.
- **세 가지 핵심 난점**: ① pseudo-action 오류는 closed-loop에서 compounding error로 폭발, ② model confidence는 pseudo-action 품질을 보장 못함, ③ noisy student update가 teacher에 averaged되면 self-distillation이 amplify되는 **feedback contamination**.

### 본 논문의 질문
**"action label이 10%뿐인 상황에서, 나머지 90% vision-language trajectory를 어떻게 reliable한 action supervision으로 전환할 것인가?"**

---

## 2. 문제 설정 (Semi-Supervised VLA Adaptation)

- Labeled set: `D_l = {(o_{1:T}, q, a_{1:T})}` (N_l개, 10%)
- Unlabeled set: `D_u = {(o_{1:T}, q)}` (N_u개, 90%) — **action만 빠짐**
- 표준 supervised VLA fine-tuning은 모든 trajectory에 action 가정 → D_u 활용 불가.
- 표준 SSL은 class label 누락 가정 → embodied action signal에 부적합.
- 본 논문은 limited-supervision-signal 관점에서 VLA adaptation을 처음으로 명시 정의.

---

## 3. 방법론 심층 분석

### 3.1 전체 구조
- **Student** πθ: gradient backprop으로 PEFT(LoRA 등) + action head + transition predictor F_ψ 업데이트.
- **Teacher** πξ: warm-up 후 student로 초기화, **gradient 없음**. BPA로만 갱신.
- **3-stage schedule**:
  1. Supervised warm-up (L_sup only, teacher 미사용).
  2. Self-distillation + reliability controller 작동, **teacher 고정**으로 reliability score 칼리브레이션.
  3. BPA teacher update 활성화.

### 3.2 Self-Distilled Pseudo-Action Learning
각 unlabeled step에서 weak view o_t^w와 strong view o_t^s 생성. Teacher는 weak view에서 pseudo-action `â_t = sg(πξ(o_t^w, q))` 생성, student는 strong view에서 학습:
```
L_sd = (1/|B_u|) Σ (1/T) Σ_t M_t · r_t · d(πθ(o_t^s, q), sg(πξ(o_t^w, q)))
```
- `d(·)`: action-token이면 KL/CE, continuous면 L1/L2.
- `M_t = I[r_t ≥ τ_r]`: reliability mask.

### 3.3 VLA-Specific Reliability Controller (핵심 1)
**Multiplicative form (의도적 strict)**:
```
r_t = r_t^vl · r_t^act · r_t^temp
```

- **r_t^vl (vision-language alignment)**: `σ(sim(z_q, z_o)/τ_vl)`. Attention map 있으면 instruction-relevant visual token으로 z_o 계산 (e.g. "place red mug on plate" → red mug/plate 토큰).
- **r_t^act (action feasibility)**: continuous는 `exp(-||â_t - â_{t-1}||_1 / τ_a) · I[||â_t||_∞ < δ_a]` — sudden jump, 비현실적 end-effector 움직임 차단. Token VLA는 action-token entropy + neighboring frame consistency.
- **r_t^temp (temporal transition consistency)**: 경량 transition predictor F_ψ가 `h(o_t), z_q, â_t`로 next visual state 예측. 예측 오차 작을수록 score 높음:
```
L_trans = (1/|B_u|) Σ (1/(T-1)) Σ_t M_t · r_t · ||h(o_{t+1}) - F_ψ(h(o_t), â_t, z_q)||_2^2
```

### 3.4 VLA Alignment Loss
세 reliability signal이 **개선되도록** 명시 학습 신호 부여:
```
L_align = (1/|B_u|) Σ (1/T) Σ_t M_t [(1 - r_t^vl) + λ_a(1 - r_t^act) + λ_t(1 - r_t^temp)]
```
→ teacher confidence가 아닌 **pseudo-action quality 자체를 향상**.

### 3.5 Final Student Objective
```
L = L_sup + λ_sd · R_u(e) · L_sd + λ_align · L_align + λ_trans · L_trans
```
- `R_u(e)`: epoch-dependent unsupervised ramp-up → 초기 noisy pseudo-action dominate 방지.

### 3.6 Bottleneck-Projected Alignment Update (핵심 2)
**기존 scalar EMA의 한계**: nuisance-sensitive visual change, unstable action correction, language-irrelevant drift를 teacher에 averaging → corrupt pseudo-action generator.

**SemiVLA의 처방**: parameter displacement `Δ = θ - ξ`를 두 그룹으로 분해:
- **G_sem (semantic alignment group)**: visual projector + vision-language PEFT.
- **G_act (action precision group)**: action head + late policy PEFT.

각각 다른 alignment signal 사용:
- Semantic group: weak/strong view feature covariance에서 channel-wise gate A_t^w, A_t^s 계산 → `r_t^ib = exp(-||A_t^w - A_t^s||_F / τ_ib)`. ω_sem = r_ib · r_vl.
- Action group: ω_act = r_act · r_temp.

Group별 adaptive update strength:
```
ρ_g = ρ_max(e) · σ(w_1 ω_g - w_2 e_g - w_3 v_g)
```
- `e_g`: teacher-student prediction deviation (KL for token VLA, action distance for continuous).
- `v_g = ||Δ_g - Δ̄_g||_2 / (||Δ̄_g||_2 + ε)`: volatility — sudden change suppress.

Teacher 갱신:
```
ξ_g ← ξ_g + ρ_g · P_g(Δ_g),  g ∈ {G_sem, G_act}
```
- **P_sem**: batch-averaged bottleneck gate로 LoRA update direction projection → 안정 vision-language channel만 전달.
- **P_act**: full direction 유지 but action feasibility + temporal reliability로 scaling → fine-grained control 보존.

→ teacher가 alignment-supported student feedback만 받음.

---

## 4. 데이터 전략

| 항목 | 내용 |
|---|---|
| Benchmarks | LIBERO (4 suite), CALVIN, SIMPLER-ENV |
| Supervision protocol | **10% labeled + 90% unlabeled** trajectory (Table 3 명시) |
| Backbone | OpenVLA (main), OpenVLA-OFT, π0-FAST, GR00T-N1.6 (확장) |
| Augmentation (strong view) | random crop, color jitter, camera noise, temporal jitter, mild viewpoint disturbance |
| Few-shot 추가 평가 | 1/5/10/20-shot adaptation |

---

## 5. 주요 실험 결과 (LIBERO + CALVIN, OpenVLA backbone, 10% labeled)

### 5.1 LIBERO 4-suite (Table 1, Table 2)

| Method | Spatial | Object | Goal | Long | **Avg.** |
|---|---:|---:|---:|---:|---:|
| OpenVLA Zero-shot | 80.0 | 69.6 | 74.0 | 55.5 | 69.8 |
| OpenVLA + Linear Probe | 81.3 | 72.4 | 76.8 | 59.6 | 72.5 |
| OpenVLA + Adapter | 84.6 | 78.2 | 81.5 | 65.7 | 77.5 |
| OpenVLA + LoRA | 86.8 | 82.1 | 84.9 | 70.3 | 81.0 |
| OpenVLA + QLoRA | 85.9 | 80.5 | 83.7 | 68.9 | 79.8 |
| OpenVLA + Full FT (7B) | 84.5 | 79.8 | 82.6 | 66.8 | 78.4 |
| OpenVLA + Selective LoRA | 87.6 | 83.4 | 85.6 | 71.8 | 82.1 |
| SemiVLA + Adapter | 88.9 | 84.7 | 87.2 | 74.6 | 83.9 |
| SemiVLA + LoRA | 91.7 | 88.6 | 90.9 | 80.5 | 87.9 |
| SemiVLA + QLoRA | 90.8 | 87.4 | 89.7 | 78.9 | 86.7 |
| **SemiVLA + Selective LoRA** | **92.4** | **89.5** | **91.8** | **82.3** | **89.0** |
| SemiVLA + Full FT | 90.5 | 87.8 | 89.2 | 78.5 | 86.5 |

- **Supervised LoRA 대비 +8.0pt**, zero-shot 대비 **+19.2pt**.
- LIBERO-Long에서 **+12.0pt** — long-horizon에서 temporal consistency 효과 극대화.
- 흥미: Full FT(7B)가 Selective LoRA(31.6M)보다 낮음 → 10% label로 7B 전체 업데이트는 overfit + pretrained VLA prior 손상.

### 5.2 CALVIN (Table 2, 평균 완료 task 수)

| Method | CALVIN avg |
|---|---:|
| OpenVLA Zero-shot | 1.32 |
| OpenVLA + LoRA | 1.78 |
| OpenVLA + Selective LoRA | 1.86 |
| **SemiVLA + Selective LoRA** | **2.58** |

→ LIBERO 학습 방식이 CALVIN에 transfer (+0.72 vs supervised, +1.26 vs zero-shot) → LIBERO overfit 아님을 입증.

### 5.3 SIMPLER-ENV (Table 3, semi-supervised 비교)

| Method | Pick | Move | Drawer | Avg. |
|---|---:|---:|---:|---:|
| OpenVLA Zero-shot | 16.3 | 46.2 | 35.6 | 32.7 |
| OpenVLA-FT | 36.8 | 48.2 | 39.1 | 41.4 |
| LAM-Pseudo | 57.0 | 66.2 | 42.5 | 55.2 |
| LARA-style Align | 60.5 | 70.8 | 43.0 | 58.1 |
| **SemiVLA-OpenVLA** | **66.0** | **78.2** | **43.0** | **62.4** |
| SemiVLA-GR00T-N1.6 | 92.0 | 87.6 | 52.5 | 77.4 |

### 5.4 Backbone scaling (Table 3, LIBERO Avg.)
- SemiVLA-OpenVLA: **89.0**
- SemiVLA-OpenVLA-OFT: **91.2**
- SemiVLA-π0-FAST: **92.0**
- SemiVLA-GR00T-N1.6: **94.1**

→ adaptation framework가 backbone-agnostic하게 효과.

---

## 6. Ablation Study

### 6.1 Component (Table 7)
| Setting | Avg. |
|---|---:|
| Supervised LoRA | 81.0 |
| + Naive pseudo-action | 83.2 (+2.2) |
| + VLA reliability controller | 86.1 (+2.9) |
| + Bottleneck-projected update | 87.6 (+1.5) |
| **Full SemiVLA** | **87.9** |

### 6.2 Reliability controller (Table 8)
| Signal | Accept ratio | Avg. |
|---|---:|---:|
| Confidence only | 52.8% | 83.2 |
| VL only | 46.5% | 84.0 |
| Action only | 48.1% | 84.5 |
| Temporal only | 44.7% | 84.8 |
| VL + Action | 39.6% | 86.0 |
| **VL + Action + Temporal** | **36.8%** | **87.9** |

→ accept ratio가 낮아져도 **quality > quantity** 명확히 입증.

### 6.3 Teacher update strategy (Table 9)
| Update | Pseudo-action error ↓ | Avg. ↑ |
|---|---:|---:|
| Frozen teacher | 0.184 | 85.2 |
| Standard EMA | 0.171 | 85.9 |
| Reliability-aware EMA | 0.154 | 86.8 |
| **BPA update** | **0.137** | **87.9** |

→ BPA는 acceptance ratio가 최고는 아니나 pseudo-action error 최저, 최종 SR 최고.

### 6.4 Labeled trajectory ratio (Table 4)
| Ratio | LoRA | SemiVLA Selective LoRA | Gain |
|---|---:|---:|---:|
| 1% | 72.6 | **81.4** | +8.8 |
| 5% | 76.8 | **86.2** | +9.4 |
| 10% | 81.0 | **89.0** | +8.0 |
| 20% | 84.7 | **91.1** | +6.4 |
| Full | 88.1 | **92.4** | +4.3 |

→ **label이 적을수록 gain 큼** (motivation 정합).

### 6.5 Few-shot (Table 5)
| Shot | OpenVLA+SelectiveLoRA | **SemiVLA+SelectiveLoRA** |
|---|---:|---:|
| 1-shot | 49.8 | **58.7** |
| 5-shot | 63.5 | **72.8** |
| 10-shot | 71.2 | **81.9** |
| 20-shot | 79.1 | **87.3** |

### 6.6 Robustness (Table 10)
- Unseen Task / Lighting / Object / Distractor 4-axis 모두 +11pt 이상.
- Distractor robustness: 62.1 → **73.9** (+11.8) — stable VL correspondence 학습.

---

## 7. 효율성 (Table 6, OpenVLA backbone, 10% labeled)

| Method | Trainable | Time | GPU Mem | Avg. |
|---|---:|---:|---:|---:|
| Full FT (7.0B) | 100% | 9.8h | 78.0GB | 78.4 |
| LoRA (79.4M) | 1.13% | 3.6h | 27.8GB | 81.0 |
| Selective LoRA (31.6M) | 0.45% | 3.0h | 24.2GB | 82.1 |
| SemiVLA + LoRA | 1.13% | 4.7h | 29.6GB | 87.9 |
| **SemiVLA + Selective LoRA** | **0.45%** | 4.2h | 24.8GB | **89.0** |

→ SemiVLA overhead는 supervised 대비 +1~1.5h, GPU memory는 +0.6GB. **Selective LoRA + SemiVLA가 가장 좋은 efficiency-performance trade-off**.

---

## 8. 핵심 인사이트

1. **Confidence ≠ Reliability (VLA에서)**: action prediction은 language grounding + physical feasibility + temporal coherence를 모두 요구하므로 confidence-only filtering(52.8% accept)이 86% 평균에 못 미침. Strict multiplicative form(36.8% accept)이 87.9% 달성.
2. **Teacher contamination이 진짜 문제**: frozen teacher(85.2)와 standard EMA(85.9) 차이가 작음 → naive averaging은 student 잡음을 그대로 전달. **directional filtering(BPA)**이 진짜 lever.
3. **Action label 희소성이 클수록 method 가치 큼**: 1% label에서 +8.8pt, full label에서도 +4.3pt regularizer 역할.
4. **Backbone-agnostic adaptation recipe**: OpenVLA → OpenVLA-OFT → π0-FAST → GR00T-N1.6 모두 monotonic 향상.

---

## 9. 한계 및 비판

- **Reliability controller hyperparameter 다수**: τ_vl, τ_a, δ_a, τ_ib, λ_a, λ_t, λ_sd, λ_align, λ_trans, τ_r, w_1/w_2/w_3, ρ_max — 실전 적용시 calibration 부담. Stage 2에서 teacher 고정으로 어느 정도 mitigation.
- **Transition predictor F_ψ의 capacity 미상세**: "lightweight"라고만 표기, architecture/parameter 미공개. Embodiment scaling 영향 불확실.
- **Real robot 미평가**: LIBERO/CALVIN/SIMPLER-ENV 시뮬레이션만. real-world DROID-style 평가 없음.
- **Code/checkpoint 미공개** (작성 시점). reproducibility 검증 어려움.
- **D_u 가정 amibguous**: "vision-language trajectory"의 quality (camera view 일치, instruction 정확도)가 reliability score에 직접 영향 → 노이즈가 큰 in-the-wild data에서의 성능 미검증.
- **장기 self-distillation 안정성**: 3-stage schedule이 finite epoch 가정. teacher가 student보다 reliable한 시점이 언제 reversal되는지 분석 부재.

---

## 10. Open Questions

- Latent Action Model (LAM, Moto)과의 **결합** 가능성 — F_ψ를 LAM-style latent action tokenizer로 대체하면?
- 진정한 in-the-wild D_u (egocentric video, YouTube manipulation)에서 r_vl이 reliable한가?
- BPA의 channel-wise gate는 LoRA에서는 trivial하지만 full FT나 다른 PEFT (DoRA, IA³)로의 일반화는?
- Inference cost는 동일하므로(teacher 폐기), test-time에 reliability controller를 활용한 safe action filtering 가능성?

---

## 11. 관련 연구 비교

| 작업 | 차이점 |
|---|---|
| OpenVLA / OpenVLA-OFT [10, 11] | 100% labeled fine-tuning 가정, SemiVLA는 그 위에 SSL layer |
| FixMatch / Mean Teacher [16, 17] | confidence/EMA-only, embodied action signal 무시 |
| Moto / LAPA [6, 24] | latent action tokenizer 학습, SemiVLA는 별도 tokenizer 불필요 |
| LARA [12] | latent action representation alignment, SemiVLA는 직접 pseudo-action에 reliability 부여 |
| PseudoAction-VLA / FixMatch-VLA (본 논문 baseline) | 단순 SSL 변형, SemiVLA가 LIBERO Avg 4-5pt 우위 |

---

## 12. 결론 및 기여 요약

1. **Problem formulation**: limited-supervision-signal 기반 semi-supervised VLA adaptation을 명시 정의 (Dl + Du protocol, Table 3).
2. **Self-distilled teacher-student framework**: VLA에 SSL을 끌어들이되 confidence-only를 폐기.
3. **VLA-specific reliability controller**: vision-language × action feasibility × temporal — 3-axis multiplicative.
4. **Bottleneck-Projected Alignment Update**: scalar EMA 대체, group-wise directional filtering으로 teacher contamination 방어.
5. **실증**: OpenVLA에서 LIBERO 81.0→89.0 (+8.0pt), CALVIN 1.86→2.58 (+0.72), SIMPLER-ENV 41.4→62.4 (+21.0pt). Backbone scaling으로 GR00T-N1.6에서 94.1% LIBERO 도달.

**핵심 메시지**: VLA adaptation의 다음 frontier는 더 큰 backbone이나 더 많은 demo가 아니라, **action-unlabeled vision-language trajectory를 reliable pseudo-action으로 전환하는 supervision design**이다.

<!-- VERIFIED: pdf -->
