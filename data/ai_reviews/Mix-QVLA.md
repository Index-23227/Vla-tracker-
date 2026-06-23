# Mix-QVLA: Task-Evidence-Aware Mixed-Precision Quantization of Vision-Language-Action Models

> **한 줄 요약**: OpenVLA / OpenVLA-OFT autoregressive 정책에 적용하는 **task-evidence-aware mixed-precision PTQ** — 네 가지 VLA 기능 경계(비전 인코더, projector, LLM, pre-action head)에서 *full-precision action-token 결정을 뒷받침하는 task evidence*가 양자화 후 얼마나 보존되는지를 gradient-weighted evidence map으로 측정, **evidence-mass + attribution distortion** 결합 손실을 soft-bottleneck으로 집계해 layer sensitivity Omega(m,b) 산출, 추가로 **trajectory-phase별 시간 민감도** Omega_tau를 결합. 결과: OpenVLA-OFT W4A4에서 메모리 **15.4 GB → 4.1 GB**, LIBERO 평균 **96.3%** (BF16 97.1%), **1.52× 추론 가속**.

---

## 1. 배경 및 동기

### VLA quantization이 LLM/VLM과 다른 이유
- VLA 출력은 *로봇 action*으로 실행되며, closed-loop interaction을 통해 다음 observation에까지 영향 — 양자화 오류가 visual grounding → 언어 추론 → action token 예측 전체를 따라 전파.
- 일반 LLM/VLM PTQ는 weight reconstruction, activation outlier 제어, salient channel 식별(GPTQ, SmoothQuant, AWQ)에 집중 — VLA의 embodied 특성을 직접 다루지 않음.

### 기존 VLA-aware quantization의 한계
- **QVLA** (Xu et al., 2026a): 최종 action 편차로 sensitivity 추정 — *결과만* 봄.
- **DyQ-VLA** (Zheng et al., 2026): 시간축 precision 적응을 kinematic proxy로 — *간접* 신호.
- **QuantVLA** (Zhang et al., 2026a): scale-calibrated PTQ — *수치 안정성* 위주.
- 공통 문제: action 편차가 작아도 *내부 evidence pathway가 망가졌을 수* 있음 — 비슷한 motion이지만 task grounding이 어긋남.

### 핵심 가설
> "Action deviation은 policy computation의 *endpoint*만 본다. 양자화가 full-precision 결정의 *내부 evidence 경로*를 보존하는지는 따로 측정해야 한다."

---

## 2. 방법론 심층 분석

### 2.1 문제 설정

- Full-precision 모델 θ_FP, 후보 양자화 변형 θ_{m,b} (layer m을 b-bit로 양자화, 나머지는 FP).
- 각 calibration sample i = (V_τ, x_τ, P)에 대해 FP 모델로 reference action-token sequence y_i* = (y*_{i,1}, ..., y*_{i,K}) 미리 계산해 고정.
- Reference action-support objective (teacher-forced log-likelihood):

```
J_i(θ; y_i*) = (1/K) Σ_k log p_θ(y*_{i,k} | y*_{i,<k}; z_i)
```

### 2.2 네 가지 기능 경계 Γ = {ν, β, ψ, α}
| 기호 | 위치 | 역할 |
|------|------|------|
| ν | Vision encoder 출력 | 시각 표상 |
| β | Projector 출력 | VL 정렬 |
| ψ | LLM (language-policy) 표상 | 멀티모달 추론 |
| α | Action head 직전 | pre-action 결정 형성 |

각 경계마다 full-precision activation 통계(μ_γ^FP, σ_γ^FP)로 정규화한 hidden Z_{i,γ}^θ를 사용.

### 2.3 Gradient-weighted task-evidence map

```
E_{i,γ}^θ = Z_{i,γ}^θ ⊙ ∇_{Z_{i,γ}^θ} J_i(θ; y_i*)
```

- 활성화 항: 어떤 feature가 *실제로 존재*하는지
- gradient 항: 그 feature 변화가 *FP 결정을 얼마나 지지*하는지
- Figure 1: vision encoder 경계 시각화 — W8 (ℓ_ev≈0.14) → W4 (0.37) → W2 (1.37)로 evidence 분포가 점점 무너짐.

### 2.4 두 가지 distortion: mass + attribution

**Evidence mass distortion**: 결정 지지 evidence의 *총량* 변화
```
Δ^mass_{i,γ}(m,b) = log [ (M_{i,γ}^{θ_{m,b}} + ε) / (M_{i,γ}^{θ_FP} + ε) ]
```
단, M_{i,γ} = (1/d_γ) Σ_j E_{i,γ,j}.

**Attribution distortion**: evidence의 *내부 분포* 변화 (Jensen–Shannon divergence)
```
a_{i,γ,j}^θ = (E_{i,γ,j} + ε) / Σ_{j'} (E_{i,γ,j'} + ε)
Δ^attr_{i,γ}(m,b) = D_JS(a_{i,γ}^{θ_FP}, a_{i,γ}^{θ_{m,b}})
```

경계별 task-evidence loss:
```
ℓ_ev_{i,γ}(m,b) = Δ^mass_{i,γ} + λ Δ^attr_{i,γ}     (λ=1)
```

> ❓ **예상 질문**: mass와 attribution을 왜 따로 보는가?
> **답변**: 총량이 비슷해도 evidence가 *다른 token/channel*로 재할당되면 task grounding이 다를 수 있다 — Figure 2(a)는 *language module*에서 task-evidence loss가 가장 크지만 *action error*는 상대적으로 작음을 보여줌 (action-only sensitivity의 맹점).

### 2.5 Soft-bottleneck 집계 (log-sum-exp)

```
L_i^SB(m,b; κ) = κ log [ (1/|Γ_i|) Σ_γ exp(ℓ_ev_{i,γ}(m,b) / κ) ]
```
κ=0.1 (강한 bottleneck 근처지만 numerically smooth). κ→0이면 max에 수렴, κ↑면 평균에 가까워짐.

**Layer sensitivity**:
```
Ω(m,b; κ) = (1/N) Σ_i L_i^SB(m,b; κ)
```

### 2.6 Temporal sensitivity Ω_τ

- 각 sample을 trajectory progress ρ_i = τ_i / (T_{r_i} - 1) ∈ [0,1]로 정규화 → Q개 phase bin으로 분할.
- Phase-wise: Ω_q(m,b; κ) = (1/|C_q|) Σ_{i ∈ C_q} L_i^SB
- Scalar 사용: **Ω_τ(m,b) = max_q Ω_q(m,b; κ)** — *최악 phase* 보호 전략.

### 2.7 Mixed-precision bit allocation (BLP)

```
min Σ_m Σ_b x_{m,b} [ α Ω(m,b) + β Ω_τ(m,b) ]
s.t. Σ_b x_{m,b} = 1           ∀m
     Σ x_{m,b} C_size(m,b) ≤ C_size^target
     Σ x_{m,b} C_bitops(m,b) ≤ C_bitops^target
     x_{m,b} ∈ {0,1}
```
- 비용: C_size = N_m b, C_bitops = MACs(m) b² (W_b A_b 가정).
- B = {2, 4, 8, 16}.
- Solver: **CVXPY + ECOS_BB** (branch-and-bound BLP). Sensitivity/cost는 calibration 단계에서 precompute.
- α=0.75, β=0.25가 최적 (Table 3b). 최종 allocation A*는 calibration 후 고정되며 *deployment 중 timestep-wise switching 안 함*.

---

## 3. 시스템/실험 세부

| 항목 | 값 |
|------|-----|
| Base policy | **OpenVLA (7B)** 및 **OpenVLA-OFT** (autoregressive token-based VLA) |
| Quantization 종류 | Post-training (no retraining) |
| Hardware | 단일 NVIDIA A100 |
| Calibration | LIBERO training demos (RGB + 로봇 상태 + instruction + timestep) |
| Bit-widths | {2, 4, 8, 16} |
| 비교 대상 | SmoothQuant, OmniQuant, AWQ, QVLA, DyQ-VLA |

---

## 4. 실험 결과 심층 분석

### 4.1 LIBERO W4A4 — OpenVLA-OFT (Table 1)
| 방법 | Spatial | Object | Goal | Long | **Avg** | Δ | Mem(GB) | Speedup |
|------|---------|--------|------|------|---------|---|---------|---------|
| BF16 | 97.6 | 98.4 | 97.9 | 94.5 | **97.1** | – | 15.4 | 1× |
| SmoothQuant | 77.2 | 70.0 | 77.8 | 68.6 | 73.4 | -23.7 | 4.9 | 1.53× |
| OmniQuant | 95.0 | 94.4 | 94.0 | 92.0 | 93.9 | -3.2 | 5.7 | 1.37× |
| QVLA | 96.2 | 97.6 | 96.4 | 93.8 | 96.0 | -1.1 | 4.5 | 1.49× |
| **Mix-QVLA** | **96.8** | **97.8** | **96.4** | **94.0** | **96.3** | **-0.8** | **4.1** | **1.52×** |

→ BF16 대비 메모리 **73% 감소**, 정확도 손실 -0.8%p, 1.52× 가속. 모든 baseline을 W4A4에서 능가.

### 4.2 LIBERO W4A4 — OpenVLA
| 방법 | Avg | Mem | Speedup |
|------|-----|-----|---------|
| BF16 | 76.5 | 15.2 | 1× |
| QVLA | 76.0 | 4.3 | 1.47× |
| DyQ-VLA | 76.1 | 4.7 | 1.51× |
| **Mix-QVLA** | **76.3** | **4.0** | **1.52×** |

### 4.3 Weight-only (W4A16, Table 2)
- OpenVLA: Mix-QVLA **76.6** (QVLA 76.5, AWQ 70.8), 4.1 GB.
- OpenVLA-OFT: Mix-QVLA **96.9** (QVLA 96.7, AWQ 92.5), 4.2 GB.

### 4.4 Ablation — sensitivity signal (Table 3a, OpenVLA ~4-bit avg)
| Act. | Ev. | Temp. | Bit↓ | Mem↓ | Avg↑ |
|------|-----|-------|------|------|------|
| ✓ |  |  | 4.00 | 4.3 | 76.0 |
|  | ✓ |  | 3.94 | 4.0 | 75.9 |
|  |  | ✓ | 3.95 | 4.1 | 75.6 |
|  | ✓ | ✓ | **3.96** | **4.0** | **76.3** |

→ Evidence 또는 temporal 단독은 action-based보다 좋지 않지만, **둘을 결합**하면 더 낮은 메모리에서 가장 높은 성공률.

### 4.5 Temporal weighting (Table 3b)
α=1.0/β=0.0: 75.9, **α=0.75/β=0.25: 76.3 (최적)**, α=0.0/β=1.0: 75.6. → temporal 신호는 *보조*여야지 *주신호*면 worst-phase layer를 과보호하다 일반 중요 layer를 희생.

---

## 5. 핵심 통찰

- Figure 2(a): **language module**이 task-evidence loss는 가장 크지만 action error는 상대적으로 작음 — action-only criterion이 *어디서* VLA 계산이 망가지는지 못 잡는다는 가장 강한 증거.
- Figure 3: 시간축 sensitivity heatmap — 어떤 layer는 trajectory 내내 fragile, 어떤 layer는 *특정 phase에서만* 민감. → time-aware allocation의 정당화.
- Soft-bottleneck (κ=0.1)이 핵심 — 단순 평균은 critical boundary의 국소 실패를 희석하고, multiplicative retention은 다중 moderate degradation에서 포화.

---

## 6. 한계 및 비판

저자가 밝힌 한계:
1. **LIBERO simulation only** — real-robot 검증 없음.
2. **OpenVLA-style policies only** — diffusion/flow-matching VLA (π0, π0.5, GR00T) 미검증.
3. Calibration 단계에 추가 forward + backward pass 필요 → action-only 대비 *offline 분석 비용 증가*.
4. **Fixed allocation** after calibration — deployment 중 timestep-wise precision switching 안 함 (DyQ-VLA와 대비).
5. Evidence map은 *diagnostic* signal이지 task success의 *인과적 보장*은 아님.

추가로 비판할 만한 점:
- α, β grid search가 매우 거친 단계 (0.25 간격) — 더 미세 조정이 추가 이득을 줄 수 있음.
- W4A4 OpenVLA에서 QVLA 대비 정확도 이득은 +0.3%p로 통계적 유의성이 미상 — 메모리/속도 이득이 본질적 셀링 포인트.
- LIBERO만으로 evidence preservation의 *task-grounding 효과*를 직접 입증하기 어려움 (RoboTwin, LIBERO-Plus 등 robustness suite 필요).

---

## 7. 위치 정립

| 축 | 위치 |
|----|------|
| Sensitivity 신호 | **Internal evidence path** (vs. action-only QVLA, kinematic proxy DyQ-VLA, scale-only QuantVLA) |
| Allocation 방식 | **Static mixed-precision BLP** (vs. DyQ-VLA dynamic) |
| 대상 백본 | **Autoregressive (OpenVLA 계열)** (vs. Ω-QVLA: diffusion / GR00T·π0.5, QuoVLA: flow-matching prefix) |
| Retraining | **불필요 (PTQ)** |

---

## 8. 재현성

- Code: 공개 없음 (논문에 repo 언급 없음).
- Solver: CVXPY + ECOS_BB (오픈소스로 재현 가능).
- Baseline: OpenVLA, OpenVLA-OFT BF16 (공개).
- 핵심 hyper: κ=0.1, λ=1.0, α=0.75, β=0.25, Q bins (수치 미명시).

---

## 9. 향후 연구

- Diffusion/flow-matching action head (π0, π0.5, GR00T)에도 task-evidence framework 확장.
- **Dynamic** mixed-precision: phase 별로 다른 allocation A_q*를 deployment 중 swap.
- Real-robot validation — closed-loop noise하에서 evidence preservation이 success rate로 직접 이어지는지.
- Evidence map의 *causal* 검증 — counterfactual layer perturbation으로 task 성공 인과성 측정.

---

## 10. 핵심 수치 표 (한눈에)

| 설정 | 백본 | Avg | Mem(GB) | Speedup |
|------|------|-----|---------|---------|
| BF16 | OpenVLA-OFT | 97.1 | 15.4 | 1× |
| W4A4 Mix-QVLA | OpenVLA-OFT | 96.3 | 4.1 | 1.52× |
| W8A8 Mix-QVLA | OpenVLA-OFT | 96.6 | 6.7 | 1.39× |
| W4A16 Mix-QVLA | OpenVLA-OFT | 96.9 | 4.2 | – |
| W4A4 Mix-QVLA | OpenVLA | 76.3 | 4.0 | 1.52× |
| W4A16 Mix-QVLA | OpenVLA | 76.6 | 4.1 | – |

---

## 11. 강점 / 약점 종합

**강점**
- 진단 가능성 — *어디서* 양자화가 망가지는지 boundary-wise로 가시화.
- 두 축(mass + attribution) 결합이 단일 metric 대비 일관된 이득.
- BLP 기반 allocation은 size + BitOps dual-budget로 깔끔.
- W4A4에서 BF16 대비 -0.8%p만 손해, 메모리 73% 감소.

**약점**
- 정확도 절대 우위는 W4A4에서 QVLA 대비 +0.3%p로 작음 — 핵심 이득은 메모리/속도.
- Real-robot, non-autoregressive VLA, time-switching 미확장.
- Evidence가 diagnostic이지 causal이 아니라는 저자 자평.

---

## 12. 결론

Mix-QVLA는 *"action 편차만으로 양자화 sensitivity를 추정하면 내부 evidence 경로가 망가졌는지 모른다"*는 문제를 정면으로 다룬 첫 PTQ 프레임워크다. **gradient-weighted evidence map + mass/attribution dual distortion + soft-bottleneck aggregation + temporal-aware BLP**의 깔끔한 조합으로, OpenVLA-OFT W4A4에서 BF16 97.1% → 96.3% (메모리 -73%, 1.52× 가속)을 달성. 한계는 LIBERO/OpenVLA 계열에 국한된 검증과 fixed allocation이지만, *VLA 양자화의 sensitivity criterion 자체*를 evidence-preservation 관점으로 재정의했다는 점에서 후속 연구(특히 diffusion/flow VLA로의 확장)의 baseline이 될 가치가 있다.

<!-- VERIFIED: pdf -->
