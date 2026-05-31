# COAST: Contrastive Conceptor Activation Steering — Unlocking Vision-Language-Action Models through Hidden States

> **한 줄 요약**: VLA의 hidden state에서 성공/실패 rollout으로 conceptor 행렬 (C = R(R + α⁻²I)⁻¹)을 닫힌 형태로 추정한 뒤, 대조적 결합 C_steer = C_success ∧ ¬C_failure 을 multiplicative gating M = (1-β)I + βC_steer 형태로 inference 시 hidden state에 적용하는 **training-free activation steering** 기법. π0.5 LIBERO-10 0.43→0.80, RoboCasa 0.40→0.56, 실로봇 DROID +40% 성능 향상.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 최근 VLA(π0, π0.5, GR00T 등)는 사전학습으로 풍부한 표상을 학습하지만, **downstream task에서는 잠재 능력이 충분히 발현되지 않음** — 특히 LIBERO-10 같은 long-horizon에서 π0.5 baseline이 0.43에 머무름.
- 기존 개선책은 (a) **fine-tuning** (계산 비용·forgetting 위험), (b) **prompt engineering** (한계 명확), (c) **CAA(Contrastive Activation Addition)** 류 LLM steering (additive vector → VLA 같이 다단계 디노이징 정책엔 부족).
- VLA hidden state의 어떤 **subspace** 가 성공을 결정하는지에 대한 mechanistic 분석 부재.

### 핵심 질문
- **VLA의 hidden state 안에 이미 "성공 방향"이 존재하는가? 그렇다면 별도 학습 없이 그것만 강조해도 성능이 오르는가?**
- **하나의 steering 기법이 flow-matching (π0.5) / autoregressive (π0-FAST) / Diffusion Policy 모두에 작동하는가?**

📌 [Figure 1 삽입] — COAST 파이프라인: 성공·실패 rollout → activation 수집 → conceptor C_success, C_failure 추정 → 대조적 결합 → multiplicative gating으로 hidden state 사출.

---

## 2. 방법론 심층 분석

### 2.1 Conceptor 기초
Conceptor C 는 활성화 벡터 집합의 주성분 부분공간에 데이터를 **soft-projection** 하는 선형 연산자다. 평균-중심화한 활성화 covariance R 로부터 닫힌 형태로:

```
C = R (R + α⁻² I)⁻¹
```

- **α (aperture)**: 정규화 강도. α↑ → conceptor가 더 많은 방향을 포함.
- 고유값 분해 관점에서 R = UΛUᵀ 이면 C = U diag(λᵢ/(λᵢ + α⁻²)) Uᵀ — 큰 분산 방향은 1에 가깝게, 작은 방향은 0에 가깝게 게이팅.

### 2.2 Contrastive Conceptor

성공 rollout과 실패 rollout 각각에서 hidden activation을 수집해 두 개의 conceptor를 추정:
- **C_success**: 성공 활성화의 주요 부분공간
- **C_failure**: 실패 활성화의 주요 부분공간

대조적 결합 (Boolean 연산 — conceptor 대수의 핵심):
```
C_steer = C_success ∧ ¬C_failure
```
이는 "성공에 있지만 실패에는 없는" 방향만 남긴다.

### 2.3 Multiplicative Gating

추론 시 hidden state h 를 다음으로 갱신:
```
h' = M h,   M = (1 - β) I + β C_steer
```
- **β ∈ [0, 1]**: steering 강도. β=0 → baseline, β=1 → 완전 사출.

### 2.4 Per-Step 변형

Flow-matching VLA(π0.5)는 다단계 디노이징을 수행. 단일 global conceptor 대신 **denoising step별 separate conceptor**(per-step)를 학습/적용하면 성능이 추가로 향상. 이는 step마다 hidden state 분포가 달라진다는 가정과 부합.

> ❓ **예상 질문**: Per-step이 step마다 별도 covariance를 추정하면 데이터가 부족하지 않는가?
> **답변**: Conceptor는 closed-form 추정이라 수십 rollout으로 충분. 또한 rollout당 다단계가 모두 같은 step index에 누적되므로 effective sample이 충분히 큼.

📌 [Figure 2/3 삽입] — Conceptor 부분공간의 시각화 (PCA projection), per-step vs global 비교.

---

## 3. 데이터 전략

- **Calibration data**: task당 성공/실패 rollout 수십 개 (구체 수는 논문 본문 참조). baseline VLA 자체에서 수집 가능 → 추가 휴먼 어노테이션 없음.
- **No additional training data**: closed-form matrix inversion만으로 conceptor 추정.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| 학습 단계 | 없음 (training-free) |
| 추론 추가 비용 | hidden_dim × hidden_dim 행렬 곱 1회/layer |
| Hyperparameter | α (aperture), β (steering strength) |
| 적용 backbone | π0.5, π0-FAST, GR00T-N1.5, Diffusion Policy |

> ❓ **예상 질문**: Multiplicative gating의 layer 선택은 어떻게 정하는가?
> **답변**: 논문에서는 후반 transformer block(action-relevant)에 적용. Ablation에서 layer-sweep 결과를 제시.

---

## 5. 실험 결과 심층 분석

### 5.1 LIBERO-10 (π0.5)

| 메트릭 | Baseline | + COAST (per-step) | Δ |
|--------|---------|-------------------|---|
| Mean SR | 0.43 | **0.80** | **+0.37** |

극적 회복 사례:
- Two Mokas: 0.07 → 0.60
- Mug+Micro: 0.13 → 0.60
- Soup+Cheese: 0.80 → 1.00

### 5.2 MetaWorld ML45 (π0.5)

| 메트릭 | Baseline | + COAST | Δ |
|--------|---------|---------|---|
| Mean SR | 0.69 | **0.94** | **+0.25** |

- pick-place-wall 0.20 → 1.00, stick-push 0.20 → 0.73 등 long-tail failure가 거의 해소.

### 5.3 RoboCasa (멀티 백본)

| Backbone | Baseline | + COAST | Δ |
|----------|---------|---------|---|
| π0.5 | 0.40 | **0.56** | +0.17 |
| GR00T N1.5 | 0.59 | **0.75** | +0.16 |
| Diffusion Policy | 0.32 | **0.46** | +0.14 |

→ **세 가지 이질적 정책(flow / AR / diffusion) 모두에서 일관된 향상** — conceptor가 backbone-agnostic 표상을 활용한다는 강력한 증거.

### 5.4 실로봇 (DROID)

π0.5 + COAST가 3개 task에서 평균 **+40%** 성공률 향상.

### 5.5 Ablation

| 변형 | 결과 |
|------|------|
| Global conceptor | per-step 대비 열위 |
| Positive-only (¬C_failure 제거) | contrastive 대비 열위 |
| Additive CAA (vector add) | conceptor (subspace projection) 대비 열위 |
| SAE 기반 feature steering | conceptor 대비 열위 |

→ **대조성(contrast)** + **subspace projection** + **per-step** 세 요소 모두 필요.

---

## 6. 관련 연구 비교

| 방법 | 형태 | Training | VLA 적용 | 효과 |
|------|------|----------|---------|------|
| CAA (LLM) | additive vector | none | × | LLM에서만 |
| RepE | linear probe + addition | linear probe 학습 | △ | 약함 |
| SAE steering | sparse feature toggle | SAE 학습 | △ | 비용 큼 |
| Fine-tuning (LoRA) | gradient | yes | ○ | forgetting 위험 |
| **COAST** | **multiplicative projection** | **none** | **○ (3종 backbone)** | **+25~40%** |

---

## 7. 한계 및 미해결 문제

1. **Calibration rollout 필요**: 성공/실패 rollout 수집이 필요하므로 baseline policy가 어느 정도는 작동해야 함. 0% 성공 task에는 적용 어려움.
2. **Task-specific conceptor**: 각 task별로 conceptor를 재추정해야 함. Task generalization 가능성은 미검증.
3. **Layer 선택의 휴리스틱**: 어떤 transformer block을 steering할지 task-dependent일 수 있음.
4. **이론적 보장 부재**: "성공 부분공간"이 정말 task-success를 인과적으로 결정하는지, 아니면 단지 correlated activation pattern인지에 대한 인과 분석 없음.
5. **Code 미공개**: NeurIPS 2026 submission으로 표기되어 있으나 현재 GitHub 공개 없음 — 재현성 우려.
6. **Real-world 평가가 3 task로 제한**: DROID 3 task만 보고. 더 다양한 환경/객체에서의 robustness 검증 필요.

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Conceptor를 VLA hidden state steering에 적용한 최초 사례, per-step 변형의 통찰 |
| **Technical depth** | ★★★★☆ — closed-form 수식이 우아하고 ablation도 체계적 |
| **Experimental rigor** | ★★★★☆ — 3종 백본 + LIBERO + MetaWorld + RoboCasa + DROID 매우 광범위 |
| **Practical impact** | ★★★★★ — Training-free이며 즉시 배포 가능, 비용 거의 0 |
| **Writing quality** | ★★★★☆ — 명확한 motivation, 표·그림 풍부 |

**강점**: Training-free로 LIBERO-10 0.43→0.80 (+37%p)이라는 극적인 향상을 보여줌. 세 가지 이질적 backbone(flow / AR / diffusion)에서 일관된 효과는 conceptor가 VLA 표상 일반에 유효함을 시사. **약점**: code 미공개, calibration rollout 의존성, layer-선택의 휴리스틱.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Conceptor가 단순한 PCA whitening과 본질적으로 다른가? | C = U diag(λ/(λ+α⁻²)) Uᵀ 는 soft-projection — α가 큰 분산만 살리고 작은 분산을 억제. PCA는 hard threshold (top-k components). Conceptor는 연속적이고 Boolean 대수(∧, ¬) 연산이 가능해 contrastive 결합이 가능. |
| 2 | C_success ∧ ¬C_failure가 underflow되지 않는가? (즉, 모든 activation이 둘 다에 속하는 경우) | 가능. 실제로 ablation에서 positive-only가 0%인 case가 있는 task에서는 contrastive가 약간만 우월. 그러나 평균적으로 contrast가 우월. |
| 3 | Per-step conceptor는 flow-matching에서 step이 보통 10-50개인데 모두 별도 추정인가? | 그렇다. 각 step에서 activation을 수집해 step-specific conceptor. 효율을 위해 step grouping (e.g., early/mid/late)도 가능 — 논문은 per-step 최적. |
| 4 | β를 task별로 튜닝해야 하는가? | 본 논문은 β를 일정 범위(0.3~0.7)에서 grid-search. 자동 튜닝은 future work. |
| 5 | 왜 layer 후반에 적용하는가? | Action-relevant feature가 후반 layer에 집중. Layer-sweep ablation에서 후반이 최적임을 확인. |
| 6 | Steering이 다른 task에 negative transfer를 일으키지 않는가? | Task-specific conceptor이므로 적용 task에만 영향. 다만 cross-task evaluation(한 task의 conceptor를 다른 task에 적용)은 본 논문에서 미검증. |
| 7 | DROID 실로봇 +40%는 어떤 task인가? | 3개 task 평균. 구체 task는 본문 Sec. 5 참조. 3개로 통계적 유의성은 제한적. |
| 8 | Code가 없다면 재현 가능한가? | Closed-form formula는 단순하지만, "어느 layer, 어느 hyperparameter" 같은 detail이 빠지면 재현이 어려움. NeurIPS 2026 채택 시 code 공개 기대. |
| 9 | Activation steering이 long-horizon에서 step간 일관성을 깨지 않는가? | Per-step conceptor가 이를 완화. 그러나 매우 긴 horizon에서는 누적 drift 가능성 — 본 논문은 LIBERO-10(10 step) 정도까지만 검증. |
| 10 | Fine-tuning과의 비교는? | 본 논문은 SFT/LoRA 비교 없이 baseline VLA만 비교. Fine-tuning 대비 효율(시간/계산)은 압도적이지만 절대 성능이 fine-tuned VLA를 능가하는지는 별도 실험 필요. |

<!-- VERIFIED: pdf -->
