# ATHENA: Accelerated Multi-Task Heterogeneous Influence Functions for Robot Data Curation

> **한 줄 요약**: Billion-parameter 멀티태스크 VLA(π₀) 파인튜닝을 위한 **influence function 기반 데이터 큐레이션 프레임워크**. (1) Kronecker-compressed gradient featurization으로 projection 비용을 O(DP) → O(√DP)로, (2) Rank-r Random Truncated Approximation으로 Hessian inversion을 O(NP²+P³) → O(NPr)로 축소, (3) 50개 task 균형을 위한 MII(Multitask Influence Interaction) score를 도입. RoboTwin 2.0에서 **50% 데이터만으로** full-data joint fine-tuning과 동등/우월(clean 43.36% vs 43.42%, randomized 17.30% vs 15.44%), 실로봇 ALOHA 6 task에서 66.7% 데이터로 68.0% 성공률(Joint-100% 60.0% 대비 +8.0%p). 총 8054.6 → 25.7 GPU-hour의 **313.4× 속도 향상**.

---

## 1. 배경 및 동기

### 문제 정의: 왜 VLA 데이터 큐레이션이 어려운가?
- VLA 모델은 large-scale 로봇 demonstration에 의존하나, **단순 데이터 스케일링은 비용만 증가하고 성능은 정체/저하**
- 핵심 질문: "billion-parameter VLA 파인튜닝에서 *어떤* demonstration을 retain할 것인가?"
- 기존 접근의 한계:
  - **Leave-one-out / Data Shapley**: 매 후보 subset마다 재학습 — 계산 불가능
  - **Quality scoring / distillation**: 효율적이나 downstream policy 성능과의 *인과* 연결 부재
  - **Gradient-based influence (CUPID, QoQ, DataMIL)**: 원리적으로 우월하나 24M policy / single-task에 머무름 — billion-parameter 멀티태스크로 직접 확장 불가

### 두 가지 scaling bottleneck
1. **계산 bottleneck**: per-sample gradient projection O(DP) + dense Hessian inversion O(NP² + P³)
2. **멀티태스크 bottleneck**: greedy single-ranking은 strong-gradient task에 편향 → task coverage 불균형

📌 [Figure 1] — ATHENA pipeline: Training demos + closed-loop rollouts → accelerated influence estimation + multitask balanced sorting → curated subset → 재학습

---

## 2. 방법론 심층 분석

### 2.1 Closed-loop performance influence (Eq. 2)
기존 influence function은 step-wise loss를 가정하나, 로봇은 sequential trajectory의 closed-loop 성공률이 중요. CUPID를 따라 demonstration-level performance influence를 다음으로 정의:

$$\hat{\Psi}_{\pi\text{-inf}}(\xi_i) = \frac{1}{m}\sum_{\tau_j \in E}\frac{R(\tau_j)}{H_i}\sum_{\hat{z}\in\tau_j}\sum_{z\in\xi_i}\Psi_{a\text{-inf}}(\hat{z}, z)$$

여기서 R(τⱼ) ∈ {1, −1}는 rollout return.

### 2.2 Kronecker-Compressed Gradient Featurization (Eq. 5)
- Linear layer의 weight gradient는 outer-product 구조: δᵢˡ(xᵢˡ)ᵀ
- 양쪽 인자를 *activation space*에서 사전 projection:
  $$\bar{g}_i^\ell = (P_{out}^\ell)^\top \delta_i^\ell (x_i^\ell)^\top P_{in}^\ell$$
- 결과: 파라미터 공간에서의 dense projection 회피 → **O(DP) → O(√DP)** + full gradient materialize 불필요

### 2.3 Random Truncated Approximation (Eq. 6)
- G ≈ Uᵣ Σᵣ Vᵣᵀ 의 randomized rank-r SVD
- ϕ̃ᵢ = Vᵣᵀ ϕᵢ 로 r-차원 부분공간에 사영
- Influence: ψ̂_RTA(z_te, z_tr) = ϕ̃ᵀ_te (Σᵣ² + λIᵣ)⁻¹ ϕ̃_tr
- **O(NP²) → O(NPr)**, r ≪ P

### 2.4 Multitask Influence Interaction (Eq. 7–10)
- Task-local: Ψ̃_π-inf^c(i)(i) — i가 자기 task rollouts에 미친 영향
- Cross-task: Ψ̃_π-inf^{all-c(i)}(i) — i가 다른 task rollouts에 미친 영향
- 두 component는 서로 다른 numerical scale을 가지므로 **rank → normalized utility**:
  - uᵢᶜ = max(ε, 1 − rᵢᶜ/n_c)
  - uᵢ^{all-c} = max(ε, 1 − rᵢ^{all-c}/(n−n_c))
- **MII score**: fᵢ^MII = uᵢᶜ · uᵢ^{all-c} (product)
- 효과: greedy single-ranking이 빠뜨리는 task-local critical example + cross-task transfer 모두 보존

> ❓ **예상 질문**: 왜 product인가? sum이 아닌가?
> **답변**: Product는 두 utility 모두 높을 때만 보존 → "자기 task에도 유익 AND 다른 task에도 유익한" demonstration이 우선. Sum이면 한쪽 utility만 높아도 살아남아 균형이 깨짐.

---

## 3. 데이터 전략

### 시뮬레이션 (RoboTwin 2.0)
- **demo_clean split**: 50 task × 50 demos = 2,500 demonstration, 16.67Hz, 총 9.34h
- Clean / Randomized 두 evaluator로 in-dist 및 generalization 동시 측정

### 실로봇 (ALOHA 6 task)
- 720 demos, 25Hz, 총 6.9h
- 난이도 3단계: Pick Fruits / Wipe Board (쉬움), Stack Bowls / Box Return (중간), Seal Stamping / Shelf Retrieval (long-horizon)
- task당 25 trial, 무작위 object position

---

## 4. 실험 결과 심층 분석

### 4.1 RoboTwin 2.0 (Section 5.2.1)
ρ = 0.1 (10% retain) 극단 budget에서:

| Method | Clean | Randomized |
|--------|-------|------------|
| Full-data joint | 43.42 | 15.44 |
| Single-task π₀ (RoboTwin 2.0 보고) | 46.42 | 16.34 |
| **ATHENA ρ=0.1** | **44.70** | **17.72** |
| **ATHENA 50% data** | **43.36** | **17.30** |

- **핵심**: 50% 데이터의 ATHENA가 full-data joint를 randomized에서 +1.86%p 초과
- ρ=0.1에서조차 full-data를 clean +1.28, randomized +2.28 능가 — 90% 데이터를 버려도 더 잘함
- 누적 50 task 기준 **45.0-point 향상**
- TAROT (whitened feature distance) / Random / Oracle 모두 ATHENA에 열등

### 4.2 실로봇 ALOHA (Section 5.2.2, Figure 4)

| Method | 평균 성공률 | 스토리지/계산 |
|--------|------------|---------------|
| Single-100% (task별 6 체크포인트) | 46.7% | 300K steps, 240GB |
| Joint-100% | 60.0% | 단일 학습 |
| Random-66.7% | 50.0% | — |
| Oracle-66.7% | 47.3% | — |
| **ATHENA-66.7%** | **68.0%** | 단일 학습 |

- ATHENA는 Joint-100% 대비 +8.0%p, Single-100% 대비 +21.3%p
- Random/Oracle은 단순 필터링이 오히려 성능 하락 — quality≠downstream success 입증

### 4.3 Computational speedup (Table 1)
| K (task 수) | Demo timesteps | w/o Accel (GPU-h) | ATHENA (GPU-h) | Speedup |
|-------------|----------------|-------------------|----------------|---------|
| 5 | 30.2K | 446.2 | 1.1 | 405.6× |
| 10 | 66.5K | 885.5 | 2.4 | 369.0× |
| 25 | 291.9K | 3297.4 | 14.0 | 235.5× |
| **50** | **560.5K** | **8054.6** | **25.7** | **313.4×** |

- TRAK-style / CUPID-style baseline 대비 압도적 차이
- task 수가 늘수록 absolute saving은 커지나 speedup multiplier는 다소 감소(데이터 크기가 RTA 이득보다 빨리 증가)

---

## 5. Ablation 분석

- **Computational ablation** (Table 1): Kronecker featurization + RTA 제거시 정확히 baseline TRAK/CUPID pipeline으로 환원 — 두 component가 313.4× speedup의 source임을 보임
- **Multitask joint fine-tuning ablation** (§6.2): joint 학습 자체의 ROI 검증 — 단 figure 텍스트만으로는 세부 수치 추출 어려움
- **부재한 ablation**:
  - MII에서 product vs sum vs max
  - Kronecker 단독 / RTA 단독의 success rate 영향 (computation은 분리됐으나 *policy 성능*은 미분리)
  - rank r 의 sensitivity sweep

---

## 6. 한계 및 미해결 문제

### 방법론적 미비점
1. **새 정책 모델이 아님**: ATHENA는 *데이터 큐레이션 방법*이며, action head/backbone은 π₀ 그대로 → "VLA model"이라기보단 "VLA training methodology"
2. **Base VLA 의존성**: π₀ 외 다른 backbone(OpenVLA, RDT 등)에서의 검증 부재
3. **RoboTwin 2.0에 국한**: LIBERO, CALVIN, SimplerEnv 등 다른 표준 benchmark 미평가
4. **Code 공개 여부 불명확**: project website만 언급
5. **Influence approximation bias**: RTA + Kronecker projection의 누적 오차가 selection 품질에 미치는 영향 정량 부재

### Attribution 문제
- ATHENA-50%가 full-data를 능가하는 현상은 "data curation 효과"인지 "노이즈 demo 제거 효과"인지 분리 불가
- demo_clean split 자체가 이미 quality control된 set — 더 noisy한 raw set에서의 효과 미검증

---

## 7. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Kronecker + RTA 조합은 LoGRA 기반 확장이나 멀티태스크 MII는 새로움 |
| **Technical depth** | ★★★★★ — 두 가지 bottleneck을 명확히 정의하고 각각 algebra로 해결 |
| **Experimental rigor** | ★★★★☆ — 50 task simulation + 6 task 실로봇, baseline 다양 (TAROT/TSS/Distillation/Oracle/Random) |
| **Practical impact** | ★★★★★ — 313.4× speedup은 billion-parameter VLA 큐레이션의 실용 임계점을 넘김 |
| **Writing quality** | ★★★★☆ — 수식 명확, 그러나 일부 figure label·축이 불명확하게 렌더링 |

**강점**:
- 두 bottleneck(grad projection, Hessian) 모두에 algebra 수준 해법
- 50% 데이터로 full-data 초과 — data curation의 진정한 ROI 입증
- Joint 학습 + curation의 결합으로 single-task pipeline 대비 storage/compute 수십 배 절감

**약점**:
- 큐레이션 *방법*이지 새 VLA 아님 → leaderboard에서 score 해석 시 주의 필요
- π₀ 단일 backbone, RoboTwin 2.0 단일 benchmark
- Ablation의 깊이 부족 (특히 MII 디자인 선택)

---

## 8. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 50% 데이터로 full-data를 *능가*하는 게 어떻게 가능? | demo_clean에도 negative-influence demo 존재. 이들 제거가 학습 dynamics 정화. randomized eval에서 generalization 증가가 결정적 |
| 2 | Kronecker compression이 정확도를 해치진 않나? | LoGRA에서 검증된 outer-product 구조 보존. 정확도 ablation 부재가 약점 |
| 3 | RTA의 rank r 은 어떻게 정하나? | 본문에 sweep 없음. Appendix 의존 |
| 4 | MII가 sum 대신 product인 이유? | 양쪽 utility 모두 높을 때만 score 보존 → strict balance enforcement |
| 5 | π₀ 외 backbone에서도 동작하나? | 미검증. Kronecker는 linear layer에 일반적이므로 원리적 적용 가능 |
| 6 | LIBERO 등 다른 benchmark에서는? | 미평가. 일반화 주장의 폭이 좁음 |
| 7 | Oracle baseline이 ATHENA에 진다는 의미는? | 사람이 정의한 quality ≠ downstream causal contribution. 데이터 가치에 대한 근본적 통찰 |
| 8 | ATHENA 자체의 compute (25.7 GPU-h)는 작은 편인가? | 8054.6 → 25.7로 313.4× 향상이나, 50K-step 파인튜닝 자체보다는 여전히 큼. 데이터 재사용 시 amortize |
| 9 | 313.4× speedup 중 Kronecker vs RTA 각각의 기여? | per-component breakdown은 Appendix C.1로 deferred — 본문 부재 |
| 10 | 실로봇 ALOHA 6 task은 통계적으로 충분한가? | task당 25 trial로 confidence interval은 ±10%p 수준 — 신중한 해석 필요 |

---

## 9. 본 연구의 기여 정리

1. **알고리즘**: Kronecker-compressed gradient featurization + rank-r RTA → 313.4× influence 계산 가속
2. **이론**: MII proposition으로 멀티태스크 데이터 큐레이션을 task-local × cross-task utility product로 공식화
3. **실험**: RoboTwin 2.0 50 task에서 50% 데이터로 full-data 초과; 실로봇 ALOHA 6 task에서 66.7% 데이터로 +8%p
4. **시스템**: π₀-based billion-parameter VLA에 직접 적용 가능한 첫 influence-function curation 프레임워크

---

## 10. 후속 연구 방향

- **Cross-backbone 검증**: OpenVLA, RDT-1B, π₀.₅ 등에 동일 파이프라인 적용
- **Online curation**: 학습 중간 checkpoint에서의 dynamic re-selection
- **MII 디자인 공간**: product 외 copula / harmonic mean 등의 비교
- **Heterogeneous data sources**: human video + sim + real 혼합에서의 cross-source influence

---

## 11. 실용적 함의

- **Robotics lab 관점**: VLA fine-tuning 시 무작정 데이터 늘리지 말고 ATHENA-style influence 큐레이션을 고려할 가치 — 50% data로 같은 성능이면 데이터 수집 비용 절반
- **Inference 비용 동일**: 큐레이션은 학습-time만 영향 → deployment에 부담 없음
- **Quality score 함정 경고**: Oracle baseline의 underperformance는 "human-judged quality"에 대한 맹신을 경계하라는 강한 신호

---

## 12. 결론

ATHENA는 billion-parameter 멀티태스크 VLA 파인튜닝에서 influence-function-based data curation을 *실현 가능*한 수준으로 끌어올린 작업이다. Kronecker compression + RTA + MII의 세 축은 각각 (1) projection cost, (2) Hessian cost, (3) multitask balance라는 명확한 bottleneck을 algebra 수준에서 해결한다. RoboTwin 2.0에서 50% 데이터로 full-data 동등, 실로봇 ALOHA에서 66.7% 데이터로 Joint-100% 대비 +8%p — data quantity 일변도 trend에 강한 반론을 제기한다.

다만 본 연구가 새로운 *정책 모델*이 아니라 *데이터 큐레이션 방법*이라는 점을 leaderboard 해석에서 분명히 해야 하며, π₀ 단일 backbone 및 RoboTwin 2.0 단일 benchmark에 머문 점은 일반화 주장의 폭을 좁힌다. 후속 작업이 cross-backbone, cross-benchmark, online curation으로 확장된다면 VLA training methodology의 표준 component로 자리잡을 가능성이 높다.

<!-- VERIFIED: pdf -->
