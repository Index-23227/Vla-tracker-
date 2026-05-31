# PAPO-VLA: Planning-Aware Policy Optimization for Vision-Language-Action Models

> **한 줄 요약**: ISCAS팀의 RL fine-tuning method로, manipulation trajectory의 step들을 "planning action"(고수준 결정)과 "execution action"(저수준 motion)으로 구분하고, causal sufficiency와 causal necessity의 harmonic mean으로 정의한 importance를 GRPO advantage 추정에 주입. LIBERO 4-suite 평균 **95.75%**(Long suite 94%), RoboTwin2.0 short/medium/long horizon에서 OpenVLA-OFT/π₀/RDT 대비 일관된 우위.

---

## 1. 배경 및 동기

### 기존 RL fine-tuning의 한계
- 기존 VLA에 대한 RL fine-tuning(예: TGRPO, GRAPE, MetaVLA)은 **모든 step에 동일한 advantage**를 부여 → 장기 horizon에서 critical decision의 학습 신호가 희석
- Manipulation trajectory에서 대부분의 step은 "단순 motion 실행"이고, 소수의 step만 task 성공/실패를 결정 ("approach to object"의 첫 직진 vs "어느 그릇에 부을지" 선택)
- 기존 GRPO는 group 내 advantage 정규화로 sparse reward 문제를 일부 완화하나, **time-step 내부의 importance 비대칭**은 다루지 않음

### 핵심 질문
- **Trajectory 내에서 "어떤 step이 planning decision인가"를 자동으로 식별할 수 있는가?**
- **이를 GRPO 같은 RL fine-tuning에 통합하여, 특히 long-horizon task에서 성능을 끌어올릴 수 있는가?**

📌 [Figure 1 삽입] — Planning action identification → causal importance → GRPO advantage 조정

---

## 2. 방법론 심층 분석

### 2.1 Planning Action Identification (Eq. 7)

각 step에 대해 score를 계산:

```
s_t = normalize(||a_t - a_{t-1}||) × g(outcome)
```

- **Action variation**: 인접 action의 차이 (방향 변화, 그리퍼 토글 등)
- **Outcome-aware gating**: 해당 action이 후속 trajectory의 성공으로 연결되는지

이 score가 threshold 이상인 step이 "planning action"으로 식별됨.

> ❓ **예상 질문**: Action variation으로 planning을 판단하는 것은 motion의 단순 jerk를 잡지 않는가?
> **답변**: 합리적 우려. 단순 변동(예: noise)을 planning으로 오인할 수 있음. Outcome-aware gating이 이를 필터링하는 역할이나, gating 자체가 GT outcome에 의존하여 학습 중 계산 비용 증가.

### 2.2 Causal Sufficiency (Eq. 8)

> "Preserving an action supports subsequent successful execution"

해당 step의 action을 **고정**한 상태에서 후속 trajectory의 평균 success rate. 높을수록 그 action이 성공의 충분조건에 가까움.

### 2.3 Causal Necessity (Eq. 9)

> "Perturbing an action degrades the outcome"

해당 step의 action에 **noise**를 주입한 상태에서 success rate 감소량. 높을수록 그 action이 필수.

### 2.4 Unified Importance (Eq. 10)

```
I_t = harmonic_mean(Sufficiency_t, Necessity_t) = 2·S·N / (S+N)
```

> ❓ **예상 질문**: 왜 harmonic mean인가? Arithmetic mean이 아니라?
> **답변**: Harmonic mean은 두 값 중 작은 쪽에 보수적. Sufficiency만 높고 Necessity 낮은 step(redundant good action)이나 그 반대(necessary but not sufficient)를 모두 페널티. F1-score와 같은 직관.

### 2.5 Modified GRPO Advantage (Eq. 11)

```
A_t^PAPO = A_t^GRPO · (1 + η · I_t)
```

- η=0.15가 optimal (range [0, 0.3])
- Planning action에 advantage가 amplify되어 gradient signal 강화

> ❓ **예상 질문**: η=0.15면 weak signal 아닌가?
> **답변**: 표면적으론 작지만, harmonic mean으로 I_t가 0~1로 정규화된 상태에서 15% 증폭은 학습 dynamics에서 큰 차이. 더 큰 η(0.3)에서는 over-weighting으로 성능 저하 — ablation 확인.

---

## 3. 데이터 및 학습 세부사항

| 항목 | 설정 |
|------|------|
| Datasets | LIBERO (Spatial, Object, Goal, Long) + RoboTwin2.0 (short/medium/long horizons) |
| Hardware | H100 GPU clusters |
| Optimizer | 미공개 |
| Steps | 미공개 |
| Base policy | 미명시 (논문은 method-agnostic; 결과는 OpenVLA-OFT class 기반) |

> ⚠️ **주요 정보 부재**: training compute(GPU 수), step count, learning rate, batch size 등 RL fine-tuning 재현에 필수적인 hyperparameter가 paper에 명시되지 않음.

---

## 4. 실험 결과 심층 분석

### 4.1 LIBERO 4-suite (Table 1)

| Task | OpenVLA | TGRPO | GRAPE | Nora | **PAPO-VLA** |
|------|---------|-------|-------|------|--------------|
| Spatial | - | - | - | - | **0.93** |
| Object | - | - | - | - | **0.98** |
| Goal | - | - | - | - | **0.98** |
| Long | - | - | - | - | **0.94** |
| **Average** | 0.76 | 0.81 | 0.80 | 0.87 | **0.96** |

→ Average 0.96 (=96%)은 Nora(0.87) 대비 +9pp, OpenVLA(0.76) 대비 +20pp.

특히 **Long suite에서 0.94**는 long-horizon에서의 planning advantage가 실제로 작동함을 시사.

### 4.2 RoboTwin2.0 (Table 2)

| Horizon | Steps | Avg Success |
|---------|-------|-------------|
| Short | 100-130 | 58.6% |
| Medium | 150-230 | 69.5% |
| Long / Extra-Long | 280-650 | 63.8% |

π₀, RDT, OpenVLA-OFT 모두 능가. 흥미롭게 Medium > Long > Short — 너무 짧으면 planning action이 적어 PAPO의 이점이 작고, 너무 길면 cascading error로 한계.

### 4.3 Ablation (Table 3)

| Variant | Spatial | Object | Goal | Long | Avg |
|---------|---------|--------|------|------|-----|
| w/o Suff. & Nec. | 0.85 | 0.88 | 0.79 | 0.53 | 0.76 |
| w/o Sufficiency | 0.89 | 0.90 | 0.87 | 0.80 | 0.87 |
| w/o Necessity | 0.88 | 0.92 | 0.89 | 0.85 | 0.89 |
| **Full PAPO-VLA** | 0.93 | 0.98 | 0.98 | 0.94 | **0.96** |

- **Long suite에서 가장 큰 차이**: w/o Suff.&Nec. 0.53 → Full 0.94 (+41pp)
- Sufficiency와 Necessity가 **상호 보완적** — 하나만 빼면 87~89%, 둘 다 빼면 76%
- Necessity > Sufficiency (Necessity 제거 시 0.89 vs Sufficiency 제거 시 0.87) — perturbation 기반 signal이 약간 더 robust

### 4.4 η Sensitivity

η ∈ [0, 0.3] 범위에서 0.15가 optimal. 0에서는 vanilla GRPO, 0.3 이상에서는 over-weighting으로 trajectory의 non-planning step 학습 신호 부족.

---

## 5. 관련 연구 비교

| Method | RL? | Step-level importance | Causal reasoning | LIBERO Avg |
|--------|-----|----------------------|-----------------|-----------|
| OpenVLA (SFT) | ✗ | ✗ | ✗ | 0.76 |
| TGRPO | ✓ | △ (token-level) | ✗ | 0.81 |
| GRAPE | ✓ | ✗ (uniform) | ✗ | 0.80 |
| Nora | ✓ | △ | ✗ | 0.87 |
| **PAPO-VLA** | **✓** | **✓ (planning-aware)** | **✓** | **0.96** |

### 핵심 차이
- Step-level **causal** importance를 명시적으로 추정한 최초의 VLA RL method
- Causal sufficiency/necessity를 RL advantage에 직접 주입하는 framework은 일반화 가능 (다른 base policy에도 적용 가능)

---

## 6. 한계 및 미해결 문제

### 방법론적 미비점
1. **Base policy 미명시**: "method-agnostic" 주장이지만 정확히 어떤 VLA 위에서 96%를 달성했는지 paper에서 명확히 식별 어려움 — 재현성 저하
2. **Training compute 미공개**: H100 cluster 외 GPU 수, step count, batch size 부재. RL fine-tuning이 비싼 점을 감안하면 critical
3. **Causal sufficiency/necessity 계산 비용**: 각 step마다 forward simulation(또는 perturbation rollout) 필요 → trajectory 길이에 비례 또는 제곱 비용. Scalability 의문
4. **Outcome gating의 의존성**: GT outcome에 의존하여 importance 계산 → online RL에서 outcome이 noisy하면 importance도 noisy
5. **Code 미공개**: GitHub repo 없음. Method 재구현에 hyperparameter 부족
6. **CALVIN, SimplerEnv, RLBench 등 미평가**: LIBERO + RoboTwin2.0만 평가 — manipulation 외 generalization 불명확
7. **Sufficiency/Necessity의 실제 계산 방식 모호**: Eq. 8-9가 expectation을 어떻게 sample하는지 (몇 개 rollout, 어떤 perturbation)에 대한 구체적 spec 부족

### Attribution 문제
- LIBERO 0.96 → 0.76 (OpenVLA SFT) 차이의 +20pp 중, RL 자체의 기여(GRPO baseline) vs PAPO 기여를 깔끔히 분리하는 baseline (vanilla GRPO with same compute) 부재
- Ablation table의 "w/o Suff.&Nec." = 0.76은 SFT-level → vanilla GRPO 결과와 거의 동일하다면 PAPO 전체가 그저 RL을 더 잘 작동시키는 trick인지, 본질적 새로움인지 의문

---

## 7. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Causal sufficiency/necessity를 RL advantage에 통합한 framework은 새로움 |
| **Technical depth** | ★★★☆☆ — Equations은 명확하나 sufficiency/necessity의 실제 estimator 구체성 부족 |
| **Experimental rigor** | ★★★☆☆ — LIBERO Avg 0.96, RoboTwin2.0 좋은 결과. 그러나 base policy/compute 미공개, public code 없음 |
| **Practical impact** | ★★★☆☆ — LIBERO Long 94%는 큰 의미. RL inference cost는 base policy와 동일하나 학습 cost는 추정 어려움 |
| **Reproducibility** | ★★☆☆☆ — Code, hyperparameter, base policy 모두 미공개 |

**강점**: Long-horizon task에서 planning action을 명시적으로 모델링하는 신선한 접근. Causal framework은 일반화 가능하며 ablation이 component별로 깔끔. **약점**: 재현성이 낮음 — base policy, training compute, causal estimator의 구현 detail 모두 모호. 96% Avg는 인상적이나 vanilla GRPO baseline (same compute, same base) 부재.

---

## 8. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Base policy가 정확히 무엇인가? | Paper에서 명시되지 않음. 결과 패턴은 OpenVLA-OFT class와 호환. method-agnostic claim의 실증 부족 |
| 2 | Causal sufficiency를 어떻게 계산하는가? Rollout 비용은? | Eq. 8에서 expectation을 어떻게 sample하는지 모호. 매 step마다 forward simulation이면 trajectory 길이 T에 대해 O(T²) 비용 |
| 3 | LIBERO 0.96은 어떤 base policy에 PAPO를 얹은 결과인가? | 명확하지 않으나 OpenVLA-OFT(94.4%) 위에서 +1.6pp 정도면 합리적. 그러나 SFT baseline이 76%로 보고되어 base가 lower라면 PAPO의 기여가 더 큼 |
| 4 | Vanilla GRPO baseline은? | Ablation의 "w/o Suff.&Nec."가 vanilla GRPO에 해당. 0.76은 SFT 수준 — PAPO 전체 효과의 의미 있는 분리 |
| 5 | Necessity가 Sufficiency보다 약간 더 중요한 이유는? | Perturbation 기반 signal이 trajectory의 실제 sensitivity를 포착. Sufficiency는 binary success/fail로 coarse |
| 6 | η=0.15는 어떻게 결정되었는가? | Grid search over [0, 0.3]. Optimal but range가 좁아 hyperparameter sensitivity 우려 |
| 7 | RoboTwin2.0 Long horizon이 Medium보다 낮은 이유는? | 280-650 step의 cascading error. PAPO도 episode 전체 horizon의 한계는 극복 못함 |
| 8 | Real-world 평가는? | 없음. Simulation only — LIBERO/RoboTwin2.0 모두 sim |
| 9 | CALVIN long-horizon에 적용하면? | 미평가. CALVIN ABC→D 같은 visual generalization은 PAPO와 직교한 문제 |
| 10 | Method가 inference 시 추가 비용을 유발하는가? | 아니오. PAPO는 학습 시 advantage 조정만 — inference policy는 base와 동일 |

<!-- VERIFIED: pdf -->
