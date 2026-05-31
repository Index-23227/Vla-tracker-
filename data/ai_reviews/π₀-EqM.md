# π₀-EqM: Equilibrium Matching for Closed-Loop Vision-Language-Action Control

> **한 줄 요약**: π₀의 flow-matching action decoder를 **시간 불변(time-invariant) 조건부 vector field**의 fixed-point가 executable action 분포와 일치하는 *Equilibrium Matching(EqM)* decoder로 교체. 추론을 forward Euler integration 대신 *residual에 대한 fixed-point iteration*으로 캐스팅 → 추론 step 수가 policy의 일부가 되는 "stationarity–executability gap"을 활용한 closed-loop 제어. RoboTwin 19-task에서 π₀ 40.4% → **50.2%** (+9.8%p), LIBERO-10에서 85.2% → **87.0%**, LIBERO 4-suite 평균 94.15% → 94.35%.

---

## 1. 배경 및 동기

### Flow-matching VLA의 구조적 한계
- π₀, Qwen-VLA 등 SOTA VLA는 시간 의존적 vector field $v_\theta(x, t, c)$를 학습 (t = flow time)
- 추론: $x_0 \to x_1$로 forward Euler integration (10~16 step)
- 문제 1: **t-conditioning이 expressivity 낭비** — 진짜로 필요한 것은 "action distribution"이며 t는 부수적
- 문제 2: **추론 step 수가 hyperparameter** — 적게 하면 부정확, 많이 하면 느림. 더 핵심은 *적은 step일 때 stationary point에 도달하지 못한 채 종료*

### 핵심 질문
- 시간 의존성을 **제거**하고 time-invariant equilibrium을 학습할 수 있는가?
- 추론 step 수를 **policy parameter**로 격상시켜 closed-loop 제어 품질을 조정할 수 있는가?

📌 [Figure 1] — π₀ frozen upstream + EqM decoder; inference as fixed-point iteration

---

## 2. 방법론 심층 분석

### 2.1 Equilibrium Matching (EqM)

표준 Flow Matching:
$$\mathcal{L}_{FM} = \mathbb{E}_{t, x_0, x_1} \|v_\theta(x_t, t, c) - (x_1 - x_0)\|^2$$

Equilibrium Matching:
$$\mathcal{L}_{EqM} = \mathbb{E}_{x, c}[\text{interpolant + structural noise objective such that}\ f^*(x^*, c) = 0]$$

- $f_\theta(x, c)$: **time-invariant** residual vector field
- $x^*$: fixed point ↔ executable action sample
- 학습 후 $f_\theta$의 fixed point set이 action distribution

### 2.2 추론: Fixed-Point Iteration
$$x^{k+1} = x^k - \eta \cdot f_\theta(x^k, c)$$

- $k = 0, 1, ..., K-1$, $K$는 추론 step 수
- $K$가 작으면 residual이 큼 → 덜 *stationary*하지만 빠른 응답
- $K$가 크면 stationary → 정확하지만 느림 → **stationarity–executability gap**

### 2.3 Stationarity–Executability Gap
- 핵심 관찰: **모든 태스크가 같은 K를 요구하지 않음**
- 빠른 closed-loop 응답이 필요한 dynamic 태스크: 작은 K, 큰 residual 허용
- Precision insertion: 큰 K, 작은 residual
- → **K는 task-dependent policy parameter**

### 2.4 π₀ upstream은 frozen
- VLM representation stack, conditioning interface 그대로
- Action decoder만 교체 → migration cost 최소화

> ❓ **예상 질문**: Time-invariance가 왜 expressivity를 늘리는가?
> **답변**: 시간 의존 vector field는 같은 (x, c)에 대해 t별로 다른 값 학습 — 시간 차원에 capacity 분산. EqM은 같은 capacity를 fixed-point 정확도에 집중 → executable manifold 표현력 ↑

> ❓ **예상 질문**: Fixed-point iteration의 수렴 보장은?
> **답변**: $f_\theta$가 contractive하면 수렴. 학습 objective가 contraction을 implicit하게 유도하나, **이론적 수렴 보장은 부재**. 실험적으로 K=8 내외에서 안정.

> ❓ **예상 질문**: Closed-loop "control"이 무슨 의미?
> **답변**: 매 action chunk마다 환경 관측을 다시 받아 fixed-point iteration 재시작. K가 작으면 frequency 높아져 closed-loop. K가 크면 open-loop에 가까움. 즉 EqM은 한 단일 모델로 control frequency tradeoff를 노출.

---

## 3. 데이터 전략
- π₀와 동일한 RoboTwin 19-task, LIBERO 4-suite training
- 별도 real-world 실험 부재 — 평가 fully simulated

---

## 4. 실험 결과 심층 분석

### 4.1 RoboTwin (Table I, 19 tasks)

| 모델 | 평균 |
|------|------|
| π₀ (flow-matching baseline) | 40.4 |
| **π₀-EqM** | **50.2 (+9.8)** |

**큰 향상이 보인 태스크:**
| Task | 향상 |
|------|------|
| pick_dual_bottles | +40 |
| place_cans_plasticbox | +33 |
| put_bottles_dustbin | +30 |

- **+9.8%p across 19 tasks는 대단히 큰 향상** — 단순한 decoder 교체로 얻기 어려운 격차
- 가장 큰 향상이 *bimanual + precision* 태스크에 집중 → fixed-point iteration이 정밀 제어에 적합함을 시사

### 4.2 LIBERO 4-suite (Table II)

| Suite | π₀ | π₀-EqM | Δ |
|-------|-----|--------|---|
| Spatial | 96.8 | 97.2 | +0.4 |
| Object | 98.8 | 98.4 | -0.4 |
| Goal | 95.8 | 94.8 | -1.0 |
| Long (10) | 85.2 | 87.0 | +1.8 |
| **Avg** | **94.15** | **94.35** | **+0.20** |

- LIBERO 평균은 거의 동일 — saturate 영역
- **LIBERO-10(Long)에서 +1.8%p**가 가장 의미 — fixed-point iteration이 long-horizon precision에 기여
- Object/Goal에서 -0.4/-1.0의 작은 손실은 simple task에서는 time-invariance가 손해

### 4.3 Stationarity–Executability Gap 분석
- 논문은 K 값에 따른 성공률 곡선 제시 (정확한 수치는 figure 의존)
- 핵심 발견: optimum K가 task별로 다름 → 단일 K hyperparameter로 묶을 수 없음

---

## 5. Ablation 분석 (한도 내)
- EqM vs π₀ flow-matching 직접 비교만 존재
- Interpolant 선택, structural noise schedule 등 세부 ablation은 정보 부족

---

## 6. 한계 및 미해결 문제

### 방법론적 미비점
1. **수렴 보장 부재**: Fixed-point iteration의 contraction 조건이 학습 중 보장되지 않음 — 일부 (x, c)에서 발산 가능성
2. **Real-world 실험 부재**: 전적으로 simulation 평가 → 실제 sensor noise / contact 동역학에서 검증 부재
3. **K 선택의 자동화 부재**: Stationarity–executability gap을 "policy parameter"로 제시하나 task별 optimal K 자동 결정 알고리즘 부재
4. **Code 미공개**: 재현성 보장 없음
5. **LIBERO Object/Goal에서 손실**: 단순 태스크에서 작은 negative — time-invariance의 만능성에 의문

### Attribution 문제
- RoboTwin +9.8%p가 **EqM 자체**인지, **π₀ baseline의 약함**인지 분리 어려움 (Qwen-VLA의 RoboTwin Hard 87.2%와 직접 비교 부재)
- Fixed-point iteration이 진짜 contractive하게 학습됐는지 정량 분석 부족

---

## 7. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Time-invariant equilibrium을 VLA action decoding에 도입은 신선 |
| **Technical depth** | ★★★★☆ — Fixed-point iteration framework가 우아 |
| **Experimental rigor** | ★★★☆☆ — Simulation만, ablation 부재, real-world 없음 |
| **Practical impact** | ★★★★☆ — Drop-in replacement, RoboTwin +9.8 매우 큼 |
| **Writing quality** | ★★★★☆ |

**강점**: Drop-in decoder 교체로 RoboTwin +9.8%p는 매우 큰 향상. Stationarity–executability gap이라는 새 control-policy primitive 제안. **약점**: 수렴 이론 부재, real-world 검증 없음, code 미공개, LIBERO 일부 suite 손실.

---

## 8. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Fixed-point iteration이 수렴하지 않으면? | 이론적 보장 부재. 실험적으로 K≤8에서 안정 보고. Degenerate (x, c)에서 행동 |
| 2 | Time-invariance가 왜 LIBERO Object/Goal에서 손해? | 단순 태스크는 short-horizon — time-conditioning이 helpful한 영역. EqM은 trade-off 존재 |
| 3 | RoboTwin +9.8%p의 통계적 유의성? | 19 tasks 평균 — variance 보고 부재. Seed 수 미명시 |
| 4 | Real-world 미평가의 영향? | Contact noise, sensor delay 등에서의 fixed-point 안정성 미검증 |
| 5 | Optimum K는 어떻게 정하나? | 수작업 — 자동화 부재. Task별 grid search 필요 |
| 6 | Qwen-VLA(RoboTwin Hard 87.2)와 비교? | 직접 비교 부재. EqM은 π₀ 기반이므로 baseline 자체 차이 |
| 7 | EqM의 학습 비용은? | 명시 부재. Time conditioning이 빠진 만큼 약간 낮을 것으로 추정 |
| 8 | "Closed-loop"의 정확한 정의? | 매 chunk마다 관측 갱신 후 iteration 재시작. 일반 flow-matching도 같은 방식 가능하므로 "Closed-loop"은 marketing 측면 |
| 9 | π₀.₅, Qwen-VLA에 EqM 적용 가능? | Decoder만 교체이므로 원리적으로 가능. 검증 부재 |

<!-- VERIFIED: pdf -->
