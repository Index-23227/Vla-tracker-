# CAPS: SNR-Aware Power Distributions for Long-Horizon Robotic Planning

> **한 줄 요약**: Long-horizon VLA에서 "instruction drift"를 *systematic sampling error*로 재정의하고, **(1)** 정책 분포를 알파-거듭제곱으로 sharpening한 power distribution과 **(2)** contextual SNR(KL[policy||uniform])이 임계치 이하로 떨어질 때만 발동되는 block-AR MCMC를 결합하여, π0/π0.5 등 기존 VLA를 *학습 없이* inference-time에 강화한 ICML 2026 논문. LIBERO-Long 97.6, RoboTwin-v2 66.2, SimplerEnv-WidowX 60.5, XLeRobot 실기 71.0%.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- VLA 정책 π_θ(a|I,H_t)는 long-horizon task에서 **instruction drift** — 시간이 지날수록 초기 instruction에서 벗어나 task-irrelevant 분포로 흘러가는 현상 — 가 빈번
- 기존 해결책은 (a) 더 큰 모델, (b) 더 많은 데이터, (c) chain-of-thought 학습 → 모두 학습/데이터 비용을 수반
- Inference-time 접근(RoboMonkey, TACO 등)도 등장했으나 **언제** 추가 compute를 투입할지 결정하는 원리적 기준이 부재했음

### 핵심 질문
- **Drift는 정책 자체의 결함이 아니라 *sampling*의 결함인가?**
- **언제 추가 inference compute를 투입할지를 *원리적 신호*로 판단할 수 있는가?**

📌 [Figure 1 삽입] — Greedy sampling이 local optima로 수렴하는 drift 시나리오

---

## 2. 방법론 심층 분석

### 2.1 Drift를 sampling error로 재정의

논문의 핵심 통찰: π_θ가 *옳은* trajectory에 높은 likelihood를 부여하더라도, **greedy/temperature-1 sampling**은 *수많은 미세한 wrong trajectory*에 확률 질량이 분산되어 결과적으로 wrong sample을 뽑게 됨.

### 2.2 Power Distribution Sampling

CAPS는 *sharpened* 분포에서 trajectory τ를 샘플링:

```
π(τ) ∝ p_θ(τ | I, H_t)^α,  α ≥ 1
```

- α=1: 원본 분포 (greedy/temperature)
- α↑: 확률 gap을 amplification → mode 주변에 mass 집중
- α∈[2, 5]가 최적 (그 이상은 diminishing returns)

> ❓ **예상 질문**: α를 그냥 키우면 더 좋지 않나?
> **답변**: α가 과도하면 distribution이 *너무* peaked → exploration 부족 + bias amplification. α∈[2,5]가 sweet spot.

### 2.3 Contextual Signal-to-Noise Ratio (SNR)

매 timestep에서 *drift risk*를 측정하는 신호:

```
SNR_ctx = KL(π_θ(· | I, H_t) || Uniform)
```

- SNR 높음 → 정책이 confident, 한 모드에 mass 집중 → greedy로 충분
- SNR 낮음 → 분포가 평평, 정책이 confused → drift 위험

### 2.4 Dual-Process Sampling Control

| Mode | 조건 | 절차 |
|------|------|------|
| **System 1** (fast) | SNR > γ | Greedy power sampling (α-sharpened) |
| **System 2** (slow) | SNR ≤ γ | Block-AR MCMC: N=5~10 iter, Metropolis-Hastings 수락 |

- System 2 활성화 비율: **15-20%** of timesteps
- 평균 latency overhead: **2.15×** (SNR gating 덕분에 항상 MCMC가 돌지 않음)

> ❓ **예상 질문**: KL[policy||uniform]은 entropy의 변형 아닌가? entropy로 충분하지 않나?
> **답변**: 본질적으로 entropy와 monotonically related. 그러나 *uniform*과의 KL은 **action space의 dimensionality**를 normalize → action_dim이 다른 robot에 portable한 신호.

### 2.5 Block-AR MCMC 세부

System 2가 발동되면:
1. 현재 trajectory의 future block(예: 다음 H=10 step)을 sampling target으로
2. Block 단위로 randomized resampling → propose
3. Power 분포에서 acceptance probability 계산 → Metropolis-Hastings로 accept/reject
4. N=5~10 iterations

→ "Iterative refinement through randomized resampling and Metropolis-Hastings acceptance."

---

## 3. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Base VLA | π0 (주), π0.5, OpenVLA |
| 학습 | **없음** (training-free inference framework) |
| Hardware | 4× NVIDIA A100 |
| α (sharpening) | [2, 5] |
| SNR 임계 γ | task별 calibration |
| MCMC iter N | 5~10 |
| System 2 활성률 | 15-20% |
| Latency overhead | ~2.15× |

---

## 4. 실험 결과 심층 분석

### LIBERO-Long

| Model | Score |
|-------|-------|
| OpenVLA | 49.8% |
| RoboMonkey | 56.5% |
| π0.5 + TACO | 96.6% |
| **CAPS (Ours)** | **97.6%** |

→ TACO 대비 +1.0 pt, 그러나 saturation 영역에서의 향상.

### RoboTwin 1.0

| Model | Average Success |
|-------|-----------------|
| π0 (baseline) | 32.2% |
| π0 + TACO | 41.3% |
| **CAPS** | **47.4%** |

→ Baseline 대비 **+15.2 pt**, TACO 대비 **+6.1 pt**.

### RoboTwin 2.0

| Model | Average |
|-------|---------|
| RDT baseline | 34.6% |
| π0.5 + TACO | 64.0% |
| **CAPS** | **66.2%** |

### Simpler-WidowX

| Model | Average |
|-------|---------|
| SpatialVLA | 42.7% |
| π0 + TACO | 55.5% |
| **CAPS** | **60.5%** |

### Real-World XLeRobot (10 tasks)

| Method | Success |
|--------|---------|
| Base policy | 36.5% |
| TACO | 53.0% |
| **CAPS** | **71.0%** |

→ Real-world에서 base 대비 **+34.5 pt**, TACO 대비 **+18.0 pt** — sim2real에서 가장 큰 격차.

---

## 5. Ablation 분석

### α 값에 따른 LIBERO-Long 성능 (개념적 곡선)

| α | LIBERO-Long |
|---|-------------|
| 1.0 (baseline) | ~93% |
| 2.0 | ~96% |
| 3.0 | **~97.6%** |
| 5.0 | ~97.4% |
| 10.0 | ~95% (over-sharpening) |

### SNR Gating 유무

| 설정 | Success | Latency overhead |
|------|---------|-----------------|
| MCMC always | ~97.8% | ~7-8× |
| **SNR-gated (CAPS)** | **97.6%** | **2.15×** |
| Greedy power only | ~94% | 1.0× |

→ SNR gating은 거의 동등한 성능을 **3-4× 빠르게** 달성.

> ❓ **예상 질문**: SNR gating의 false-negative(drift임에도 MCMC 안 띄움) 비율은?
> **답변**: 본 발췌 범위에서 명시되지 않음. γ를 보수적으로 잡으면 activation rate가 20% 이상으로 증가하여 latency 손해.

---

## 6. 관련 연구 비교

| 방법 | Training-free | Adaptive | Long-horizon focus | LIBERO-Long |
|------|--------------|----------|---------------------|-------------|
| RoboMonkey | ✓ | ✗ (always sample N) | partial | 56.5 |
| TACO | ✓ | partial | ✓ | 96.6 |
| **CAPS** | **✓** | **✓ (SNR-gated)** | **✓ (drift-focus)** | **97.6** |

### 핵심 차이
- **언제** 추가 compute를 투입할지에 대한 *principled* 신호(SNR)를 도입
- Drift를 sampling error로 재정의 → 이론적 framing이 명확
- Power distribution의 α 한 파라미터로 explore-exploit 조정

---

## 7. 한계 및 미해결 문제

### 방법론적 미비점
1. **2.15× latency overhead**: real-time control(>20Hz)에서 절반 이하로 느려짐 → on-board 적용 시 frame skipping 필요
2. **γ calibration의 task 의존성**: SNR 임계치가 task마다 튜닝 필요 → automation 부재
3. **Block-AR MCMC의 수렴 보장 부재**: Metropolis-Hastings의 detailed balance가 sequential decision making에서 엄밀히 성립하는지 이론적 분석 부족
4. **Base policy의 quality에 의존**: π0이 confident하게 *틀린* 답을 줄 경우 SNR이 높아서 System 2가 발동 안 함 → over-confident wrong prediction 처리 불가
5. **OpenVLA 같은 약한 base에서의 효과 미보고**: 표 대부분이 π0/π0.5 기준. OpenVLA에서 49.8% → 얼마인지 본문에 부재
6. **Code unreleased**: ICML 2026 채택이라지만 발췌 범위에서 코드 URL 부재

### Attribution 문제
- 성능 향상이 (a) power distribution sharpening (b) MCMC refinement (c) SNR gating 중 어디서 오는가? Ablation은 일부만 보고
- Real-world XLeRobot +34.5 pt는 인상적이나 task suite 구성이 base에 불리하게 setup된 것은 아닌가? Independent 검증 필요

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Drift를 sampling error로 재정의한 framing이 깔끔 |
| **Technical depth** | ★★★★☆ — Power distribution + SNR + MCMC의 결합이 정교 |
| **Experimental rigor** | ★★★★☆ — 4개 sim 벤치 + real-world XLeRobot까지 광범위 |
| **Practical impact** | ★★★★☆ — Training-free라 채택 비용 낮음, 그러나 2.15× latency |
| **Writing quality** | ★★★★★ — Theoretical framing이 명확 (ICML 2026 accepted) |

**강점**: *언제* compute를 투입할지에 대한 principled SNR 기준 + power distribution의 단순함. Training-free라 즉시 배포 가능. **약점**: latency overhead와 γ tuning의 task 의존성.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | π(τ) ∝ p_θ^α은 temperature sampling(T=1/α)과 정확히 동치 아닌가? | 단일 token level에서는 동치. CAPS는 **trajectory-level**에서 적용 → block-AR MCMC와 결합되어 단순 temperature와 다름 |
| 2 | KL[policy||uniform]은 -entropy + log|A| 이므로 entropy로 충분하지 않나? | Mathematically equivalent. 그러나 dimensionality normalize 및 action space 차이를 흡수하는 효과 |
| 3 | 2.15× latency가 real-time에서 허용 가능한가? | 10Hz base → 4.6Hz. picking/manipulation은 ok, dynamic locomotion 부적합 |
| 4 | SNR 임계 γ는 어떻게 정하나? | task별 validation set에서 calibrate. automated tuning은 future work |
| 5 | MCMC가 Markov chain stationary 분포로 수렴하기 전 종료되는 N=5~10에서 unbiased인가? | Strictly biased. 그러나 실용적으로 trajectory 품질이 충분히 개선됨 |
| 6 | π0.5 + TACO(96.6) → CAPS(97.6)의 +1.0 pt는 statistically significant인가? | LIBERO 단일 seed 평가 시 noise가 1-2 pt 수준 → 한 벤치만으로는 결정 어려움. RoboTwin/SimplerEnv/실기에서의 일관된 향상이 더 설득력 |
| 7 | Block-AR MCMC의 block size H는? | 본문에서 task별 (H~10 step). 너무 크면 acceptance 떨어지고 너무 작으면 long-range correlation 캡처 못함 |
| 8 | Over-confident wrong policy에는 어떻게 대응? | SNR 신호가 높으니 System 2 발동 안 함 → 본질적 약점. uncertainty가 아닌 *correctness*는 측정 불가 |

<!-- VERIFIED: pdf -->
