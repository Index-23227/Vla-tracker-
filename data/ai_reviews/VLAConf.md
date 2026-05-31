# VLAConf: Calibrated Task-Success Confidence for Vision-Language-Action Models

> **한 줄 요약**: Frozen VLA의 internal hidden state에 두 층 MLP one-class discriminator를 얹어 Coin-Flip Network 목적함수로 학습하고 Platt scaling으로 보정하는 경량 confidence head. Single-pass로 discrete/continuous action space 모두 지원하며 pi-0.5 백본 기준 LIBERO 성공률 91.3%에 대해 online Brier 0.0668의 우수한 calibration을 달성.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- VLA를 실로봇에 배포할 때 "실패 가능성"을 사전 예측하는 confidence score가 필수 (safety, human handover)
- 기존 접근: **Ensemble + variance** — 다수 forward pass 필요, latency 폭발
- 기존 접근: **Token-level perplexity** — discrete action에만 적용, continuous flow-matching head에서는 정의 불가
- 결과: **단일 forward로 success probability를 추정하는 universal한 방법 부재**

### 핵심 질문
- **Action output space가 아닌 hidden representation에서 confidence를 뽑을 수 있는가?**
- **Discrete/continuous 양 종류의 VLA에 동일한 framework가 적용되는가?**

📌 [Figure 1 삽입] — VLAConf: frozen VLA hidden state → step-conditioned one-class scoring → Platt calibration

---

## 2. 방법론 심층 분석

### 2.1 Coin-Flip Network (CFN) 목적함수

- Success-only 데모로부터 학습된 **one-class anomaly detector**
- Hidden state h_t를 입력받아, "랜덤 coin vector" v ~ Uniform({+1,-1}^d)와 dot product를 예측하는 MSE regression
- 직관: success trajectory의 representation manifold 위에서는 모든 coin과 일관된 inner product을 학습 가능, failure는 manifold 밖이라 inconsistent → MSE residual = anomaly score

> ❓ **예상 질문**: 왜 one-class detection이 success/failure 양 클래스 분류보다 나은가?
> **답변**: Failure rollout은 분포가 매우 다양(다양한 실패 모드) → 양 클래스 분류는 class imbalance와 mode coverage 어려움. One-class는 success manifold만 학습하면 됨. Coin-Flip은 self-supervised로 representation diversity를 induce.

### 2.2 Step-conditioned Aggregation

- 매 step t의 anomaly score s_t를 trajectory prefix [s_1,...,s_t]에 대해 aggregate
- Normalization horizon K (96~256 steps)로 sliding window 평균
- Step-conditioning: 초기/말기 step의 confidence semantic이 다름을 반영

### 2.3 Platt Scaling Calibration

- Raw anomaly score → success probability 변환을 위해 2-parameter Platt sigmoid 학습
- 별도의 outcome-labeled rollout으로 calibration set 구성
- Brier score, NLL 평가 용이

> ❓ **예상 질문**: Temperature scaling이나 isotonic regression 대신 Platt을 쓴 이유?
> **답변**: Platt(2 params)이 가장 적은 calibration set으로도 안정. Isotonic은 monotonicity 가정으로 over-fit 위험.

### 2.4 Coin-vector dim d=64

- Coin vector 차원 d가 너무 작으면 manifold representation 부족, 너무 크면 noise
- d=64가 sweet spot

---

## 3. 데이터 전략

### Training Data
- **Success-only demonstrations** from LIBERO standard suites
- Frozen VLA로부터 hidden state extract (학습 중 backbone 비고정 X)
- Backbone: **OpenVLA-OFT** (discrete) + **pi-0.5** (continuous flow-matching)

### Calibration Data
- 별도의 outcome-labeled rollouts (성공/실패 라벨링)
- Platt parameter 2개만 학습 → 작은 set 충분

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Backbone | OpenVLA-OFT, pi-0.5 (둘 다 frozen) |
| Confidence head | 2-layer MLP, d=64 |
| Optimizer | AdamW |
| Calibration | Platt scaling (2 parameters) |
| Inference | 64.9ms (OpenVLA-OFT), 164.7ms (pi-0.5) — single forward |

---

## 5. 실험 설계 및 평가 프로토콜

평가 지표가 **calibration 중심**:
1. **Brier score** (낮을수록 좋음) — squared error of confidence vs outcome
2. **NLL** (negative log likelihood) — proper scoring rule
3. **Pre-Exec vs Online**: 실행 전 예측 vs 실행 중 갱신
4. **Robustness**: LIBERO-Pro (perturbation), LIBERO-Plus (extended)

---

## 6. 실험 결과 심층 분석

### LIBERO Standard (Table 핵심)

| Metric | OpenVLA-OFT | pi-0.5 |
|--------|-------------|--------|
| Success Rate | 78.2% | **91.3%** |
| Pre-Exec Brier ↓ | 0.1614 | **0.0821** |
| Online Brier ↓ | 0.1073 | **0.0668** |
| Pre-Exec NLL ↓ | 0.4991 | 0.3141 |
| Online NLL ↓ | 0.3335 | 0.2501 |
| Inference (ms) | 64.9 | 164.7 |

- pi-0.5에서 online Brier 0.0668은 매우 좋은 calibration (perfect = 0)
- OpenVLA-OFT가 base success rate가 낮지만, calibration의 절대 품질도 떨어짐 → "calibration은 model quality와 entangle"

### Robustness — LIBERO-Pro / LIBERO-Plus

| 설정 | OpenVLA-OFT Online Brier |
|-----|-------------------------|
| Standard | 0.1073 |
| LIBERO-Pro | **0.0452** |
| LIBERO-Plus | **0.0325** |

- 의외로 challenging variant에서 Brier가 **더 낮음**
- 해석: 실패가 명확한 patten으로 발생 → confidence head가 더 잘 discriminate
- 또는 success rate 자체가 낮아져 base rate Brier limit이 작아짐

> ❓ **예상 질문**: 어려운 task에서 Brier가 낮아지는 게 실용적으로 의미가 있는가?
> **답변**: Yes - 어려운 task일수록 실패 가능성 사전 경고가 중요. 다만 success rate가 매우 낮을 때는 trivial한 "항상 실패 예측" baseline 대비 lift를 봐야 함.

### Real-World

| 메트릭 | 값 |
|--------|-----|
| Total successes | 115/151 (76.2%) |
| Average Brier | 0.1595 |
| Average NLL | 0.4885 |

- 3개 실로봇 task에서 deployment
- Sim과 비슷한 calibration 품질 보존

---

## 7. Ablation 분석

### Coin-vector dimension

| d | Brier |
|---|-------|
| 16 | 약화 |
| 64 | **최적** |
| 256 | 약화 |

### Normalization horizon K

- K=96~256 사이 안정
- K=32 이하: short horizon noise
- K>256: stale signal

### Step-conditioned vs unconditioned

- Step-conditioning 빼면 early-step에서 false confidence 발생

---

## 8. 관련 연구 비교

| 방법 | Single-pass | Continuous action 지원 | Frozen backbone |
|------|------------|---------------------|---------------|
| Token Perplexity | ✓ | ✗ | ✓ |
| MC Dropout | ✗ (N forward) | ✓ | ✗ |
| Ensemble | ✗ (N forward) | ✓ | ✗ |
| KnowNo-style | △ | △ | ✓ |
| **VLAConf** | **✓** | **✓** | **✓** |

### 핵심 차이
- **Universal**: discrete + continuous 모두 지원
- **Efficient**: single forward (64ms~165ms)
- **Frozen**: 기존 VLA에 plug-in 가능

---

## 9. 한계 및 미해결 문제

### 방법론적 미비점
1. **Calibration이 task와 entangle**: OpenVLA-OFT의 calibration 절대 품질이 pi-0.5보다 나쁨 — base model quality와 confidence quality가 분리되지 않음
2. **Success-only 학습의 가정**: Failure 데이터를 활용하지 않음 → diverse failure mode detection이 implicit
3. **Step-conditioning의 디자인 선택**: K=96~256은 LIBERO trajectory 길이에 맞춤. Long-horizon task(CALVIN, 수백 step)에서는 재튜닝 필요
4. **2개 backbone만 검증**: Octo, RT-2, OpenVLA-vanilla 등 추가 검증 부재
5. **Out-of-distribution task에서의 robustness**: LIBERO-Plus는 같은 distribution 변형. 완전히 다른 환경(예: 실외, lighting 급변)에서는 미평가

### Attribution 문제
- LIBERO-Pro/Plus에서 Brier가 낮아지는 현상이 **VLAConf의 calibration 우수성** 때문인지, **base rate 변화**에 따른 trivial baseline shift 때문인지 분리 어려움
- Step-conditioning의 정확한 기여도(ablation에서 부분적 언급)는 metric 단위로 분리되지 않음

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Coin-Flip Network on hidden state 적용은 fresh, but anomaly detection의 응용 변형 |
| **Technical depth** | ★★★★☆ — One-class + Platt + step-conditioning 구조 체계적 |
| **Experimental rigor** | ★★★★☆ — 2개 backbone, sim+real, 4가지 metric, robustness variants |
| **Practical impact** | ★★★★★ — Plug-in으로 어떤 VLA에도 적용 가능, single forward |
| **Writing quality** | ★★★★☆ — 명확한 motivation, metric-rich |

**강점**: VLA의 deployment safety를 위한 **단일 forward**, **continuous/discrete 통합**, **frozen backbone** plug-in. Real-world deployment까지 검증. **약점**: Base model quality와의 entanglement, failure data 미활용, long-horizon task 미검증.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Coin-Flip 대신 Masked Autoencoder reconstruction loss를 써도 되지 않나? | Yes, 가능. CFN의 advantage는 simple MSE objective + diverse coin sampling. Reconstruction은 high-D output prediction이라 학습 비용 큼 |
| 2 | Platt이 2-parameter라 underfit 가능성? | 가능. Isotonic regression이나 spline이 더 표현력 있지만 calibration set 작을 때 unstable. Platt이 robust한 default |
| 3 | LIBERO-Plus Brier 0.0325는 trivial baseline 대비 얼마나 향상? | Base rate calibration("predict mean success rate constant")과 비교 부재 — 절대값만 보고하는 한계 |
| 4 | OpenVLA-OFT Brier가 pi-0.5보다 나쁜 이유? | Hidden state 품질이 더 낮음 + discrete action manifold가 less smooth. Backbone quality dependency |
| 5 | Continuous action(pi-0.5)에서 hidden state는 어디서 추출? | Diffusion/flow head 전 단계의 backbone hidden state. 명확한 layer 정의가 paper에서 핵심 detail |
| 6 | Real-world에서 71/151 실패 중 confidence가 사전 경고를 준 비율? | Pre-Exec Brier 0.1595는 mid-level. Coverage-risk curve가 더 informative하지만 paper에서 부분만 보고 |
| 7 | 다른 calibration metric (ECE, ACE)에서는? | Brier/NLL만 보고. ECE는 binning artifact 있지만 deployment intuition 강함 — missing |
| 8 | Confidence head 학습이 backbone 행동에 영향을 주는가? | Backbone frozen이라 영향 없음. 그러나 deployment에서 confidence threshold 기반 abort 시, downstream behavior에 영향 |

<!-- VERIFIED: pdf -->
