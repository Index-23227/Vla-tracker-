# CF-VLA: Efficient Coarse-to-Fine Action Generation for Vision-Language-Action Policies

> **한 줄 요약**: Flow matching 기반 VLA의 다단계 ODE 적분을 **coarse(endpoint velocity 예측 → action prior 초기화) + fine(단일 step refinement)** 두 단계로 분해하여, action sampling latency를 π0.5 대비 **75.4% 단축(29.17 → 7.81 ms, 약 128 Hz)** 하면서 LIBERO 96.5%, CALVIN avg-len 3.67을 달성한 PaliGemma 3B VLA.

---

## 1. 배경 및 동기

### 기존 Flow-matching VLA의 구조적 비효율
- π0, π0.5, GR00T-N1 등 flow matching 계열 VLA는 학습이 안정적이고 continuous action을 다루기 좋으나, **inference 시 수~수십 step의 ODE 적분**이 필요
- 가우시안 노이즈에서 시작하는 단일 흐름이 (a) action manifold 전역 정렬과 (b) 국소 잔차 보정을 **한꺼번에** 수행해야 함 → 두 역할이 충돌하면서 step 수를 줄이기 어려움
- 실제 로봇에서 closed-loop control은 50 Hz 이상이 권장되는데, π0.5의 action sampling latency는 약 29 ms로 30 Hz 수준이 한계

### 핵심 질문
- **"global alignment"와 "local refinement"를 분리하면 step 수를 줄이면서도 정확도를 유지할 수 있는가?**
- **Action prior를 명시적으로 모델링하여 노이즈가 아닌 "구조화된 초기값"에서 출발할 수 있는가?**

📌 [Figure 1] Coarse stage(endpoint velocity 분포 학습) → Fine stage(single-step refinement) 두 단계 분해도

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

- **Backbone**: PaliGemma (SigLIP + Gemma family), 검증용으로 PaliGemma 2 (Gemma 2 기반)도 사용
- **Action expert**: 약 3B 파라미터의 VLA. 입력은 멀티뷰 RGB + 자연어 명령 + (선택적) proprioception
- **Action head**: Flow matching을 두 단계로 재구성
  - Coarse stage: endpoint velocity의 **조건부 posterior** (mean μ, variance σ²)를 학습 → Gaussian noise를 구조화된 초기값 ε̃로 변환
  - Fine stage: ε̃에서 시작해 **고정 시간 1-step refinement**로 최종 action을 산출

### 2.2 Coarse Stage — Action-Prior-Guided Initialization

기존 flow matching은 t=0 에서 Gaussian noise z₀ ~ N(0, I) 로 시작해 t=1 에서 action a를 얻는다. CF-VLA는 이 과정을 다음과 같이 재구성:

- VLM이 (관측, 명령)을 바탕으로 endpoint velocity **분포의 posterior** q(v | o, ℓ) = N(μ_θ, σ²_θ) 를 예측
- 학습 목표는 이 posterior가 데이터에서 유도된 **target velocity 분포 p(v)** 와 일치하도록 KL divergence를 최소화
- 결과적으로 초기값 ε̃ = μ_θ + σ_θ · ξ (ξ ~ N(0, I)) 는 action manifold에 **이미 정렬된** 분포에서 샘플링됨

> ❓ **예상 질문**: 왜 noise 대신 endpoint **velocity** 의 분포를 학습하는가?
> **답변**: Velocity는 flow matching ODE의 자연스러운 학습 신호. 데이터에서 유도된 target velocity는 action manifold의 tangent 정보를 담고 있어, 단순히 action 자체를 회귀하는 것보다 flow 구조와 호환된다. 또한 분산 σ²를 함께 학습함으로써 다중 모드 데이터(multi-modal demonstrations)에서도 mode collapse를 막을 수 있다.

### 2.3 Fine Stage — Single-Step Local Refinement

- ε̃를 입력으로 받아 **하나의 refinement step**만으로 최종 action a를 출력
- 일반적인 flow matching의 N-step Euler/RK 적분 대신 **고정 시간 Δt = 1** 의 단일 step
- 이 stage는 coarse 출력에 남은 잔차(residual)만 보정하면 되므로 작은 네트워크로 충분

> ❓ **예상 질문**: 1-step refinement만으로 정확도가 보장되는가? Consistency model이나 distilled flow와 무엇이 다른가?
> **답변**: Consistency model은 사전 학습된 다단계 모델을 distill하는 후처리 방식이라 학습 비용이 두 배. CF-VLA는 **처음부터** coarse(분포 매칭)와 fine(잔차 보정)을 별도 목적함수로 동시에 최적화하므로 distillation gap이 없다. 실험적으로 LIBERO 96.5% / CALVIN 3.67로 다단계 baseline과 동등 이상.

### 2.4 학습 단계 (Phase I → Phase II)

- **Phase I**: Coarse stage 단독 학습. Controlled proxy input에서 endpoint velocity의 mean/variance를 안정화
- **Phase II**: Coarse + Fine 결합 학습. Refinement가 coarse stage의 실제 출력 위에서 동작하도록 joint optimization

> ❓ **예상 질문**: Phase I 없이 처음부터 joint training하면 안 되는가?
> **답변**: Ablation Table 3에서 "w/o Phase I" 95.8% vs 전체 96.5% — 차이는 작지만 long-horizon LIBERO-Long에서 더 두드러짐. Phase I이 coarse의 variance 추정을 안정화하지 않으면 Phase II 초기 단계에서 refinement가 잡음투성이 입력을 받아 수렴이 느려진다.

---

## 3. 데이터셋

| 영역 | 데이터 | 규모 |
|------|--------|------|
| LIBERO | Replay-filtered demonstrations | 4 suite × 10 task × 50 demo |
| CALVIN | D split (D→D protocol) | 표준 split |
| Real-robot | 5 tasks: Pick X, Put X into Box, Wipe Table, Pour Water, Fold Towel into Thirds | 25-50 demos/task |

- **Replay filtering**: LIBERO 원본은 노이즈 demonstration이 포함되어 있어 replay로 성공한 trajectory만 선별 (최근 OpenVLA-OFT 이후 표준화된 평가 protocol)
- CALVIN은 **D→D** (single split) 만 보고. ABC→D, ABCD→D 결과는 미보고

> ❓ **예상 질문**: CALVIN D→D만 보고하고 ABC→D는 왜 빠졌는가?
> **답변**: ABC→D는 환경 변화(distribution shift)에 대한 generalization을 측정하는 더 어려운 setting. 논문이 D→D만 보고한 것은 generalization보다 **inference 효율성**이 본 논문의 핵심 메시지이기 때문으로 보임. 다만 reviewer가 ABC→D 결과를 요구할 가능성 높음.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Hardware | 8 × NVIDIA A100 (FSDP) |
| Global batch size | 16 (2 per GPU) |
| Learning rate | 5e-5 |
| LIBERO 학습 steps | 60,000 |
| 평가 trial | 시뮬레이션 task당 10 trial |
| Phase I / II 손실 가중치 | λ_I = λ_II = 0.1 |
| Coarse noise variance | σ²_noise = 0.01235 |
| KL temperature γ | 0.01 |
| Divergence | KL > NLL (sweep 결과) |

---

## 5. 실험 결과 심층 분석

### 5.1 LIBERO (Table 1, replay-filtered)

| 모델 | Spatial | Object | Goal | Long | **Avg** |
|------|---------|--------|------|------|---------|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0.5 (10-step FM) | 96.8 | 98.8 | 95.4 | 85.8 | 94.2 |
| MIP | 95.4 | 98.4 | 94.6 | 81.6 | 92.5 |
| **CF-VLA** | **98.0** | **99.2** | **96.6** | **92.0** | **96.5** |

- 가장 두드러진 향상은 **LIBERO-Long (+6.2 vs π0.5)**. Long-horizon에서 step 수를 줄이면 보통 성능이 떨어지는데 CF-VLA는 오히려 향상

### 5.2 CALVIN D→D (Table 2)

| 모델 | 1 inst | 2 | 3 | 4 | 5 | **Avg len** |
|------|--------|---|---|---|---|------------|
| RoboFlamingo | 82.4 | 61.9 | 46.6 | 33.1 | 23.5 | 2.47 |
| GR-1 | 94.9 | 89.6 | 84.4 | 78.9 | 73.1 | 4.21 |
| π0.5 | 88.4 | 75.6 | 65.2 | 58.1 | 49.5 | 3.37 |
| **CF-VLA** | **91.1** | **80.2** | **71.8** | **66.2** | **57.3** | **3.67** |

- D→D setting에서 π0.5 대비 +0.30 avg len. GR-1(autoregressive video pre-training)에는 못 미침 — backbone scale 차이

### 5.3 Real-Robot (5 tasks)

| 모델 | 평균 SR |
|------|--------|
| MIP | 63.5% |
| π0.5 | 79.0% |
| **CF-VLA** | **83.0%** |

- 5개 task 평균. 실패 패턴: bimanual deformable에서 gripper 조기 release, pouring 시 병이 컵을 가려 visual feedback 감소

### 5.4 ⚡ Inference Latency (논문의 핵심 셀링 포인트)

| 모델 | Action sampling | Hz (env) |
|------|----------------|----------|
| π0.5 (10-step) | 29.17 ms | ~34 |
| **CF-VLA (coarse 3.78 + fine 4.03)** | **7.81 ms** | **~128** |

- **75.4% latency 감소**. Closed-loop 50 Hz+ 요구사항을 여유 있게 충족

---

## 6. Ablation 분석

### Table 3 — LIBERO ablation

| 변형 | LIBERO Avg |
|------|-----------|
| Full CF-VLA | **96.5** |
| w/o Phase II | 94.6 |
| w/o refinement (coarse only) | 94.9 |
| w/o variance modeling (σ²=const) | 95.2 |
| w/o Phase I | 95.8 |

- **Phase II coupling이 가장 큼 (-1.9)**. Coarse와 fine을 joint optimize해야 함
- Refinement 자체가 +1.6 기여 → 1-step refinement가 단순 redundancy가 아님
- Variance modeling -1.3 → 다중 모드 demonstration에 중요

### Table 4 — CALVIN ablation
- Long-horizon에서는 variance modeling의 기여가 LIBERO보다 큼. 5-instruction success rate에서 -3% 이상 차이

### Table 5 — Cross-backbone (PaliGemma 2)

| Backbone | LIBERO Avg |
|----------|-----------|
| PaliGemma + standard FM | 93.9 |
| PaliGemma 2 + standard FM | 94.1 |
| **PaliGemma 2 + CF-VLA** | **94.6** |

- 다른 backbone에서도 CF-VLA 구조의 이득이 유지됨 (+0.5 ~ +0.7)

### Tables 6-9 — Hyperparameter sweep

| Hyper | Best | 의미 |
|-------|------|------|
| σ²_noise | 0.01235 | Coarse noise variance |
| γ | 0.01 | KL temperature |
| λ_I, λ_II | 0.1, 0.1 | Phase loss 가중치 |
| Divergence | KL > NLL | 분포 매칭에 KL 우세 |

> ❓ **예상 질문**: σ²_noise=0.01235라는 매우 구체적인 값은 어떻게 정했는가? Overfit 위험?
> **답변**: Grid search 결과로 보임. 이 정도 정밀도는 cross-domain 일반화에서 fragile할 수 있다. Real-robot 5-task에서 동일 hyper로 작동한 것은 긍정적이나, 새로운 robot embodiment에서 재튜닝이 필요할지 평가 부재.

---

## 7. 한계 및 미해결 문제

1. **CALVIN ABC→D, ABCD→D 미보고**: Distribution shift generalization 평가 부재
2. **Real-robot이 5 task로 제한**: π0.5의 수십 task 평가에 비해 적음. 25-50 demo/task로 fine-tuning 의존도 높음
3. **Bimanual deformable 실패**: Towel folding 등 deformable object에서 gripper 타이밍 실수. 두 단계 분해가 contact-rich 동역학에는 부족
4. **Backbone scale 비교 부재**: 7B/13B로 scale up했을 때 coarse-to-fine 분해의 이점이 유지되는지 미검증
5. **Step 수 동적 조절 미지원**: π0.5는 5-step / 10-step trade-off가 가능하나 CF-VLA는 항상 2-step 고정 → high-precision task에서 추가 step으로 정확도 트레이드오프 불가
6. **Endpoint velocity의 정의 ambiguity**: Action chunk 전체의 평균 velocity인지 마지막 step인지 명확하지 않음 (논문 본문 § 3.2 표기 모호)

### Attribution 문제
- Inference 가속의 75.4% 중 얼마가 (a) coarse posterior 학습, (b) single-step refinement, (c) PaliGemma backbone의 효율성에 기인하는지 분리 불가
- Coarse-only가 이미 94.9%이므로, **fine refinement는 +1.6의 marginal contribution** — "two-stage가 필수"라는 강주장에는 추가 증거 필요

---

## 8. 관련 연구 비교

| 모델 | Action head | Inference steps | LIBERO Avg | Latency |
|------|------------|----------------|-----------|---------|
| OpenVLA | Autoregressive discrete | N tokens | 76.5 | ~150 ms |
| π0 | Flow matching | 10 | 91.7 | ~25 ms |
| π0.5 | Flow matching | 10 | 94.2 | 29.2 ms |
| Consistency-Policy | Consistency distillation | 1-4 | ~92 | ~10 ms |
| **CF-VLA** | **Coarse-to-fine FM** | **2 (coarse + fine)** | **96.5** | **7.81 ms** |

### 핵심 차이
- Consistency Policy / distilled flow는 **post-hoc** distillation → 학습 비용 두 배
- CF-VLA는 **처음부터** 두 단계 구조로 학습 → distillation gap 없음
- π0.5 대비 동등 이상의 성능을 1/4 latency로 달성

---

## 9. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Coarse(분포 매칭) / Fine(잔차 보정) 분해 자체는 자연스러우나 VLA 맥락에서 처음 체계화 |
| **Technical depth** | ★★★★☆ — KL-based posterior, two-phase training, variance modeling의 통합 |
| **Experimental rigor** | ★★★☆☆ — LIBERO/CALVIN/실로봇 모두 평가했으나 CALVIN ABC→D 부재, real-robot 5 task로 제한 |
| **Practical impact** | ★★★★★ — 7.81 ms (128 Hz) inference는 closed-loop 로봇 제어에 즉시 적용 가능 |
| **Writing quality** | ★★★★☆ — 명확한 구조, ablation 풍부 |

**강점**: VLA inference 효율성 문제를 정면으로 다루며, distillation 없이 처음부터 효율적 구조를 학습하는 새로운 paradigm 제시. LIBERO 96.5%는 same-backbone π0.5(94.2%)보다 명확히 우세. **약점**: CALVIN ABC→D 등 generalization 평가 부재, real-robot task 다양성 부족, hyperparameter sensitivity 우려.

---

## 10. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Consistency Policy / Shortcut Models와 본질적 차이는? | Consistency는 post-hoc distillation, CF-VLA는 from-scratch joint optimization. Distillation gap 없음 |
| 2 | Coarse-only가 94.9%인데 fine stage(+1.6)가 정말 필요한가? | LIBERO-Long에서는 fine의 기여가 더 크다. Long-horizon에서 누적 오차 보정 역할 |
| 3 | π0.5 대비 같은 LIBERO 성능이라면 단순히 더 큰 모델/더 많은 데이터의 효과 아닌가? | 동일 PaliGemma 3B backbone, 동일 LIBERO replay-filtered split. Table 5의 cross-backbone에서도 동일 backbone에서 +0.5~0.7 |
| 4 | Endpoint velocity posterior가 다중 모드일 때 단일 Gaussian으로 충분한가? | 핵심 약점. σ² 학습으로 mode coverage 시도하나, 진정한 multi-modal demonstration에서는 mode-averaging 가능성 |
| 5 | CALVIN D→D만 보고하는 이유는 ABC→D 성능이 약해서인가? | 가능성 있음. Generalization 평가 부재가 reviewer 주요 지적 사항이 될 것 |
| 6 | Inference 7.81 ms가 정확히 어디서 측정되었는가? Backbone forward는 포함? | "Action sampling" 명시 — VLM forward는 별도. End-to-end latency는 backbone forward + 7.81 ms로 더 큼 |
| 7 | Real-robot 5 task로 robust한 일반화 주장이 가능한가? | 한계. Pick/Place/Wipe/Pour/Fold만으로는 contact-rich, dexterous 일반화 미검증 |
| 8 | σ²_noise=0.01235 같은 hyper가 새 embodiment에서 재튜닝 필요한가? | 미평가. Sim-to-real, cross-embodiment에서 hyper sensitivity가 실용성의 핵심 변수 |
| 9 | Coarse stage가 KL 대신 NLL일 때 왜 성능이 떨어지는가? | KL은 분포 전체를 매칭, NLL은 데이터 likelihood만 — multi-modal 분포에서 KL이 mode coverage에 우세 |
| 10 | Phase I 학습 비용은? Full pipeline 학습 시간 증가? | 논문 미보고. Phase I이 짧다면 marginal, 길다면 학습 비용 2배 우려 |

<!-- VERIFIED: pdf -->
