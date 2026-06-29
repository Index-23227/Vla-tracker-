# Action ControlNet: A Lightweight Delay-Aware Adapter for Smooth Asynchronous Control in Vision-Language-Action Models

> **한 줄 요약**: 비동기(asynchronous) chunked VLA 실행에서 inference latency 동안 이미 실행된 motion suffix("delay action")를 ControlNet 스타일 residual 조건으로 인코딩하여, 대부분 frozen된 generative action head(diffusion/flow matching)에 주입함으로써 chunk handoff 불연속을 줄이는 경량 어댑터. 전체 파라미터의 약 20%만 학습하여 Kinetix(0.79 avg, delay>0)와 Meta-World MT50(0.74 avg)에서 full delay-conditioned retraining(Training-RTC, 100% 학습)과 거의 동등한 robustness를 달성하고, SO-ARM101 실로봇에서 20/20 성공.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 대형 backbone + 반복적 generative action head를 쓰는 VLA는 **inference latency**가 커서 고주파 closed-loop 제어가 어려움
- **동기(synchronous) 실행**: 추론이 끝날 때까지 로봇이 대기 → stop-and-go 정지 구간 발생, wall-clock 시간 증가
- **비동기 실행**(SmolVLA의 Naïve Async 등): 추론과 실행을 overlap하지만, 다음 chunk가 **이미 stale해진 관측**에서 생성되어 chunk 경계(handoff)에서 불연속, jitter, contact-rich 작업 실패 발생
- 기존 대응책의 한계: runtime inpainting/interpolation은 heuristic·architecture-dependent, correction head는 특정 action representation에 강결합, training-time delay simulation/full-policy adaptation은 학습 cost 급증하여 pretrained policy 재사용 이점을 약화

### 핵심 질문
- **handoff mismatch를 전역적 task 이해 실패가 아닌 "국소(local) 경계 조건 문제"로 다룰 수 있는가?**
- **backbone을 건드리지 않고 action head에만 소수 파라미터를 추가하여 delay robustness를 얻을 수 있는가?**

📌 [Figure 1 삽입] — 비동기 VLA 제어 setting 개요 및 SO-ARM101 clean-the-table 실로봇 rollout (Naïve Async vs ACNet)

---

## 2. 방법론 심층 분석

### 2.1 핵심 통찰
inference delay는 주로 **국소 handoff mismatch**를 유발할 뿐, task semantic의 전역적 실패가 아니다. 관측·언어는 여전히 task와 coarse plan을 올바르게 지정하고, 불신뢰 component는 "이미 실행 중인 motion으로부터의 continuation"이다. 따라서 전체 visuomotor mapping을 재학습하지 말고, **task context는 frozen backbone에 맡기고 motion context(delay action)만 action head에 조건으로 주입**한다.

### 2.2 문제 형식화
- chunk horizon: `H = d + e + r` (d=inference delay, e=relaunch interval, r=optional future suffix)
- delay action: 직전 chunk에서 delay 구간 동안 **실제 실행된 suffix** `a_delay = {a^(e)_{t-e}, ..., a^(e+d-1)_{t-e}} ∈ R^{d×da}`
- 목표: `η* = arg min E[L_pred(â,a*) + λ·L_bd(â, a_delay)]`, with |η|≪|θ|
- 경계 손실 `L_bd = ||â^(d) - a^(e+d-1)_{t-e}||²` — 새 chunk의 첫 실행 action이 직전 chunk의 마지막 실행 action과 매끄럽게 이어지도록

### 2.3 ACNet 아키텍처 (4가지 설계 요건)
| 요건 | 내용 |
|------|------|
| R1 | pretrained backbone 보존(frozen) |
| R2 | 실행된 suffix 인코딩 |
| R3 | 보정을 action head 내부에서 국소적으로 적용 |
| R4 | 샘플된 delay 전반에 효율적 |

1. **Delay-Action Encoding**: 길이 d의 실행 suffix를 H-step tensor로 padding하되 미관측 미래 위치를 **learnable token** `p_j`로 채움(zero/noise padding의 모호성 회피). 경량 transformer encoder `E_φ` + action expert와 **동일한 terminal temporal pooling**으로 `c_t = Pool(E_φ(ã_delay)) ∈ R^{dc}` 산출.
2. **Residual Injection**: action expert의 주입 블록 집합 S에 대해 `h'_l = h_l + Z_{φ,l}(c_t)`. residual이라 원래 predictor가 default mode로 유지되고, delay cue가 무의미하면 0으로 수렴 가능. 1차 근사상 ACNet은 전체 chunk 재구성이 아니라 downstream 효과가 원하는 boundary correction과 맞는 residual 방향만 만들면 됨(Jacobian 논증, Eq.12).
3. **주입 위치**: Evo-1의 8-layer DiT-style flow-matching expert에서 **최종 블록만**(S={L}) 주입 — 출력에 가장 가까워 handoff 경계에 직접 영향.

📌 [Figure 3 삽입] — ACNet 아키텍처: frozen backbone + main expert, side branch로 delay action 인코딩 후 projection으로 residual 주입

### 2.4 학습 목표
- backbone은 delay마다 재forward할 필요 없음: `B_ω(o_t,l)` latent를 한 번 캐싱하여 여러 delay에 재사용 → delay coverage↑, 연산 amortize
- flow-matching: `τ ~ Beta(2,2)`, `z ~ U([-1,1])`, `x_τ=(1-τ)z+τx_0`, 목표 `L_FM = E||（x_0 - z) - v^{ACNet}_{ψ,η}(x_τ,τ,c_{t,d})||²`

---

## 3. 데이터 전략

| Track | 데이터 | 비고 |
|-------|--------|------|
| Kinetix | RTC 평가 프로토콜 [4], Kinetix 시뮬레이터 | RTC/Training-RTC는 π₀ 기반 |
| Meta-World MT50 | 전체 50 task | ACNet은 Evo-1 backbone |
| 실로봇 SO-ARM101 | 50 training rollouts, 2 task | put cube into box / clean table |

학습 스케줄(ACNet, Evo-1): stage1 1 epoch → stage2 24 epoch로 backbone 학습 후 **frozen**, 이후 ACNet+terminal pooling만 **8 epoch** 학습. 실로봇은 10 epoch 최적화.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Backbone | Evo-1 (lightweight VLA, CVPR 2026) |
| Action expert | 8-layer DiT-style flow-matching |
| 주입 블록 | 최종 블록 1개 (S={L}) |
| Padding | learnable token |
| Trainable 비율 | **~20%** of total params (Kinetix 기준) |
| Hardware | single RTX 4080 SUPER (Meta-World) |
| 평가 delay | Kinetix d∈{0,1,2,3,4}; MT50 d∈{0,5,10,15} |

---

## 5. 실험 설계 및 평가 프로토콜

4가지 deployment scheme 비교(동일 비동기 프로토콜):
- **Naïve Async**: 연속 chunk 직접 stitching
- **RTC** [4]: committed action freeze + 나머지 suffix inpainting
- **Training-RTC** [17]: delay-conditioned 학습(100% 파라미터)
- **ACNet**(제안): delay-aware residual adapter (~20% 파라미터)

지표: task success rate(주), trainable 파라미터 비율(Kinetix), end-to-end latency·achieved control frequency(MT50), translational jerk(부드러움). Meta-World benchmark-wide: H=50, e=25.

---

## 6. 실험 결과 심층 분석

### Kinetix (Table I) — delay별 success rate, Avg는 d>0
| Method | d=0 | 1 | 2 | 3 | 4 | Avg(d>0) | Params% |
|--------|-----|---|---|---|---|----------|---------|
| Naïve Async | 0.89 | 0.74 | 0.69 | 0.55 | 0.46 | 0.61 | 0 |
| RTC | 0.91 | 0.75 | 0.80 | 0.72 | 0.61 | 0.72 | 0 |
| Training-RTC | 0.89 | 0.88 | 0.83 | 0.79 | 0.70 | **0.80** | 100 |
| **ACNet** | 0.90 | 0.87 | 0.84 | 0.76 | 0.68 | **0.79** | **~20** |

- ACNet 0.79 vs Training-RTC 0.80 — 격차 0.01에 불과하나 **trainable 파라미터는 약 80% 적음**

### Meta-World MT50 (Table II)
| Method | d=0 | 5 | 10 | 15 | Avg | Lat. | Freq. |
|--------|-----|---|----|----|-----|------|-------|
| Naïve Async | 0.80 | 0.71 | 0.70 | 0.70 | 0.70 | 73ms | 13.6Hz |
| RTC | 0.79 | 0.72 | 0.72 | 0.71 | 0.71 | 159ms | 6.28Hz |
| Training-RTC | 0.80 | 0.77 | 0.74 | 0.73 | 0.74 | 134ms | 7.46Hz |
| **ACNet** | 0.81 | 0.76 | 0.74 | 0.73 | **0.74** | **91ms** | **11.0Hz** |

- ACNet 0.74로 Training-RTC와 동률, RTC(0.71)·Naïve(0.70) 상회
- latency 91ms로 RTC(159ms)·Training-RTC(134ms)보다 낮음 — 단, 논문도 인정하듯 이는 **Evo-1 backbone이 π₀보다 가볍기 때문**이지 ACNet side branch 자체가 빨라서가 아님

### 실로봇 SO-ARM101 (Table III, 10 trials/task)
| Method | Put cube into box | Clean table | Overall |
|--------|-------------------|-------------|---------|
| Naïve Async | 9/10 (90%) | 8/10 (80%) | 17/20 (85%) |
| **ACNet** | 10/10 (100%) | 10/10 (100%) | **20/20 (100%)** |

### 부드러움 (jerk, Fig.4)
nut-assembly-v3 / plate-slide-back-v3 (H=50, d=10)에서 ACNet이 handoff 경계 주변에서 더 평탄한 jerk profile → cross-chunk transition이 매끄러움.

---

## 7. Ablation 분석

Meta-World MT50, Evo-1 backbone 기반 두 가지 질문:

1. **주입 위치(Fig.6, Fig.7a)**: K=1000 flow-matching step에서 블록별 token-update 크기를 측정한 결과 **최종 블록이 가장 활성** → late layer가 chunk 최종 refinement에 가장 기여. final-block conditioning이 delay 전반에서 최고, layer 0 등 early-layer 주입은 robustness 저하. delay action을 task 표현의 early perturbation이 아닌 **국소 boundary cue**로 써야 함을 뒷받침.
2. **future padding 방식(Fig.7b)**: **learnable token**이 zero/noise padding을 모든 delay에서 일관되게 상회. 미관측 슬롯을 모호함 없이 명시적으로 표시하는 minimal-intervention 설계 지지.

---

## 8. 관련 연구 비교

| 접근 | 방식 | 파라미터 비용 | architecture 결합 |
|------|------|--------------|------------------|
| Naïve Async (SmolVLA) | stale 관측으로 다음 chunk | 0 | 무관 |
| RTC [4] | committed freeze + inpainting | 0(runtime) | 결합 강함 |
| 경량 correction head [15] | delay-error 보정 head | 소 | action repr 결합 |
| VLASH [16] | future-state 예측 | 중 | 결합 |
| Training-RTC [17] | delay-conditioned full 학습 | 100% | 무관하나 cost↑ |
| **ACNet** | **frozen head에 residual 주입** | **~20%** | diffusion/flow 모두 호환 |

핵심 차이: ACNet은 ControlNet [5]에서 영감받아 **action-head-level residual adapter**로, backbone 보존 + plug-and-play + generative head 호환을 동시 달성.

---

## 9. 한계 및 미해결 문제

1. **Latency 비교의 교란**: MT50에서 ACNet의 낮은 latency(91ms)는 Evo-1 backbone 덕분이며 RTC/Training-RTC는 π₀ 기반 → **동일 backbone 상의 공정한 head-to-head 부재**. 논문도 명시적으로 인정.
2. **Training-RTC 대비 정확도 우위 없음**: success는 사실상 동률(Kinetix 0.79 vs 0.80). 강점은 정확도가 아니라 **파라미터 효율성**.
3. **실로봇 규모 제한**: 2 task × 10 trials = 20 trial. 통계적 power 낮음(20/20도 신뢰구간 넓음).
4. **delay 범위 제한**: 평가 delay가 작고 고정 분포. 큰/time-varying delay에 대한 robustness는 future work로 명시.
5. **Kinetix/Meta-World만**: LIBERO/CALVIN/SimplerEnv 같은 표준 manipulation 벤치마크 점수 없음 → 본 tracker의 cross-model 비교에서 directly 비교 불가.
6. **Code 미공개**: 재현성 위험.
7. **diffusion head 호환 주장 미검증**: "diffusion·flow matching 모두 호환" 주장하나 실험은 flow-matching(Evo-1)에 한정.

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — handoff를 boundary-conditioning 문제로 재정의하고 ControlNet식 residual을 action head에 적용한 관점이 깔끔 |
| **Technical depth** | ★★★★☆ — 형식화(L_bd), Jacobian 1차 논증, latent 캐싱, learnable padding 등 디테일 견고 |
| **Experimental rigor** | ★★★☆☆ — Kinetix+MT50+실로봇 3 track, 체계적 ablation. 다만 backbone 교란·실로봇 trial 수·표준 벤치 부재 |
| **Practical impact** | ★★★★☆ — pretrained VLA 재사용하며 ~20% 파라미터로 delay robustness 확보, 비동기 고주파 배포에 실용적 |
| **Writing quality** | ★★★★☆ — 동기→형식화→방법→실험 흐름 명료 |

**강점**: 전체 재학습(Training-RTC, 100%) 대비 약 20% 파라미터로 거의 동등한 delay robustness. frozen backbone 보존 + generative head 호환 + plug-and-play. 실로봇에서 명확한 정성·정량 개선(20/20).
**약점**: Training-RTC 대비 정확도 우위 없음, latency 비교의 backbone 교란, 표준 벤치 부재, 작은 실로봇 표본, code 미공개.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | ACNet의 진짜 이점은 정확도인가 효율인가? | 정확도는 Training-RTC와 동률(0.79~0.80). 본질적 기여는 ~20% 파라미터라는 **adaptation cost 절감**이다 |
| 2 | MT50 latency 91ms가 RTC 159ms보다 낮은 건 ACNet 덕분인가? | 아니다. 논문도 인정하듯 Evo-1(경량) backbone vs π₀ 차이. 동일 backbone이면 ACNet은 작은 encoder/projection overhead만 추가 |
| 3 | "diffusion·flow matching 모두 호환"인데 실험은? | flow-matching(Evo-1)만 검증. diffusion head 실증은 없음 — 호환성은 설계상 주장 수준 |
| 4 | 왜 최종 블록만 주입하나? | Fig.6에서 final block의 token-update가 최대 → 출력에 가장 가까워 handoff에 직접 영향. early-layer(특히 layer0) 주입은 robustness 저하 |
| 5 | learnable padding이 zero/noise보다 나은 이유는? | 미관측 미래 슬롯을 **모호함 없이 명시**. zero는 "정지 action"과 혼동, noise는 인위적 변동 유발. Fig.7b에서 전 delay 우위 |
| 6 | 실로봇 20/20의 통계적 신뢰성은? | 2 task×10 trial로 표본 작음. 100% 성공도 신뢰구간 넓어 강한 일반화 결론은 주의 |
| 7 | 더 크거나 time-varying한 delay에서도 동작하나? | 미검증. 평가 delay 분포가 작고 고정적 — future work로 명시 |
| 8 | 표준 LIBERO/SimplerEnv 점수가 없는 이유와 영향은? | 본 연구는 비동기 delay robustness가 초점이라 Kinetix/MT50/실로봇만 사용. 결과적으로 cross-model 표준 비교가 어려움 |

<!-- VERIFIED: pdf -->
