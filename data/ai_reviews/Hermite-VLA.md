# Hermite-VLA: Hermite Curves as Trajectory Priors for Vision-Language-Action Models

> **한 줄 요약**: action chunk를 "per-timestep 제어값의 평평한 행렬"이 아니라 **끝점 위치·속도로 정의되는 조각별 3차 Hermite 곡선**으로 재파라미터화하여 intra-chunk smoothness와 inter-chunk 경계 연속성을 표현 자체에 내장하고, 세 가지 통합 수준(discrete token / continuous scaffold / auxiliary regularization)을 비교한 결과 **추론 비용 0인 학습 전용 정규화(Hermite-VLA_Reg)가 가장 강력**하다는 것을 보인 연구 (LIBERO 95.9 → 98.7, LIBERO-plus 85.7 → 90.9, 실로봇 63.4 → 90.0).

- **arXiv**: 2608.01265 (2026-08-02, cs.RO) — IEEE TPAMI 투고 형식
- **저자/소속**: Qi Lv, Jianming Xing, Zhao Yang, Mingyuan Yao, Yinan Shi, Yawei Jueluo, Mike Zheng Shou, Xiang Deng — HIT(선전), Jiangsu Cytoderm, NUS Show Lab
- **Project page**: https://aopolin-lv.github.io/Hermite/

---

## 1. 배경 및 동기

기존 VLA는 backbone과 decoding objective(autoregressive token / diffusion / flow matching)에서는 크게 갈라지지만, **action chunk를 per-timestep 제어값의 평평한 행렬로 본다는 가정은 공유**한다. 즉 "어떻게 예측하는가"만 다르고 "무엇을 표현하는가"는 동일하다. 논문은 이 표현 자체를 문제 삼으며 두 가지 결함을 지적한다(Sec. 1).

1. **Intra-chunk incoherence**: chunk 내부의 각 timestep이 독립적으로 파라미터화되므로 학습은 자연히 pointwise reconstruction loss에 의존한다. 결과적으로 per-timestep 오차가 비슷하면 매끄러운 궤적과 들쭉날쭉한(jagged) 궤적이 동일한 페널티를 받고, 고주파 진동이 충분히 억제되지 않는다.
2. **Inter-chunk discontinuity**: closed-loop replanning이 독립적으로 디코딩된 chunk들을 경계 제약 없이 이어붙인다. 특히 **위치는 chunk의 좌표로 직접 노출되지만 handover 지점의 속도는 두 chunk를 잇는 유한차분**이라 어느 쪽 파라미터 집합의 함수도 아니며, 따라서 표준 behavior cloning 목적함수에서 전혀 감시되지 않는다.

저자들은 이를 "대형 backbone과 충분한 데이터로 극복 가능한 capability ceiling"이 아니라 **누락된 inductive bias**로 규정하며, 그 부재가 분포 변화(distribution shift)와 실제 물리 실행에서 가장 크게 드러난다고 본다.

---

## 2. 방법론 — Hermite trajectory operator

**핵심 아이디어**: chunk의 궤적을 조각별 3차 Hermite 곡선으로 표현한다. 각 세그먼트는 양 끝점의 **위치 p와 속도 v**로 완전히 규정되고, 이산 chunk는 고정 선형 기저 행렬 `H`를 boundary 좌표 `θ`에 곱하는 것만으로 복원된다: `a_c = H·θ`.

이 설계가 위 두 결함을 직접 겨냥한다.
- Hermite 파라미터가 끝점 위치·속도를 **직접** 인코딩하므로, 연속성을 지배하는 경계 조건이 action 표현의 명시적 구성요소가 된다. 인접 세그먼트 knot에 일치하는 상태를 부여하면 디코더가 seamless transition 쪽으로 편향된다.
- 구조적으로 각 세그먼트는 **C1 연속**이며, 저차 다항 기저가 표현력을 저주파 성분으로 제한하여 jagged prediction을 원천 차단한다.

연산자는 고정·미분가능·저비용(행렬 곱 1회)이므로 기존 VLA에 삽입하기 쉽다. Proposition 3.3은 seam discontinuity가 인접 두 chunk의 예측 오차와 시연 자체의 across-seam 증분으로 **상한**이 잡히며(`||θ̂ - θ_LS(a)||₂ ≤ ||Hθ̂ - a||₂ / σ_min(H)`), 따라서 궤적 공간 목적함수로 직접 제어된다는 것을 보인다. 저자들은 두 가지 caveat도 명시한다: (a) 오차가 chunk 단위로만 누적되므로 seam을 **좁힐 뿐 제거하지는 못하고**, (b) 보장은 residual 더하기 이전의 Hermite scaffold에 대한 것이며 Reg 변형에서는 공유 feature를 통한 inductive bias로만 작용한다.

---

## 3. 세 가지 변형 (Sec. 3.3)

논문은 "동일한 연산자를 discrete/continuous 모두에 넣을 수 있는가", "연속 모델링에서 prior가 배포 시 예측에 반드시 참여해야 하는가"라는 중첩된 두 질문을 설계한다.

| 변형 | 통합 수준 | 동작 | 추론 시 |
|---|---|---|---|
| **Hermite-VLA_DH** | discrete AR | θ의 각 좌표를 256 bin으로 양자화 → chunk당 `4·K·D_c` 토큰을 K개 세그먼트 그룹으로 묶어 **segment-level autoregressive** 예측(세그먼트 내부 4D_c 토큰은 병렬). gripper는 별도 head | 경량 action head 내부 K=2 루프만 돌고 backbone은 1회 forward |
| **Hermite-VLA_CH** | continuous (deployed) | flow matching의 clean action 추정을 `Hθ + r`(Hermite scaffold + per-timestep residual)로 분해. θ는 horizon 평균 feature에 MLP, r은 timestep별 `W_r h_τ` | scaffold+residual head 사용 |
| **Hermite-VLA_Reg** | auxiliary objective only | 기존 flow-matching loss에 `λ·||H·θ_aux − a_gt||²` 만 추가. 배포 head와 샘플링 절차는 **그대로** | auxiliary branch 폐기 → 오버헤드 0 |

FAST[47]와의 차이가 명확하다. FAST의 DCT 계수는 horizon 전체의 **전역 기술자**라 chunk 경계 상태에 대한 손잡이가 없는 반면, Hermite 좌표는 **국소 경계 위치·속도**라서 연속성이 요구되는 replanning 인터페이스와 표현이 정렬된다.

---

## 4. 실험 설정

- **시뮬레이션**: 표준 LIBERO(Spatial/Object/Goal/10, 각 10 task, task당 50 rollout = suite당 500 episode), LIBERO-plus(7종 test-time perturbation, 필터링·카테고리 균형 후 10,030 instance).
- **실로봇**: 3개 플랫폼 4개 task, 각 15회 rollout, 30 Hz end-effector 제어. Task 1 Franka 단완(832 demo, 6.3h), Task 2 Cybopal 단완(1,006 demo, 4.1h), Task 3 Cybopal 양완(805 demo, 4.6h), Task 4 ARX 양완 수건 접기(723 demo, 3.2h). 모두 3단계(open→place→close, Task 4는 pick→fold→place)로 정의되어 부분 진척 측정 가능.
- **구현**: π-series backbone(PaliGemma VLM + action expert), ~3.3B. K=2 세그먼트. LIBERO는 T=10, W=5, D=7, D_c=6(binary gripper 채널 제외). 실로봇은 T=50, W=20. AdamW, warmup 1k → peak 5e-5 → 3e-5, batch 256, 30k step, 8×H100. λ=10, 추론 시 N=10 denoising step.
- **핵심 통제**: π0-FAST, π0.5, 세 Hermite 변형 **모두 동일한 학습 프레임워크·세팅**을 공유하고 head와 loss만 다르다.

---

## 5. 표준 LIBERO 결과 (Table 2)

| Method | Params | Objective | Spatial | Object | Goal | LIBERO-10 | Avg |
|---|---|---|---|---|---|---|---|
| OpenVLA | ~7B | AR CE | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| WorldVLA | ~7B | AR CE | 87.6 | 96.2 | 83.4 | 60.0 | 81.8 |
| π0-FAST | ~3.3B | AR CE | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| **Hermite-VLA_DH** | ~3.3B | AR CE | 96.0 | 98.6 | 96.4 | 90.4 | **95.4** |
| Diffusion Policy | 157M | Diffusion | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| Octo | 93M | Diffusion | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| DiT Policy | 334M | Diffusion | 84.2 | 96.3 | 85.4 | 63.8 | 82.4 |
| OpenVLA-OFT | ~7B | L1 regression | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| π0.5 | ~3.3B | Flow matching | 98.4 | 96.0 | 97.8 | 91.4 | 95.9 |
| Hermite-VLA_CH | ~3.3B | Flow matching | 99.2 | 99.2 | 96.4 | 96.0 | 97.7 |
| **Hermite-VLA_Reg** | ~3.3B | FM + Hermite reg. | 99.6 | 99.2 | 99.0 | 96.8 | **98.7** |

가장 큰 이득은 **장기 horizon LIBERO-10**에 집중된다. Reg는 91.4 → 96.8(+5.4%p), DH는 60.2 → 90.4(+30.2%p). 저자들의 해석: LIBERO-10은 고정 실행 윈도우에서 replanning seam이 빈번하게 발생하는데, 평평한 표현은 chunk 경계를 넘는 궤적 수준 제약 수단이 아예 없다는 것.

---

## 6. LIBERO-plus 강건성 (Table 3)

| Method | Camera | Robot | Language | Light | Background | Noise | Layout | Instance Avg |
|---|---|---|---|---|---|---|---|---|
| OpenVLA | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| WorldVLA | 0.1 | 27.9 | 41.6 | 43.7 | 19.8 | 10.9 | 38.0 | 25.3 |
| π0-FAST | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| **Hermite-VLA_DH** | 54.3 | 36.9 | 70.4 | 92.8 | 86.4 | 78.8 | 77.6 | **69.4** |
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| π0.5 | 75.8 | 79.4 | 83.3 | 95.5 | 95.0 | 89.6 | 87.0 | 85.7 |
| Hermite-VLA_CH | 78.7 | 83.3 | 97.7 | 87.0 | 85.9 | 73.9 | 86.8 | 85.0 |
| **Hermite-VLA_Reg** | 89.2 | 85.4 | 87.3 | 97.7 | 96.7 | 93.8 | 89.9 | **90.9** |

Reg는 7개 중 6개 카테고리에서 1위이며 **모든 카테고리에서 π0.5를 상회**한다. 이득은 Camera(+13.4%p)와 Robot(+6.0%p) 같은 기하학적 교란에서 크고, Light(+2.2%p)·Background(+1.7%p) 같은 외형 변화에서는 작다 — action 쪽 prior라는 성격과 정확히 일치하는 프로파일. 반면 CH는 평균 85.0으로 **baseline보다 낮아지는 카테고리(Light, Background, Noise)가 존재**한다. 사전학습된 action predictor가 이미 dense action을 잘 모델링하는 경우, Hermite 구조는 **예측을 대체하기보다 보완적 supervision으로 쓰는 쪽이 낫다**는 논문의 핵심 논지를 뒷받침한다.

---

## 7. 실로봇 결과 (Table 4)

| Task | π0-FAST | π0.5 | DH | CH | **Reg** |
|---|---|---|---|---|---|
| Task 1 (Franka, pot) | 60.0 | 86.7 | 80.0 | 86.7 | **100.0** |
| Task 2 (Cybopal 단완, pot) | 20.0 | 26.7 | 33.3 | 60.0 | **66.7** |
| Task 3 (Cybopal 양완, pot) | 40.0 | 46.7 | 46.7 | 80.0 | **93.3** |
| Task 4 (ARX 양완, 수건 접기) | 80.0 | 93.3 | 86.7 | 100.0 | **100.0** |
| **평균 SR** | 50.0 | 63.4 | 61.7 | 81.7 | **90.0** |

단계별 분해가 특히 유익하다. 1단계(뚜껑 열기/물체 집기)는 π0-FAST를 제외한 모든 방법이 거의 100%로 해결하며, 최종 SR을 좌우하는 것은 **중간 단계(정밀 배치 또는 접기)**다. Task 2에서 Reg는 **1단계 통과율이 baseline보다 낮음에도(80.0 vs 86.7)** 최종 SR이 높다 — 즉 이득이 "조작 지점에 더 잘 도달해서"가 아니라 **조작 실행 자체의 품질**에서 온다는 강한 증거다.

---

## 8. Ablation (Table 5-9)

**정규화 강도 λ (Table 5, LIBERO)** — 종 모양 곡선. λ=0(baseline) 95.9 → λ=1 97.4 → λ=5 98.2 → **λ=10 98.7** → λ=20 97.5. 동시에 median jerk는 0.000460 → 0.000443 → 0.000422 → **0.000410** → 0.000418. λ가 0에서 10으로 갈 때 jerk가 **단조 감소**하는 것이 인상적이며, λ=20의 하락은 auxiliary loss가 주 flow-matching objective와 경쟁하기 시작함을 시사한다.

**세그먼트 수 K (Table 6, λ=10)** — K=1: 93.7, **K=2: 98.7**, K=3: 96.6, K=4: 95.8. 역시 종 모양. T=10 고정 하에서 knot을 과다 배치하면 세그먼트 span이 몇 timestep으로 줄어 경계 속도가 인접 프레임에 강하게 결합되고 저주파 smoothness prior가 희석된다는 설명. LIBERO-10에서 K=2가 K=1보다 +8.8%p로 격차가 가장 크다.

**CH의 supervision 설계 (Table 7)** — θ-space + 전역 최소제곱 타깃 91.5 → θ-space + 국소 finite-difference 96.4 → 궤적 공간 supervision(implicit LS) 96.2 → **궤적 공간 + per-timestep residual 97.7**. Row 1의 실패가 타깃 형태가 아니라 **θ-space의 unweighted 좌표 표현** 탓임을 Row 3이 분리해 보인다(궤적 공간 supervision은 normal matrix `HᵀH`로 최적화 metric을 재가중). residual은 cubic spline span 밖의 고주파 미세 조정을 담당한다.

**경계 기저 비교 (Table 8, K=2)** — Polynomial 96.7 (jerk 0.000447), Bernstein 96.9 (0.000439), B-spline 97.6 (0.000422), **Hermite 98.7 (0.000410)**. 즉 이득이 "저차 다항식으로 부드럽게 만들었기 때문"이 아니라 **좌표가 물리적으로 의미 있는 경계 위치·속도라서**임을 보인다(action normalization 통계와도 자연스럽게 정렬).

**추론 비용 (Table 9, 단일 H100 80GB, 1,000회 평균)** — π0-FAST 238.2 ms / 4.2 FPS / 7.57 GB, **Hermite-VLA_DH 28.9 ms / 34.6 FPS / 6.66 GB**, π0.5 48.4 ms / 20.7 FPS / 6.63 GB, CH 49.4 ms / 20.2 FPS / 6.64 GB, **Reg 48.6 ms / 20.6 FPS / 6.85 GB**. Reg는 baseline 대비 사실상 무비용(+0.2 ms). DH는 backbone을 1회만 forward하고 K=2 루프로 compact token만 디코딩해 π0-FAST 대비 **8.2배 빠르다**.

---

## 9. 궤적 품질 분석 (Table 10-12, Fig. 7)

성공률이 비슷해도 실행 kinematics는 질적으로 다를 수 있다는 문제의식에서, 저자들은 2,000회 LIBERO rollout과 실로봇 실행에 대해 jerk, SPARC(속도 프로파일의 spectral arc length), seam discontinuity를 측정한다.

**성공/실패별 jerk (Table 11)**: π0.5는 성공 0.00046 / 실패 0.00070, 실패 82회. CH는 0.00047 / 0.00052, 실패 46회. **Reg는 0.00040 / 0.00048, 실패 26회**로 두 부분집합 모두에서 가장 매끄럽고 실패도 가장 적다. 저자들은 인과 방향(거친 운동이 실패를 유발하는가, OOD 상태가 불안정 제어를 유발하는가)이 완전히 해소되지는 않는다고 정직하게 인정한다.

**실로봇 seam discontinuity (Table 12, W=20, T=50)**: 모든 정책이 handover 지점에서 내부 스텝 대비 유의미하게 큰 불연속을 보이며(ρ = 5.5–8.6), Reg는 π0.5 대비 seam median을 Task 2에서 **0.72×**, Task 3에서 **0.48×**로 줄인다. Fig. 7은 두 Hermite 변형이 실로봇 가속도/jerk RMS에서 π0.5보다 통계적으로 유의하게(p < 10⁻⁴) 매끄럽고, Reg가 가장 낮고 가장 일관적임을 보인다.

---

## 10. 강점과 한계

**강점**
- **표현 수준의 문제 제기가 명확**하다. backbone/디코더 논쟁과 직교하는 축(무엇을 표현하는가)을 짚고, 동일 학습 프레임워크로 세 통합 수준을 통제 비교한 점이 실험 설계상 가장 큰 미덕이다.
- **"prior는 runtime constraint가 아니라 learning inductive bias"**라는 결론이 반직관적이면서도 실용적이다. 배포 파이프라인을 건드리지 않고 loss 한 항만 추가해 LIBERO +2.8%p, LIBERO-plus +5.2%p, 실로봇 +26.6%p를 얻는다.
- **성공률을 넘어선 kinematic 측정**(jerk, SPARC, seam ratio)을 시뮬·실기 양쪽에서 제시하고, Table 8의 기저 비교로 "단순 평활화" 대안 가설을 배제했다.
- 이론(Prop. 3.3의 seam 상한)과 그 한계를 스스로 명시하는 태도가 좋다.

**한계**
- **CH가 LIBERO-plus 일부 카테고리에서 baseline보다 나쁘다**(Light 87.0 vs 95.5, Background 85.9 vs 95.0, Noise 73.9 vs 89.6). 이를 "보완적 supervision이 낫다"로 해석하지만, 명시적 scaffold가 왜 특정 외형 교란에서 취약해지는지에 대한 기전 분석은 부족하다.
- **DH의 Camera 성능 퇴행**(54.3 vs π0-FAST 65.1)도 평균 개선에 가려져 있다. 양자화(256 bin) 오차가 카메라 시점 변화와 상호작용하는지 확인되지 않았다.
- **벤치마크 범위가 LIBERO 계열로 한정**된다. CALVIN, SimplerEnv, RoboTwin 등 다른 시뮬 벤치마크가 없어 일반성 주장이 제한적이다.
- **실로봇 통계력이 약하다**. task당 15 rollout이면 66.7%와 60.0%의 차이는 1회 시도에 해당한다. Table 4의 세부 순위는 조심스럽게 읽어야 한다.
- K, λ, 기저 ablation이 모두 **표준 LIBERO에서만** 수행되어, 실로봇(T=50, W=20)에서 K=2가 여전히 최적인지는 검증되지 않았다.
- 코드/가중치가 논문 시점에 공개 URL로 확인되지 않는다(project page만 제시).

---

## 11. 다른 연구와의 위치

- **FAST(π0-FAST)** 대비: 둘 다 chunk 표현 수준의 개입이지만 FAST의 DCT 계수는 전역 기술자로 경계 상태를 노출하지 않는다. Hermite는 국소 경계 좌표라 replanning 인터페이스와 정렬된다. 실제로 DH가 동일 backbone에서 85.5 → 95.4로 앞선다.
- **Spline Policy [53]** (동시기 연구) 대비: Spline Policy는 spline 파라미터를 학습·배포 **양쪽 모두**에서 정책 출력으로 사용한다. 본 논문은 그 축(explicit output vs implicit prior)을 명시적으로 ablation하여 **implicit 쪽이 우세**하다는 결론을 낸다 — primitive-as-policy 계열이 접근하지 못하는 zero-cost 배포 영역.
- **Seam 보정 계열 [9,54,55,56]** 대비: 이들은 평평한 chunk를 유지한 채 외부 모듈로 이미 디코딩된 출력을 사후 수정한다. Hermite는 경계 상태를 **주 디코더 내부에서** 모델링한다.
- **Movement primitive + diffusion [51,52]** 대비: 유사한 고전 파라미터화 부활이지만, 본 논문은 사전학습된 대형 VLA에 어느 **아키텍처 수준**에서 주입할지를 체계적으로 탐색한 점이 차별적이다.

---

## 12. 종합 평가

**평점: 8.5 / 10**

아이디어의 참신성 자체는 중간 정도다 — Hermite 곡선도, 보조 손실도 새로운 도구는 아니다. 이 논문의 가치는 **"어디에 넣을 것인가"를 통제된 실험으로 답했다는 데** 있다. 동일 backbone·동일 학습 세팅에서 discrete token / explicit scaffold / auxiliary regularization 세 지점을 나란히 놓고, 가장 침습적이지 않은 선택이 가장 강하다는 결과를 내놓은 것은 VLA 설계 실무에 바로 쓰이는 지식이다. λ와 K의 종 모양 곡선, 기저 비교(Table 8), 성공/실패별 jerk 분해(Table 11), seam ratio(Table 12)까지 주장별로 대응하는 증거가 붙어 있고, CH의 열위나 인과 방향 미해소 같은 불리한 결과도 숨기지 않는다.

감점 요인은 벤치마크 편중(LIBERO 계열 only), 실로봇 15 rollout의 낮은 통계력, CH/DH 퇴행 사례에 대한 기전 설명 부족, 그리고 ablation이 시뮬에만 국한된 점이다. 그럼에도 "trajectory 구조는 표현이 아니라 학습 신호로 쓸 때 가장 값싸고 강력하다"는 메시지는 재현·이식이 쉬워 후속 연구에 실질적 영향을 줄 가능성이 높다.

**핵심 takeaway 3가지**
1. Action chunk의 경계 **속도**는 표준 BC 목적함수에서 아무도 감시하지 않는 자유도이며, 이것이 장기 horizon 실패의 구조적 원인이다.
2. 궤적 prior를 배포 정책의 출력 형태로 강제하면(CH) 사전학습된 dense action 예측 능력과 충돌할 수 있다. **보조 손실로만 주입하면(Reg)** 충돌 없이 이득만 취한다.
3. Hermite가 Bernstein/B-spline/전역 다항식을 이기는 이유는 평활화 강도가 아니라 **좌표가 경계 위치·속도라는 물리량이어서** action 정규화 통계와 정렬되기 때문이다(98.7 vs 96.7–97.6).

<!-- VERIFIED: pdf -->
