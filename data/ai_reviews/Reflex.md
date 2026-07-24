# Reflex: Real-Time VLA Control through Streaming Inference — 리뷰

arXiv: 2607.14695 · ICML 2026 (PMLR 306) · Beijing University of Posts and Telecommunications
코드: https://github.com/9yc/Reflex

## 1. 개요

Reflex는 새로운 정책 모델이 아니라, 기존 flow-matching VLA(Pi0, Pi0.5)를 위한 실시간 스트리밍 추론 프레임워크다. Flow matching 정책은 정밀한 연속 제어를 약속하지만, 반복적 denoising 특성 때문에 실시간 로보틱스와 근본적으로 충돌한다. Reflex는 이를 알고리즘-런타임 공동 설계로 해결한다.

## 2. 문제 정의: 제어-추론 격차

로봇 조작은 부드러운 궤적을 위해 50-100Hz 제어 루프를 요구하지만, 최신 VLA는 추론당 100-200ms(즉 5-10Hz)가 걸려 한 자릿수의 주파수 격차가 존재한다. Action chunking으로 이를 완화하지만, 긴 chunk는 나중 action이 오래된 관측에 실행되는 staleness 문제를 낳는다. 핵심 통찰은 병목이 계산 시간이 아니라 실행을 멈추는 동기적 대기라는 점이다.

## 3. 캐시 재사용 딜레마

Transformer의 KV 캐싱은 복잡도를 O(n^2)에서 O(n)으로 줄이지만, flow matching에서는 실패한다. timestep 임베딩 t가 모든 레이어를 조건화하여 denoising step마다 내부 상태가 바뀌고, 캐시된 특징이 즉시 무효화된다. 결과적으로 느린 재계산 vs 수학적으로 부정확한 캐시 재사용이라는 트레이드오프가 강제된다.

## 4. 핵심 통찰: Timestep-Invariance Property

perception encoder는 flow matching timestep과 함수적으로 독립적(∂Enc/∂t_k = 0)이다. 무거운 VLM 백본은 timestep-불변, 경량 action expert는 timestep-의존적이라는 아키텍처적 분리가 Reflex 캐시 분할 전략의 토대다.

## 5. Partitioned Attention (정확성, §3.1)

컨텍스트를 세 영역으로 분할한다. (1) Static Prefix: 시스템 명령 토큰 l_1:L, 한 번 계산 후 영구 고정. (2) Sliding History: 최근 N개 관측의 FIFO 큐(LIBERO는 N=10, 약 300ms). (3) Dynamic Suffix: 매 denoising cycle마다 리셋되는 전이적 flow 상태 x(k). 이 분할로 고정 입력·고정 관측 창에 대해 full-batch attention과 동일한 출력을 내면서 O(1) 캐시 업데이트가 가능하다(부록 A.1 증명). Incremental Prefill과 Manual Cache Merging으로 메모리 재할당 오버헤드를 제거한다.

## 6. AdaRMSNorm (안정성, §3.2)

50Hz 연속 운용 시 모델은 offline 훈련보다 50배 자주 고분산 초기화 노이즈(x ~ N(0,I))에 노출되어 BFloat16 underflow와 activation collapse를 일으킨다. AdaRMSNorm(x,c) = x/RMS(x) ⊙ γ(c), γ(c)=1+MLP(c)로, c=[t_k, s_t]는 sinusoidal timestep과 proprioceptive 상태를 concat한다. RMS는 FP32로 계산하고 gating MLP는 BFloat16으로 운용하는 엄격한 mixed-precision 가드레일과 Robust Dtype Inference를 적용한다.

## 7. 비동기 파이프라인 (처리량, §3.3)

Thread A(Vision)는 VLM 백본으로 카메라 프레임을 연속 인코딩(10-30Hz)해 KV 쌍을 공유 캐시에 push하는 producer, Thread B(Policy)는 flow matching 정책으로 action chunk를 생성하는 consumer다. 비동기 실행의 상태 정렬 문제는 Future-Conditional State Prediction으로 처리한다: 오래된 센서값 대신 마지막 명령 action을 사용(ŝ_{t+Δ} ≈ a^cmd_t). Algorithm 1의 Adaptive Overlap Scheduling이 실시간 지연 측정으로 lookahead K를 동적 조정한다.

## 8. 시스템 최적화 (§3.4)

Operator Fusion(QKV 및 SwiGLU Gate/Up 프로젝션 융합)으로 레이어당 커널 실행 수를 50% 줄여 15-20% 속도 향상. 정적 Ring Buffer 아키텍처로 동적 할당을 제거해 O(1) 메모리 접근과 결정론적 지연을 보장한다.

## 9. 실험 설정

Base Model: Pi0 계열 두 종 — Pi0.5(2.3B: PaliGemma 2B 백본 + 300M action expert), 더 큰 Pi0(3.1B). 벤치마크: LIBERO(4개 category)와 Kinetix(물리 반응형). 지표: inference latency, reaction latency, stall rate, peak memory, success rate. 베이스라인: Standard, Naive Cache, Async-Naive.

## 10. 주요 결과

- 효율: Pi0.5 기준 LIBERO에서 2.58x 속도 향상(135.2ms→52.4ms), Pi0는 2.73x. Peak VRAM 27%(LIBERO)/24%(Kinetix) 절감.
- 반응 지연: 최대 54% 감소(151.9ms→82.5ms), stall rate 100%→0%, 안정적 50Hz.
- LIBERO 성능(Pi0.5+Reflex): Spatial 83.2, Object 80.4, Goal 82.0, Long 72.4 (평균 79.5). 동기 베이스라인과 parity 유지, LIBERO-Long에서 +3.6% 향상.
- Kinetix: Pi0.5 +7.4%, Pi0 +6.7%.
- 실물 로봇(AgileX PiPer, 180 에피소드): Pick-Place +11pp, Articulated +14pp, Dynamic Recovery +17pp.

## 11. Ablation

- Partitioned Attention: Naive caching은 MSE>1.0로 파국적 실패(성공률 12.5%), Reflex는 MSE=0.00으로 oracle과 정확히 일치(85.4%). 최대 단일 지연 감소원(135.2ms→61.5ms).
- AdaRMSNorm: BF16 baseline은 120-220 step에서 붕괴, AdaRMSNorm은 >2000 step 안정(NaN/Inf 없음, +0.4ms).
- Context window: K=10이 지연-정확도 최적점. K=50에서는 속도 향상이 1x 이하로 하락.
- Future-State Predictor: 4 delay step에서 +11pp(predictor 제거 대비), +22pp(Async-Naive 대비).

## 12. 총평

Reflex는 "stop-think-act" 사이클이 VLA의 본질적 한계가 아니라 시스템 설계 선택임을 보인다. 정확성 보장(Partitioned Attention)은 고정 입력·고정 관측 창에 한정되며, 비동기 스케줄링·future-state 예측·mixed-precision 동작은 형식적 보장이 아닌 경험적 평가로 다뤄진다는 점을 저자 스스로 명확히 한다. 새 정책이 아니라 배포 인프라 기여로, 정확도 향상보다 systems-side 이득이 핵심이다. flow-matching VLA 실시간 배포에 실용적 가치가 크다.

<!-- VERIFIED: pdf -->
