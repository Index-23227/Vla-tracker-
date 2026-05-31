# Pre-VLA: Preemptive Runtime Verification for Reliable Vision-Language-Action and World-Model Rollouts 세미나 리뷰

> **한 줄 요약**: VLA 정책과 world-model rollout 직전에 "이 action chunk를 실제로 실행/상상해도 안전한가?"를 사전 판단하는 경량 dual-branch verifier(Pre-VLA)를 RynnVLA-002 위에 얹어, LIBERO 4개 suite 평균 closed-loop 성공률을 30.79% → 37.62%로 끌어올린 runtime verification 프레임워크.

---

## 1. 배경 및 동기

- 대규모 VLA 모델과 generative world model은 long-horizon embodied intelligence를 빠르게 발전시켰지만, 학습 기반 action 생성에는 본질적인 불확실성이 남는다. 잘못된 action은 실제 실행 시 물리적 실패를 야기하거나, world-model rollout 단계에서 redundant rendering 비용과 함께 잘못된 상상으로 오차를 누적시킨다.
- 따라서 "실행 직전(preemptive)"에 candidate action chunk의 품질을 평가하는 **runtime verification** 계층이 필요하며, 이때 verifier 자체가 가벼우면서도 안전성과 효용을 동시에 판단해야 한다는 것이 Pre-VLA의 출발점이다.

## 2. 핵심 아이디어

- **Preemptive verification**: 정책이 산출한 action chunk를 실행하거나 world model에 넣기 *전에* validity를 평가하는 unified 아키텍처.
- **Dual-branch head**: 하나의 효율적인 multimodal backbone(modality-aware pooling 사용) 위에 (a) safety confidence와 (b) critic-derived advantage score를 동시에 예측하는 가벼운 dual-branch 헤드를 둠.
- **Multi-task objective**: 심각한 class imbalance와 불안정한 boundary를 다루기 위해 Focal classification + advantage regression + soft-threshold calibration을 결합해 학습.
- **Dual-mode preemptive resampling scheduler**: 배포 시 저품질 action을 필터링하고, 제한된 compute budget 안에서 adaptive resampling을 trigger.
- **World-model rollout 보호**: 단순히 실제 로봇 실행만이 아니라, world-model 기반 imagined rollout에서도 누적 오차(error accumulation)를 완화하는 데 사용.

## 3. 방법론 요약

- 기반 정책은 RynnVLA-002이며, Pre-VLA는 그 위에 얹는 verification 모듈이다.
- Backbone은 multimodal observation을 받아 modality-aware pooling으로 표현을 압축하고, 그 위의 dual-branch head가 candidate action chunk에 대해 (1) 안전 여부(분류)와 (2) advantage(회귀)를 예측한다.
- 학습은 안전/비안전 비율의 imbalance와 경계 근처 라벨 불안정성을 가정한 multi-task loss(Focal + advantage regression + soft-threshold calibration)로 수행된다.
- 추론 시에는 dual-mode preemptive resampling scheduler가 동작한다: verifier 점수가 임계 이하면 action을 기각하고 정책에 재샘플링을 요청하되, 전체 compute budget을 초과하지 않도록 조절한다. 평균 forward verification 시간은 action chunk당 **183.9 ms**.

## 4. 실험 결과

- **LIBERO (closed-loop, 4 suites 평균)**: RynnVLA-002 baseline **30.79% → Pre-VLA 37.62%** (절대 +6.83%p 향상). 개별 suite별 점수는 abstract에 미명시.
- **Task execution steps**: 동일 task 완수에 필요한 step 수가 감소(구체 수치 abstract에 미명시).
- **Verification latency**: action chunk당 평균 forward verification time 183.9 ms.
- **World-model rollout**: error accumulation이 완화됨(정량값 abstract에 미명시).
- 다른 benchmark(CALVIN/SimplerEnv 등)와의 비교: abstract에 미명시.

## 5. 한계 및 의의

- **한계**:
  - LIBERO 단일 benchmark에서만 정량 결과가 abstract에 보고되어, generalization은 추가 검증 필요.
  - Per-suite breakdown, 실패 모드 분석, world-model rollout에서의 정량적 error reduction 수치 등이 abstract만으로는 확인 불가.
  - Verifier 자체의 학습/배포 비용(파라미터 수, 학습 데이터 구성)이 abstract에 미명시.
  - Base policy인 RynnVLA-002에 강하게 결합된 결과로 보이며, 다른 VLA backbone(OpenVLA, π₀ 등)에 대한 plug-and-play 효과는 미검증.
- **의의**:
  - VLA를 "단발 정책"이 아니라 "정책 + runtime verifier" 시스템으로 보는 관점은 안전한 실세계 배포에 직접적으로 기여한다.
  - World-model rollout의 hallucination/error accumulation을 사전 필터링하는 setup은 Dreamer 계열 world model과 VLA를 결합하는 최근 흐름에서 실용적 ingredient가 될 수 있다.
  - 약 184 ms의 verification overhead로 LIBERO 평균 성공률을 +6.83%p 끌어올린 점은, 경량 verifier가 비용 대비 매력적인 ingredient임을 시사한다.

<!-- VERIFIED: abstract-only -->
