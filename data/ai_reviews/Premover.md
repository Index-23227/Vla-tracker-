# Premover: Fast Vision-Language-Action Control by Acting Before Instructions Are Complete

> **한 줄 요약**: 사용자가 명령어를 입력하는 동안 idle 상태로 낭비되는 수 초의 시간을 frozen VLA 위의 가벼운 두 projection head로 precomputation에 활용하여, LIBERO에서 wall-clock 13.6% 단축(34.0s→29.4s)에도 baseline과 동등한 95.1% 성공률을 유지하는 streaming-prefix VLA 모듈.

---

## 1. 배경 및 동기

- 현재 VLA 정책은 사용자가 명령을 **완전히 입력한 뒤** 행동을 시작한다고 가정해 평가되지만, 실제 배포에서는 사용자가 입력하는 데 수 초가 소요되어 정책이 **idle 상태**로 남는다.
- 이 idle window를 단순히 흘려보내지 않고 **유용한 precomputation**으로 전환할 수 있다면 latency를 크게 줄일 수 있다는 것이 동기.
- 단순히 부분 prompt에 곧장 행동을 개시하면("naive premoving") 성공률이 무너지는 문제가 있어, **언제 행동을 시작할지** 결정하는 메커니즘이 필요하다.

---

## 2. 핵심 아이디어

- VLA backbone은 **frozen**으로 유지하고 추가 학습 비용 최소화.
- Backbone의 중간 layer를 공유 공간으로 보내는 **두 개의 작은 projection head**:
  - 하나는 **image patch**용, 다른 하나는 **language token**용.
- 두 projection의 결과로 얻는 **focus map**은 **simulator-rendered target-object segmentation mask**로 감독.
- Focus map은 다음 step의 image token을 **per-patch reweighting** 하는 데 사용.
- **단일 scalar readiness threshold**가 streaming prefix들 위에서 함께 학습되어, "지금 행동해도 안전한가?"를 판정.

---

## 3. 방법론 요약

- 학습 시 streaming prefix(부분 명령) 입력들을 시뮬레이션 환경에서 다양하게 생성.
- 각 prefix에서 image / language projection이 만든 focus map을 GT segmentation mask로 supervise → 모듈이 target-object에 주목하도록 유도.
- Readiness scalar는 prefix들과 행동 성공/실패 신호를 결합해 학습되어, 충분히 합리적인 명령 수준에 도달했을 때만 정책을 가동.
- Inference 시: 사용자가 입력하는 동안 backbone은 partial prompt로 focus map을 계속 갱신하다가, readiness scalar가 threshold를 넘으면 본격적으로 action을 출력.

---

## 4. 실험 결과

- **LIBERO benchmark suite**:
  - 평균 wall-clock: **34.0s → 29.4s** (**-13.6%**).
  - 성공률: full-prompt baseline **95.0%** vs Premover **95.1%** → 사실상 동등.
  - **Naive premoving** (Premover의 readiness 메커니즘 없이 그냥 일찍 행동): **66.4%** 로 붕괴.
- LIBERO 하위 suite별(spatial/object/goal/long) 수치는 abstract에 미명시.

---

## 5. 한계 및 의의

- **의의**:
  - VLA latency를 정책 성능 손실 없이 줄이는 **시스템적 관점**의 기여 — 모델 자체보다 **상호작용 타이밍**을 재설계.
  - Backbone을 freeze한 lightweight 모듈이라 다양한 VLA에 쉽게 부착 가능.
  - Naive premoving과의 비교(66.4% vs 95.1%)는 "언제 행동할지"의 readiness 학습이 본질적 contribution임을 명확히 보여줌.
- **한계**:
  - Supervision이 **simulator-rendered mask**에 의존하므로, 실세계 분포로의 transfer가 abstract 단계에서는 검증되지 않음.
  - LIBERO만 평가되어 더 긴 horizon, multi-step instruction에서의 일반화는 미확인.
  - Wall-clock 절약량(약 4.6s)은 명령 입력 시간이 길수록 의미가 큰데, voice 명령 등 다른 입력 modality에서의 ablation은 abstract에 미명시.
- LIBERO에서 **속도-정확도 trade-off를 사실상 깬** 결과로, VLA 실시간 배포 측면에서 실용적 가치가 명확하다.

<!-- VERIFIED: abstract-only -->
