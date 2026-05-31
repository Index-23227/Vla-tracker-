# IntentVLA: Short-Horizon Intent Modeling for Aliased Robot Manipulation 세미나 리뷰

> **한 줄 요약**: 동일한 visual-language 관측이라도 인간 시연자의 "단기 의도(short-horizon intent)"에 따라 후속 action chunk가 달라지는 multimodal imitation 데이터의 본질에 주목해, 최근 관측 history를 압축한 intent representation으로 chunk 생성을 conditioning하는 IntentVLA. 또한 RoboTwin2 위에 ambiguity-aware 12-task benchmark인 AliasBench를 함께 제안한다.

---

## 1. 배경 및 동기

- 로봇 imitation 데이터는 본질적으로 multimodal이다: 비슷한 visual-language 관측 뒤에 서로 다른 action chunk가 따라오는 경우가 흔한데, 이는 시연자의 단기 의도, task phase, 최근 context에 따라 행동이 달라지기 때문이다.
- 기존 **frame-conditioned VLA 정책**은 현재 관측과 instruction만으로 각 chunk를 추론하므로, partial observability 하에서 인접한 replanning step마다 서로 다른 의도를 resampling하게 되고, 이로 인해 **inter-chunk conflict**와 불안정한 실행이 발생한다는 것이 문제 의식이다.

## 2. 핵심 아이디어

- **History-conditioned VLA**: 최근 visual 관측들을 압축한 **short-horizon intent representation**을 별도로 인코딩하고, 이를 chunk 생성의 condition으로 사용.
- **Intent를 명시적으로 한정**: long-horizon goal이 아니라 의도적으로 **short-horizon**으로 한정해, 의도 추정 자체가 또 다른 long-horizon 문제로 발산하지 않도록 함.
- **AliasBench 제안**: RoboTwin2 위에 구축한 **12-task ambiguity-aware benchmark**. 학습 데이터와 평가 환경을 matched 시켜, 순수하게 **short-horizon observation aliasing**(같은 관측, 다른 의도) 효과만 isolate해 측정.
- **다중 benchmark 검증**: AliasBench, SimplerEnv, LIBERO, RoboCasa 전반에서 rollout stability 향상과 강력한 VLA baseline 대비 성능 우위를 주장.

## 3. 방법론 요약

- 정책은 (current observation + instruction)만이 아니라, 최근 visual frame들을 입력으로 받아 **compact intent vector**를 생성하는 history encoder를 함께 사용한다.
- 이 intent vector는 action chunk decoder의 conditioning input으로 들어가, 인접 chunk가 같은 의도를 공유하도록 유도한다 → 결과적으로 inter-chunk conflict 감소.
- AliasBench는 RoboTwin2 환경에서 12개의 task를 구성하되, 동일한 관측이 서로 다른 의도/단기 목표에 대응하는 sample을 의도적으로 포함시켜 aliasing을 stress-test한다.
- 코드는 https://github.com/ZGC-EmbodyAI/IntentVLA 에 공개되어 있어, AliasBench 셋업과 학습 파이프라인을 재현할 수 있을 것으로 기대된다.

## 4. 실험 결과

- **AliasBench (RoboTwin2 기반, 12 tasks)**: IntentVLA가 강력한 VLA baseline을 능가(구체 수치는 abstract에 미명시).
- **SimplerEnv**: rollout stability 향상 및 baseline 대비 성능 우위(구체 점수 abstract에 미명시).
- **LIBERO**: IntentVLA가 baseline을 outperform(per-suite 점수 abstract에 미명시).
- **RoboCasa**: baseline 대비 성능 우위(구체 점수 abstract에 미명시).
- 정량 점수는 모두 abstract에 미명시이며, 본문/표 확인이 필요하다.

## 5. 한계 및 의의

- **한계**:
  - Abstract 단계에서는 four-benchmark에서의 정량적 비교 수치, baseline 정체(OpenVLA / π₀ / RoboTwin 자체 baseline 등)가 명확히 드러나지 않는다.
  - "Short-horizon intent"의 길이/세그먼테이션 정의, history window 크기, intent representation의 supervision 방식 등 구현 세부는 본문 확인이 필요.
  - AliasBench는 저자들이 직접 제안한 benchmark이므로, 외부 모델로의 일반화와 over-fitting 가능성에 대한 sanity check가 필요하다.
  - intent를 단기로 한정한 설계는 long-horizon planning이 중요한 task(예: 다단계 조립)에서는 별도의 hierarchical layer가 필요할 수 있다.
- **의의**:
  - "VLA imitation data는 본질적으로 multimodal"이라는 관찰을 매우 명확하게 정형화하고, observation aliasing을 isolate해 측정 가능한 benchmark(AliasBench)로 만든 점은 평가 인프라 측면에서 중요한 기여다.
  - History-conditioning과 intent representation은 다른 VLA 백본(OpenVLA, RDT, π₀)에도 비교적 쉽게 plug-in 될 수 있는 컴포넌트이며, 일반화 잠재력이 크다.
  - 공식 GitHub 공개(`ZGC-EmbodyAI/IntentVLA`)로 인해 RoboTwin2 + AliasBench 평가 셋업이 커뮤니티 표준으로 수용될 가능성이 있다.

<!-- VERIFIED: abstract-only -->
