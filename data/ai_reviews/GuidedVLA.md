# GuidedVLA: Specifying Task-Relevant Factors via Plug-and-Play Action Attention Specialization

> **한 줄 요약**: VLA의 action decoder를 단일 학습기가 아닌 기능별 컴포넌트의 조합으로 보고, 개별 attention head를 객체 grounding · 공간 기하 · 시간적 skill 논리 같은 수동 정의 보조 신호로 명시적으로 감독해 spurious correlation을 줄이는 plug-and-play 프레임워크.

---

## 1. 배경 및 동기

- 기존 VLA 모델은 강력한 VLM 위에 action을 하나의 modality로 정렬하지만, 학습은 end-to-end supervision에 의존하여 action decoding이 **task-relevant feature를 암묵적으로**만 학습한다.
- 명시적 가이드가 없는 상태에서는 모델이 시각적 shortcut이나 환경 노이즈 같은 **허위 상관(spurious correlation)** 에 과적합되어 일반화가 제한된다.
- 저자들은 action decoder의 학습 과정 자체에 **task-relevant factor를 명시적으로 주입**할 수 있는지를 묻는다.

---

## 2. 핵심 아이디어

- Action decoder를 **monolithic learner가 아닌 기능 모듈의 조립체**로 재해석.
- 개별 attention head를 사람이 정의한 보조 신호로 감독 → 각 head가 **서로 다른 factor**를 담당.
- 초기 instantiation은 세 가지 specialized head를 사용:
  - **Object grounding** (조작 대상 객체)
  - **Spatial geometry** (공간 기하 관계)
  - **Temporal skill logic** (시간적 skill/sub-task 구조)
- Plug-and-play 설계로 기존 VLA backbone에 부착 가능.
- 학습된 factor 품질과 task 성능 사이의 **양의 상관관계**를 정량적으로 검증한다고 주장.

---

## 3. 방법론 요약

- VLA backbone(VLM + action decoder) 위에서, decoder의 attention head들을 그룹화하고 각 그룹마다 **다른 auxiliary supervision**을 부여.
- Object grounding head는 객체 마스크/위치, spatial geometry head는 공간 관계, temporal skill logic head는 sub-task 진행 정보를 예측하도록 학습.
- 메인 action loss와 보조 supervision loss가 결합되어, decoder가 **decoupled, high-quality feature**를 형성하도록 유도.
- 구체적 backbone, 파라미터 수, head 수 등 구현 세부는 abstract에 미명시.

---

## 4. 실험 결과

- 시뮬레이션 및 실제 로봇 실험에서 강력한 VLA baseline 대비 **in-domain / out-of-domain 모두에서 성공률 향상**.
- Specialized factor의 품질이 task 성능과 양의 상관을 가짐을 보임.
- LIBERO/CALVIN 등 구체적 벤치마크 수치는 abstract에 미명시.

---

## 5. 한계 및 의의

- **의의**: VLA action decoder에 **명시적 inductive bias**를 부여하는 일반화 가능한 패러다임을 제시. 세 head는 초기 instantiation일 뿐이며, 다른 factor(force, contact, affordance 등)로 확장 가능.
- **한계**:
  - Auxiliary supervision 신호(객체 마스크, 공간 관계 라벨 등)를 어떻게 확보할지에 대한 **데이터 비용** 문제가 남는다 (abstract에서는 simulation 기반으로 보임).
  - 수동 정의 factor에 의존하므로, **사람이 미처 고려하지 못한 factor**가 중요한 task에서는 효과가 제한될 수 있음.
  - 구체적 수치, 모델 크기, 학습 코스트가 abstract에 공개되지 않아 실용성 평가 보류.
- **위상**: RSS 2026 채택. "VLA decoder의 attention head를 modular하게 감독한다"는 관점은 향후 interpretable / controllable VLA 연구의 baseline으로 자리잡을 가능성이 있다.

<!-- VERIFIED: abstract-only -->
