# PriorVLA: Prior-Preserving Adaptation for Vision-Language-Action Models

> **한 줄 요약**: 사전학습된 VLA의 prior가 full fine-tuning 과정에서 좁은 학습 분포로 끌려가는 문제를 막기 위해, **frozen Prior Expert + 학습 가능한 Adaptation Expert** 이중 전문가 구조와 Expert Query를 도입. full fine-tuning 대비 **25% 파라미터만 갱신**하면서 LIBERO 99.1%, RoboTwin 2.0-Hard에서 pi0.5 대비 +11pt, 10-shot OOD에서 pi0.5 대비 +22pt.

---

## 1. 배경 및 동기

대규모 사전학습된 VLA는 generalist 로봇 조작의 유망한 foundation이지만, 다운스트림 적용 시 **full fine-tuning**이 일반적이며 이는 사전학습을 단순한 초기화로만 사용해 **광범위한 prior를 좁은 학습 분포 패턴으로 이동**시키는 부작용을 낳는다. 그 결과 in-distribution 성능은 올라가도 OOD/few-shot 일반화가 손상된다. PriorVLA는 "prior를 보존하면서도 적응한다"는 명시적 목표 아래, prior를 **읽기 전용 자원**으로 분리하고 그 위에 별도의 적응 모듈을 학습시키는 방향을 택한다.

## 2. 핵심 아이디어

- **Frozen Prior Expert + Adaptation Expert**: 사전학습된 전문가를 동결해 read-only prior로 사용하고, 별도의 Adaptation Expert만 다운스트림용으로 학습.
- **Expert Queries**: 학습 가능한 query가 pretrained VLM으로부터 **scene prior**, Prior Expert로부터 **motor prior**를 각각 수집해 Adaptation Expert로 주입.
- **PEFT 효율**: full fine-tuning이 갱신하는 파라미터의 **25%만 업데이트**.
- **OOD/few-shot 강점**: prior 보존 효과로 OOD와 few-shot 설정에서 가장 큰 이득.
- **다중 검증**: RoboTwin 2.0, LIBERO 시뮬레이터에 더해 8개 real-world task, 2개 embodiment에서 검증.

## 3. 방법론 요약

PriorVLA는 사전학습된 generalist VLA(예: pi0.5)를 그대로 동결한 채 **Prior Expert**로 사용하고, 그 옆에 학습 가능한 **Adaptation Expert**를 둔다. 입력 장면에 대해 사전학습된 VLM이 scene prior를, Prior Expert가 motor prior를 산출하며, **Expert Query**라는 학습 가능한 토큰들이 두 종류의 prior를 추출해 Adaptation Expert에 통합한다. Adaptation Expert는 이 prior 신호를 가이드로 사용해 다운스트림 태스크에 특화된 행동을 학습하므로, prior가 갱신되지 않고도 다운스트림 성능을 끌어올릴 수 있다. 결과적으로 갱신 대상 파라미터가 full fine-tuning의 25%로 줄어들고, prior가 보존되므로 OOD/few-shot에서 일반화가 더 잘 유지된다.

## 4. 실험 결과

- **LIBERO**: 평균 성공률 **99.1%**.
- **RoboTwin 2.0-Hard**: pi0.5 대비 **+11pt** (절대 점수는 abstract에 미명시).
- **Real-world (8 tasks × 2 embodiments, standard data)**:
  - In-distribution: **81%**
  - OOD: **57%**
- **Few-shot (태스크당 10 demos)**:
  - ID: **48%** (pi0.5 대비 **+24pt**)
  - OOD: **32%** (pi0.5 대비 **+22pt**)
- 갱신 파라미터: full fine-tuning의 **25%**.
- 모델 파라미터 수 절대값, 코드 공개 여부는 abstract에 미명시. 프로젝트 페이지(https://priorvla.github.io/)는 존재.

## 5. 한계 및 의의

PriorVLA는 "generalist VLA를 어떻게 downstream에 옮기는가"라는 실용적으로 매우 중요한 질문에 대해, **prior 보존이라는 명시적 목표를 아키텍처 수준에서 강제**한다는 점이 핵심 기여다. 단순한 PEFT(LoRA/Adapter)와 달리, Prior Expert를 별도 분기로 두고 Expert Query로 prior를 명시적으로 추출/주입하는 설계는 OOD/few-shot 성능 향상의 메커니즘을 설명력 있게 만든다. 특히 10-shot OOD에서 pi0.5 대비 +22pt는 실배포에서 데이터 부족 상황의 가치를 직접 보여주는 강한 결과다. 다만 abstract만으로는 (a) Prior Expert가 pi0.5에 한정되는지 다른 backbone에서도 성립하는지, (b) RoboTwin 2.0-Hard 절대 점수, (c) Expert Query 수와 위치에 대한 ablation, (d) 25% 파라미터의 구체적 위치(어느 layer/모듈인지), (e) Prior Expert를 함께 학습시켰을 때와의 정량 비교를 확인할 수 없다. 코드 공개가 abstract에서 확인되지 않아 재현성은 본문/프로젝트 페이지에서 후속 확인이 필요하다.

<!-- VERIFIED: abstract-only -->
